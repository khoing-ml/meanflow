import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch
from diffusers.models import FlaxAutoencoderKL
from optax._src.alias import *

NUM_CLASSES = 1000


def create_imagenet_dataloader(imagenet_root, split, batch_size, image_size, num_workers=4, for_fid=False, hf_dataset_name=None):
    """Create ImageNet dataloader for the specified split.
    
    Args:
        imagenet_root: Path to ImageNet dataset (ignored if hf_dataset_name is provided)
        split: Dataset split ('train', 'val', 'validation', etc.)
        batch_size: Batch size for dataloader
        image_size: Target image size
        num_workers: Number of dataloader workers
        for_fid: Whether to prepare data for FID computation
        hf_dataset_name: HuggingFace dataset name (e.g., 'imagenet-1k'). If provided, loads from HF Hub.
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torchvision import datasets, transforms

    from utils.input_pipeline import center_crop_arr, loader, worker_init_fn
    from utils.logging_util import log_for_0
    
    if for_fid:
        # For FID: only center crop, return numpy array directly
        def fid_transform(pil_image):
            cropped = center_crop_arr(pil_image, image_size)
            return np.array(cropped)  # PIL -> numpy [0,255] uint8
        transform = fid_transform
    else:
        # For latent computation: full preprocessing pipeline
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    
    if hf_dataset_name:
        # Load from HuggingFace Hub
        from datasets import load_dataset
        from PIL import Image
        from io import BytesIO
        
        log_for_0(f"Loading dataset from HuggingFace: {hf_dataset_name}, split={split}")
        
        # Map common split names (val -> validation for HF)
        hf_split = 'validation' if split == 'val' else split
        
        # Load the dataset from HuggingFace
        hf_dataset = load_dataset(hf_dataset_name, split=hf_split, trust_remote_code=True)
        log_for_0(f"Loaded {len(hf_dataset)} samples from HuggingFace")
        
        # Create a wrapper to apply transforms
        class HFDatasetWrapper(torch.utils.data.Dataset):
            def __init__(self, hf_dataset, transform, image_column='image', label_column='label'):
                self.hf_dataset = hf_dataset
                self.transform = transform
                self.image_column = image_column
                self.label_column = label_column
            
            def __len__(self):
                return len(self.hf_dataset)
            
            def __getitem__(self, idx):
                item = self.hf_dataset[idx]
                
                # Get image (could be bytes, PIL, or dict)
                img = item[self.image_column]
                if isinstance(img, bytes):
                    img = Image.open(BytesIO(img)).convert('RGB')
                elif isinstance(img, dict) and 'bytes' in img:
                    img = Image.open(BytesIO(img['bytes'])).convert('RGB')
                elif not isinstance(img, Image.Image):
                    img = Image.fromarray(img).convert('RGB')
                
                # Apply transform
                if self.transform:
                    img = self.transform(img)
                
                # Get label
                label = item[self.label_column]
                
                return img, label
        
        dataset = HFDatasetWrapper(hf_dataset, transform)
    else:
        # Load from local ImageFolder structure
        dataset = datasets.ImageFolder(
            os.path.join(imagenet_root, split),
            transform=transform,
            loader=loader,
        )
    
    log_for_0(f"Dataset {split} (FID={for_fid}): {dataset}")
    
    rank = jax.process_index()
    num_replicas = jax.process_count()
    log_for_0(f"Distributed setup: rank={rank}, num_replicas={num_replicas}")
    log_for_0(f"JAX devices: {jax.devices()}")
    log_for_0(f"JAX local devices: {jax.local_devices()}")
    
    # Check distributed setup
    if num_replicas == 1:
        log_for_0("WARNING: Only 1 process detected - running in single-worker mode!")
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=False,
    )
    
    log_for_0(f"DistributedSampler: total_samples={len(dataset)}, samples_per_replica={len(sampler)}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Return the per-worker dataset size (distributed size) and total dataset size
    return dataloader, len(sampler), len(dataset)


def prepare_batch_data_encode(batch):
    """Prepare batch data for VAE encoding."""
    image, label = batch
    
    # Convert BCHW -> BHWC
    image = image.permute(0, 2, 3, 1)  # BCHW -> BHWC
    
    # Pad batch to be divisible by local device count
    local_device_count = jax.local_device_count()
    batch_size = image.shape[0]
    
    if batch_size % local_device_count != 0:
        # Pad the batch to make it divisible by local_device_count
        pad_size = local_device_count - (batch_size % local_device_count)
        
        # Pad with zeros (or repeat last sample)
        image_pad = torch.zeros((pad_size,) + image.shape[1:], dtype=image.dtype)
        label_pad = torch.zeros(pad_size, dtype=label.dtype)
        
        image = torch.cat([image, image_pad], dim=0)
        label = torch.cat([label, label_pad], dim=0)
        
        # Store original batch size for later use
        original_batch_size = batch_size
    else:
        original_batch_size = batch_size
    
    # Reshape to (local_devices, device_batch_size, height, width, channels)
    image = image.reshape((local_device_count, -1) + image.shape[1:])
    label = label.reshape(local_device_count, -1)
    
    return {
        'image': image.numpy(),
        'label': label.numpy(),
        'original_batch_size': original_batch_size,
    }


def compute_latent_dataset(imagenet_root, output_dir, vae_type, batch_size, image_size, overwrite=False, hf_dataset_name=None, splits=None):
    
    from tqdm import tqdm

    from utils.logging_util import log_for_0
    
    log_for_0("Starting latent dataset computation...")
    
    # Determine splits to process
    if splits is None:
        if hf_dataset_name:
            # For HuggingFace datasets, try to process all available splits
            splits = ['train', 'validation', 'test']
        else:
            # For local datasets, default to train split only
            splits = ['train']
    
    log_for_0(f"Splits to process: {splits}")
    
    # Calculate latent size from image size (VAE downsampling factor is 8)
    if image_size % 8 != 0:
        raise ValueError(f"Image size {image_size} must be divisible by 8 for VAE encoding")
    
    latent_size = image_size // 8
    log_for_0(f"Image size: {image_size} -> Latent size: {latent_size}x{latent_size}")
    
    # Initialize VAE
    log_for_0(f"Loading VAE model: sd-vae-ft-{vae_type}-flax")
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(f"pcuenq/sd-vae-ft-{vae_type}-flax") 
    
    # Create encode function
    def encode_fn(vae_model, vae_params, batch):
        latent_dist = vae_model.apply(
            {'params': vae_params}, 
            jnp.transpose(batch, (0, 3, 1, 2)),  # BHWC -> BCHW
            method=FlaxAutoencoderKL.encode
        ).latent_dist
        return jnp.concatenate((latent_dist.mean, latent_dist.std), axis=-1)
    
    p_encode_fn = jax.pmap(
        partial(encode_fn, vae, vae_params),
        axis_name='batch',
    )
    
    for split in splits:
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Check if already exists and not overwriting
        if not overwrite and os.path.exists(split_output_dir) and os.listdir(split_output_dir):
            log_for_0(f"Split {split} already exists, skipping...")
            continue
            
        log_for_0(f"Processing {split} split...")
        
        # Create dataloader
        dataloader, dataset_size, total_samples = create_imagenet_dataloader(
            imagenet_root, split, batch_size, image_size, hf_dataset_name=hf_dataset_name
        )
        
        # Calculate starting index for this worker to avoid filename conflicts
        rank = jax.process_index()
        num_replicas = jax.process_count()
        
        # Calculate the starting index based on distributed sampling logic
        # DistributedSampler distributes samples evenly across workers
        samples_per_worker = total_samples // num_replicas
        remainder = total_samples % num_replicas
        
        # Workers with rank < remainder get one extra sample
        if rank < remainder:
            start_idx = rank * (samples_per_worker + 1)
        else:
            start_idx = rank * samples_per_worker + remainder
        
        log_for_0(f"Worker {rank}: processing {dataset_size} samples, starting from index {start_idx}")
        log_for_0(f"Total samples: {total_samples}, samples per worker: {samples_per_worker}, remainder: {remainder}")
        
        # Process batches
        sample_idx = start_idx
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {split}")):
            # Prepare batch
            batch_data = prepare_batch_data_encode(batch)
            
            # Encode to latents
            latents = p_encode_fn(batch_data['image']).reshape(-1, latent_size, latent_size, 8)
            labels = batch_data['label'].reshape(-1)
            
            # Only process the original batch size (ignore padded samples)
            original_batch_size = batch_data['original_batch_size']
            
            # Save individual samples
            for i in range(original_batch_size):
                if sample_idx >= start_idx + dataset_size:
                    break
                    
                # Transpose latent to match expected shape (B, C, H, W)
                latent = torch.tensor(np.array(latents[i]))
                latent = latent.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                
                sample_data = {
                    'image': latent,
                    'label': torch.tensor(np.array(labels[i]))
                }
                
                filename = f"{sample_idx:08d}.pt"
                filepath = os.path.join(split_output_dir, filename)
                torch.save(sample_data, filepath)
                
                sample_idx += 1
            
            if batch_idx % 100 == 0:
                log_for_0(f"Progress {split}: {batch_idx} batches, {sample_idx} samples")
        
        log_for_0(f"Completed {split} split: {sample_idx} samples saved")