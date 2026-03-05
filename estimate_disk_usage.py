#!/usr/bin/env python3
"""
Script to estimate disk usage for prepare_data.sh and launch.sh

This script estimates:
1. Data preparation disk usage (latent dataset)
2. Training disk usage (checkpoints, logs)
"""

import os
import argparse
from pathlib import Path

# ImageNet dataset statistics
IMAGENET_STATS = {
    'train': 1281167,      # Training set size
    'validation': 50000,    # Validation set size
}

def estimate_latent_size_per_image(image_size):
    """
    Estimate size of one latent representation file.
    
    VAE reduces image by factor of 8, so:
    - 256x256 image -> 32x32 latents
    - 512x512 image -> 64x64 latents
    - 1024x1024 image -> 128x128 latents
    
    Each latent contains: mean + std (both 8 channels)
    Total channels: 8 (for mean) + 8 (for std) = 16 channels
    
    Each .pt file contains: dict with 'image' tensor + 'label'
    - image tensor: (16, latent_h, latent_w) as float32
    - label: int64
    - PyTorch overhead: ~400 bytes per file
    """
    latent_size = image_size // 8
    channels = 16  # mean (8 channels) + std (8 channels)
    
    # Float32: 4 bytes per value
    latent_bytes = channels * latent_size * latent_size * 4
    
    # Add overhead for .pt file format (~400 bytes)
    overhead = 400
    
    return latent_bytes + overhead


def estimate_prepare_data_disk_usage(image_size=256, splits=['train', 'validation']):
    """
    Estimate disk usage for prepare_data.sh
    
    Args:
        image_size: Image size (256, 512, or 1024)
        splits: List of splits to process
    
    Returns:
        Dictionary with size estimates in GB
    """
    print("=" * 70)
    print("PREPARE_DATA.SH DISK USAGE ESTIMATION")
    print("=" * 70)
    
    latent_size_per_image = estimate_latent_size_per_image(image_size)
    
    total_bytes = 0
    split_sizes = {}
    
    for split in splits:
        if split not in IMAGENET_STATS:
            print(f"Warning: Unknown split '{split}'. Skipping.")
            continue
            
        num_images = IMAGENET_STATS[split]
        split_size_bytes = num_images * latent_size_per_image
        split_size_gb = split_size_bytes / (1024**3)
        
        split_sizes[split] = split_size_gb
        total_bytes += split_size_bytes
        
        latent_dim = image_size // 8
        print(f"\n{split.upper()} split:")
        print(f"  - Number of images: {num_images:,}")
        print(f"  - Latent size: {latent_dim}x{latent_dim}")
        print(f"  - Channels: 16 (8 mean + 8 std)")
        print(f"  - Size per image: {latent_size_per_image:,} bytes (~{latent_size_per_image/1024:.2f} KB)")
        print(f"  - Total: {split_size_gb:.2f} GB")
    
    total_gb = total_bytes / (1024**3)
    
    print("\n" + "-" * 70)
    print(f"TOTAL LATENT DATASET SIZE: {total_gb:.2f} GB")
    print("-" * 70)
    
    return {
        'total_gb': total_gb,
        'split_sizes': split_sizes,
        'latent_size_per_image': latent_size_per_image,
        'image_size': image_size
    }


def estimate_training_disk_usage(
    max_steps=500000,
    checkpoint_interval=5000,
    model_size='medium'
):
    """
    Estimate disk usage for launch.sh (training)
    
    Args:
        max_steps: Maximum training steps
        checkpoint_interval: Steps between checkpoints
        model_size: 'small', 'medium', or 'large'
    
    Returns:
        Dictionary with size estimates in GB
    """
    print("\n" + "=" * 70)
    print("LAUNCH.SH DISK USAGE ESTIMATION (Training)")
    print("=" * 70)
    
    # Model checkpoint sizes (estimates for different model sizes)
    # These include model weights + optimizer state
    checkpoint_sizes_gb = {
        'small': 1.5,    # ~Small DiT (maybe 50-100M params)
        'medium': 4.0,   # ~Medium DiT (maybe 300-500M params)
        'large': 8.0,    # ~Large DiT (maybe 1B+ params)
    }
    
    if model_size not in checkpoint_sizes_gb:
        print(f"Warning: Unknown model size '{model_size}'. Using 'medium'")
        model_size = 'medium'
    
    checkpoint_size = checkpoint_sizes_gb[model_size]
    num_checkpoints = max_steps // checkpoint_interval
    
    # Training outputs
    total_checkpoint_size = checkpoint_size * num_checkpoints
    
    # Logs and outputs (typically small)
    logs_size_gb = 0.1
    output_size_gb = 0.5
    
    total_training_gb = total_checkpoint_size + logs_size_gb + output_size_gb
    
    print(f"\nModel size: {model_size}")
    print(f"  - Checkpoint size: {checkpoint_size:.2f} GB")
    print(f"  - Number of checkpoints: {num_checkpoints}")
    print(f"  - Total checkpoint storage: {total_checkpoint_size:.2f} GB")
    print(f"\nLogs and outputs:")
    print(f"  - Logs: ~{logs_size_gb:.2f} GB")
    print(f"  - Output samples: ~{output_size_gb:.2f} GB")
    
    print("\n" + "-" * 70)
    print(f"TOTAL TRAINING SIZE: {total_training_gb:.2f} GB")
    print("-" * 70)
    
    return {
        'total_gb': total_training_gb,
        'checkpoint_total_gb': total_checkpoint_size,
        'logs_gb': logs_size_gb,
        'output_gb': output_size_gb,
        'num_checkpoints': num_checkpoints,
        'checkpoint_size_gb': checkpoint_size,
        'model_size': model_size
    }


def print_summary(data_info, training_info):
    """Print comprehensive summary"""
    print("\n" + "=" * 70)
    print("TOTAL DISK USAGE SUMMARY")
    print("=" * 70)
    
    total = data_info['total_gb'] + training_info['total_gb']
    
    print(f"\n1. Latent Dataset (prepare_data.sh):")
    print(f"   Image size: {data_info['image_size']}x{data_info['image_size']}")
    for split, size in data_info['split_sizes'].items():
        print(f"   - {split}: {size:.2f} GB")
    print(f"   Subtotal: {data_info['total_gb']:.2f} GB")
    
    print(f"\n2. Training Output (launch.sh):")
    print(f"   Model size: {training_info['model_size']}")
    print(f"   - Checkpoints: {training_info['checkpoint_total_gb']:.2f} GB ({training_info['num_checkpoints']} checkpoints)")
    print(f"   - Logs: {training_info['logs_gb']:.2f} GB")
    print(f"   - Outputs: {training_info['output_gb']:.2f} GB")
    print(f"   Subtotal: {training_info['total_gb']:.2f} GB")
    
    print(f"\n{'GRAND TOTAL: ' + str(round(total, 2)) + ' GB':^70}")
    print("=" * 70)
    
    # Show what you can fit in common storage sizes
    print("\nStorage recommendations:")
    storage_options = [256, 512, 1024, 2048]
    for storage_gb in storage_options:
        if total <= storage_gb:
            print(f"  ✓ {storage_gb} GB SSD is sufficient (with {storage_gb - total:.1f} GB margin)")
        else:
            print(f"  ✗ {storage_gb} GB SSD is NOT sufficient (need {total - storage_gb:.1f} GB more)")


def main():
    parser = argparse.ArgumentParser(description='Estimate disk usage for prepare.sh and launch.sh')
    parser.add_argument('--image-size', type=int, default=256, 
                       choices=[256, 512, 1024],
                       help='Image size for processing (default: 256)')
    parser.add_argument('--splits', type=str, default='train,validation',
                       help='Splits to process (comma-separated, default: train,validation)')
    parser.add_argument('--max-steps', type=int, default=500000,
                       help='Maximum training steps (default: 500000)')
    parser.add_argument('--checkpoint-interval', type=int, default=5000,
                       help='Steps between checkpoints (default: 5000)')
    parser.add_argument('--model-size', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Model size estimate (default: medium)')
    
    args = parser.parse_args()
    
    # Parse splits
    splits = [s.strip() for s in args.splits.split(',')]
    
    # Run estimations
    data_info = estimate_prepare_data_disk_usage(args.image_size, splits)
    training_info = estimate_training_disk_usage(
        args.max_steps,
        args.checkpoint_interval,
        args.model_size
    )
    
    # Print summary
    print_summary(data_info, training_info)


if __name__ == '__main__':
    main()
