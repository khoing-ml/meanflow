"""Wandb utilities for experiment tracking.

Adapted from https://github.com/kvfrans/shortcut-models and https://github.com/dibyaghosh/jaxrl_m
"""
import datetime
import os
import tempfile
import time

import jax
import ml_collections
import numpy as np
import wandb
from absl import flags


def _to_numpy(value):
    """Convert JAX/array-like values to host NumPy arrays when possible."""
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(jax.device_get(value))
    except Exception:
        return value


def get_flag_dict():
    """Extract all flags as a dictionary."""
    flag_dict = {}
    try:
        flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
        for k in flag_dict:
            if isinstance(flag_dict[k], ml_collections.ConfigDict):
                flag_dict[k] = flag_dict[k].to_dict()
    except:
        pass  # If no flags are defined
    return flag_dict


def default_wandb_config():
    """Get default wandb configuration."""
    config = ml_collections.ConfigDict()
    config.offline = False  # Syncs online or not?
    config.project = "meanflow"  # WandB Project Name
    config.entity = None  # Which entity to log as (default: your own user)
    config.group = None  # Group name for organizing runs
    config.name = None  # Run name (will be formatted with config values)
    config.run_id = None  # For resuming a run
    config.tags = []  # List of tags
    config.notes = ""  # Notes about the run
    config.mode = "online"  # online, offline, or disabled
    return config


def setup_wandb(
    config_dict,
    wandb_config=None,
    entity=None,
    project="meanflow",
    group=None,
    name=None,
    run_id=None,
    tags=None,
    notes="",
    offline=False,
    mode="online",
    **additional_init_kwargs
):
    """Initialize wandb with proper configuration.
    
    Args:
        config_dict: Dictionary of hyperparameters to log
        wandb_config: WandB config dict (if using ml_collections)
        entity: WandB entity name
        project: WandB project name
        group: Group name for organizing runs
        name: Run name
        run_id: ID for resuming a run
        tags: List of tags for the run
        notes: Notes about the run
        offline: Whether to run in offline mode
        mode: online, offline, or disabled
        **additional_init_kwargs: Additional kwargs for wandb.init
    
    Returns:
        wandb.Run object
    """
    # Override with wandb_config if provided
    if wandb_config is not None:
        entity = wandb_config.get('entity', entity)
        project = wandb_config.get('project', project)
        group = wandb_config.get('group', group)
        name = wandb_config.get('name', name)
        run_id = wandb_config.get('run_id', run_id)
        tags = wandb_config.get('tags', tags) or []
        notes = wandb_config.get('notes', notes)
        offline = wandb_config.get('offline', offline)
        mode = wandb_config.get('mode', mode)
    
    if offline:
        mode = "offline"
    
    # Generate unique identifier
    unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_identifier += f"_{np.random.randint(0, 1000000):06d}"
    
    # Format name with config values if needed
    if name is not None and '{' in name:
        try:
            name = name.format(**config_dict)
        except KeyError:
            pass  # Keep original name if formatting fails
    
    # Create experiment ID
    if name is not None:
        experiment_id = f"{name}_{unique_identifier}"
    else:
        experiment_id = unique_identifier
    
    # Set wandb directory
    if os.path.exists("/kaggle/working"):
        wandb_output_dir = "/kaggle/working/wandb"
        os.makedirs(wandb_output_dir, exist_ok=True)
    elif os.path.exists("/tmp"):
        wandb_output_dir = "/tmp/wandb"
        os.makedirs(wandb_output_dir, exist_ok=True)
    else:
        wandb_output_dir = tempfile.mkdtemp()
    
    # Prepare init kwargs
    init_kwargs = dict(
        config=config_dict,
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        id=experiment_id if run_id is None else run_id,
        name=name,
        notes=notes,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
        resume="allow" if run_id is None else "must",
    )
    init_kwargs.update(additional_init_kwargs)
    
    # Initialize wandb
    run = wandb.init(**init_kwargs)
    
    # Update config with flags if available
    flag_dict = get_flag_dict()
    if flag_dict:
        wandb.config.update(flag_dict, allow_val_change=True)
    
    return run


def log_metrics(metrics_dict, step=None, prefix=None, commit=True):
    """Log metrics to wandb.
    
    Args:
        metrics_dict: Dictionary of metrics to log
        step: Training step
        prefix: Prefix to add to all metric names (e.g., 'train/', 'valid/')
        commit: Whether to commit the metrics immediately
    """
    if not wandb.run:
        return
    
    # Add prefix if provided
    if prefix:
        metrics_dict = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}
    
    wandb.log(metrics_dict, step=step, commit=commit)


def log_images(images_dict, step=None, prefix=None, commit=True):
    """Log images to wandb.
    
    Args:
        images_dict: Dictionary mapping names to images (numpy arrays or PIL Images)
        step: Training step
        prefix: Prefix to add to all image names
        commit: Whether to commit immediately
    """
    if not wandb.run:
        return
    
    # Convert to wandb.Image objects
    wandb_images = {}
    for name, img in images_dict.items():
        key = f"{prefix}/{name}" if prefix else name
        wandb_images[key] = wandb.Image(_to_numpy(img))
    
    wandb.log(wandb_images, step=step, commit=commit)


def finish_wandb():
    """Finish the wandb run."""
    if wandb.run:
        wandb.finish()


def log_histograms(data_dict, step=None, prefix=None):
    """Log histograms to wandb.
    
    Args:
        data_dict: Dictionary mapping names to arrays
        step: Training step number
        prefix: Prefix to add to histogram names
    """
    if not wandb.run:
        return
    
    wandb_histograms = {}
    for name, data in data_dict.items():
        key = f"{prefix}/{name}" if prefix else name
        # Convert to host NumPy array before creating histogram.
        data = _to_numpy(data)
        if hasattr(data, '__array__'):
            data = np.asarray(data).flatten()
        wandb_histograms[key] = wandb.Histogram(data)
    
    wandb.log(wandb_histograms, step=step)


def log_gradients(grads, step=None):
    """Log gradient statistics to wandb.
    
    Args:
        grads: Gradient pytree
        step: Training step number
    """
    if not wandb.run:
        return
    
    # Flatten gradients to get statistics per layer
    grad_dict = {}
    for key_path, value in jax.tree_util.tree_flatten_with_path(grads)[0]:
        # Convert key path to string
        key_str = '/'.join(str(k.key) for k in key_path)
        if hasattr(value, 'shape') and value.size > 0:
            grad_dict[f"{key_str}/mean"] = float(np.mean(np.abs(value)))
            grad_dict[f"{key_str}/max"] = float(np.max(np.abs(value)))
            grad_dict[f"{key_str}/std"] = float(np.std(value))
    
    log_metrics(grad_dict, step=step, prefix='grads', commit=False)


def log_parameters(params, step=None):
    """Log parameter statistics to wandb.
    
    Args:
        params: Parameter pytree
        step: Training step number
    """
    if not wandb.run:
        return
    
    # Flatten parameters to get statistics per layer
    param_dict = {}
    for key_path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
        # Convert key path to string
        key_str = '/'.join(str(k.key) for k in key_path)
        if hasattr(value, 'shape') and value.size > 0:
            param_dict[f"{key_str}/mean"] = float(np.mean(np.abs(value)))
            param_dict[f"{key_str}/max"] = float(np.max(np.abs(value)))
            param_dict[f"{key_str}/std"] = float(np.std(value))
    
    log_metrics(param_dict, step=step, prefix='params', commit=False)


def watch_model(model, log_freq=1000):
    """Watch model parameters and gradients (similar to wandb.watch).
    
    Args:
        model: Flax model
        log_freq: How often to log histograms
    """
    if not wandb.run:
        return
    
    # This is a placeholder - actual implementation would integrate with training loop
    # In practice, use log_gradients() and log_parameters() in the training step
    pass
