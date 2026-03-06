# Weights & Biases (wandb) Integration Guide

This document explains the comprehensive wandb integration in the meanflow repository, adapted from best practices in [kvfrans/shortcut-models](https://github.com/kvfrans/shortcut-models).

## Features

### 1. Automatic Experiment Tracking
- **Run Management**: Automatic unique run IDs with timestamps
- **Resume Capability**: Resume interrupted training runs
- **Config Logging**: All hyperparameters automatically logged
- **System Info**: JAX version, device count, process info

### 2. Comprehensive Metrics Logging

#### Training Metrics (logged every `log_per_step` steps)
- **Loss Metrics** (prefix: `train/`)
  - `loss`: Total training loss
  - `v_loss`: Velocity loss (monitoring only)
  
- **Optimization Metrics** (prefix: `optim/`)
  - `lr`: Current learning rate
  - `ema_decay`: EMA decay value
  
- **Gradient & Parameter Norms** (prefix: `norms/`)
  - `grad_norm`: Global gradient norm (L2)
  - `param_norm`: Global parameter norm (L2)
  - `update_norm`: Global update norm (L2)
  
- **Performance Metrics** (prefix: `perf/`)
  - `steps_per_second`: Training throughput
  
- **Metadata** (prefix: `meta/`)
  - `epoch`: Current epoch number

#### Evaluation Metrics
- **FID Scores** (prefix: `eval/`)
  - `fid_ema`: FID score using EMA parameters
  - `num_samples`: Number of samples used for FID calculation

#### Sample Statistics
- **Generated Sample Stats** (prefix: `sample_stats/`)
  - `mean`: Mean pixel value
  - `std`: Standard deviation
  - `min`: Minimum pixel value
  - `max`: Maximum pixel value

### 3. Image Logging

#### Training Samples (prefix: `samples/`)
- `generated_samples`: Grid visualization of generated images
- Logged every `sample_per_epoch` epochs

#### Evaluation Samples (prefix: `eval/`)
- `fid_samples`: Grid of samples used in FID computation
- Logged during FID evaluation

### 4. Configuration

#### In `configs/default.py`:
```python
config.wandb = wandb = ml_collections.ConfigDict()
wandb.project = 'meanflow'              # Project name
wandb.entity = None                      # Your username/team
wandb.group = None                       # Group related runs
wandb.name = 'meanflow_{dataset}_{model}' # Run name (supports formatting)
wandb.run_id = None                     # For resuming runs
wandb.tags = []                         # Tags for organization
wandb.notes = ''                        # Run description
wandb.mode = 'online'                   # online, offline, or disabled
wandb.log_model = False                 # Save model checkpoints to wandb
```

#### In YAML configs (e.g., `configs/run_b4.yml`):
```yaml
wandb:
    project: 'meanflow'
    entity: null  # Set to your wandb username
    group: 'imagenet-dit-b4'
    name: 'dit_b4_bs256_lr1e-4'
    run_id: null
    tags: ['dit-b4', 'imagenet', 'meanflow']
    notes: 'DiT-B/4 training on ImageNet latents'
    mode: 'online'
    log_model: False
```

## Usage

### Basic Training with Wandb
```bash
python main.py --config=configs/run_b4.yml --workdir=/path/to/logs
```

### Disable Wandb
```bash
# Method 1: Set mode in config
wandb:
    mode: 'disabled'

# Method 2: Use environment variable
WANDB_MODE=disabled python main.py --config=...

# Method 3: Set offline mode
wandb:
    mode: 'offline'
```

### Resume a Run
```bash
# Get the run_id from wandb (looks like: 'dit_b4_bs256_lr1e-4_20260305_123456_123456')
# Update config:
wandb:
    run_id: 'your-run-id-here'

# Then run training
python main.py --config=configs/run_b4.yml --workdir=/path/to/logs
```

### Organize with Groups and Tags
```yaml
wandb:
    group: 'ablation-study'  # Groups related experiments
    tags: ['dit-b4', 'ablation', 'lr-sweep']
    notes: 'Testing learning rate of 1e-4'
```

## Metric Organization

All metrics are organized with prefixes for easy filtering in wandb UI:

```
train/
├── loss
└── v_loss

norms/
├── grad_norm
├── param_norm
└── update_norm

optim/
├── lr
└── ema_decay

perf/
└── steps_per_second

eval/
├── fid_ema
└── num_samples

samples/
└── generated_samples

sample_stats/
├── mean
├── std
├── min
└── max

meta/
└── epoch
```

## Best Practices

### 1. Always Check Process Index
Only rank 0 process logs to wandb:
```python
if jax.process_index() == 0:
    log_metrics(metrics, step=step)
```

### 2. Convert JAX Arrays to NumPy
```python
metrics = jax.tree_util.tree_map(lambda x: float(x.mean()), metrics)
```

### 3. Use Descriptive Names
```yaml
name: 'dit_{model_size}_bs{batch_size}_lr{learning_rate}'
```

### 4. Tag Your Experiments
```yaml
tags: ['baseline', 'dit-xl', 'imagenet-256']
```

### 5. Add Notes
```yaml
notes: 'Baseline run with standard hyperparameters'
```

## Advanced Features

### Custom Metric Logging
```python
from utils.wandb_util import log_metrics, log_images

# Log custom metrics
custom_metrics = {
    'custom_loss': loss_value,
    'custom_accuracy': acc_value,
}
log_metrics(custom_metrics, step=step, prefix='custom')

# Log custom images
custom_images = {
    'comparison': comparison_grid,
    'attention_map': attention_vis,
}
log_images(custom_images, step=step, prefix='visualizations')
```

### System Information
Automatically logged:
- JAX version
- Total device count
- Number of processes
- Local device count

### Metric Filtering in wandb UI
Use the prefix system to filter metrics:
- Show only training losses: Filter for `train/*`
- Show only norms: Filter for `norms/*`
- Show only performance: Filter for `perf/*`

## API Reference

### `setup_wandb(config_dict, **kwargs)`
Initialize wandb run with configuration.

**Parameters:**
- `config_dict`: Dictionary of all hyperparameters
- `project`: Project name
- `entity`: wandb username/team
- `group`: Group name
- `name`: Run name
- `run_id`: For resuming runs
- `tags`: List of tags
- `notes`: Run description
- `mode`: 'online', 'offline', or 'disabled'

**Returns:** wandb.Run object

### `log_metrics(metrics_dict, step=None, prefix=None, commit=True)`
Log metrics to wandb.

**Parameters:**
- `metrics_dict`: Dictionary of metrics
- `step`: Training step number
- `prefix`: Prefix to add (e.g., 'train/', 'eval/')
- `commit`: Whether to commit immediately

### `log_images(images_dict, step=None, prefix=None, commit=True)`
Log images to wandb.

**Parameters:**
- `images_dict`: Dictionary mapping names to images
- `step`: Training step number
- `prefix`: Prefix to add
- `commit`: Whether to commit immediately

### `finish_wandb()`
Finish the wandb run and upload final data.

## Troubleshooting

### Issue: "wandb: ERROR Error while calling W&B API"
**Solution:** Check your internet connection or use offline mode:
```yaml
wandb:
    mode: 'offline'
```

### Issue: Multiple processes logging
**Solution:** Ensure only rank 0 logs:
```python
if jax.process_index() == 0:
    setup_wandb(...)
```

### Issue: Large image uploads slowing training
**Solution:** Reduce image logging frequency:
```yaml
training:
    sample_per_epoch: 50  # Instead of 10
```

### Issue: Want to log model checkpoints
**Solution:** Enable model logging:
```yaml
wandb:
    log_model: True
```

## Comparison with Reference Implementation

Features adopted from [kvfrans/shortcut-models](https://github.com/kvfrans/shortcut-models):

✅ **Implemented:**
- Process-aware logging (only rank 0)
- Metric organization with prefixes
- Gradient/parameter norm tracking
- Image logging with visualization
- Resume capability
- Config management
- System info logging
- Proper JAX array conversion

✅ **Adapted for meanflow:**
- MeanFlow-specific loss metrics
- Latent space statistics
- EMA parameter tracking
- FID evaluation logging
- Sample quality monitoring

## Example Output

When training, you'll see organized metrics in wandb:

```
📊 Metrics Dashboard
├── train/loss: 0.234
├── train/v_loss: 0.156
├── norms/grad_norm: 1.234
├── norms/param_norm: 45.678
├── norms/update_norm: 0.0123
├── optim/lr: 0.0001
├── optim/ema_decay: 0.9999
├── perf/steps_per_second: 12.3
├── eval/fid_ema: 23.45
└── sample_stats/mean: 127.5

🖼️ Media
├── samples/generated_samples (every 10 epochs)
└── eval/fid_samples (every FID eval)
```

## References

- [kvfrans/shortcut-models](https://github.com/kvfrans/shortcut-models)
- [wandb Documentation](https://docs.wandb.ai/)
- [JAX Multi-host Programming](https://jax.readthedocs.io/en/latest/multi_process.html)
