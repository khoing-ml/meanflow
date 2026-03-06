# Wandb Integration - Quick Start Guide

## 🚀 Quick Start (2 minutes)

### 1. Install wandb (if not already installed)
```bash
pip install wandb
```

### 2. Login to wandb (first time only)
```bash
wandb login
```

### 3. Update your config
Edit `configs/run_b4.yml` or create your own:
```yaml
wandb:
    project: 'meanflow'
    entity: 'your-username'  # Replace with your wandb username
    name: 'my_first_run'
    mode: 'online'
```

### 4. Run training
```bash
python main.py --config=configs/run_b4.yml --workdir=./logs
```

### 5. View results
Open your browser to: `https://wandb.ai/your-username/meanflow`

---

## 🎯 Common Use Cases

### Test Locally (Offline Mode)
```yaml
wandb:
    mode: 'offline'
```
Then sync later:
```bash
wandb sync wandb/offline-run-*
```

### Disable Wandb Completely
```yaml
wandb:
    mode: 'disabled'
```

### Resume a Crashed Run
1. Get run_id from wandb UI (e.g., `dit_b4_20260305_123456`)
2. Update config:
```yaml
wandb:
    run_id: 'dit_b4_20260305_123456'
```
3. Run training with same config

### Organize Experiments
```yaml
wandb:
    group: 'hyperparameter-sweep'
    tags: ['lr-1e-4', 'bs-256', 'baseline']
    notes: 'Baseline run with standard hyperparameters'
```

---

## 📊 What Gets Logged

### Every 100 steps (default)
- **Losses**: `train/loss`, `train/v_loss`
- **Norms**: `norms/grad_norm`, `norms/param_norm`, `norms/update_norm`
- **Optimization**: `optim/lr`, `optim/ema_decay`
- **Performance**: `perf/steps_per_second`

### Every 10 epochs (default)
- **Generated samples**: `samples/generated_samples` (grid visualization)
- **Sample stats**: `sample_stats/{mean,std,min,max}`

### During FID evaluation
- **FID score**: `eval/fid_ema`
- **FID samples**: `eval/fid_samples` (sample grid)

---

## ⚙️ Configuration Options

### Minimal Config
```yaml
wandb:
    project: 'meanflow'
    mode: 'online'
```

### Full Config
```yaml
wandb:
    project: 'meanflow'                           # Required
    entity: 'your-team'                           # Optional: team/user
    group: 'experiment-group'                     # Optional: group related runs
    name: 'descriptive-run-name'                  # Optional: custom name
    run_id: null                                  # Optional: for resuming
    tags: ['baseline', 'dit-b4']                  # Optional: for filtering
    notes: 'Detailed description of this run'     # Optional: notes
    mode: 'online'                                # online/offline/disabled
    log_model: False                              # Log checkpoints to wandb
    log_histograms: False                         # Log param histograms (slow!)
```

---

## 🐛 Troubleshooting

### "wandb: ERROR Error while calling W&B API"
**Solution**: Check internet or use offline mode:
```bash
export WANDB_MODE=offline
```

### Too many metrics/slow logging
**Solution**: Reduce logging frequency in config:
```yaml
training:
    log_per_step: 200  # Default: 100
    sample_per_epoch: 20  # Default: 10
```

### Want to see more details
**Solution**: Enable histogram logging (warning: slow):
```yaml
wandb:
    log_histograms: True
```

---

## 📖 Detailed Documentation

- **Full Guide**: [`docs/WANDB_GUIDE.md`](../docs/WANDB_GUIDE.md)
- **Implementation Details**: [`docs/WANDB_IMPLEMENTATION.md`](../docs/WANDB_IMPLEMENTATION.md)

---

## ✅ Quick Test

Test the integration without running full training:

```bash
# Run the test script
bash scripts/test_wandb.sh

# Check offline run was created
ls /tmp/meanflow_wandb_test_*/wandb/

# Sync to wandb cloud (optional)
wandb sync /tmp/meanflow_wandb_test_*/wandb/offline-*
```

---

## 💡 Pro Tips

1. **Use descriptive names**:
   ```yaml
   name: 'dit_xl_bs512_lr2e-4_cosine'
   ```

2. **Tag your experiments**:
   ```yaml
   tags: ['ablation', 'architecture', 'dit-xl']
   ```

3. **Group related runs**:
   ```yaml
   group: 'learning-rate-sweep'
   ```

4. **Add notes for context**:
   ```yaml
   notes: 'Testing effect of batch size on FID score'
   ```

5. **Use offline mode for debugging**:
   ```yaml
   mode: 'offline'  # Fast local logging
   ```
   Then sync later:
   ```bash
   wandb sync wandb/offline-run-*
   ```

---

## 📝 Example Complete Config

```yaml
model:
    cls: DiT_B_4

dataset:
    root: /kaggle/temp/output

training:
    batch_size: 256
    learning_rate: 0.0001
    num_epochs: 240
    log_per_step: 100
    sample_per_epoch: 10
    fid_per_epoch: 10

wandb:
    project: 'meanflow-imagenet'
    entity: 'my-research-team'
    group: 'dit-b4-baseline'
    name: 'dit_b4_bs256_lr1e-4_{dataset}_{seed}'
    tags: ['baseline', 'dit-b4', 'imagenet-latents']
    notes: 'Baseline DiT-B/4 on ImageNet latents with standard hyperparameters'
    mode: 'online'
    log_histograms: False
```

---

## 🎓 Next Steps

1. ✅ Run your first experiment with wandb
2. 📊 Explore metrics in wandb dashboard
3. 🔍 Compare multiple runs side-by-side
4. 📈 Track improvements over time
5. 🚀 Share results with your team

---

**Need help?** Check the [full documentation](../docs/WANDB_GUIDE.md) or open an issue!
