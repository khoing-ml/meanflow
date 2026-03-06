# Wandb Integration Summary

## 🎉 Implementation Complete!

I've successfully implemented comprehensive wandb logging for your meanflow repository, adapting best practices from [kvfrans/shortcut-models](https://github.com/kvfrans/shortcut-models).

---

## 📁 Files Created

### New Files
1. **`utils/wandb_util.py`** - Core wandb utilities
   - `setup_wandb()` - Initialize wandb with configuration
   - `log_metrics()` - Log metrics with prefixes
   - `log_images()` - Log images/visualizations
   - `log_histograms()` - Log parameter histograms
   - `finish_wandb()` - Clean up wandb run

2. **`docs/WANDB_GUIDE.md`** - Complete guide (150+ lines)
   - Features overview
   - Configuration options
   - Usage examples
   - API reference
   - Troubleshooting

3. **`docs/WANDB_IMPLEMENTATION.md`** - Implementation details (300+ lines)
   - What was changed
   - Metrics logged
   - Features implemented
   - Comparison with reference repo

4. **`docs/WANDB_QUICKSTART.md`** - Quick start guide
   - 2-minute setup
   - Common use cases
   - Example configs
   - Pro tips

5. **`scripts/test_wandb.sh`** - Test script
   - Verify integration works
   - Offline mode testing
   - Quick validation

---

## 📝 Files Modified

### 1. `train.py` - Enhanced Training Loop
**Added:**
- Import wandb utilities
- Comprehensive metric tracking:
  - Gradient norm (L2)
  - Parameter norm (L2)
  - Update norm (L2)
  - EMA decay value
- Organized metric logging with prefixes:
  - `train/*` - Training losses
  - `norms/*` - Gradient/parameter norms
  - `optim/*` - Optimization metrics
  - `perf/*` - Performance metrics
  - `eval/*` - Evaluation metrics
  - `samples/*` - Generated samples
- Better wandb initialization
- Sample statistics logging
- Optional histogram logging
- Improved visualization logging
- Proper wandb teardown

### 2. `configs/default.py` - Config Schema
**Added:**
```python
config.wandb = wandb = ml_collections.ConfigDict()
wandb.project = 'meanflow'
wandb.entity = None
wandb.group = None
wandb.name = 'meanflow_{dataset}_{model}'
wandb.run_id = None
wandb.tags = []
wandb.notes = ''
wandb.mode = 'online'
wandb.log_model = False
wandb.log_histograms = False
```

### 3. `configs/run_b4.yml` - Example Config
**Added:**
```yaml
wandb:
    project: 'meanflow'
    entity: null
    group: 'imagenet-dit-b4'
    name: 'dit_b4_bs256_lr1e-4'
    tags: ['dit-b4', 'imagenet', 'meanflow']
    mode: 'online'
```

---

## 📊 Metrics Now Logged

### Training (every `log_per_step` steps)
| Metric | Description | Prefix |
|--------|-------------|--------|
| `loss` | Total training loss | `train/` |
| `v_loss` | Velocity loss | `train/` |
| `grad_norm` | Global gradient L2 norm | `norms/` |
| `param_norm` | Global parameter L2 norm | `norms/` |
| `update_norm` | Global update L2 norm | `norms/` |
| `lr` | Learning rate | `optim/` |
| `ema_decay` | EMA decay value | `optim/` |
| `steps_per_second` | Training throughput | `perf/` |
| `epoch` | Current epoch | `meta/` |

### Evaluation
| Metric | Description | Prefix |
|--------|-------------|--------|
| `fid_ema` | FID with EMA params | `eval/` |
| `num_samples` | Number of FID samples | `eval/` |

### Samples (every `sample_per_epoch` epochs)
| Metric | Description | Prefix |
|--------|-------------|--------|
| `generated_samples` | Sample grid image | `samples/` |
| `mean` | Mean pixel value | `sample_stats/` |
| `std` | Std dev | `sample_stats/` |
| `min` | Min pixel value | `sample_stats/` |
| `max` | Max pixel value | `sample_stats/` |

### Optional Histograms (every 1000 steps if enabled)
| Metric | Description | Prefix |
|--------|-------------|--------|
| (key layers) | Parameter distributions | `params/` |

**Total: 15+ metrics** organized in 8 categories

---

## ✨ Key Features

### 🔧 Implemented from kvfrans/shortcut-models
- ✅ Process-aware logging (only rank 0)
- ✅ Metric organization with prefixes
- ✅ Gradient/parameter norm tracking
- ✅ Image logging
- ✅ Config management
- ✅ Resume capability
- ✅ System info logging
- ✅ Proper JAX array handling

### 🎯 MeanFlow-Specific Adaptations
- ✅ MeanFlow loss metrics (v_loss)
- ✅ Latent space statistics
- ✅ EMA parameter tracking
- ✅ FID evaluation logging
- ✅ Sample quality monitoring
- ✅ Velocity field metrics

### 🚀 Additional Enhancements
- ✅ Optional histogram logging
- ✅ Sample statistics tracking
- ✅ Update norm tracking
- ✅ Better image organization
- ✅ Comprehensive documentation

---

## 🚀 Quick Start

### 1. Install wandb
```bash
pip install wandb
wandb login
```

### 2. Configure
Edit `configs/run_b4.yml`:
```yaml
wandb:
    entity: 'your-username'  # Add your username
```

### 3. Run
```bash
python main.py --config=configs/run_b4.yml --workdir=./logs
```

### 4. View
Open: `https://wandb.ai/your-username/meanflow`

---

## 📚 Documentation

1. **Quick Start**: [`docs/WANDB_QUICKSTART.md`](docs/WANDB_QUICKSTART.md)
   - 2-minute setup guide
   - Common use cases
   - Pro tips

2. **Complete Guide**: [`docs/WANDB_GUIDE.md`](docs/WANDB_GUIDE.md)
   - All features explained
   - Configuration options
   - API reference
   - Troubleshooting

3. **Implementation Details**: [`docs/WANDB_IMPLEMENTATION.md`](docs/WANDB_IMPLEMENTATION.md)
   - What changed
   - Metrics logged
   - Comparison with reference

---

## 🧪 Testing

### Quick Test (Offline Mode)
```bash
bash scripts/test_wandb.sh
```

This will:
- Create a test config
- Run training in offline mode
- Verify wandb initialization
- Show where to find logs

---

## 🎓 Example Workflows

### Experiment Tracking
```yaml
wandb:
    group: 'architecture-study'
    tags: ['dit-b4', 'baseline']
    notes: 'Baseline run for comparison'
```

### Hyperparameter Sweep
```yaml
wandb:
    group: 'lr-sweep'
    name: 'dit_b4_lr{lr}'
    tags: ['sweep', 'learning-rate']
```

### Debugging
```yaml
wandb:
    mode: 'offline'  # Fast local logging
    log_histograms: True  # Detailed analysis
```

### Production Run
```yaml
wandb:
    project: 'meanflow-production'
    group: 'final-models'
    tags: ['production', 'dit-xl']
    mode: 'online'
```

---

## 🎯 Benefits

### For Research
- ✅ Track all experiments automatically
- ✅ Compare runs side-by-side
- ✅ Share results with collaborators
- ✅ Reproduce experiments exactly

### For Development
- ✅ Monitor training in real-time
- ✅ Detect gradient issues early
- ✅ Identify performance bottlenecks
- ✅ Debug model behavior

### For Production
- ✅ Track model performance over time
- ✅ A/B test different configurations
- ✅ Document what works
- ✅ Resume interrupted training

---

## 🔧 Configuration Examples

### Minimal (Just get started)
```yaml
wandb:
    project: 'meanflow'
    mode: 'online'
```

### Development (Fast iteration)
```yaml
wandb:
    project: 'meanflow-dev'
    mode: 'offline'
    tags: ['dev', 'experimental']
```

### Production (Full tracking)
```yaml
wandb:
    project: 'meanflow-prod'
    entity: 'research-team'
    group: 'production-models'
    name: 'dit_xl_final_{date}'
    tags: ['production', 'dit-xl', 'v1.0']
    notes: 'Final production model - DO NOT DELETE'
    mode: 'online'
    log_histograms: False
```

---

## 📈 What You'll See in Wandb

### Dashboard Sections
1. **Overview** - Run summary and system info
2. **Charts** - All metrics organized by prefix
3. **System** - GPU/CPU usage, memory
4. **Logs** - Console output
5. **Files** - Config files and code
6. **Media** - Generated samples

### Metric Categories
- **train/** - Training losses (2 metrics)
- **norms/** - Gradient/parameter norms (3 metrics)
- **optim/** - Optimization (2 metrics)
- **perf/** - Performance (1 metric)
- **eval/** - Evaluation (2 metrics)
- **samples/** - Visualizations (1 metric)
- **sample_stats/** - Statistics (4 metrics)

---

## 🤝 Contributing

To add more metrics:

1. **In training step**:
```python
metrics['my_metric'] = value
```

2. **In logging section**:
```python
if jax.process_index() == 0:
    log_metrics({'my_metric': value}, 
                step=step, 
                prefix='my_category')
```

---

## 📞 Support

- **Documentation**: See `docs/WANDB_*.md` files
- **Quick Help**: Run `bash scripts/test_wandb.sh`
- **Wandb Docs**: https://docs.wandb.ai/
- **Issues**: Open a GitHub issue

---

## ✅ Checklist - Done!

- [x] Create wandb utility module
- [x] Enhance training loop with metrics
- [x] Add wandb config to default config
- [x] Update example YAML config
- [x] Track gradient norms
- [x] Track parameter norms
- [x] Track update norms
- [x] Organize metrics with prefixes
- [x] Add sample statistics
- [x] Improve image logging
- [x] Add histogram logging (optional)
- [x] Write comprehensive documentation
- [x] Create quick start guide
- [x] Create test script
- [x] Add usage examples
- [x] Document all features

---

## 🎉 Summary

**You now have production-ready wandb integration with:**
- 15+ tracked metrics
- 8 organized categories
- Beautiful visualizations
- Complete documentation
- Test scripts
- Example configs

**Ready to track your experiments!** 🚀

---

**Questions?** Check the documentation or run the test script!
