#!/bin/bash
# Quick test script for wandb integration
# This runs a short test with offline wandb to verify everything works

set -e

echo "==================================="
echo "Testing Wandb Integration (Offline)"
echo "==================================="

# Set wandb to offline mode
export WANDB_MODE=offline

# Create test output directory
TEST_DIR="/tmp/meanflow_wandb_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo ""
echo "Output directory: $TEST_DIR"
echo ""

# Create a minimal test config
cat > "$TEST_DIR/test_config.yml" <<EOF
model:
    cls: DiT_S_4

dataset:
    root: /kaggle/temp/output
    image_size: 16

training:
    log_per_step: 10
    checkpoint_per_epoch: 100
    num_epochs: 2
    learning_rate: 0.0001
    batch_size: 64
    adam_b2: 0.95
    ema_type: 'const'
    sample_on_training: True
    sample_per_epoch: 1
    fid_per_epoch: 100
    seed: 42

fid:
    device_batch_size: 20
    num_samples: 1000
    cache_ref: /kaggle/temp/output/benjamin-paine_imagenet-1k-128x128_fid_stats.npz
    on_training: False

sampling:
    num_steps: 1

wandb:
    project: 'meanflow-test'
    entity: null
    group: 'integration-test'
    name: 'test_run'
    mode: 'offline'
    log_histograms: False

load_from: null
eval_only: False
EOF

echo "Test config created at: $TEST_DIR/test_config.yml"
echo ""
echo "Running training for 2 epochs..."
echo ""

# Run training (this will likely fail if data not available, but will test wandb setup)
python main.py \
    --config="$TEST_DIR/test_config.yml" \
    --workdir="$TEST_DIR" || {
    echo ""
    echo "Training may have failed due to missing data, but wandb should have initialized."
    echo "Check for wandb offline files in: $TEST_DIR/wandb/offline-*"
    echo ""
}

# Check if wandb was initialized
if [ -d "$TEST_DIR/wandb" ]; then
    echo "✅ SUCCESS: Wandb directory created"
    echo ""
    echo "Wandb offline run files:"
    ls -lh "$TEST_DIR/wandb/"
    echo ""
    echo "To sync to wandb cloud:"
    echo "  wandb sync $TEST_DIR/wandb/offline-*"
else
    echo "❌ ERROR: Wandb directory not found"
    exit 1
fi

echo ""
echo "Test workspace: $TEST_DIR"
echo "To clean up: rm -rf $TEST_DIR"
echo ""
echo "==================================="
echo "Test Complete!"
echo "==================================="
