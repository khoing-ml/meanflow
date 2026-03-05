#!/bin/bash

# Note: You should also update the dataset.root and fid.cache_ref in configs/run_b4_config.yml
export DATA_ROOT="/kaggle/input/datasets/ayaroshevskiy/downsampled-imagenet-128x128"
export LOG_DIR="./logs"

export now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
export JOBNAME=${now}_${salt}_$1
export LOG_DIR=$LOG_DIR/$USER/$JOBNAME

mkdir -p ${LOG_DIR}
# sudo chmod 777 -R ${LOG_DIR}

python3 main.py \
    --workdir=${LOG_DIR} \
    --config=configs/load_config.py:run_b4 \
    2>&1 | tee -a $LOG_DIR/output.log
