#!/bin/bash

LOG_ROOT=/home/nesc-gy/nyf/code/cvhw/runs

# 找到最新生成的目录
LATEST_LOG=$(ls -td $LOG_ROOT/*/ | head -1)

# 启动 TensorBoard
tensorboard --logdir="$LATEST_LOG"