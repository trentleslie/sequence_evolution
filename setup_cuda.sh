#!/bin/bash

# CUDA paths
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Additional CUDA library paths
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# CUDA compiler paths
export CUDA_BIN_PATH=$CUDA_HOME/bin
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export CUDACXX=$CUDA_HOME/bin/nvcc

# Set CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Additional environment variables
export CUDA_MODULE_LOADING=LAZY

# Clear any existing PyTorch CUDA cache
rm -rf ~/.cache/torch/hub ~/.cache/torch/kernels
