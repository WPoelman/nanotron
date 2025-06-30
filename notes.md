# Test on interactive node

```
interactive-gpu
python scripts/2_generate_config_small_llama_debug.py
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 scripts/3_nanotron_run_train.py --config-file config/train/debug/config_small_llama_in_maxmin_a_debug.yaml
```

# conda and cuda
`conda install nvidia/label/cuda-12.8.1::cuda`

# torch
`uv pip install torch --index-url https://download.pytorch.org/whl/cu128`
`uv pip install -e .`
`uv pip install datasets transformers datatrove[io] numba wandb`

# Fused kernels
`uv pip install ninja triton "flash-attn>=2.5.0" --no-build-isolation`
