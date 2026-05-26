"""Baseline Llama-style pretraining configuration.

This is the control setting used in the paper: standard causal softmax
self-attention with RMSNorm, SwiGLU and RoPE, but without HeadNorm.
"""

out_dir = "out/baseline"
wandb_log = False
wandb_project = "headnorm"
wandb_run_name = "baseline"

# 12 * 1024 * 5 * 8 = 491,520 tokens per optimizer step on 8 GPUs.
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

max_iters = 600000
lr_decay_iters = 600000

eval_interval = 1000
eval_iters = 200
log_interval = 10

learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 2000
weight_decay = 1e-1
grad_clip = 1.0

n_layer = 12
n_head = 12
n_embd = 768
n_kv_head = None

enable_headnorm = False
headnorm_shared_weights = False
