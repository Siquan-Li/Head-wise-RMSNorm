"""Baseline Llama-style pretraining configuration.

This is the control setting used in the paper: standard causal softmax
self-attention with RMSNorm, SwiGLU and RoPE, but without HeadNorm.
"""

out_dir = "out/baseline"
wandb_log = False
wandb_project = "headnorm"
wandb_run_name = "baseline"

# Model Architecture (Table 3)
n_layer = 12
n_head = 12
n_embd = 768
n_kv_head = None
intermediate_size = 3072
vocab_size = 50304

# Training Hyperparameters (Table 4)
batch_size = 12
block_size = 4096
gradient_accumulation_steps = 5

max_iters = 40000
lr_decay_iters = 40000

eval_interval = 1000
eval_iters = 200
log_interval = 10

learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 2000
weight_decay = 0.1
grad_clip = 1.0
precision = "bfloat16"

enable_headnorm = False
headnorm_shared_weights = False
