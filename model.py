"""Llama-style decoder with optional HeadNorm."""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# RoPE helper functions.
# -----------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# -----------------------------------------------------------------------------
# HeadRMSNorm (New Feature)
# -----------------------------------------------------------------------------

class HeadRMSNorm(nn.Module):
    """
    RMSNorm applied specifically to the head dimension of the attention output.
    Input shape: (Batch, n_head, T, head_dim)
    """
    def __init__(self, n_head, head_dim, shared_weights=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.shared_weights = shared_weights
        self.n_head = n_head
        self.head_dim = head_dim
        
        # Initialization note: parameters will be overwritten by calibration later
        if shared_weights:
            # One set of weights shared across all heads
            self.weight = nn.Parameter(torch.ones(head_dim))
        else:
            # Independent weights for each head
            self.weight = nn.Parameter(torch.ones(n_head, head_dim))

    def forward(self, x):
        # x: (B, n_head, T, head_dim)
        
        # 1. Calculate RMS on the last dimension (head_dim)
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        
        # 2. Apply affine transformation (weight)
        if self.shared_weights:
            # weight: (head_dim) -> broadcasts to (..., head_dim)
            return self.weight * x_normed
        else:
            # weight: (n_head, head_dim)
            # Need to reshape for broadcasting against (B, n_head, T, head_dim)
            # view as (1, n_head, 1, head_dim)
            return self.weight.view(1, self.n_head, 1, self.head_dim) * x_normed

# -----------------------------------------------------------------------------
# Model Components
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.q_size = self.n_head * self.head_dim
        self.kv_size = self.n_kv_head * self.head_dim
        
        self.c_attn = nn.Linear(config.n_embd, self.q_size + 2 * self.kv_size, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        
        # --- HeadNorm Configuration ---
        self.enable_headnorm = config.enable_headnorm
        if self.enable_headnorm:
            self.head_norm = HeadRMSNorm(
                n_head=self.n_head, 
                head_dim=self.head_dim, 
                shared_weights=config.headnorm_shared_weights
            )
        # ------------------------------

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention.")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, freqs_cis):
        B, T, C = x.size()

        # 1. Project
        qkv = self.c_attn(x)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=2)
        
        # 2. Reshape
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        # 3. Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # 4. Transpose (B, nh, T, hs)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 5. GQA
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k[:, :, None, :, :].expand(B, self.n_kv_head, n_rep, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_kv_head, n_rep, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)

        # 6. Attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v 
        
        # --- Apply HeadNorm (Insertion Point) ---
        if self.enable_headnorm:
            # y is (B, n_head, T, head_dim)
            y = self.head_norm(y)
        # ----------------------------------------
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

    @torch.no_grad()
    def init_headnorm_stats(self, device='cpu'):
        """
        Custom initialization for HeadNorm.
        Run a dummy forward pass with standard normal input.
        Initialize head_norm weights based on the standard deviation of the FIRST token 
        across the hidden dimension.
        """
        if not self.enable_headnorm:
            return

        # Create a dummy input
        # Shape: (1, 64, n_embd) - We only need the first token, but we simulate a small batch
        dummy_x = torch.randn(1, 64, self.n_embd, device=device)
        
        # Run projection part of forward
        qkv = self.c_attn(dummy_x)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=2)
        
        # Reshape to (B, T, n_head, head_dim)
        q = q.view(1, 64, self.n_head, self.head_dim)
        k = k.view(1, 64, self.n_kv_head, self.head_dim)
        v = v.view(1, 64, self.n_kv_head, self.head_dim)
        
        # RoPE (Optional for magnitude stats but good for consistency)
        # We need freqs_cis for T=64
        # Assuming we can grab it from parent or recompute quickly. 
        # For simplicity in this isolated function, we skip RoPE or use simple one if needed.
        # Ideally, we should just assume the magnitude impact of RoPE is negligible on initialization std.
        # Let's Apply RoPE if we want to be strict, but for initialization stats, 
        # raw Q/K/V magnitude matters most. Let's proceed with raw Q/K/V to avoid dependency hell 
        # inside this method, OR just transpose and calculate.
        
        q = q.transpose(1, 2) # (B, nh, T, hs)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # GQA expansion
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k[:, :, None, :, :].expand(1, self.n_kv_head, n_rep, 64, self.head_dim).reshape(1, self.n_head, 64, self.head_dim)
            v = v[:, :, None, :, :].expand(1, self.n_kv_head, n_rep, 64, self.head_dim).reshape(1, self.n_head, 64, self.head_dim)
            
        # Attention
        # y shape: (1, n_head, 64, head_dim)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        
        # --- Modifications Start Here ---
        
        # 1. Select the FIRST token only (t=0)
        # y_first shape: (1, n_head, head_dim)
        y_first = y[:, :, 0, :]
        
        # 2. Calculate Std across the hidden dimension (dim=-1)
        # We want a scalar representing the "scale" of the activation vector.
        # y_std shape: (1, n_head)
        y_std = y_first.std(dim=-1, unbiased=False)
        
        # 3. Initialize Weights Uniformly
        if self.head_norm.shared_weights:
            # Average the std across all heads to get a single global scalar
            # scalar_std shape: scalar
            scalar_std = y_std.mean()
            # Fill the entire weight vector with this single value
            self.head_norm.weight.data.fill_(scalar_std)
        else:
            # y_std is (1, n_head), we need to fill (n_head, head_dim)
            # Each head gets its own scalar std, but that scalar is applied uniformly to all its dims
            # Transpose to (n_head, 1) and expand
            head_scales = y_std.view(self.n_head, 1).expand(self.n_head, self.head_dim)
            self.head_norm.weight.data.copy_(head_scales)
            
        print(f"Initialized HeadNorm weights (First Token Strategy). Mean Scale: {y_std.mean().item():.4f}")

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.intermediate_size
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.rms_1(x), freqs_cis)
        x = x + self.mlp(self.rms_2(x))
        return x

@dataclass
class LlamaConfig:
    block_size: int = 4096
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_head: int = None
    intermediate_size: int = 3072
    dropout: float = 0.0
    bias: bool = False
    precision: str = "bfloat16"
    # --- New Config Params ---
    enable_headnorm: bool = False
    headnorm_shared_weights: bool = True # If True, share weights across heads. If False, per-head weights.

class Llama(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Init params
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # RoPE Precomputation
        head_dim = config.n_embd // config.n_head
        freqs_cis = precompute_freqs_cis(head_dim, config.block_size * 2) 
        self.register_buffer('freqs_cis', freqs_cis)

        # --- HeadNorm Calibration ---
        if config.enable_headnorm:
            print("Calibrating HeadNorm weights based on random init forward pass...")
            for block in self.transformer.h:
                # We can perform calibration on CPU or GPU if available
                # Assuming model is on CPU during init usually, but let's be safe
                block.attn.init_headnorm_stats()
        # ----------------------------

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        x = self.transformer.wte(idx) 
        x = self.transformer.drop(x)
        
        freqs_cis = self.freqs_cis[:t]

        for block in self.transformer.h:
            x = block(x, freqs_cis)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
    
    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        
        if hasattr(self, 'freqs_cis'):
            self.freqs_cis = self.freqs_cis[:block_size]

        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
    

    # ... (Generate and Optimizer methods remain unchanged) ...
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
