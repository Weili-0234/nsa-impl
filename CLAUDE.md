# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NSA (Native Sparse Attention) — a PyTorch + Triton + FlexAttention implementation of DeepSeek's Native Sparse Attention (arXiv:2502.11089). Combines three attention mechanisms via learned gating: compression, selection, and sliding window attention.

## Setup

```bash
uv sync
```

Requires Python 3.13+ and a CUDA-capable GPU (Triton kernels).

## Architecture

Three attention components are fused in `nsa/nsa.py:nsa_func()`:

1. **Compression** (`nsa/compression.py`): Mean-pools K/V into blocks, applies `torch.nn.attention.flex_attention` with block masking. Wrapped with `@torch.compile`.

2. **Selection** (`nsa/selection.py`): Selects top-K blocks via `parallel_nsa_topk` (from FLA), then runs attention over selected blocks. Has two backward variants:
   - `'two-pass'` (default): Uses FLA's `ParallelNSAFunction`
   - `'one-pass'`: Custom Triton kernel with atomic gradient accumulation

3. **Sliding Window**: Uses `flash_attn_func` with causal masking.

**Fusion**: `output = g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swd`

## Key Dependencies

- **flash-attn**: Sliding window attention
- **flash-linear-attention (FLA)**: `parallel_nsa_topk`, `ParallelNSAFunction`, `mean_pooling`
- **Triton**: Custom GPU kernels for selection attention forward/backward

## Tensor Conventions

- Query: `(B, M, H, D)` — batch, query length, heads, head dim
- Key/Value: `(B, N, G, D)` — batch, kv length, kv groups (GQA), head dim
- Supports Grouped Query Attention (GQA) where `H` is a multiple of `G`

## Code Notes

- `nsa/` has no `__init__.py`; modules use relative imports (`from compression import ...`)
- Selection attention Triton kernels use `@triton.autotune` over num_warps (1, 2, 4, 8)
- The custom autograd `SelectionAttention` class handles forward/backward with proper log-sum-exp numerical stability
