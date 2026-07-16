import torch

import flag_attn

# replace the default config function
from flag_attn import flash


def get_fwd_config(B, H, M, N, D, causal):
    return (64, 64, 1, 4)


flash.get_fwd_config = get_fwd_config

B, H, M, N, D = 2, 16, 4096, 4096, 128
causal = True

q = torch.randn(B, H, M, D, dtype=torch.bfloat16, device="cuda:0", requires_grad=True)
k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda:0", requires_grad=True)
v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda:0", requires_grad=True)

o = flag_attn.flash_attention(q, k, v, causal=causal)
go = torch.randn_like(o)
gq, gk, gv = torch.autograd.grad(o, (q, k, v), go)
