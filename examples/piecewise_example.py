import torch

from flag_attn import piecewise_attention

B, H, T, D = 2, 16, 8192, 128
dist_threshold = T // 2

q1 = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
q2 = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
k1 = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
k2 = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
v = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
o = piecewise_attention(q1, k1, q2, k2, v, dist_threshold, causal=True)
print(o)

go = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0")
gq1, gk1, gq2, gk2, gv = torch.autograd.grad(o, (q1, k1, q2, k2, v), go)
print(gq1)
