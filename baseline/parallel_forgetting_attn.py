# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Official ACP forward baseline for the default Forgetting Attention cases."""

import torch
import torch.nn.functional as F


OP_NAME = "parallel_forgetting_attn"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# id, B, M, N, HQ, H, D, dtype, scale: benchmark DEFAULT_IDS.
CASES = (
    ("S001", 1, 63, 63, 1, 1, 64, "float16", 1.0),
    ("S003", 3, 1024, 1024, 8, 2, 60, "float16", 0.1),
    ("S011", 4, 4096, 4096, 32, 32, 64, "bfloat16", 1.3),
    ("S024", 4, 4096, 4096, 32, 32, 128, "bfloat16", 1.3),
    ("S031", 2, 1, 1, 8, 8, 64, "bfloat16", 1.3),
    ("S034", 2, 65, 65, 8, 8, 64, "bfloat16", 1.3),
    ("S064", 4, 1020, 2098, 2, 2, 64, "bfloat16", 1.3),
)


def native():
    try:
        from flag_attn.forgetting_attention.naive import forgetting_attention
    except (ImportError, OSError):
        return None

    def official_acp(q, k, v, gate, *, scale, threshold):
        d = q.shape[-1]
        padded = 1 << (d - 1).bit_length()
        if d != padded:
            q, k, v = (F.pad(t, (0, padded - d)) for t in (q, k, v))
        groups = q.shape[2] // k.shape[2]
        if groups != 1:
            k, v = (t.repeat_interleave(groups, dim=2) for t in (k, v))
        out = forgetting_attention(
            q, k, v, gate, head_first=False, seq_start=None,
            sm_scale=scale, adaptive_threshold=threshold,
        )
        return out if d == padded else out[..., :d].contiguous()

    return official_acp


def grid():
    return [
        dict(zip(("id", "B", "M", "N", "HQ", "H", "D", "dtype", "scale"), row))
        for row in CASES
    ]


def unsupported_reason(binding, dtype):
    if str(dtype) != f"torch.{binding['dtype']}":
        return "benchmark does not run this dtype for this case"
    return None


def build_inputs(binding, dtype, device):
    generator = torch.Generator(device=device).manual_seed(0)
    b, m, n, hq, h, d = (binding[k] for k in ("B", "M", "N", "HQ", "H", "D"))
    q = torch.randn((b, m, hq, d), device=device, dtype=dtype, generator=generator)
    k = torch.randn((b, n, h, d), device=device, dtype=dtype, generator=generator)
    v = torch.randn((b, n, h, d), device=device, dtype=dtype, generator=generator)
    gate = F.logsigmoid(
        torch.empty((b, n, hq), device=device, dtype=torch.float32)
        .uniform_(0.0, 10.0, generator=generator)
    )
    return (q, k, v, gate), {"scale": binding["scale"], "threshold": -10.0}


def key_shape(binding):
    dims = [binding[k] for k in ("B", "M", "N", "HQ", "H", "D")]
    return [dims, {"scale": binding["scale"], "threshold": -10.0}]


def config(binding, dtype):
    b, m, n, hq, h, d = (binding[k] for k in ("B", "M", "N", "HQ", "H", "D"))

    def tensor(shape, tensor_dtype=dtype):
        return {"shape": shape, "dtype": str(tensor_dtype)}

    return {
        "op": OP_NAME,
        "dtype": str(dtype),
        "dims": dict(binding),
        "inputs": {
            "q": tensor([b, m, hq, d]), "k": tensor([b, n, h, d]),
            "v": tensor([b, n, h, d]), "gate": tensor([b, n, hq], torch.float32),
            "scale": {"scalar": binding["scale"]},
            "threshold": {"scalar": -10.0},
        },
        "outputs": {"output": tensor([b, m, hq, d])},
        "mutation": "none",
        "graph": False,
    }
