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

"""Workload-aware launch presets for Inkling paged relative attention.

Launch configurations are specialized by GPU architecture, query workload,
KV work, head dimensions, and GQA packing. H100 values are benchmark seeds and
must be validated against the workload benchmark and Nsight reports.
"""

from __future__ import annotations

import torch


def _detect_arch() -> str:
    """Return the active CUDA compute-capability key."""
    if not torch.cuda.is_available():
        return "sm90"

    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def get_preset(
    max_seqlen_q: int,
    *,
    max_seqlen_k: int | None = None,
    head_dim_q: int = 128,
    head_dim_v: int = 128,
    q_heads_per_kv_head: int = 1,
    arch: str | None = None,
) -> dict[str, int]:
    """Return a launch configuration specialized by workload class.

    Args:
        max_seqlen_q:
            Maximum query length in the batch.
        max_seqlen_k:
            Synchronization-free upper bound for KV length. The wrapper may use
            page-table capacity instead of reading cache_seqlens back to CPU.
        head_dim_q:
            Q/K head dimension.
        head_dim_v:
            V/output head dimension.
        q_heads_per_kv_head:
            GQA group size.
        arch:
            Explicit architecture such as "sm90", or None for auto-detection.
    """
    arch = _detect_arch() if arch is None else arch
    k = max_seqlen_q if max_seqlen_k is None else max_seqlen_k
    wide = max(head_dim_q, head_dim_v) > 128
    scheduler_rows = max_seqlen_q * q_heads_per_kv_head

    if arch == "sm90":
        # Single-token decode:
        # Pack the GQA heads into the M dimension so one paged K/V traversal
        # serves multiple Q heads. Long KV uses a wider K tile.
        if max_seqlen_q == 1:
            block_q = 16 if q_heads_per_kv_head <= 16 else 32
            block_k = 64 if k < 8192 else 128

            return {
                "BLOCK_Q": block_q,
                "BLOCK_K": block_k,
                "num_warps": 4 if not wide else 8,
                "num_stages": 3,
            }

        # Small-Q/chunked workloads:
        # Keep the FP32 output accumulator small and use Split-KV to expose
        # additional parallelism where the caller selects num_splits > 1.
        if scheduler_rows <= 64:
            return {
                "BLOCK_Q": 32,
                "BLOCK_K": 64,
                "num_warps": 4,
                "num_stages": 3,
            }

        # H100 prefill A/B configuration:
        #
        # The former 128x64, 8-warp, 4-stage configuration produced:
        #   254 registers/thread
        #   131.58 KiB dynamic shared memory/CTA
        #   12.5% theoretical occupancy
        #
        # Reducing BLOCK_Q to 64 shortens the live FP32 accumulator and score
        # fragments. Four warps and three stages reduce register/shared-memory
        # pressure and increase the query-tile grid.
        return {
            "BLOCK_Q": 64,
            "BLOCK_K": 64,
            "num_warps": 4,
            "num_stages": 3,
        }

    # Portable fallback for non-SM90 devices.
    if max_seqlen_q <= 16:
        return {
            "BLOCK_Q": 16,
            "BLOCK_K": 32,
            "num_warps": 4,
            "num_stages": 2,
        }

    return {
        "BLOCK_Q": 32,
        "BLOCK_K": 32,
        "num_warps": 4,
        "num_stages": 2,
    }


def get_tle_preset(
    max_seqlen_q: int,
    *,
    max_seqlen_k: int | None = None,
    head_dim_q: int = 128,
    head_dim_v: int = 128,
    q_heads_per_kv_head: int = 1,
    arch: str | None = None,
) -> dict[str, int]:
    """Return a launch preset whose query tile is valid for Hopper WGMMA.

    The TLE macro kernel masks query rows internally, so short decode and
    prefill workloads can use the same 64-row WGMMA tile as larger workloads.
    Triton keeps using :func:`get_preset` and its smaller tiles where those are
    faster.
    """
    cfg = get_preset(
        max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        head_dim_q=head_dim_q,
        head_dim_v=head_dim_v,
        q_heads_per_kv_head=q_heads_per_kv_head,
        arch=arch,
    )
    if cfg["BLOCK_Q"] % 64:
        cfg = {**cfg, "BLOCK_Q": 64}
    return cfg
