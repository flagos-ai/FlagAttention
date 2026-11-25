# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA MSA specializations loaded by the vendor registry.

Shared FP8 QKV and Blackwell persistent/FP4/NVFP4 implementations reuse
selectors and BF16-Q kernels from flag_attn.minimax_sparse_attention.
"""
