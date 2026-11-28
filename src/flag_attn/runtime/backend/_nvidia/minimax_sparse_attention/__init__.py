# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA MSA specializations loaded by the vendor registry.

PDL decode, TLE radix TopK, FP8 QKV and Blackwell persistent/FP4/NVFP4
implementations reuse portable helpers from flag_attn.minimax_sparse_attention.
Hopper TMA/WGMMA prefill lives in the sibling hopper architecture package.
"""
