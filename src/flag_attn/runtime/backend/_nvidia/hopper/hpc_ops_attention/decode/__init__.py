# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from flag_attn.hpc_ops_attention.decode import (
    DecodeWorkload, PureTritonMTP1Workspace, PureTritonMTPWorkspace,
    attention_decode_pure_triton_mtp, attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp_workspace, prepare_pure_triton_mtp1_workspace,
)

try:
    import triton.experimental.tle.language as tle
    from triton.experimental.tle.language.gpu import types as gpu_types
    HAS_TLE = hasattr(tle, "device_mesh")
except (ImportError, AttributeError):
    tle = None
    gpu_types = None
    HAS_TLE = False

USE_TLE = HAS_TLE and hasattr(tle, "device_mesh")

