# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""In-place FP16 exp2 and FP32 row sums for the split Wall PV kernel."""

from pathlib import Path

from triton.experimental.tle.raw import dialect


@dialect(
    name="cann",
    file=str(Path(__file__).with_name("ops") / "wall_exp_sum.cpp"),
    extern_func_name="wall_exp_sum_inplace",
    pipeline={"core": "vector", "pipe": "PIPE_V"},
)
def exp_sum_op(*args, **kwargs):
    ...
