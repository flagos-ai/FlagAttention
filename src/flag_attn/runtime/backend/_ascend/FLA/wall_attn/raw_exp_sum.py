# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in registry for the BS=128 TLE-Raw exp/sum primitive.

Call build_and_register() before compiling a kernel that calls
raw.call('ascend_wall_exp_sum', score, out=(prob_bf16, sum_fp32, scratch_fp32)).
Scratch is a distinct [128] fp32 tensor; returned scratch contents are undefined.
Input score is natural-log units. Production attention currently uses log2 units.
"""
import os
from pathlib import Path
import shutil
import subprocess

import triton.language.extra.cann.extension as al


def build_and_register(output_dir=None):
    """Build plain+mix vector bitcode, link it, then register the raw op.

    Honors the active ccec on PATH and TLE_CANN_TEMPLATE_INCLUDE. No NPU access.
    Use a fresh TRITON_CACHE_DIR for each experimental process.
    """
    include = Path(os.environ['TLE_CANN_TEMPLATE_INCLUDE'])
    lib = include.parent / 'lib'
    compiler = shutil.which('ccec')
    if compiler is None:
        raise RuntimeError('ccec is not on PATH; activate the CANN environment')
    linker = Path(compiler).parent / 'llvm-link'
    target = Path(output_dir or os.environ['TRITON_CACHE_DIR']) / 'wall_exp_sum'
    target.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / 'ops' / 'wall_exp_sum.cpp'
    common = [compiler, '-O2', '-x', 'cce', '--cce-auto-sync=off',
              '--cce-aicore-only', '--cce-generic-addrspace=off',
              '-mllvm', '-disable-llvm-optzns', '--cce-aicore-arch=dav-c220-vec',
              '--cce-enable-print', '--cce-enable-sanitizer', '-std=c++17',
              '-I', str(include), '-I', str(lib)]
    objects = []
    for mode, extra in [('plain', []), ('mix', ['-cce-enable-mix', '-mllvm', '-enable-mix=true'])]:
        obj = target / f'wall_exp_sum.{mode}.bc'
        subprocess.run(common + extra + [str(source), '-emit-llvm', '-c', '-o', str(obj)],
                       check=True, capture_output=True, text=True)
        objects.append(str(obj))
    bitcode = target / 'wall_exp_sum.bc'
    subprocess.run([str(linker), *objects, '-o', str(bitcode)], check=True,
                   capture_output=True, text=True)

    @al.register_custom_op
    class wall_exp_sum:
        core = al.CORE.VECTOR
        pipe = al.PIPE.PIPE_V
        mode = al.MODE.SIMD

        def __init__(self, score, out=None):
            self.symbol = 'custom_wall_exp_sum'
            self.bitcode = str(bitcode)

    return bitcode
