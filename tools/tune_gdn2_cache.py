#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run native GDN2 Autotune and export measured winners as project-style YAML."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARGUMENTS = ["--full", "--dtype", "bfloat16", "--warmup", "20", "--rep", "100"]
KERNELS = ("_chunk_gdn2_fwd_intra_infer_kernel", "_chunk_gdn2_fwd_h_o_infer_kernel")


def collect_tuned_configs(cache: Path) -> dict:
    """Export exactly one measured winner per complete native workload key."""
    result = {}
    for name in KERNELS:
        entries = []
        for path in sorted(cache.rglob(f"{name}.autotune.json")):
            data = json.loads(path.read_text())
            timings = [
                item
                for item in data["configs_timings"]
                if item[1] and all(math.isfinite(t) and t > 0 for t in item[1])
            ]
            if not timings:
                raise ValueError(f"No finite candidate measurements in {path}")
            best = min(timings, key=lambda item: item[1])[0]
            if best.get("pre_hook") is not None or best.get("ir_override") is not None:
                raise ValueError(f"Unsupported launch options in {path}")
            unknown = set(best) - {
                "kwargs",
                "num_warps",
                "num_stages",
                "num_ctas",
                "maxnreg",
                "pre_hook",
                "ir_override",
            }
            if unknown:
                raise ValueError(
                    f"Unsupported launch options {sorted(unknown)} in {path}"
                )
            entry = {"WORKLOAD": data["key"], "META": best["kwargs"]}
            entry.update(
                {
                    field: best[field]
                    for field in ("num_warps", "num_stages", "num_ctas", "maxnreg")
                    if field in best
                }
            )
            matches = [
                item for item in entries if item["WORKLOAD"] == entry["WORKLOAD"]
            ]
            if matches and matches[0] != entry:
                raise ValueError(
                    f"Conflicting measured winners for {name} workload={entry['WORKLOAD']}"
                )
            if not matches:
                entries.append(entry)
        if entries:
            result[name] = entries
    if not result:
        raise ValueError("No measured GDN2 TLE Autotune results found")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="External native Autotune/JIT cache",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="New external YAML file; never overwritten",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Convert existing native results without running a workload",
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Original Benchmark flags after --",
    )
    args = parser.parse_args(argv)
    cache, output = args.cache_dir.resolve(), args.output_file.resolve()
    project = PROJECT_ROOT.resolve()
    if any(path == project or project in path.parents for path in (cache, output)):
        raise ValueError("Cache and output must be outside the repository")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output == cache or cache in output.parents:
        raise ValueError("YAML output must be outside the runtime cache")
    if not args.export_only:
        cache.mkdir(parents=True, exist_ok=True)
        arguments = args.benchmark_args
        if arguments and arguments[0] == "--":
            arguments = arguments[1:]
        arguments = arguments or DEFAULT_ARGUMENTS
        # A separate empty candidate file selects the unchanged full search spaces.
        # Never rename or modify the shipped production YAML to perform tuning.
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="gdn2-full-search-",
            suffix=".yaml",
            dir=cache.parent,
            delete=False,
        ) as empty:
            empty.write("{}\n")
        environment = dict(
            os.environ,
            TRITON_CACHE_DIR=str(cache),
            TRITON_CACHE_AUTOTUNING="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUNBUFFERED="1",
            FLAG_ATTN_GDN2_TUNE_CONFIG=empty.name,
        )
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(project / "src"), str(project), environment.get("PYTHONPATH", "")]
        )
        command = [
            sys.executable,
            "-c",
            "import sys; from benchmark.test_gdn2_benchmark import run_benchmark; run_benchmark(sys.argv[1:])",
            *arguments,
        ]
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="gdn2-offline-",
            suffix=".log",
            dir=cache.parent,
            delete=False,
        ) as log:
            print(
                f"cache_dir={cache}\nbenchmark_args={arguments}\nlog={log.name}",
                flush=True,
            )
            completed = subprocess.run(
                command,
                cwd=project,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode:
            print(
                f"Benchmark failed (exit {completed.returncode}); log retained at {log.name}",
                file=sys.stderr,
            )
            return completed.returncode
    document = collect_tuned_configs(cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(
            "# Copyright 2026 FlagOS Contributors\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
            "# One measured GDN2 TLE winner per complete native WORKLOAD key.\n"
        )
        yaml.safe_dump(document, stream, sort_keys=False)
    print(
        f"exported_kernels={len(document)} workloads={sum(map(len, document.values()))} output_file={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
