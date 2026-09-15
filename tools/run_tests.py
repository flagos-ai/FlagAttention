#!/usr/bin/env python3

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

"""Unified FlagAttention accuracy and performance test scheduler.

The operator inventory lives in ``conf/operators.yaml``.  In addition to the
metadata shared with FlagGems inventories, each entry contains ``tests`` (a
list of pytest paths/nodeids) and may contain ``benchmark`` (an executable
Python benchmark script).

Each requested GPU gets one worker process.  A worker exposes only its assigned
physical GPU to its test subprocess via ``CUDA_VISIBLE_DEVICES``; tests
therefore consistently use logical ``cuda:0``.  Operators are pulled from a
shared queue, so faster GPUs/workloads naturally pick up more work.

Output layout::

    <output>/
    |-- summary.json
    |-- summary0.json
    `-- <operator>/
        |-- accuracy_result.json
        |-- accuracy_junit.xml
        |-- performance_result.json
        |-- accuracy_stdout.log       # with --dump-output
        |-- accuracy_stderr.log       # with --dump-output
        |-- performance_stdout.log    # with --dump-output
        |-- performance_stderr.log    # with --dump-output
        `-- performance_artifacts/
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import platform
import queue as queue_module
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from importlib import metadata
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "conf" / "operators.yaml"

TIMEOUT = -100
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[93m"
CYAN = "\033[36m"
DIM = "\033[2m"
NC = "\033[0m"

IS_TTY = sys.stdout.isatty()
WORKER_PROCESSES: list[Any] = []
ACTIVE_CHILD: subprocess.Popen[bytes] | None = None


class InventoryError(ValueError):
    """Raised when the operator inventory is invalid."""


def pinfo(message: str) -> None:
    print(f"{GREEN}[INFO]{NC} {message}", flush=True)


def pwarn(message: str) -> None:
    print(f"{YELLOW}[WARN]{NC} {message}", flush=True)


def perror(message: str) -> None:
    print(f"{RED}[ERROR]{NC} {message}", file=sys.stderr, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o755)


def write_json(path: Path, data: dict[str, Any], *, atomic: bool = False) -> None:
    ensure_dir(path.parent)
    if not atomic:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        return

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def _relative_file(root: Path, value: str, field: str, op_id: str) -> Path:
    file_part = value.split("::", 1)[0]
    candidate = (root / file_part).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise InventoryError(
            f"operator {op_id!r} has {field} path outside the repository: {value!r}"
        ) from exc
    if not candidate.is_file():
        raise InventoryError(
            f"operator {op_id!r} references missing {field} file: {file_part!r}"
        )
    return candidate


def load_inventory(path: Path = INVENTORY) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise InventoryError(f"cannot load {path}: {exc}") from exc

    if not isinstance(document, dict) or not isinstance(document.get("ops"), list):
        raise InventoryError(f"{path} must contain a top-level 'ops' list")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(document["ops"]):
        if not isinstance(raw, dict):
            raise InventoryError(f"ops[{index}] must be a mapping")

        op_id = raw.get("id")
        if not isinstance(op_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", op_id):
            raise InventoryError(f"ops[{index}].id is missing or unsafe: {op_id!r}")
        if op_id in seen:
            raise InventoryError(f"duplicate operator id: {op_id!r}")
        seen.add(op_id)

        stages = raw.get("stages")
        if not isinstance(stages, list) or not stages:
            raise InventoryError(f"operator {op_id!r} must have a non-empty stages list")
        if not all(isinstance(stage, dict) and len(stage) == 1 for stage in stages):
            raise InventoryError(
                f"operator {op_id!r} stages must be single-key mappings"
            )
        current_stage = next(iter(stages[-1]))
        if current_stage not in {"alpha", "beta", "stable", "removed"}:
            raise InventoryError(
                f"operator {op_id!r} has unsupported current stage {current_stage!r}"
            )

        tests = raw.get("tests", [])
        if not isinstance(tests, list) or not all(
            isinstance(selector, str) and selector.strip() for selector in tests
        ):
            raise InventoryError(f"operator {op_id!r} tests must be a list of strings")
        tests = [selector.strip() for selector in tests]
        for selector in tests:
            _relative_file(ROOT, selector, "test", op_id)

        benchmark_value = raw.get("benchmark")
        benchmarks_value = raw.get("benchmarks")
        if benchmark_value is not None and benchmarks_value is not None:
            raise InventoryError(
                f"operator {op_id!r} cannot define both benchmark and benchmarks"
            )
        if benchmark_value is None:
            benchmarks = benchmarks_value or []
        elif isinstance(benchmark_value, str):
            benchmarks = [benchmark_value]
        else:
            raise InventoryError(f"operator {op_id!r} benchmark must be a string")
        if not isinstance(benchmarks, list) or not all(
            isinstance(item, str) and item.strip() for item in benchmarks
        ):
            raise InventoryError(
                f"operator {op_id!r} benchmarks must be a list of strings"
            )
        benchmarks = [item.strip() for item in benchmarks]
        for benchmark in benchmarks:
            _relative_file(ROOT, benchmark, "benchmark", op_id)

        benchmark_requires = raw.get("benchmark_requires", [])
        if not isinstance(benchmark_requires, list) or not all(
            isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", item)
            for item in benchmark_requires
        ):
            raise InventoryError(
                f"operator {op_id!r} benchmark_requires must be module names"
            )

        normalized.append(
            {
                **raw,
                "id": op_id,
                "tests": tests,
                "benchmarks": benchmarks,
                "benchmark_requires": benchmark_requires,
                "current_stage": current_stage,
            }
        )

    return normalized


def _requested_ids(args: argparse.Namespace) -> list[str] | None:
    if args.ops:
        return [item.strip() for item in args.ops.split(",") if item.strip()]

    if args.op_list_file:
        try:
            lines = Path(args.op_list_file).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise InventoryError(
                f"cannot read operator list {args.op_list_file!r}: {exc}"
            ) from exc
        requested = []
        for line in lines:
            value = line.partition("#")[0].strip()
            if value:
                requested.append(value)
        return requested

    return None


def select_operators(
    catalog: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    requested = _requested_ids(args)
    by_id = {operator["id"]: operator for operator in catalog}

    if requested is not None:
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise InventoryError(f"unknown operator id(s): {', '.join(unknown)}")
        selected = [by_id[op_id] for op_id in requested]
    else:
        requested_stages = [
            value.strip().lower() for value in args.stages.split(",") if value.strip()
        ]
        supported = {"alpha", "beta", "stable", "removed", "all"}
        invalid = sorted(set(requested_stages) - supported)
        if invalid:
            raise InventoryError(f"unsupported stage(s): {', '.join(invalid)}")
        if not requested_stages:
            requested_stages = ["stable"]
        effective_stages = (
            {"alpha", "beta", "stable"}
            if "all" in requested_stages
            else set(requested_stages)
        )
        selected = [
            operator
            for operator in catalog
            if operator["current_stage"] in effective_stages
        ]

    if args.start:
        selected = [operator for operator in selected if operator["id"] >= args.start]

    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for operator in selected:
        if operator["id"] not in seen:
            deduplicated.append(operator)
            seen.add(operator["id"])
    return deduplicated


def print_operator_list(operators: list[dict[str, Any]]) -> None:
    if not operators:
        print("No operators selected.")
        return
    width = max(len(operator["id"]) for operator in operators)
    for operator in operators:
        benchmark = "yes" if operator["benchmarks"] else "no"
        print(
            f"{operator['id']:<{width}}  "
            f"stage={operator['current_stage']:<6}  "
            f"tests={len(operator['tests']):>2}  benchmark={benchmark}"
        )


def probe_environment() -> dict[str, Any]:
    try:
        os_release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        os_release = {}

    environment: dict[str, Any] = {
        "architecture": platform.machine(),
        "os_name": os_release.get("ID", platform.system()),
        "os_release": os_release.get("VERSION_ID", platform.release()),
        "python": platform.python_version(),
    }

    if importlib.util.find_spec("pytest") is None:
        raise RuntimeError("pytest is not installed; install FlagAttention's test extra")
    try:
        environment["pytest"] = metadata.version("pytest")
    except metadata.PackageNotFoundError:
        environment["pytest"] = "unknown"

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch cannot be imported: {exc}") from exc

    device_count = torch.cuda.device_count()
    torch_info: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": device_count,
        "devices": [],
    }
    for index in range(device_count):
        try:
            torch_info["devices"].append(torch.cuda.get_device_name(index))
        except Exception:
            torch_info["devices"].append("unknown")
    environment["torch"] = torch_info

    if not torch_info["cuda_available"] or device_count == 0:
        raise RuntimeError("FlagAttention tests require at least one CUDA device")
    if importlib.util.find_spec("triton") is None:
        raise RuntimeError("Triton cannot be imported; install a compatible Triton runtime")
    try:
        environment["triton"] = metadata.version("triton")
    except metadata.PackageNotFoundError:
        environment["triton"] = "compatible runtime"

    try:
        flag_attn_version = metadata.version("flag_attn")
    except metadata.PackageNotFoundError:
        flag_attn_version = "source tree"
    environment["flag_attn"] = {"version": flag_attn_version}
    return environment


def parse_gpu_ids(specification: str, device_count: int) -> list[int]:
    if specification.strip().lower() == "all":
        return list(range(device_count))

    parts = [part.strip() for part in specification.split(",") if part.strip()]
    if not parts:
        raise ValueError("the GPU list is empty")
    try:
        gpu_ids = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("GPU IDs must be non-negative integers") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("GPU IDs must be non-negative integers")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must not contain duplicates")

    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id >= device_count]
    if invalid:
        raise ValueError(
            f"GPU ID(s) outside detected range 0..{device_count - 1}: "
            f"{', '.join(map(str, invalid))}"
        )
    return gpu_ids


def subprocess_environment(root: Path, gpu_id: int) -> dict[str, str]:
    environment = os.environ.copy()
    current_mask = environment.get("CUDA_VISIBLE_DEVICES", "")
    visible_devices = [value.strip() for value in current_mask.split(",") if value.strip()]
    # GPU IDs are logical IDs in the process that launches this runner.  Preserve
    # an existing container/scheduler mask by mapping the requested logical ID
    # back to its token (which may be a physical index, UUID, or MIG identifier).
    selected_device = visible_devices[gpu_id] if visible_devices else str(gpu_id)
    environment["CUDA_VISIBLE_DEVICES"] = selected_device
    environment["PYTHONUNBUFFERED"] = "1"
    source_path = str(root / "src")
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path + os.pathsep + current_pythonpath
        if current_pythonpath
        else source_path
    )
    return environment


def terminate_child(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run_command(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout: int,
    output_dir: Path,
    flavor: str,
    dump_output: bool,
    append: bool = False,
) -> tuple[int, float, int]:
    """Run one command and return (exit_code, duration_seconds, output_bytes)."""

    global ACTIVE_CHILD

    ensure_dir(output_dir)
    stdout_stream: Any
    stderr_stream: Any
    if dump_output:
        mode = "ab" if append else "wb"
        stdout_stream = (output_dir / f"{flavor}_stdout.log").open(mode)
        stderr_stream = (output_dir / f"{flavor}_stderr.log").open(mode)
        header = f"[CMD] {shlex.join(command)}\n[CWD] {cwd}\n\n".encode()
        stderr_stream.write(header)
        stderr_stream.flush()
    else:
        stdout_stream = tempfile.TemporaryFile(mode="w+b", dir=output_dir)
        stderr_stream = tempfile.TemporaryFile(mode="w+b", dir=output_dir)

    stdout_start = stdout_stream.tell()
    stderr_start = stderr_stream.tell()
    started = time.monotonic()
    try:
        ACTIVE_CHILD = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=(os.name == "posix"),
        )
        try:
            ACTIVE_CHILD.wait(timeout=timeout)
            exit_code = ACTIVE_CHILD.returncode
        except subprocess.TimeoutExpired:
            terminate_child(ACTIVE_CHILD)
            exit_code = TIMEOUT
    except Exception as exc:
        message = f"failed to start command: {type(exc).__name__}: {exc}\n".encode()
        stderr_stream.write(message)
        stderr_stream.flush()
        exit_code = -1
    finally:
        duration = time.monotonic() - started
        stdout_stream.flush()
        stderr_stream.flush()
        output_bytes = max(0, stdout_stream.tell() - stdout_start) + max(
            0, stderr_stream.tell() - stderr_start
        )
        ACTIVE_CHILD = None
        stdout_stream.close()
        stderr_stream.close()
    return exit_code, duration, output_bytes


def _element_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def parse_junit(path: Path, exit_code: int) -> dict[str, Any]:
    if exit_code == TIMEOUT:
        return {
            "status": "Timeout",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "details": {},
        }
    if not path.is_file():
        status = "NotFound" if exit_code == 5 else "Error"
        return {
            "status": status,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0 if status == "NotFound" else 1,
            "skipped": 0,
            "details": {"error": "pytest did not produce JUnit XML"},
        }

    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return {
            "status": "Error",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "details": {"error": f"invalid JUnit XML: {exc}"},
        }

    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    details: dict[str, list[dict[str, str]]] = {}
    testcases = [element for element in root.iter() if _element_name(element) == "testcase"]
    for testcase in testcases:
        classname = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "unknown")
        nodeid = f"{classname}::{name}" if classname else name
        outcome = "passed"
        outcome_element: ET.Element | None = None
        for child in testcase:
            child_name = _element_name(child)
            if child_name == "failure":
                outcome = "failed"
                outcome_element = child
                break
            if child_name == "error":
                outcome = "errors"
                outcome_element = child
                break
            if child_name == "skipped":
                outcome = "skipped"
                outcome_element = child
                break
        counts[outcome] += 1
        if outcome_element is not None:
            message = outcome_element.attrib.get("message") or (
                outcome_element.text or ""
            ).strip()
            details.setdefault(outcome, []).append(
                {"test": nodeid, "reason": message[:4000]}
            )

    total = sum(counts.values())
    if exit_code in {2, 3, 4} or exit_code < 0:
        status = "Error"
    elif counts["errors"]:
        status = "Error"
    elif counts["failed"] or exit_code == 1:
        status = "Failed"
    elif exit_code == 5 or total == 0:
        status = "NotFound"
    elif exit_code != 0:
        status = "Error"
    elif counts["passed"]:
        status = "Passed"
    else:
        status = "Skipped"

    if exit_code != 0:
        details.setdefault("pytest", []).append(
            {"test": "pytest", "reason": f"exit code {exit_code}"}
        )
    return {"status": status, "total": total, **counts, "details": details}


def run_accuracy(
    operator: dict[str, Any], gpu_id: int, config: dict[str, Any]
) -> dict[str, Any]:
    op_id = operator["id"]
    output_root = Path(config["output_dir"])
    op_dir = output_root / op_id
    ensure_dir(op_dir)
    result_path = op_dir / "accuracy_result.json"
    junit_path = op_dir / "accuracy_junit.xml"

    if not operator["tests"]:
        result = {
            "status": "NotFound",
            "exit_code": 5,
            "duration": 0.0,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "details": {"reason": "no tests configured"},
        }
        write_json(result_path, result)
        return result

    if junit_path.exists():
        junit_path.unlink()
    command = [
        config["python"],
        "-m",
        "pytest",
        *operator["tests"],
        f"--junitxml={junit_path}",
        "-p",
        "no:cacheprovider",
        "--tb=short",
        "-ra",
    ]
    if config["dump_output"]:
        # Match the reference runners' -s behavior so prints from passing tests
        # are present in accuracy_stdout.log as well as failure diagnostics.
        command.append("-s")
    exit_code, duration, _ = run_command(
        command,
        cwd=Path(config["root"]),
        environment=subprocess_environment(Path(config["root"]), gpu_id),
        timeout=config["accuracy_timeout"],
        output_dir=op_dir,
        flavor="accuracy",
        dump_output=config["dump_output"],
    )
    result = parse_junit(junit_path, exit_code)
    result.update(
        {
            "exit_code": exit_code,
            "duration": round(duration, 3),
            "command": command,
            "data_file": (
                str(junit_path.relative_to(output_root)) if junit_path.exists() else None
            ),
        }
    )
    write_json(result_path, result)
    return result


def run_performance(
    operator: dict[str, Any], gpu_id: int, config: dict[str, Any]
) -> dict[str, Any]:
    op_id = operator["id"]
    output_root = Path(config["output_dir"])
    op_dir = output_root / op_id
    result_path = op_dir / "performance_result.json"
    benchmarks = operator["benchmarks"]

    if config["skip_benchmarks"]:
        result = {
            "status": "Skipped",
            "exit_code": 0,
            "duration": 0.0,
            "details": {"reason": "benchmarks disabled by --skip-benchmarks"},
        }
        write_json(result_path, result)
        return result
    if not benchmarks:
        result = {
            "status": "NotFound",
            "exit_code": 0,
            "duration": 0.0,
            "details": {"reason": "no benchmark configured"},
        }
        write_json(result_path, result)
        return result

    missing_requirements = [
        module
        for module in operator["benchmark_requires"]
        if importlib.util.find_spec(module) is None
    ]
    if missing_requirements:
        result = {
            "status": "Skipped",
            "exit_code": 0,
            "duration": 0.0,
            "details": {
                "reason": "missing optional benchmark module(s): "
                + ", ".join(missing_requirements)
            },
        }
        write_json(result_path, result)
        return result

    artifacts_dir = op_dir / "performance_artifacts"
    ensure_dir(artifacts_dir)
    records: list[dict[str, Any]] = []
    for index, benchmark in enumerate(benchmarks):
        script = (Path(config["root"]) / benchmark).resolve()
        command = [config["python"], "-u", str(script)]
        artifacts_before = {
            path.relative_to(artifacts_dir): (path.stat().st_mtime_ns, path.stat().st_size)
            for path in artifacts_dir.rglob("*")
            if path.is_file()
        }
        exit_code, duration, output_bytes = run_command(
            command,
            cwd=artifacts_dir,
            environment=subprocess_environment(Path(config["root"]), gpu_id),
            timeout=config["benchmark_timeout"],
            output_dir=op_dir,
            flavor="performance",
            dump_output=config["dump_output"],
            append=index > 0,
        )
        artifacts_after = {
            path.relative_to(artifacts_dir): (path.stat().st_mtime_ns, path.stat().st_size)
            for path in artifacts_dir.rglob("*")
            if path.is_file()
        }
        artifacts_changed = sorted(
            str(path)
            for path, signature in artifacts_after.items()
            if artifacts_before.get(path) != signature
        )
        if exit_code == TIMEOUT:
            status = "Timeout"
        elif exit_code != 0:
            status = "Failed"
        elif output_bytes == 0 and not artifacts_changed:
            status = "Skipped"
        else:
            status = "Passed"
        records.append(
            {
                "script": benchmark,
                "command": command,
                "status": status,
                "exit_code": exit_code,
                "duration": round(duration, 3),
                "output_bytes": output_bytes,
                "artifacts_changed": artifacts_changed,
                "measurement": (
                    "artifacts"
                    if artifacts_changed
                    else "console"
                    if output_bytes
                    else "none"
                ),
            }
        )

    statuses = {record["status"] for record in records}
    if "Timeout" in statuses:
        overall_status = "Timeout"
    elif "Failed" in statuses:
        overall_status = "Failed"
    elif statuses == {"Skipped"}:
        overall_status = "Skipped"
    else:
        overall_status = "Passed"
    result = {
        "status": overall_status,
        "exit_code": next(
            (record["exit_code"] for record in records if record["exit_code"] != 0),
            0,
        ),
        "duration": round(sum(record["duration"] for record in records), 3),
        "details": records,
        "artifacts_dir": str(artifacts_dir.relative_to(output_root)),
    }
    write_json(result_path, result)
    return result


def error_result(reason: str) -> dict[str, Any]:
    return {
        "status": "Error",
        "exit_code": -1,
        "duration": 0.0,
        "details": {"error": reason},
    }


def worker_proc(
    gpu_id: int,
    work_queue: Any,
    display_queue: Any,
    config: dict[str, Any],
) -> None:
    def stop_worker(signum: int, _frame: Any) -> None:
        terminate_child(ACTIVE_CHILD)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_worker)
    signal.signal(signal.SIGINT, stop_worker)
    worker_results: dict[str, Any] = {}
    summary_path = Path(config["output_dir"]) / f"summary{gpu_id}.json"
    try:
        while True:
            operator = work_queue.get()
            if operator is None:
                break
            op_id = operator["id"]
            try:
                display_queue.put(("start", gpu_id, "accuracy", op_id))
                accuracy = run_accuracy(operator, gpu_id, config)
            except Exception as exc:
                accuracy = error_result(f"{type(exc).__name__}: {exc}")
                write_json(
                    Path(config["output_dir"]) / op_id / "accuracy_result.json",
                    accuracy,
                )
            display_queue.put(
                (
                    "done",
                    gpu_id,
                    "accuracy",
                    op_id,
                    accuracy["status"],
                    accuracy["duration"],
                )
            )

            try:
                display_queue.put(("start", gpu_id, "performance", op_id))
                performance = run_performance(operator, gpu_id, config)
            except Exception as exc:
                performance = error_result(f"{type(exc).__name__}: {exc}")
                write_json(
                    Path(config["output_dir"]) / op_id / "performance_result.json",
                    performance,
                )
            display_queue.put(
                (
                    "done",
                    gpu_id,
                    "performance",
                    op_id,
                    performance["status"],
                    performance["duration"],
                )
            )

            worker_results[op_id] = {
                "customized": True,
                "accuracy": accuracy,
                "performance": performance,
            }
            write_json(summary_path, worker_results, atomic=True)
    finally:
        display_queue.put(("exit", gpu_id))


def _status_text(status: str, duration: float) -> str:
    mapping = {
        "Passed": (GREEN, "OK"),
        "Failed": (RED, "FAILED"),
        "Timeout": (RED, "TIMEOUT"),
        "Error": (RED, "ERROR"),
        "NotFound": (YELLOW, "NOTFOUND"),
        "Skipped": (YELLOW, "SKIPPED"),
    }
    color, label = mapping.get(status, (YELLOW, status.upper()))
    return f"{color}[{label:<8} {duration:>7.1f}s]{NC}"


def display_loop(
    display_queue: Any,
    workers: dict[int, Any],
    operator_count: int,
) -> None:
    finished_workers: set[int] = set()
    completed_steps = 0
    total_steps = operator_count * 2
    while len(finished_workers) < len(workers):
        try:
            message = display_queue.get(timeout=1)
        except queue_module.Empty:
            for gpu_id, process in workers.items():
                if gpu_id not in finished_workers and not process.is_alive():
                    perror(
                        f"GPU {gpu_id} worker exited unexpectedly "
                        f"with code {process.exitcode}"
                    )
                    finished_workers.add(gpu_id)
            continue

        kind = message[0]
        if kind == "exit":
            gpu_id = message[1]
            finished_workers.add(gpu_id)
            if IS_TTY:
                pinfo(f"GPU {gpu_id} worker finished")
            continue
        if kind == "start":
            if not IS_TTY:
                _, gpu_id, phase, op_id = message
                timestamp = dt.datetime.now().strftime("%H:%M:%S")
                pinfo(f"[{timestamp}][GPU {gpu_id}] {phase:<11} {op_id} ...")
            continue
        if kind != "done":
            continue

        _, gpu_id, phase, op_id, status, duration = message
        completed_steps += 1
        percentage = completed_steps * 100 // total_steps if total_steps else 100
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        print(
            f"{GREEN}[INFO]{NC} [{timestamp}][GPU {gpu_id}] "
            f"{phase:<11} {op_id:<40} {_status_text(status, duration)} "
            f"({percentage:>3}%)",
            flush=True,
        )


def terminate_workers() -> None:
    for process in WORKER_PROCESSES:
        if process.is_alive():
            process.terminate()
    for process in WORKER_PROCESSES:
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def aggregate_results(
    operators: list[dict[str, Any]],
    gpu_ids: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for gpu_id in gpu_ids:
        path = output_dir / f"summary{gpu_id}.json"
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as stream:
                worker_results = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            perror(f"cannot read {path}: {exc}")
            continue
        if isinstance(worker_results, dict):
            results.update(worker_results)

    for operator in operators:
        if operator["id"] not in results:
            reason = "operator was not completed by any worker"
            results[operator["id"]] = {
                "customized": True,
                "accuracy": error_result(reason),
                "performance": error_result(reason),
            }
    return results


def write_incomplete_summary(
    *,
    status: str,
    reason: str,
    started_at: dt.datetime,
    environment: dict[str, Any],
    operators: list[dict[str, Any]],
    gpu_ids: list[int],
    output_dir: Path,
) -> None:
    finished_at = dt.datetime.now()
    duration = round((finished_at - started_at).total_seconds(), 2)
    write_json(
        output_dir / "summary.json",
        {
            "status": status,
            "reason": reason,
            "timestamp": finished_at.strftime("%Y-%m-%d %H:%M:%S"),
            "start_time": started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": str(dt.timedelta(seconds=int(duration))),
            "total_duration_seconds": duration,
            "env": environment,
            "result": aggregate_results(operators, gpu_ids, output_dir),
        },
        atomic=True,
    )


def configure_colors(mode: str) -> None:
    global RED, GREEN, YELLOW, CYAN, DIM, NC
    use_colors = mode == "always" or (mode == "auto" and IS_TTY)
    if not use_colors:
        RED = GREEN = YELLOW = CYAN = DIM = NC = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "FlagAttention operator accuracy and performance scheduler. "
            "Operators are loaded from conf/operators.yaml and distributed "
            "across the requested GPUs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--ops", help="comma-separated operator IDs")
    source.add_argument(
        "--op-list-file", metavar="FILE", help="operator IDs, one per line"
    )
    parser.add_argument(
        "--start",
        metavar="OP_ID",
        help="only run selected operator IDs lexicographically >= OP_ID",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        metavar="GPUS",
        help='comma-separated GPU IDs, or "all"',
    )
    parser.add_argument(
        "--stages",
        default="stable",
        metavar="STAGES",
        help="comma-separated alpha,beta,stable,removed, or all",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        metavar="DIR",
        help="result directory (both spellings are supported)",
    )
    parser.add_argument(
        "--dump-output",
        action="store_true",
        help="save subprocess stdout and stderr for every operator",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="run accuracy tests only",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        metavar="SECONDS",
        help="timeout for each operator's pytest command",
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="timeout for each benchmark script",
    )
    parser.add_argument(
        "--list-ops",
        action="store_true",
        help="list selected inventory entries without probing GPUs",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="ANSI color mode",
    )
    return parser


def has_failures(results: dict[str, Any]) -> bool:
    bad_statuses = {"Failed", "Error", "Timeout"}
    return any(
        record.get(phase, {}).get("status") in bad_statuses
        for record in results.values()
        for phase in ("accuracy", "performance")
    )


def main(argv: list[str] | None = None) -> int:
    def raise_interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, raise_interrupt)
    args = build_parser().parse_args(argv)
    configure_colors(args.color)
    if args.timeout <= 0 or args.benchmark_timeout <= 0:
        perror("timeouts must be positive integers")
        return 2

    try:
        catalog = load_inventory()
        operators = select_operators(catalog, args)
    except InventoryError as exc:
        perror(str(exc))
        return 2

    if args.list_ops:
        print_operator_list(operators)
        return 0
    if not operators:
        perror("no operators selected")
        return 2

    started_at = dt.datetime.now()
    pinfo(f"Test started at ... {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        environment = probe_environment()
        gpu_ids = parse_gpu_ids(args.gpus, environment["torch"]["device_count"])
    except (RuntimeError, ValueError) as exc:
        perror(str(exc))
        return 2

    if len(gpu_ids) > len(operators):
        gpu_ids = gpu_ids[: len(operators)]
    pinfo(
        f"Testing {len(operators)} operators on GPU(s) "
        f"{', '.join(map(str, gpu_ids))}"
    )

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = started_at.strftime("%Y%m%d_%H%M")
        output_dir = (Path.cwd() / f"logs_results_{stamp}").resolve()
    ensure_dir(output_dir)

    # Known log files are per-run data.  Remove them for selected operators so
    # an output directory reused without --dump-output never exposes stale logs.
    for operator in operators:
        op_dir = output_dir / operator["id"]
        for log_name in (
            "accuracy_stdout.log",
            "accuracy_stderr.log",
            "performance_stdout.log",
            "performance_stderr.log",
        ):
            log_path = op_dir / log_name
            if log_path.is_file():
                log_path.unlink()

    config = {
        "root": str(ROOT),
        "output_dir": str(output_dir),
        "python": sys.executable,
        "dump_output": args.dump_output,
        "skip_benchmarks": args.skip_benchmarks,
        "accuracy_timeout": args.timeout,
        "benchmark_timeout": args.benchmark_timeout,
    }
    write_json(
        output_dir / "run_config.json",
        {
            "operators": [operator["id"] for operator in operators],
            "gpus": gpu_ids,
            "stages": args.stages,
            "dump_output": args.dump_output,
            "skip_benchmarks": args.skip_benchmarks,
            "accuracy_timeout": args.timeout,
            "benchmark_timeout": args.benchmark_timeout,
        },
    )
    # Initialize this run's summaries before workers start.  This prevents a
    # reused output directory from contributing stale per-GPU results if a
    # worker exits before completing its first operator.
    for gpu_id in gpu_ids:
        write_json(output_dir / f"summary{gpu_id}.json", {})
    write_json(
        output_dir / "summary.json",
        {
            "status": "running",
            "start_time": started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "env": environment,
            "result": {},
        },
        atomic=True,
    )

    context = get_context("spawn")
    work_queue = context.Queue()
    display_queue = context.Queue()
    for operator in operators:
        work_queue.put(operator)
    for _ in gpu_ids:
        work_queue.put(None)

    workers: dict[int, Any] = {}
    try:
        for gpu_id in gpu_ids:
            process = context.Process(
                target=worker_proc,
                args=(gpu_id, work_queue, display_queue, config),
                name=f"flagattention-gpu-{gpu_id}",
            )
            process.start()
            workers[gpu_id] = process
            WORKER_PROCESSES.append(process)
        display_loop(display_queue, workers, len(operators))
        for process in workers.values():
            process.join()
    except KeyboardInterrupt:
        pwarn("Interrupted; terminating workers ...")
        terminate_workers()
        write_incomplete_summary(
            status="interrupted",
            reason="received an interrupt signal",
            started_at=started_at,
            environment=environment,
            operators=operators,
            gpu_ids=gpu_ids,
            output_dir=output_dir,
        )
        return 130
    except Exception as exc:
        perror(f"worker startup or scheduling failed: {type(exc).__name__}: {exc}")
        terminate_workers()
        write_incomplete_summary(
            status="error",
            reason=f"{type(exc).__name__}: {exc}",
            started_at=started_at,
            environment=environment,
            operators=operators,
            gpu_ids=gpu_ids,
            output_dir=output_dir,
        )
        return 1
    finally:
        work_queue.close()
        display_queue.close()

    finished_at = dt.datetime.now()
    duration = round((finished_at - started_at).total_seconds(), 2)
    results = aggregate_results(operators, gpu_ids, output_dir)
    summary = {
        "status": "failed" if has_failures(results) else "passed",
        "timestamp": finished_at.strftime("%Y-%m-%d %H:%M:%S"),
        "start_time": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration": str(dt.timedelta(seconds=int(duration))),
        "total_duration_seconds": duration,
        "env": environment,
        "result": results,
    }
    write_json(output_dir / "summary.json", summary, atomic=True)
    pinfo(f"Results written to {output_dir}")
    pinfo(f"Total elapsed time ... {summary['total_duration']} ({duration}s)")
    if summary["status"] == "failed":
        perror("Test run completed with failures; see summary.json")
        return 1
    pinfo("Test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
