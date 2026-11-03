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

"""State-reset fixtures for GDN CUDA Graph measurements."""

import statistics
import time

import torch


def measure_providers(providers, *, state=None, cache="warm", warmup_ms=1000, rep_ms=100, rounds=7):
    """Return GPU milliseconds; state restoration and cache eviction are untimed."""
    original = state.clone() if state is not None else None
    eviction = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8) if cache == "cold" else None
    measurements = {}

    def capture(fn):
        if state is not None:
            state.copy_(original)
        fn()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            start.record()
            fn()
            end.record()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        return graph, start, end

    def samples(name, budget_ms):
        graph, start, end = measurements[name]
        stop = time.perf_counter() + budget_ms / 1000
        values = []
        while len(values) < 5 or time.perf_counter() < stop:
            if state is not None:
                state.copy_(original)
            if eviction is not None:
                eviction.zero_()
            graph.replay()
            end.synchronize()
            values.append(start.elapsed_time(end))
        return values

    try:
        for name, fn in providers.items():
            measurements[name] = capture(fn)
        for name in providers:
            samples(name, warmup_ms)
        medians = {name: [] for name in providers}
        counts = {name: [] for name in providers}
        names = list(providers)
        for index in range(rounds):
            order = names[index % len(names) :] + names[: index % len(names)]
            for name in order:
                values = samples(name, rep_ms)
                medians[name].append(statistics.median(values))
                counts[name].append(len(values))
        return {name: statistics.median(values) for name, values in medians.items()}, {
            "round_medians_ms": medians,
            "samples_per_round": counts,
            "warmup_ms": warmup_ms,
            "rep_ms": rep_ms,
            "cache": cache,
            "scope": "one operator call per CUDA Graph; external events; reset/eviction excluded",
        }
    finally:
        if state is not None:
            state.copy_(original)
