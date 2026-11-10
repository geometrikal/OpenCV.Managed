#!/usr/bin/env python3
"""Merges BenchmarkDotNet and cv2 timings into one markdown table.

Inputs:
* ``BenchmarkDotNet.Artifacts/results/*-report-full-compressed.json``
* ``results/cv2-<mode>.json`` (written by ``tools/bench_cv2.py``)

Output: ``results/comparison.md``. The ratio columns are
managed time / cv2 time; a value above 1 means the managed port is slower.

Usage (from the Geometrikal.OpenCV.Managed.Benchmark directory):
    python tools/compare.py
"""

import argparse
import glob
import json
import os

MODES = ("baseline", "single", "default")
MODE_LABELS = {
    "baseline": "cv2 1T SSE no-IPP",
    "single": "cv2 1T AVX2+IPP",
    "default": "cv2 default",
}


def load_dotnet(artifacts):
    rows = {}
    env = None
    for path in glob.glob(os.path.join(artifacts, "results", "*-report-full-compressed.json")):
        with open(path) as f:
            data = json.load(f)
        env = env or data.get("HostEnvironmentInfo")
        for b in data["Benchmarks"]:
            stats = b.get("Statistics")
            if not stats:
                continue
            case = f"{b['Type'].removesuffix('Benchmarks')}.{b['Method']}"
            size = int(b["Parameters"].split("=")[1])
            alloc = (b.get("Memory") or {}).get("BytesAllocatedPerOperation")
            rows[(case, size)] = {"median_ns": stats["Median"], "alloc": alloc}
    return rows, env


def load_cv2(results_dir):
    out = {}
    for mode in MODES:
        path = os.path.join(results_dir, f"cv2-{mode}.json")
        if os.path.exists(path):
            with open(path) as f:
                out[mode] = json.load(f)
    return out


def fmt_time(ns):
    if ns is None:
        return "-"
    if ns >= 1e6:
        return f"{ns / 1e6:.2f} ms"
    return f"{ns / 1e3:.1f} us"


def fmt_ratio(a, b):
    if a is None or b is None or b == 0:
        return "-"
    return f"{a / b:.2f}x"


def main():
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=os.path.join(here, "BenchmarkDotNet.Artifacts"))
    parser.add_argument("--mt-artifacts", default=os.path.join(here, "BenchmarkDotNet.Artifacts-mt"),
                        help="results of `dotnet run -c Release -- --threads all`; optional")
    parser.add_argument("--results", default=os.path.join(here, "results"))
    args = parser.parse_args()

    dotnet, env = load_dotnet(args.artifacts)
    dotnet_mt, _ = load_dotnet(args.mt_artifacts) if os.path.isdir(args.mt_artifacts) else ({}, None)
    cv2_runs = load_cv2(args.results)
    cv2 = {mode: {(r["case"], r["size"]): r["median_ns"] for r in run["results"]} for mode, run in cv2_runs.items()}
    modes = [m for m in MODES if m in cv2]

    keys = sorted(set(dotnet) | {k for m in modes for k in cv2[m]},
                  key=lambda k: (k[0].split(".")[0], k[1], k[0]))

    lines = ["# Managed port vs cv2 benchmark", ""]
    if env:
        lines.append(f"- .NET: {env.get('RuntimeVersion')}, {env.get('Architecture')}, "
                     f"CPU: {env.get('ProcessorName')} ({env.get('LogicalCoreCount')} logical cores)")
    for m in modes:
        run = cv2_runs[m]
        lines.append(f"- {MODE_LABELS[m]}: cv2 {run['cv2']}, threads={run['threads']}, ipp={run['ipp']}, "
                     f"OPENCV_CPU_DISABLE='{run['cpu_disable']}', Python {run['python']}")
    lines += [
        "- Times are medians per call. Ratio = managed / cv2; above 1.00x means the managed port is slower.",
        "- Managed is single-threaded. Managed MT uses all cores (`--threads all`) and is compared with cv2 default.",
        "- Drawing cases draw 64 shapes per call; cv2 times include the Python loop overhead.",
        "",
    ]

    section = None
    header = ["Case", "Size", "Managed", "Managed alloc"] + [MODE_LABELS[m] for m in modes] \
        + [f"Ratio vs {MODE_LABELS[m]}" for m in modes]
    if dotnet_mt:
        header += ["Managed MT", "Ratio MT vs cv2 default"]
    for case, size in keys:
        group = case.split(".")[0]
        if group != section:
            section = group
            lines += [f"## {group}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        d = dotnet.get((case, size), {})
        managed = d.get("median_ns")
        alloc = d.get("alloc")
        alloc_text = "-" if alloc is None else f"{alloc / 1024:.0f} KB"
        cells = [case.split(".", 1)[1], str(size), fmt_time(managed), alloc_text]
        cells += [fmt_time(cv2[m].get((case, size))) for m in modes]
        cells += [fmt_ratio(managed, cv2[m].get((case, size))) for m in modes]
        if dotnet_mt:
            mt = dotnet_mt.get((case, size), {}).get("median_ns")
            cells += [fmt_time(mt), fmt_ratio(mt, cv2.get("default", {}).get((case, size)))]
        lines.append("| " + " | ".join(cells) + " |")
        if (case, size) == keys[-1] or keys[keys.index((case, size)) + 1][0].split(".")[0] != group:
            lines.append("")

    os.makedirs(args.results, exist_ok=True)
    out = os.path.join(args.results, "comparison.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWritten to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
