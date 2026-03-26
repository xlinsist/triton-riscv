#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "python" / "examples"

MODES = ("vir_vector", "linalg_loops")
WORKLOADS = ("matmul", "softmax", "layernorm", "vecadd")


@dataclass
class Case:
    name: str
    params: dict[str, Any]


def _cases_for(workload: str) -> list[Case]:
    if workload == "matmul":
        return [
            Case("n256", {"n": 256}),
            Case("n384", {"n": 384}),
            Case("n512", {"n": 512}),
            Case("n640", {"n": 640}),
            Case("n768", {"n": 768}),
        ]
    if workload == "softmax":
        return [
            Case("m1024_n1024", {"m": 1024, "n": 1024}),
            Case("m2048_n1024", {"m": 2048, "n": 1024}),
            Case("m2048_n2048", {"m": 2048, "n": 2048}),
        ]
    if workload == "layernorm":
        return [
            Case("m1151_n1024", {"m": 1151, "n": 1024}),
            Case("m1151_n4096", {"m": 1151, "n": 4096}),
            Case("m1151_n8192", {"m": 1151, "n": 8192}),
        ]
    if workload == "vecadd":
        return [
            Case("numel_2p20", {"numel": 2**20}),
            Case("numel_2p22", {"numel": 2**22}),
            Case("numel_2p24", {"numel": 2**24}),
        ]
    raise ValueError(f"Unsupported workload: {workload}")


def _required_env() -> list[str]:
    return ["TRITON_SHARED_OPT_PATH", "BUDDY_MLIR_BINARY_DIR", "LLVM_BINARY_DIR"]


def _snapshot_env() -> dict[str, str]:
    keys = [
        "TRITON_PLUGIN_DIRS",
        "TRITON_SHARED_OPT_PATH",
        "BUDDY_MLIR_BINARY_DIR",
        "LLVM_BINARY_DIR",
        "TRITON_RISCV_LOWERING_MODE",
        "TORCH_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    return {k: os.environ.get(k, "") for k in keys}


def _git_snapshot() -> dict[str, str]:
    out = {}
    for key, cmd in (
        ("head", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ("status_short", ["git", "status", "--short"]),
    ):
        try:
            value = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()
        except Exception:
            value = ""
        out[key] = value
    return out


def _kernel_name(workload: str) -> str:
    return {
        "matmul": "matmul_kernel",
        "softmax": "softmax_kernel",
        "layernorm": "_layer_norm_fwd_fused",
        "vecadd": "add_kernel",
    }[workload]


def _find_latest_kernel_cache_files(triton_home: Path, kernel: str) -> dict[str, Path]:
    cache_root = triton_home / ".triton" / "cache"
    patterns = ["ttir", "ttsharedir", "llir", "obj", "json", "source"]
    out = {}
    for ext in patterns:
        candidates = list(cache_root.rglob(f"{kernel}.{ext}"))
        if not candidates:
            continue
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        out[ext] = candidates[0]
    return out


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _worker_run(workload: str, case: dict[str, Any], warmup: int, repeats: int) -> dict[str, Any]:
    sys.path.insert(0, str(EXAMPLES_DIR))

    import torch
    import benchmark

    benchmark.select_cpu_backend()
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

    if workload == "matmul":
        import test_matmul as module

        n = int(case["n"])
        a = torch.randn((n, n), device="cpu", dtype=torch.float32)
        b = torch.randn((n, n), device="cpu", dtype=torch.float32)
        fn = lambda: module.matmul(a, b)
    elif workload == "softmax":
        import test_softmax as module

        m = int(case["m"])
        n = int(case["n"])
        x = torch.randn((m, n), device="cpu", dtype=torch.float32)
        fn = lambda: module.softmax(x)
    elif workload == "layernorm":
        import test_layernorm as module

        m = int(case["m"])
        n = int(case["n"])
        x_shape = (m, n)
        w_shape = (n,)
        weight = torch.rand(w_shape, dtype=torch.float16, device="cpu", requires_grad=False)
        bias = torch.rand(w_shape, dtype=torch.float16, device="cpu", requires_grad=False)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=torch.float16, device="cpu")
        fn = lambda: module.LayerNorm.apply(x, w_shape, weight, bias, 1e-5, "cpu")
    elif workload == "vecadd":
        import test_vec_add as module

        numel = int(case["numel"])
        x = torch.rand(numel, device="cpu", dtype=torch.float32)
        y = torch.rand(numel, device="cpu", dtype=torch.float32)
        fn = lambda: module.add(x, y)
    else:
        raise ValueError(f"Unsupported workload: {workload}")

    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    return {
        "workload": workload,
        "case": case,
        "warmup": warmup,
        "repeats": repeats,
        "times": times,
        "avg_s": statistics.fmean(times),
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "std_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "p95_s": sorted(times)[max(0, int(0.95 * len(times)) - 1)],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIR-vs-loops A/B performance benchmark.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for worker subprocesses")
    parser.add_argument("--run-id", default="", help="Custom run id (default: timestamp)")
    parser.add_argument("--out-root", default=str(Path(__file__).resolve().parents[1] / "ab-vir-vs-loops"))
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOADS), choices=list(WORKLOADS))
    parser.add_argument("--independent-runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-workload", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-warmup", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-repeats", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def _set_thread_env(env: dict[str, str], threads: int) -> None:
    t = str(threads)
    env["TORCH_NUM_THREADS"] = t
    env["OMP_NUM_THREADS"] = t
    env["OPENBLAS_NUM_THREADS"] = t
    env["MKL_NUM_THREADS"] = t
    env["NUMEXPR_NUM_THREADS"] = t


def main() -> int:
    args = _parse_args()

    if args.worker:
        case = json.loads(args.worker_case)
        result = _worker_run(
            args.worker_workload,
            case,
            warmup=args.worker_warmup,
            repeats=args.worker_repeats,
        )
        print(json.dumps(result))
        return 0

    missing = [k for k in _required_env() if not os.environ.get(k)]
    if missing:
        raise SystemExit("Missing required env vars: " + ", ".join(missing))

    run_id = args.run_id or time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = Path(args.out_root) / run_id
    raw_dir = run_dir / "raw"
    reports_dir = run_dir / "reports"
    configs_dir = run_dir / "configs"
    artifacts_dir = run_dir / "artifacts"
    homes_dir = run_dir / "homes"
    dumps_dir = run_dir / "dumps"

    for d in (raw_dir, reports_dir, configs_dir, artifacts_dir, homes_dir, dumps_dir):
        d.mkdir(parents=True, exist_ok=True)

    env_snapshot = _snapshot_env()
    env_snapshot["TRITON_PLUGIN_DIRS"] = env_snapshot.get("TRITON_PLUGIN_DIRS") or str(REPO_ROOT)

    _write_json(configs_dir / "env.json", env_snapshot)
    _write_json(configs_dir / "git.json", _git_snapshot())
    _write_json(
        configs_dir / "run_config.json",
        {
            "run_id": run_id,
            "out_root": str(Path(args.out_root).resolve()),
            "workloads": args.workloads,
            "modes": list(MODES),
            "independent_runs": args.independent_runs,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "threads": args.threads,
        },
    )

    measurements = []

    for mode in MODES:
        for workload in args.workloads:
            cases = _cases_for(workload)
            for case in cases:
                for run_idx in range(args.independent_runs):
                    env = os.environ.copy()
                    _set_thread_env(env, args.threads)
                    env["TRITON_PLUGIN_DIRS"] = env.get("TRITON_PLUGIN_DIRS", str(REPO_ROOT))
                    env["TRITON_RISCV_LOWERING_MODE"] = mode

                    triton_home = homes_dir / mode / workload / case.name / f"run_{run_idx:02d}"
                    dump_dir = dumps_dir / mode / workload / case.name / f"run_{run_idx:02d}"
                    triton_home.mkdir(parents=True, exist_ok=True)
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    env["TRITON_HOME"] = str(triton_home)
                    env["TRITON_SHARED_DUMP_PATH"] = str(dump_dir)

                    cmd = [
                        args.python,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--worker-workload",
                        workload,
                        "--worker-case",
                        json.dumps(case.params),
                        "--worker-warmup",
                        str(args.warmup),
                        "--worker-repeats",
                        str(args.repeats),
                    ]
                    worker_out = subprocess.check_output(cmd, text=True, env=env)
                    result = json.loads(worker_out)
                    result.update(
                        {
                            "mode": mode,
                            "case_name": case.name,
                            "run_idx": run_idx,
                            "triton_home": str(triton_home),
                            "dump_dir": str(dump_dir),
                        }
                    )
                    measurements.append(result)

                    kernel = _kernel_name(workload)
                    cache_files = _find_latest_kernel_cache_files(triton_home, kernel)
                    dst = artifacts_dir / mode / workload / case.name / f"run_{run_idx:02d}"
                    dst.mkdir(parents=True, exist_ok=True)
                    (dst / "lowering_mode.txt").write_text(mode + "\n")
                    for ext, src in cache_files.items():
                        _copy_if_exists(src, dst / f"{kernel}.{ext}")
                    _copy_if_exists(dump_dir / "buddy-opt.args.txt", dst / "buddy-opt.args.txt")
                    for name in ("tt.mlir", "ttshared.mlir", "ll.mlir", "ll.ir"):
                        _copy_if_exists(dump_dir / name, dst / name)

                    print(
                        f"[{mode}] {workload}/{case.name} run={run_idx:02d} "
                        f"median={result['median_s']:.6f}s avg={result['avg_s']:.6f}s"
                    )

    jsonl_path = raw_dir / "measurements.jsonl"
    with jsonl_path.open("w") as f:
        for m in measurements:
            f.write(json.dumps(m, sort_keys=True) + "\n")

    csv_path = raw_dir / "measurements.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "workload",
                "case_name",
                "run_idx",
                "avg_s",
                "median_s",
                "min_s",
                "max_s",
                "std_s",
                "p95_s",
                "triton_home",
                "dump_dir",
            ]
        )
        for m in measurements:
            writer.writerow(
                [
                    m["mode"],
                    m["workload"],
                    m["case_name"],
                    m["run_idx"],
                    m["avg_s"],
                    m["median_s"],
                    m["min_s"],
                    m["max_s"],
                    m["std_s"],
                    m["p95_s"],
                    m["triton_home"],
                    m["dump_dir"],
                ]
            )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for m in measurements:
        key = (m["mode"], m["workload"], m["case_name"])
        grouped.setdefault(key, []).append(m)

    summary_rows = []
    for key, rows in sorted(grouped.items()):
        avgs = [r["avg_s"] for r in rows]
        meds = [r["median_s"] for r in rows]
        summary_rows.append(
            {
                "mode": key[0],
                "workload": key[1],
                "case_name": key[2],
                "independent_runs": len(rows),
                "avg_of_avg_s": statistics.fmean(avgs),
                "median_of_avg_s": statistics.median(avgs),
                "std_of_avg_s": statistics.pstdev(avgs) if len(avgs) > 1 else 0.0,
                "avg_of_median_s": statistics.fmean(meds),
            }
        )

    summary_csv = raw_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "workload",
                "case_name",
                "independent_runs",
                "avg_of_avg_s",
                "median_of_avg_s",
                "std_of_avg_s",
                "avg_of_median_s",
            ]
        )
        for r in summary_rows:
            writer.writerow(
                [
                    r["mode"],
                    r["workload"],
                    r["case_name"],
                    r["independent_runs"],
                    r["avg_of_avg_s"],
                    r["median_of_avg_s"],
                    r["std_of_avg_s"],
                    r["avg_of_median_s"],
                ]
            )

    _write_json(raw_dir / "summary.json", summary_rows)
    print(f"Wrote measurements: {csv_path}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Artifacts root: {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
