#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/zxl/ruyi/triton-riscv
cd "${REPO_ROOT}/backend"

PYTHON_BIN=${PYTHON_BIN:-/tmp/triton-venv/bin/python}
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python3
fi

export TRITON_PLUGIN_DIRS=${TRITON_PLUGIN_DIRS:-${REPO_ROOT}}
export TRITON_SHARED_OPT_PATH=${TRITON_SHARED_OPT_PATH:-/tmp/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt}
if [[ -z "${BUDDY_MLIR_BINARY_DIR:-}" ]]; then
  if [[ -x /tmp/buddy-wrap/buddy-opt ]]; then
    export BUDDY_MLIR_BINARY_DIR=/tmp/buddy-wrap
  else
    export BUDDY_MLIR_BINARY_DIR=/home/zxl/ruyi/buddy-mlir/build/bin
  fi
fi
export LLVM_BINARY_DIR=${LLVM_BINARY_DIR:-/home/zxl/ruyi/buddy-mlir/llvm/build/bin}

RUN_ID=${RUN_ID:-run-$(date +%Y%m%d-%H%M%S)-min3-rerun2-w5-r10}
RUN_DIR=experiments/ab-vir-vs-loops/${RUN_ID}
LOG=experiments/ab-vir-vs-loops/${RUN_ID}.log
export RUN_ID
export PYTHON_BIN

echo "RUN_ID=${RUN_ID}" | tee "${LOG}"
echo "CONFIG: cases=matmul/n512 softmax/m1024_n1024 layernorm/m1151_n4096, independent-runs=1, warmup=5, repeats=10, threads=1" | tee -a "${LOG}"

/usr/bin/time -f "elapsed=%E user=%U sys=%S maxrss_kb=%M" \
  "${PYTHON_BIN}" - <<'PY' 2>&1 | tee -a "${LOG}"
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

repo_root = Path("/home/zxl/ruyi/triton-riscv")
backend_dir = repo_root / "backend"
tool = backend_dir / "experiments" / "tools" / "run_ab_bench.py"
run_id = os.environ["RUN_ID"]
run_dir = backend_dir / "experiments" / "ab-vir-vs-loops" / run_id
raw_dir = run_dir / "raw"
reports_dir = run_dir / "reports"
configs_dir = run_dir / "configs"
artifacts_dir = run_dir / "artifacts"
homes_dir = run_dir / "homes"
dumps_dir = run_dir / "dumps"
for d in (raw_dir, reports_dir, configs_dir, artifacts_dir, homes_dir, dumps_dir):
    d.mkdir(parents=True, exist_ok=True)

python_bin = os.environ.get("PYTHON_BIN", sys.executable)
modes = ("vir_vector", "linalg_loops")
cases = [
    ("matmul", "n512", {"n": 512}),
    ("softmax", "m1024_n1024", {"m": 1024, "n": 1024}),
    ("layernorm", "m1151_n4096", {"m": 1151, "n": 4096}),
]
warmup = 5
repeats = 10
threads = 1
independent_runs = 1

def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

env_snapshot_keys = [
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
write_json(configs_dir / "env.json", {k: os.environ.get(k, "") for k in env_snapshot_keys})
write_json(
    configs_dir / "run_config.json",
    {
        "run_id": run_id,
        "out_root": str((backend_dir / "experiments" / "ab-vir-vs-loops").resolve()),
        "workloads": sorted({w for w, _, _ in cases}),
        "modes": list(modes),
        "independent_runs": independent_runs,
        "warmup": warmup,
        "repeats": repeats,
        "threads": threads,
        "selected_cases": [{"workload": w, "case_name": c, "params": p} for w, c, p in cases],
    },
)

def run_cmd(cmd, env):
    out = subprocess.check_output(cmd, text=True, env=env)
    return json.loads(out)

def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

def kernel_name(workload: str) -> str:
    return {
        "matmul": "matmul_kernel",
        "softmax": "softmax_kernel",
        "layernorm": "_layer_norm_fwd_fused",
    }[workload]

def find_latest_kernel_cache_files(triton_home: Path, kernel: str):
    cache_root = triton_home / ".triton" / "cache"
    exts = ("ttir", "ttsharedir", "llir", "obj", "json", "source")
    out = {}
    for ext in exts:
        cands = list(cache_root.rglob(f"{kernel}.{ext}"))
        if not cands:
            continue
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        out[ext] = cands[0]
    return out

measurements = []
for mode in modes:
    for workload, case_name, params in cases:
        for run_idx in range(independent_runs):
            env = os.environ.copy()
            for k in ("TORCH_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[k] = str(threads)
            env["TRITON_PLUGIN_DIRS"] = env.get("TRITON_PLUGIN_DIRS", str(repo_root))
            env["TRITON_RISCV_LOWERING_MODE"] = mode

            triton_home = homes_dir / mode / workload / case_name / f"run_{run_idx:02d}"
            dump_dir = dumps_dir / mode / workload / case_name / f"run_{run_idx:02d}"
            triton_home.mkdir(parents=True, exist_ok=True)
            dump_dir.mkdir(parents=True, exist_ok=True)
            env["TRITON_HOME"] = str(triton_home)
            env["TRITON_SHARED_DUMP_PATH"] = str(dump_dir)

            cmd = [
                python_bin,
                str(tool),
                "--worker",
                "--worker-workload",
                workload,
                "--worker-case",
                json.dumps(params),
                "--worker-warmup",
                str(warmup),
                "--worker-repeats",
                str(repeats),
            ]
            result = run_cmd(cmd, env)
            result.update(
                {
                    "mode": mode,
                    "case_name": case_name,
                    "run_idx": run_idx,
                    "triton_home": str(triton_home),
                    "dump_dir": str(dump_dir),
                }
            )
            measurements.append(result)
            print(
                f"[{mode}] {workload}/{case_name} run={run_idx:02d} "
                f"median={result['median_s']:.6f}s avg={result['avg_s']:.6f}s"
            )

            kernel = kernel_name(workload)
            cache_files = find_latest_kernel_cache_files(triton_home, kernel)
            dst = artifacts_dir / mode / workload / case_name / f"run_{run_idx:02d}"
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "lowering_mode.txt").write_text(mode + "\n")
            for ext, src in cache_files.items():
                copy_if_exists(src, dst / f"{kernel}.{ext}")
            copy_if_exists(dump_dir / "buddy-opt.args.txt", dst / "buddy-opt.args.txt")
            for name in ("tt.mlir", "ttshared.mlir", "ll.mlir", "ll.ir"):
                copy_if_exists(dump_dir / name, dst / name)

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

grouped = {}
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

write_json(raw_dir / "summary.json", summary_rows)
write_json(
    configs_dir / "git.json",
    {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip(),
        "branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, text=True).strip(),
        "status_short": subprocess.check_output(["git", "status", "--short"], cwd=repo_root, text=True).strip(),
    },
)
print(f"Wrote measurements: {csv_path}")
print(f"Wrote summary: {summary_csv}")
print(f"Artifacts root: {artifacts_dir}")
PY

"${PYTHON_BIN}" -u experiments/tools/collect_ir_signals.py "${RUN_DIR}" 2>&1 | tee -a "${LOG}"
"${PYTHON_BIN}" -u experiments/tools/make_report.py "${RUN_DIR}" 2>&1 | tee -a "${LOG}"

cp "${RUN_DIR}/reports/EXPER-PERFORMANCE.md" experiments/EXPER-PERFORMANCE.md

echo "RUN_DIR=${RUN_DIR}" | tee -a "${LOG}"
echo "TOP_REPORT=experiments/EXPER-PERFORMANCE.md" | tee -a "${LOG}"

export SUMMARY_CSV="${RUN_DIR}/raw/summary.csv"
"${PYTHON_BIN}" - <<'PY' | tee -a "${LOG}"
import csv
import os

summary_csv = os.environ["SUMMARY_CSV"]
rows = list(csv.DictReader(open(summary_csv)))
m = {(r["mode"], r["workload"], r["case_name"]): float(r["avg_of_avg_s"]) for r in rows}

targets = [
    ("matmul", "n512"),
    ("softmax", "m1024_n1024"),
    ("layernorm", "m1151_n4096"),
]
print("core 3 case:")
for w, c in targets:
    vir = m[("vir_vector", w, c)]
    loops = m[("linalg_loops", w, c)]
    print(f"- {w} {c}: vir={vir:.6f}s, loops={loops:.6f}s, loops/vir={loops/vir:.4f}x")
PY
