#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_OUT_ROOT="${RUN_OUT_ROOT:-${REPO_ROOT}/experiments/ab-vir-vs-loops}"
RUN_AB_BENCH="${RUN_AB_BENCH:-${REPO_ROOT}/experiments/tools/run_ab_bench.py}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export TRITON_PLUGIN_DIRS=${TRITON_PLUGIN_DIRS:-${REPO_ROOT}}
if [[ -z "${TRITON_SHARED_OPT_PATH:-}" ]]; then
  echo "ERROR: TRITON_SHARED_OPT_PATH is not set." >&2
  exit 1
fi
if [[ -z "${BUDDY_MLIR_BINARY_DIR:-}" ]]; then
  echo "ERROR: BUDDY_MLIR_BINARY_DIR is not set." >&2
  exit 1
fi
if [[ -z "${LLVM_BINARY_DIR:-}" ]]; then
  echo "ERROR: LLVM_BINARY_DIR is not set." >&2
  exit 1
fi

RUN_ID=${RUN_ID:-run-$(date +%Y%m%d-%H%M%S)-min3-w5-r10-easy}
RUN_DIR="${RUN_OUT_ROOT}/${RUN_ID}"
RAW_DIR=${RUN_DIR}/raw

echo "RUN_ID=${RUN_ID}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "RUN_OUT_ROOT=${RUN_OUT_ROOT}"
echo "CONFIG: cases=matmul/n512 softmax/m1024_n1024 layernorm/m1151_n4096, independent-runs=1, warmup=5, repeats=10, threads=1"

export RUN_ID
export PYTHON_BIN
export REPO_ROOT
export RUN_OUT_ROOT
export RUN_AB_BENCH
/usr/bin/time -f "elapsed=%E user=%U sys=%S maxrss_kb=%M" \
  "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
run_out_root = Path(os.environ["RUN_OUT_ROOT"])
tool = Path(os.environ["RUN_AB_BENCH"])
run_id = os.environ["RUN_ID"]
run_dir = run_out_root / run_id
raw_dir = run_dir / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

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

measurements = []
for mode in modes:
    for workload, case_name, params in cases:
        for run_idx in range(independent_runs):
            env = os.environ.copy()
            for k in ("TORCH_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[k] = str(threads)
            env["TRITON_PLUGIN_DIRS"] = env.get("TRITON_PLUGIN_DIRS", str(repo_root))
            env["TRITON_RISCV_LOWERING_MODE"] = mode

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
            out = subprocess.check_output(cmd, text=True, env=env)
            result = json.loads(out)

            result.update(
                {
                    "mode": mode,
                    "workload": workload,
                    "case_name": case_name,
                    "run_idx": run_idx,
                }
            )
            measurements.append(result)
            print(
                f"[{mode}] {workload}/{case_name} run={run_idx:02d} "
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

(raw_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n")

lookup = {(r["mode"], r["workload"], r["case_name"]): r for r in summary_rows}
targets = [
    ("matmul", "n512"),
    ("softmax", "m1024_n1024"),
    ("layernorm", "m1151_n4096"),
]
lines = []
lines.append("core 3 case:")
for workload, case_name in targets:
    vir = float(lookup[("vir_vector", workload, case_name)]["avg_of_avg_s"])
    loops = float(lookup[("linalg_loops", workload, case_name)]["avg_of_avg_s"])
    lines.append(
        f"- {workload} {case_name}: vir={vir:.6f}s, loops={loops:.6f}s, loops/vir={loops/vir:.4f}x"
    )
text = "\n".join(lines) + "\n"
print(f"Wrote measurements: {csv_path}")
print(f"Wrote summary: {summary_csv}")
print(f"RAW_DIR={raw_dir}")
print(text, end="")
(raw_dir / "final_result.txt").write_text(text)
PY

echo "DONE: ${RAW_DIR}"
