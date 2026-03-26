#!/usr/bin/env bash
set -euo pipefail

cd /home/zxl/ruyi/triton-riscv/backend

export TRITON_PLUGIN_DIRS=${TRITON_PLUGIN_DIRS:-/home/zxl/ruyi/triton-riscv}
export TRITON_SHARED_OPT_PATH=${TRITON_SHARED_OPT_PATH:-/tmp/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt}
export BUDDY_MLIR_BINARY_DIR=${BUDDY_MLIR_BINARY_DIR:-/home/zxl/ruyi/buddy-mlir/build/bin}
export LLVM_BINARY_DIR=${LLVM_BINARY_DIR:-/home/zxl/ruyi/buddy-mlir/llvm/build/bin}

RUN_ID=${RUN_ID:-run-$(date +%Y%m%d-%H%M%S)}
RUN_DIR=experiments/ab-vir-vs-loops/${RUN_ID}
LOG=experiments/ab-vir-vs-loops/${RUN_ID}.log

echo "RUN_ID=${RUN_ID}" | tee "$LOG"

/usr/bin/time -f "elapsed=%E user=%U sys=%S maxrss_kb=%M" \
  /tmp/triton-venv/bin/python -u experiments/tools/run_ab_bench.py \
  --python /tmp/triton-venv/bin/python \
  --workloads matmul softmax layernorm vecadd \
  --independent-runs 10 --warmup 5 --repeats 50 --threads 1 \
  --run-id "$RUN_ID" 2>&1 | tee -a "$LOG"

/tmp/triton-venv/bin/python -u experiments/tools/collect_ir_signals.py "$RUN_DIR" 2>&1 | tee -a "$LOG"
/tmp/triton-venv/bin/python -u experiments/tools/make_report.py "$RUN_DIR" 2>&1 | tee -a "$LOG"

# Keep a top-level, up-to-date summary for quick access.
cp "$RUN_DIR/reports/EXPER-PERFORMANCE.md" experiments/EXPER-PERFORMANCE.md

echo "RUN_DIR=$RUN_DIR" | tee -a "$LOG"
echo "TOP_REPORT=experiments/EXPER-PERFORMANCE.md" | tee -a "$LOG"
