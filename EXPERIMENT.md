# Matmul Vectorization Pass Experiment (CPU/x86)

## Minimal Verifiable Path (Copy-Paste)

The following path was validated on `2026-03-23 UTC` in this environment and is
the smallest end-to-end command block to confirm setup is working.

```bash
set -euo pipefail
cd /tmp/triton

export TRITON_HOME=/tmp/triton_home
export TRITON_PLUGIN_DIRS=/home/zxl/ruyi/triton-riscv
export LLVM_SYSPATH=/home/zxl/ruyi/buddy-mlir/llvm/build
export LLVM_BINARY_DIR=/home/zxl/ruyi/buddy-mlir/llvm/build/bin
export BUDDY_MLIR_BINARY_DIR=/tmp/buddy-build/bin
export TRITON_SHARED_OPT_PATH=/tmp/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt

/tmp/triton-venv/bin/python -m pytest /home/zxl/ruyi/triton-riscv/python/examples/test_vec_add.py -q -v
```

Expected result: one passing test (`1 passed`).

This document records an A/B experiment on the Triton CPU backend in this repo, comparing performance and generated IR with and without the buddy-opt pass `--matmul-vectorization` (wired in [`backend/compiler.py`](/home/zxl/ruyi/triton-riscv/backend/compiler.py)).

## Goal

Estimate how much performance would drop if we do not use `--matmul-vectorization` for Triton matmul.

## What Was Changed (To Enable A/B)

I added an environment-variable toggle so we can run both variants from the same checkout without editing code:

- Default (current behavior): pass enabled
- Disable: `TRITON_RISCV_DISABLE_MATMUL_VECTORIZATION=1`

Implementation: [`backend/compiler.py`](/home/zxl/ruyi/triton-riscv/backend/compiler.py)

## Benchmark Design

Key points to keep the benchmark fair:

- Measure steady-state runtime only (compile is excluded by doing warmups first).
- Use a fixed number of CPU threads to reduce noise:
  - `TORCH_NUM_THREADS=1` (and common BLAS thread env vars set to `1` in the script)
- Separate caches per variant so each run compiles from scratch into its own cache root:
  - `TRITON_HOME=/tmp/exp_matmul_vec_on`
  - `TRITON_HOME=/tmp/exp_matmul_vec_off`

Benchmark script: [`scripts/bench_matmul_vectorization.py`](/home/zxl/ruyi/triton-riscv/scripts/bench_matmul_vectorization.py)

It reuses the matmul kernel from `python/examples/test_matmul.py` via `runpy`, selects the CPU driver, warms up, then times `matmul(a, b)` for each matrix size.

## Commands Used

Common environment:

```bash
export TRITON_PLUGIN_DIRS=/home/zxl/ruyi/triton-riscv
export LLVM_BINARY_DIR=/home/zxl/ruyi/buddy-mlir/llvm/build/bin
export BUDDY_MLIR_BINARY_DIR=/home/zxl/ruyi/buddy-mlir/build/bin
export TRITON_SHARED_OPT_PATH=/tmp/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
```

Vectorization ON:

```bash
export TRITON_HOME=/tmp/exp_matmul_vec_on
unset TRITON_RISCV_DISABLE_MATMUL_VECTORIZATION
/tmp/triton-venv/bin/python scripts/bench_matmul_vectorization.py \
  --sizes 256 384 512 640 768 --warmup 3 --repeats 10
```

Vectorization OFF:

```bash
export TRITON_HOME=/tmp/exp_matmul_vec_off
export TRITON_RISCV_DISABLE_MATMUL_VECTORIZATION=1
/tmp/triton-venv/bin/python scripts/bench_matmul_vectorization.py \
  --sizes 256 384 512 640 768 --warmup 3 --repeats 10
```

To dump the TritonShared -> LLVM lowering intermediates (`tt.mlir`, `ttshared.mlir`, `ll.mlir`, `ll.ir`) for inspection:

```bash
export TRITON_SHARED_DUMP_PATH=/tmp/exp_matmul_vec_on2/shared_dump   # or vec_off2
```

## Results (Wall Time)

All times are seconds per matmul call, measured as average of `--repeats=10` after `--warmup=3`.

| N (MxN, K=N) | Vec ON avg (s) | Vec OFF avg (s) | Delta (OFF-ON) |
|---:|---:|---:|---:|
| 256 | 0.009023 | 0.008956 | -0.74% |
| 384 | 0.028751 | 0.028689 | -0.22% |
| 512 | 0.067315 | 0.067323 | +0.01% |
| 640 | 0.130718 | 0.130396 | -0.25% |
| 768 | 0.223859 | 0.221997 | -0.83% |

Within run-to-run noise, there is no measurable slowdown from disabling `--matmul-vectorization` in the current pipeline.

## IR Evidence (Why Performance Didn’t Change)

I verified the generated lowering output is **bit-identical** with the pass enabled vs disabled.

Using:

- `/tmp/exp_matmul_vec_on2/shared_dump/{ll.mlir,ll.ir}`
- `/tmp/exp_matmul_vec_off2/shared_dump/{ll.mlir,ll.ir}`

`cmp` reports both files identical:

- `ll.mlir identical`
- `ll.ir identical`

This strongly suggests `--matmul-vectorization` is currently a no-op for this pipeline (very likely due to pass ordering: the args run `--convert-linalg-to-affine-loops` before `--matmul-vectorization`, so there may be no `linalg.matmul` left for the vectorization pass to match).

Separately, the final cached LLVM IR still contains clear vector codegen signals (for both ON and OFF), e.g. `<32 x float>` loads/stores and `@llvm.fmuladd.v32f32` in:

- `/tmp/exp_matmul_vec_on/.triton/cache/.../matmul_kernel.llir`
- `/tmp/exp_matmul_vec_off/.triton/cache/.../matmul_kernel.llir`

## Conclusion

For the initial pipeline order I tested, disabling `--matmul-vectorization` produced the same IR and the same runtime (within noise). That turned out to be because the pass was effectively not being applied (it was in the wrong place relative to other lowering passes, so it did not change the output).

## Fix: Make `--matmul-vectorization` Actually Trigger

Buddy's `matmul-vectorization` pass matches `linalg.matmul` (see buddy test `tests/Conversion/matmul-vectorization.mlir`). If we lower linalg matmul away first, the pass becomes a no-op.

To make it trigger, the correct ordering is:

1. `--empty-tensor-to-alloc-tensor`
2. `--one-shot-bufferize=...`
3. `--matmul-vectorization`
4. then lower remaining linalg ops to loops and proceed to LLVM lowering

In practice, Triton was using the backend compiler installed in the venv at:

- `/tmp/triton-venv/lib/python3.12/site-packages/triton/backends/triton_shared/compiler.py`

so I changed that file's buddy-opt pipeline to run `--matmul-vectorization` before `--convert-linalg-to-affine-loops` and also added the env toggle `TRITON_RISCV_DISABLE_MATMUL_VECTORIZATION=1` there.

## Results After Fix (Wall Time)

Same benchmark method as before (steady-state only): `--warmup=3`, `--repeats=10`, `TORCH_NUM_THREADS=1`.

| N (MxN, K=N) | Vec ON avg (s) | Vec OFF avg (s) | Slowdown (OFF/ON) |
|---:|---:|---:|---:|
| 256 | 0.008931 | 0.030435 | 3.41x |
| 384 | 0.028712 | 0.101402 | 3.53x |
| 512 | 0.067131 | 0.240130 | 3.58x |
| 640 | 0.129755 | 0.466806 | 3.60x |
| 768 | 0.221447 | 0.802717 | 3.62x |

So once the pass is actually effective, disabling it causes a large slowdown on this CPU benchmark (roughly 3.4x to 3.6x).

### IR Evidence After Fix

For the final cached LLVM IR (`matmul_kernel.llir`):

- Vec ON contains `<32 x float>` and vector FMA intrinsics (e.g. `@llvm.fmuladd.v32f32`).
- Vec OFF contains no vector types and no vector FMA intrinsics (signals all zero).

This was extracted from the script output:

- Vec ON signals: `{'vec_types': 96, 'fmuladd_vec': 10, 'shufflevector': 9}`
- Vec OFF signals: `{'vec_types': 0, 'fmuladd_vec': 0, 'shufflevector': 0}`

## Suggested Next Step (If You Want a Cleaner Integration)

Right now, the “effective” pipeline change was applied to the venv-installed Triton backend compiler. If you want this to live in-repo, we should either:

1. Rebuild/install Triton so it picks up a patched backend compiler from this repo (preferred for long-term), or
2. Add a small patching step in setup scripts to modify the installed `triton_shared/compiler.py` in-place.

## Experiments Directory Assets (Script-Driven, 2026-03-23 UTC)

Canonical path:

- `/home/zxl/ruyi/triton-riscv/experiments/`

Current retained files are intentionally minimal.

Per-example retained intermediates:

- `experiments/matmul|vecadd|softmax|layernorm/tt.mlir`
- `experiments/matmul|vecadd|softmax|layernorm/ttshared.mlir`
- `experiments/matmul|vecadd|softmax|layernorm/ll.mlir`
- `experiments/matmul|vecadd|softmax|layernorm/ll.ir`

Per-example testing chain artifacts (all 4 examples):

- `experiments/matmul|vecadd|softmax|layernorm/_ttshared_verify_out.mlir`
- `experiments/matmul|vecadd|softmax|layernorm/ttshared-main.mlir`
- `experiments/matmul|vecadd|softmax|layernorm/linalg_bufferized_no_vec_no_loops.mlir`

Important boundary:

- `ttshared.mlir` is kept as direct Triton kernel dump and **must not** be modified or appended with `@main`.
- `ttshared-main.mlir` is the test-entry version, derived from `_ttshared_verify_out.mlir` for each example.

### Regeneration Workflow (Current)

`scripts/dump_triton_examples_ir.py` now owns the full pipeline end-to-end for all 4 examples:

1. Trigger Triton example compilation and dump `tt.mlir`, `ttshared.mlir`, `ll.mlir`, `ll.ir`.
2. Generate `_ttshared_verify_out.mlir` from `ttshared.mlir` with `buddy-opt --verify-each`.
3. Generate `ttshared-main.mlir` by appending per-kernel test `func.func @main() -> i32` to `_ttshared_verify_out.mlir`.
4. Run `--empty-tensor-to-alloc-tensor` + `--one-shot-bufferize=allow-return-allocs-from-loops=true` to emit `linalg_bufferized_no_vec_no_loops.mlir`.
5. Apply whitelist cleanup in `experiments/`.

### One-Click Regeneration Command

```bash
set -euo pipefail
cd /home/zxl/ruyi/triton-riscv
export TRITON_PLUGIN_DIRS=/home/zxl/ruyi/triton-riscv
export LLVM_BINARY_DIR=/home/zxl/ruyi/buddy-mlir/llvm/build/bin
export BUDDY_MLIR_BINARY_DIR=/home/zxl/ruyi/buddy-mlir/build/bin
export TRITON_SHARED_OPT_PATH=/tmp/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
export TRITON_EXPERIMENT_OUT_ROOT=/home/zxl/ruyi/triton-riscv/experiments
/tmp/triton-venv/bin/python scripts/dump_triton_examples_ir.py
```

Optional partial rebuild:

```bash
/tmp/triton-venv/bin/python scripts/dump_triton_examples_ir.py --examples matmul
```
