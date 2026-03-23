## 1. 阶段结论（基于 EXPERIMENT.md 最新版本）

- 已将 `experiments/` 资产生成流程收敛为单脚本：`scripts/dump_triton_examples_ir.py`。
- `ttshared.mlir` 保持为 Triton 直接导出产物，不再承载测试入口。
- 四个示例（`matmul/vecadd/softmax/layernorm`）统一生成三件测试链路文件：
- `_ttshared_verify_out.mlir`
- `ttshared-main.mlir`
- `linalg_bufferized_no_vec_no_loops.mlir`
- 主边界已固定：
- 不含 `main`：`ttshared.mlir`、`_ttshared_verify_out.mlir`
- 含 `main`：`ttshared-main.mlir`、`linalg_bufferized_no_vec_no_loops.mlir`

## 2. 本周关键变更

- 更新文档：[EXPERIMENT.md](/home/zxl/ruyi/triton-riscv/EXPERIMENT.md)
- 章节标题更新为：`## Experiments Directory Assets (Script-Driven, 2026-03-23 UTC)`。
- 复现说明从“多段内联命令”改为“单脚本驱动”。
- 更新脚本：[dump_triton_examples_ir.py](/home/zxl/ruyi/triton-riscv/scripts/dump_triton_examples_ir.py)
- 脚本现已覆盖完整流程：dump、verify、注入 main、bufferize、白名单清理、边界校验。

## 3. 一键命令（当前标准流程）

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

可选：只重建单个示例。

```bash
/tmp/triton-venv/bin/python scripts/dump_triton_examples_ir.py --examples matmul
```

## 4. 当前产物快照

- `experiments/matmul/`
- `tt.mlir`
- `ttshared.mlir`
- `ll.mlir`
- `ll.ir`
- `_ttshared_verify_out.mlir`
- `ttshared-main.mlir`
- `linalg_bufferized_no_vec_no_loops.mlir`

- `experiments/vecadd/`
- `tt.mlir`
- `ttshared.mlir`
- `ll.mlir`
- `ll.ir`
- `_ttshared_verify_out.mlir`
- `ttshared-main.mlir`
- `linalg_bufferized_no_vec_no_loops.mlir`

- `experiments/softmax/`
- `tt.mlir`
- `ttshared.mlir`
- `ll.mlir`
- `ll.ir`
- `_ttshared_verify_out.mlir`
- `ttshared-main.mlir`
- `linalg_bufferized_no_vec_no_loops.mlir`

- `experiments/layernorm/`
- `tt.mlir`
- `ttshared.mlir`
- `ll.mlir`
- `ll.ir`
- `_ttshared_verify_out.mlir`
- `ttshared-main.mlir`
- `linalg_bufferized_no_vec_no_loops.mlir`
