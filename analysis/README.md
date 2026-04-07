# softmax LLVMIR O0 vs O2 分析记录

## 目录说明
本次样例数据位于：`analysis/softmax-llvmir-o0-vs-o2/`

包含文件：
- `tt.mlir`: Triton TTIR
- `ttshared.mlir`: Triton Shared IR
- `ll.mlir`: LLVM Dialect MLIR
- `ll.ir`: `mlir-translate` 产出的原始 LLVM IR（优化前）
- `ll.opt.O2.ir`: 在 `ll.ir` 上执行 `opt -passes=default<O2>` 后的 LLVM IR
- `buddy-opt.args.txt`: `ttshared -> ll.mlir` 的 pass 参数
- `softmax-pre-vs-o2.diff`: `ll.ir` 与 `ll.opt.O2.ir` 的 unified diff
- `softmax-pre.sig.txt`: 优化前关键指令信号（malloc/alloca/memcpy 等）
- `softmax-o2.sig.txt`: O2 后关键指令信号

## 结论
- 已在 `backend/compiler.py` 增加 LLVM 优化级别开关：`TRITON_RISCV_LLVM_OPT_LEVEL`（默认 `2`）。
- `python/examples/test_softmax.py` 在当前 `vir_vector` 路径会先报 `vir.store` 类型错误，无法作为该路径下 O2 对比样例。
- 采用 `TRITON_RISCV_LOWERING_MODE=linalg_loops` 路径，`test_softmax('cpu')` 在 O0/O2 均可跑通。
- 对 `ll.ir` 运行 O2 后，LLVMIR 规模明显下降：
  - 行数：`293 -> 222`
  - 字节数：`12812 -> 9832`
- 该 softmax 样例的 LLVMIR 中本身不含 `llvm.memcpy`，因此本例看不到“memcpy 被 O2 消除”的现象；但可见大量 IR 结构收敛（如 `insertvalue` 等冗余构造减少）。

## 复现命令（本机路径）
```bash
# 1) 生成优化前中间表示（O0，linalg_loops）
TRITON_HOME=/tmp/triton_home \
TRITON_CACHE_DIR=/tmp/triton-cache-o0 \
TRITON_PLUGIN_DIRS=/home/zxl/ruyi/triton-riscv \
LLVM_BINARY_DIR=/home/zxl/ruyi/buddy-mlir/llvm/build/bin \
BUDDY_MLIR_BINARY_DIR=/tmp/buddy-build/bin \
TRITON_SHARED_OPT_PATH=/tmp/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt \
TRITON_SHARED_DUMP_PATH=/tmp/triton-riscv-softmax-o0 \
TRITON_RISCV_LLVM_OPT_LEVEL=0 \
TRITON_RISCV_LOWERING_MODE=linalg_loops \
/tmp/triton-venv/bin/python - <<'PY'
import sys
sys.path.insert(0, '/home/zxl/ruyi/triton-riscv/python/examples')
import benchmark
benchmark.select_cpu_backend()
import test_softmax
test_softmax.test_softmax('cpu')
print('softmax O0 linalg_loops ok')
PY

# 2) 在同一份 ll.ir 上跑 O2，得到优化后 LLVMIR
/home/zxl/ruyi/buddy-mlir/llvm/build/bin/opt \
  -passes='default<O2>' -S \
  /tmp/triton-riscv-softmax-o0/ll.ir \
  -o /tmp/triton-riscv-softmax-o0/ll.opt.O2.ir

# 3) 对比
wc -l /tmp/triton-riscv-softmax-o0/ll.ir /tmp/triton-riscv-softmax-o0/ll.opt.O2.ir
wc -c /tmp/triton-riscv-softmax-o0/ll.ir /tmp/triton-riscv-softmax-o0/ll.opt.O2.ir
diff -u /tmp/triton-riscv-softmax-o0/ll.ir /tmp/triton-riscv-softmax-o0/ll.opt.O2.ir > /tmp/softmax-pre-vs-o2.diff || true
```
