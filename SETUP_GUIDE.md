# Triton-RISC-V 环境构建指南

本文给出 `triton-riscv` 的推荐环境搭建路径，并与仓库内 `INSTALLATION.md`、`EXPERIMENT.md` 保持一致。

验证基线：`2026-03-23 UTC`（当前机器已通过最小端到端用例 `test_vec_add.py`）。

## 1. 目标与编译链

`triton-riscv` 是 Triton 的后端插件，负责从 Triton IR 继续降级到 CPU/RISC-V 可执行目标。

```text
Python @triton.jit
  -> TTIR
  -> Triton-Shared / Linalg MLIR
  -> Buddy lowering
  -> LLVM IR
  -> native object
```

## 2. 推荐目录布局

```text
/home/zxl/ruyi/
  triton-riscv/   # 本仓库
  triton/         # 上游 triton
  buddy-mlir/     # buddy-compiler

/tmp/
  triton-venv/    # Triton Python 虚拟环境
  triton_home/    # Triton cache/home
  buddy-build/    # buddy-opt out-of-tree build
```

## 3. 一次性准备（从零开始）

### 3.1 克隆仓库并对齐 Triton 版本

```bash
cd /home/zxl/ruyi
git clone https://github.com/RuyiAI-Stack/triton-riscv.git triton-riscv
git clone https://github.com/triton-lang/triton.git triton
cd triton && git checkout "$(cat ../triton-riscv/triton-hash.txt)"

git clone https://github.com/buddy-compiler/buddy-mlir.git buddy-mlir
cd buddy-mlir && git submodule update --init llvm
```

### 3.2 给 Triton 打补丁（去除 GPU 后端硬依赖）

```bash
/home/zxl/ruyi/triton-riscv/scripts/apply_patches.sh /home/zxl/ruyi/triton
```

### 3.3 构建 Buddy 的 LLVM/MLIR/Clang

```bash
mkdir -p /home/zxl/ruyi/buddy-mlir/llvm/build
cd /home/zxl/ruyi/buddy-mlir/llvm/build

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DCLANG_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release

ninja -j"$(nproc)" clang mlir-opt mlir-translate llc opt llvm-config FileCheck omp
```

### 3.4 构建 `buddy-opt`

```bash
rm -rf /tmp/buddy-build
mkdir -p /tmp/buddy-build
cd /tmp/buddy-build

cmake -G Ninja /home/zxl/ruyi/buddy-mlir \
  -DMLIR_DIR=/home/zxl/ruyi/buddy-mlir/llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/zxl/ruyi/buddy-mlir/llvm/build/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=OFF

ninja -j"$(nproc)" buddy-opt
```

### 3.5 构建并安装 Triton（加载 `triton-riscv` 插件）

```bash
rm -rf /tmp/triton
git clone --no-hardlinks /home/zxl/ruyi/triton /tmp/triton
/home/zxl/ruyi/triton-riscv/scripts/apply_patches.sh /tmp/triton

rm -rf /tmp/triton-venv
/usr/bin/python3.12 -m venv /tmp/triton-venv --without-pip
curl -sSLo /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
/tmp/triton-venv/bin/python /tmp/get-pip.py "pip<25" "setuptools<80" wheel

/tmp/triton-venv/bin/python -m pip install "cmake>=3.20,<4.0" ninja pybind11 lit pytest-xdist

mkdir -p /tmp/triton_home
cd /tmp/triton
export TRITON_HOME=/tmp/triton_home
export TRITON_PLUGIN_DIRS=/home/zxl/ruyi/triton-riscv
export LLVM_SYSPATH=/home/zxl/ruyi/buddy-mlir/llvm/build
/tmp/triton-venv/bin/python -m pip install --no-build-isolation -vvv .
```

## 4. 运行时环境变量

运行测试或例子前，统一设置：

```bash
export TRITON_HOME=/tmp/triton_home
export TRITON_PLUGIN_DIRS=/home/zxl/ruyi/triton-riscv
export LLVM_SYSPATH=/home/zxl/ruyi/buddy-mlir/llvm/build

export LLVM_BINARY_DIR=/home/zxl/ruyi/buddy-mlir/llvm/build/bin
export BUDDY_MLIR_BINARY_DIR=/tmp/buddy-build/bin
export TRITON_SHARED_OPT_PATH=/tmp/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
```

## 5. 当前机器最小验证通路（已验证）

以下命令在 `2026-03-23 UTC` 已通过。

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

预期结果：`1 passed`。

## 6. 常见问题

### 6.1 `buddy-opt` 链接缺少 MLIR 测试库

如果出现：

- `cannot find -lMLIRTestTransforms`
- `cannot find -lMLIRTestTransformDialect`

请按 `INSTALLATION.md` 的“MLIR test libs link issue”小节处理（通过 stub archive 方案规避）。

### 6.2 修改后端 pipeline 后结果不更新

修改 `backend/compiler.py` 后清理当前使用的 Triton cache，例如：

```bash
rm -rf /tmp/triton_home/.triton/cache
```
