# Installation (Triton-RISC-V)

This repo is a Triton backend plugin. You build:

1. LLVM/MLIR/Clang (from Buddy’s pinned `llvm-project`)
2. `buddy-opt` (Buddy Compiler)
3. Triton (patched to make GPU backends optional) + `triton-riscv` plugin discovery

This document reflects a working, minimal build on 2026-03-23 (UTC).

## Validation Status (2026-03-23 UTC)

Validated on the current machine with:

- `/tmp/triton-venv/bin/python -V` -> `Python 3.12.3`
- `/home/zxl/ruyi/buddy-mlir/llvm/build/bin/clang++ --version` -> `clang version 22.0.0git`
- `/tmp/buddy-build/bin/buddy-opt --version` -> `LLVM version 22.0.0git`
- `TRITON_HOME=/tmp/triton_home TRITON_PLUGIN_DIRS=/home/zxl/ruyi/triton-riscv LLVM_SYSPATH=/home/zxl/ruyi/buddy-mlir/llvm/build /tmp/triton-venv/bin/python -c "import triton; print('triton', triton.__version__)"` -> `triton 3.4.0`
- `/tmp/triton-venv/bin/python -m pytest /home/zxl/ruyi/triton-riscv/python/examples/test_vec_add.py -q -v` -> `1 passed`

## Build Log (From README, With Minimal Adjustments)

The build sequence below follows `README.md` first, then applies only required
environment-specific adjustments. Keep this section as an execution log.

| Time (UTC) | Step | Command | Result | Notes |
|---|---|---|---|---|
| 2026-03-23 | Toolchain sanity | `/tmp/triton-venv/bin/python -V` | pass | Python runtime is available |
| 2026-03-23 | LLVM sanity | `/home/zxl/ruyi/buddy-mlir/llvm/build/bin/clang++ --version` | pass | Buddy LLVM build is usable |
| 2026-03-23 | Buddy sanity | `/tmp/buddy-build/bin/buddy-opt --version` | pass | `buddy-opt` is usable |
| 2026-03-23 | Triton import | `... python -c "import triton; ..."` | pass | Triton package works with plugin env vars |
| 2026-03-23 | E2E minimal test | `pytest .../test_vec_add.py -q -v` | pass | End-to-end compile/run verified |

## Layout

Repository layout used by these steps:

- `triton-riscv` (this repo): `/home/zxl/ruyi/triton-riscv`
- `triton` (upstream): `/home/zxl/ruyi/triton`
- `buddy-mlir`: `/home/zxl/ruyi/buddy-mlir`

Build/staging directories (chosen to avoid writing into `$HOME`):

- Staged triton source: `/tmp/triton`
- Triton venv: `/tmp/triton-venv`
- Triton home/cache: `/tmp/triton_home`
- Buddy build: `/tmp/buddy-build`
- Buddy fake libs (link workaround): `/tmp/buddy-fake-libs`

## 0. Prerequisites

You need a working build toolchain (`g++`, `cmake`, `ninja`, etc). LLVM is built
from source so it will take time and disk space.

## 1. Clone Repos

```bash
cd /home/zxl/ruyi
git clone https://github.com/RuyiAI-Stack/triton-riscv.git triton-riscv
git clone https://github.com/triton-lang/triton.git triton
cd triton && git checkout "$(cat ../triton-riscv/triton-hash.txt)"

git clone https://github.com/buddy-compiler/buddy-mlir.git buddy-mlir
cd buddy-mlir && git submodule update --init llvm
```

## 2. Patch Triton (make GPU backends optional)

Patch the main Triton checkout:

```bash
/home/zxl/ruyi/triton-riscv/scripts/apply_patches.sh /home/zxl/ruyi/triton
```

## 3. Build LLVM/MLIR/Clang (Buddy’s pinned LLVM)

Configure and build a minimal toolchain that provides:
`mlir-translate`, `llc`, `opt`, `clang++`.

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

Sanity check:

```bash
/home/zxl/ruyi/buddy-mlir/llvm/build/bin/clang++ --version | head -n 2
/home/zxl/ruyi/buddy-mlir/llvm/build/bin/mlir-translate --help | head -n 5
```

## 4. Build `buddy-opt` (Buddy Compiler)

Build Buddy out-of-tree under `/tmp`:

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
```

### Note: MLIR test libs link issue

With MLIR tests disabled in the LLVM build, `buddy-opt` may try to link:
`-lMLIRTestTransforms` and `-lMLIRTestTransformDialect`. If you see a link error
like:

```text
/usr/bin/ld: cannot find -lMLIRTestTransforms
/usr/bin/ld: cannot find -lMLIRTestTransformDialect
```

Workaround used in this environment: provide empty stub archives and point the
linker to them via `LIBRARY_PATH`.

```bash
rm -rf /tmp/buddy-fake-libs
mkdir -p /tmp/buddy-fake-libs
printf 'void buddy_dummy(void){}\n' > /tmp/buddy-fake-libs/buddy_dummy.c
cc -c /tmp/buddy-fake-libs/buddy_dummy.c -o /tmp/buddy-fake-libs/buddy_dummy.o
ar rcs /tmp/buddy-fake-libs/libMLIRTestTransforms.a /tmp/buddy-fake-libs/buddy_dummy.o
ar rcs /tmp/buddy-fake-libs/libMLIRTestTransformDialect.a /tmp/buddy-fake-libs/buddy_dummy.o

cd /tmp/buddy-build
LIBRARY_PATH=/tmp/buddy-fake-libs ninja -j"$(nproc)" buddy-opt
```

Result:

```bash
/tmp/buddy-build/bin/buddy-opt --version | head -n 2
```

## 5. Build and Install Triton (with `triton-riscv` plugin)

### 5.1 Stage Triton into `/tmp`

This avoids cross-device hardlink issues and keeps build artifacts under `/tmp`:

```bash
rm -rf /tmp/triton
git clone --no-hardlinks /home/zxl/ruyi/triton /tmp/triton
/home/zxl/ruyi/triton-riscv/scripts/apply_patches.sh /tmp/triton
```

### 5.2 Create a Python venv for Triton

This Triton checkout requires Python >= 3.8. The system `python3.12` works.
If your system Python lacks `ensurepip`, bootstrap `pip` via `get-pip.py`.

```bash
rm -rf /tmp/triton-venv
/usr/bin/python3.12 -m venv /tmp/triton-venv --without-pip

curl -sSLo /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
/tmp/triton-venv/bin/python /tmp/get-pip.py "pip<25" "setuptools<80" wheel

/tmp/triton-venv/bin/python -m pip install "cmake>=3.20,<4.0" ninja pybind11 lit pytest-xdist
```

### 5.3 Install Triton

Important: set `TRITON_HOME` to a writable directory, otherwise Triton may try
to write under `~/.triton` (which can be unwritable in restricted environments).

```bash
rm -rf /tmp/triton_home
mkdir -p /tmp/triton_home

cd /tmp/triton
export TRITON_HOME=/tmp/triton_home
export TRITON_PLUGIN_DIRS=/home/zxl/ruyi/triton-riscv
export LLVM_SYSPATH=/home/zxl/ruyi/buddy-mlir/llvm/build

/tmp/triton-venv/bin/python -m pip install --no-build-isolation -vvv .
```

Sanity check:

```bash
/tmp/triton-venv/bin/python -c "import triton; print('triton', triton.__version__)"
```

## 6. Runtime Environment Variables (for compiling kernels)

```bash
export TRITON_HOME=/tmp/triton_home
export TRITON_PLUGIN_DIRS=/home/zxl/ruyi/triton-riscv
export LLVM_SYSPATH=/home/zxl/ruyi/buddy-mlir/llvm/build

export LLVM_BINARY_DIR=/home/zxl/ruyi/buddy-mlir/llvm/build/bin
export BUDDY_MLIR_BINARY_DIR=/tmp/buddy-build/bin

export TRITON_SHARED_OPT_PATH=/tmp/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
```

Quick check:

```bash
"$TRITON_SHARED_OPT_PATH" --version | sed -n '1,3p'
"$BUDDY_MLIR_BINARY_DIR/buddy-opt" --version | head -n 2
```

## 7. Run the Minimal Example

### 7.1 Install PyTorch (required by examples)

The example tests import PyTorch. Install a PyTorch build compatible with your
Python in `/tmp/triton-venv`:

```bash
/tmp/triton-venv/bin/python -m pip install torch
```

### 7.2 Where to find intermediate IR artifacts

Triton writes compilation caches under `TRITON_HOME/.triton/`:

- Cache root: `${TRITON_HOME}/.triton/cache/`
- Dump root: `${TRITON_HOME}/.triton/dump/`

To force IR dumps (including `.ttir`, `.llir`, `.ttsharedir`, and `*.obj`), set:

```bash
export TRITON_HOME=/tmp/triton_home
export TRITON_DUMP_DIR=/tmp/triton_home/.triton/dump
export TRITON_KERNEL_DUMP=1
```

After running a kernel once, you can locate artifacts via:

```bash
find "$TRITON_HOME/.triton" -type f \( -name '*.ttir' -o -name '*.llir' -o -name '*.ttsharedir' -o -name '*.obj' -o -name '*.json' \) | sort
```

Example (x86 run in this environment):

- `vec_add` artifacts were found under:
  `/tmp/triton_home/.triton/cache/<hash>/add_kernel.ttir` and
  `/tmp/triton_home/.triton/cache/<hash>/add_kernel.llir` (plus `.obj`, `.json`, `.ttsharedir`)
- dumps were also written under:
  `/tmp/triton_home/.triton/dump/<hash>/add_kernel.ttir` and
  `/tmp/triton_home/.triton/dump/<hash>/add_kernel.llir`

### 7.3 Run a single test

```bash
cd /tmp/triton
export TRITON_HOME=/tmp/triton_home
export TRITON_PLUGIN_DIRS=/home/zxl/ruyi/triton-riscv
export LLVM_SYSPATH=/home/zxl/ruyi/buddy-mlir/llvm/build
export LLVM_BINARY_DIR=/home/zxl/ruyi/buddy-mlir/llvm/build/bin
export BUDDY_MLIR_BINARY_DIR=/tmp/buddy-build/bin
export TRITON_SHARED_OPT_PATH=/tmp/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt

/tmp/triton-venv/bin/python -m pytest /home/zxl/ruyi/triton-riscv/python/examples/test_vec_add.py -v
```

### 7.4 Run matmul and confirm vectorization

```bash
export TRITON_HOME=/tmp/triton_home_matmul
export TRITON_DUMP_DIR=/tmp/triton_home_matmul/.triton/dump
export TRITON_KERNEL_DUMP=1

cd /tmp/triton
/tmp/triton-venv/bin/python -m pip install numpy scipy torch --index-url https://download.pytorch.org/whl/cpu
/tmp/triton-venv/bin/python -m pytest /home/zxl/ruyi/triton-riscv/python/examples/test_matmul.py -v
```

Artifacts:

- `TTIR`: `/tmp/triton_home_matmul/.triton/cache/<hash>/matmul_kernel.ttir`
- `LLVM IR`: `/tmp/triton_home_matmul/.triton/cache/<hash>/matmul_kernel.llir`

Vectorization evidence to look for in `*.llir`:

- LLVM vector types like `<32 x float>`
- vector ops like `shufflevector`, `insertelement`, `extractelement`
- vector fused multiply-add like `@llvm.fmuladd.v32f32`

Example (using a Buddy build in `/home/...`):

```bash
export TRITON_HOME=/tmp/triton_home_matmul_buddyhome
export TRITON_DUMP_DIR=/tmp/triton_home_matmul_buddyhome/.triton/dump
export TRITON_KERNEL_DUMP=1

export BUDDY_MLIR_BINARY_DIR=/home/zxl/ruyi/buddy-mlir/build/bin

cd /tmp/triton
/tmp/triton-venv/bin/python -m pytest /home/zxl/ruyi/triton-riscv/python/examples/test_matmul.py -v
```

In this environment the artifacts landed at:

- `/tmp/triton_home_matmul_buddyhome/.triton/cache/3WC55O5Q5NUQYMC7MHH6XSST6DGTDT66VC7SOYXVD4OLVECXYBKA/matmul_kernel.llir`
- `/tmp/triton_home_matmul_buddyhome/.triton/cache/3WC55O5Q5NUQYMC7MHH6XSST6DGTDT66VC7SOYXVD4OLVECXYBKA/matmul_kernel.ttir`

## Notes

- If you change the Buddy pass pipeline (e.g. `triton-riscv/backend/compiler.py`),
  clear the Triton cache directory you are using, e.g. `rm -rf /tmp/triton_home/.triton/cache`.
- Logs from the successful run are kept under `/tmp`:
  `/tmp/buddy-llvm-*.log`, `/tmp/buddy-build.log`, `/tmp/triton-pip-install.log`.

## One-Click Minimal Verification (Copy-Paste)

Run this block from any shell. It uses overridable variables and validates the
smallest end-to-end path (`test_vec_add.py`).

```bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
TRITON_RISCV_DIR="${TRITON_RISCV_DIR:-$REPO_ROOT}"
TRITON_DIR="${TRITON_DIR:-$(cd "$TRITON_RISCV_DIR/../triton" && pwd)}"
BUDDY_DIR="${BUDDY_DIR:-$(cd "$TRITON_RISCV_DIR/../buddy-mlir" && pwd)}"

TRITON_VENV="${TRITON_VENV:-/tmp/triton-venv}"
TRITON_HOME="${TRITON_HOME:-/tmp/triton_home}"
BUDDY_BUILD="${BUDDY_BUILD:-/tmp/buddy-build}"
TRITON_STAGE="${TRITON_STAGE:-/tmp/triton}"

BUILD_DIR="${BUILD_DIR:-$TRITON_STAGE/build/cmake.linux-x86_64-cpython-3.12}"
TRITON_SHARED_OPT_PATH="${TRITON_SHARED_OPT_PATH:-$BUILD_DIR/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt}"

export TRITON_HOME
export TRITON_PLUGIN_DIRS="$TRITON_RISCV_DIR"
export LLVM_SYSPATH="$BUDDY_DIR/llvm/build"
export LLVM_BINARY_DIR="$BUDDY_DIR/llvm/build/bin"
export BUDDY_MLIR_BINARY_DIR="$BUDDY_BUILD/bin"
export TRITON_SHARED_OPT_PATH

"$TRITON_VENV/bin/python" -m pytest "$TRITON_RISCV_DIR/python/examples/test_vec_add.py" -q -v
```

Expected output includes:

- `1 passed`
