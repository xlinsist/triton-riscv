# triton-riscv

Triton compiler for RISC-V platforms.

This repository is forked from [triton-shared](https://github.com/microsoft/triton-shared) and provides a Triton compiler backend for RISC-V. The upstream triton-shared repo is no longer maintained, so this project is developed independently under the name triton-riscv.

Triton acts as a frontend (Python AST → TTIR); triton-riscv handles the rest of the pipeline (TTIR → Linalg → LLVM IR → native object). No NVIDIA or AMD toolchain is required.

**Compilation pipeline:**

```
Python @triton.jit
  └─► TTIR          (triton: ast_to_ttir + make_ttir passes)
        └─► Linalg MLIR   (triton-shared-opt --triton-to-linalg-experimental)
              └─► LLVM MLIR  (Buddy Compiler lowering passes)
                    └─► LLVM IR  (mlir-translate --mlir-to-llvmir)
                          └─► native .o  (llc -filetype=obj)
```

## Clone

Set `TRITON_PLUGIN_DIRS` to your `triton-riscv` directory so that Triton can discover the plugin:

```sh
export TRITON_PLUGIN_DIRS=$(pwd)/triton-riscv

git clone https://github.com/RuyiAI-Stack/triton-riscv.git triton-riscv
git clone https://github.com/triton-lang/triton.git
cd triton && git checkout $(cat ../triton-riscv/triton-hash.txt)
```

Apply the build-system patches from `triton-riscv/patches/` using the helper script. These patches make the NVIDIA/AMD LLVM codegen libraries conditional on their backends being enabled, so the build succeeds without any GPU toolchain.

```sh
cd triton
/path/to/triton-riscv/scripts/apply_patches.sh ./
```

The script is idempotent: re-running it on an already-patched tree prints `SKIPPED (patch already applied)` per patch and exits cleanly.

## Prerequisites

### Create a virtual environment

```sh
# From the triton root directory
python -m venv .venv --prompt triton-riscv
source .venv/bin/activate
```

### Install dependencies

1. Install the [dependencies](https://github.com/buddy-compiler/buddy-mlir?tab=readme-ov-file#llvmmlir-dependencies) required by the Ruyi Buddy Compiler.

2. Install triton-riscv Python dependencies:

   ```sh
   pip install pytest-xdist pybind11 setuptools
   ```

3. Build the Buddy Compiler — [Getting started](https://github.com/buddy-compiler/buddy-mlir?tab=readme-ov-file#getting-started)

## Build

```sh
cd triton

# Point triton's plugin discovery at triton-riscv.
export TRITON_PLUGIN_DIRS=/path/to/triton-riscv

# Use a custom LLVM build instead of the one triton downloads.
export LLVM_SYSPATH=/path/to/buddy-mlir/llvm/build

# Install.
pip install --no-build-isolation -vvv .
```

> **Tip – incremental rebuilds**: After the first full build you can skip the
> slow CMake reconfigure by building directly in the CMake output directory:
>
> ```sh
> BUILD_DIR=$(ls -d build/cmake.linux-*-cpython-*)
> cmake --build $BUILD_DIR -j$(nproc)
> # then re-install so the new .so is picked up by Python:
> pip install --no-build-isolation -vvv .
> ```

Build artifacts are placed under `triton/build/{current_cmake_version}/third_party/triton_shared`.

## Set runtime environment variables

Before running tests or Triton kernels, set these in your environment:

```sh
# Path to the triton-shared-opt binary produced by the build.
BUILD_DIR=$(ls -d build/cmake.linux-*-cpython-*)
export TRITON_SHARED_OPT_PATH=$(pwd)/${BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt

# Directory containing buddy-opt (buddy-mlir build's bin/).
export BUDDY_MLIR_BINARY_DIR=/path/to/buddy-mlir/build/bin

# Directory containing mlir-translate, llc, opt, clang++ (LLVM build's bin/).
export LLVM_BINARY_DIR=/path/to/buddy-mlir/llvm/build/bin
```

After changing the buddy-opt pass pipeline (e.g. in `backend/compiler.py`), please Triton's cache (e.g. `rm -rf ~/.triton/cache`).

## Run the example test suite

Ensure PyTorch is available in your virtual environment; on RISC-V we recommend building PyTorch from source or cross-compiling it.

```sh
pytest ../triton-riscv/python/examples/ \
    --ignore=../triton-riscv/python/examples/test_core.py \
    --ignore=../triton-riscv/python/examples/test_annotations.py \
    -v
```

To run a single test:

```sh
pytest ../triton-riscv/python/examples/test_vec_add.py -v
```

## FAQ

Building on RISC-V often runs into dependency issues; we're happy to help if you run into trouble.

### Fortran library mismatch when building PyTorch

Preload the correct Fortran library by adding the following to your virtualenv’s `activate` script (e.g. `~/triton/.venv/bin/activate`):

```sh
export LD_PRELOAD=/usr/lib64/libgfortran.so.5:$LD_PRELOAD
```
