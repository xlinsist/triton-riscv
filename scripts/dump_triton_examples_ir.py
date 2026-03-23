import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "python" / "examples"
ALL_EXAMPLES = ["matmul", "vecadd", "softmax", "layernorm"]
KEEP_FILES = {
    "tt.mlir",
    "ttshared.mlir",
    "ll.mlir",
    "ll.ir",
    "_ttshared_verify_out.mlir",
    "ttshared-main.mlir",
    "linalg_bufferized_no_vec_no_loops.mlir",
}

MAIN_SNIPPETS = {
    "matmul": """  func.func @main() -> i32 {
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %a = memref.alloc() : memref<32x16xf32>
    %b = memref.alloc() : memref<16x64xf32>
    %c = memref.alloc() : memref<32x64xf32>
    %a_u = memref.cast %a : memref<32x16xf32> to memref<*xf32>
    %b_u = memref.cast %b : memref<16x64xf32> to memref<*xf32>
    %c_u = memref.cast %c : memref<32x64xf32> to memref<*xf32>
    call @matmul_kernel(%a_u, %b_u, %c_u, %c32_i32, %c64_i32, %c16_i32, %c16_i32, %c64_i32, %c64_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    memref.dealloc %a : memref<32x16xf32>
    memref.dealloc %b : memref<16x64xf32>
    memref.dealloc %c : memref<32x64xf32>
    return %c0_i32 : i32
  }""",
    "vecadd": """  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %x = memref.alloc() : memref<1024xf32>
    %y = memref.alloc() : memref<1024xf32>
    %z = memref.alloc() : memref<1024xf32>
    %x_u = memref.cast %x : memref<1024xf32> to memref<*xf32>
    %y_u = memref.cast %y : memref<1024xf32> to memref<*xf32>
    %z_u = memref.cast %z : memref<1024xf32> to memref<*xf32>
    call @add_kernel(%x_u, %y_u, %z_u, %c1024_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32, i32) -> ()
    memref.dealloc %x : memref<1024xf32>
    memref.dealloc %y : memref<1024xf32>
    memref.dealloc %z : memref<1024xf32>
    return %c0_i32 : i32
  }""",
    "softmax": """  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %out = memref.alloc() : memref<256xf32>
    %inp = memref.alloc() : memref<256xf32>
    %out_u = memref.cast %out : memref<256xf32> to memref<*xf32>
    %inp_u = memref.cast %inp : memref<256xf32> to memref<*xf32>
    call @softmax_kernel(%out_u, %inp_u, %c256_i32, %c256_i32, %c256_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) : (memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    memref.dealloc %out : memref<256xf32>
    memref.dealloc %inp : memref<256xf32>
    return %c0_i32 : i32
  }""",
    "layernorm": """  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32
    %eps = arith.constant 1.000000e-05 : f32
    %x = memref.alloc() : memref<512xf16>
    %y = memref.alloc() : memref<512xf16>
    %w = memref.alloc() : memref<512xf16>
    %b = memref.alloc() : memref<512xf16>
    %mean = memref.alloc() : memref<1xf32>
    %rstd = memref.alloc() : memref<1xf32>
    %x_u = memref.cast %x : memref<512xf16> to memref<*xf16>
    %y_u = memref.cast %y : memref<512xf16> to memref<*xf16>
    %w_u = memref.cast %w : memref<512xf16> to memref<*xf16>
    %b_u = memref.cast %b : memref<512xf16> to memref<*xf16>
    %mean_u = memref.cast %mean : memref<1xf32> to memref<*xf32>
    %rstd_u = memref.cast %rstd : memref<1xf32> to memref<*xf32>
    call @_layer_norm_fwd_fused(%x_u, %y_u, %w_u, %b_u, %mean_u, %rstd_u, %c512_i32, %c512_i32, %eps, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) : (memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf32>, memref<*xf32>, i32, i32, f32, i32, i32, i32, i32, i32, i32) -> ()
    memref.dealloc %x : memref<512xf16>
    memref.dealloc %y : memref<512xf16>
    memref.dealloc %w : memref<512xf16>
    memref.dealloc %b : memref<512xf16>
    memref.dealloc %mean : memref<1xf32>
    memref.dealloc %rstd : memref<1xf32>
    return %c0_i32 : i32
  }""",
}


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"ERROR: env {name} is required")
    return value


def _buddy_opt() -> str:
    return str(Path(_require_env("BUDDY_MLIR_BINARY_DIR")) / "buddy-opt")


def _run_cmd(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def _setup_python_path() -> None:
    sys.path.insert(0, str(EXAMPLES_DIR))


def _select_cpu_backend() -> None:
    import benchmark

    benchmark.select_cpu_backend()


def _set_stable_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))


def _dump_one(example: str, out_dir: Path) -> None:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*"):
        if f.is_file():
            f.unlink()

    os.environ["TRITON_SHARED_DUMP_PATH"] = str(out_dir)
    os.environ["TRITON_HOME"] = f"/tmp/exp_dump_{example}_{time.time_ns()}"

    if example == "matmul":
        import test_matmul as m

        m.matmul(
            torch.randn((128, 128), device="cpu", dtype=torch.float32),
            torch.randn((128, 128), device="cpu", dtype=torch.float32),
        )
        return
    if example == "vecadd":
        import test_vec_add as m

        n = 1 << 16
        m.add(
            torch.rand(n, device="cpu", dtype=torch.float32),
            torch.rand(n, device="cpu", dtype=torch.float32),
        )
        return
    if example == "softmax":
        import test_softmax as m

        m.softmax(torch.randn((256, 256), device="cpu", dtype=torch.float32))
        return
    if example == "layernorm":
        import test_layernorm as m

        x_shape = (64, 512)
        w_shape = (x_shape[-1],)
        dtype = torch.float16
        device = "cpu"
        eps = 1e-5
        weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
        m.LayerNorm.apply(x, w_shape, weight, bias, eps, device)
        return
    raise SystemExit(f"Unknown example: {example}")


def _verify_dump_exists(example: str, out_dir: Path) -> None:
    for name in ("tt.mlir", "ttshared.mlir", "ll.mlir", "ll.ir"):
        path = out_dir / name
        if not path.exists():
            raise SystemExit(f"{example}: missing {path}")


def _emit_verify_out(example: str, out_dir: Path) -> Path:
    out = out_dir / "_ttshared_verify_out.mlir"
    _run_cmd([_buddy_opt(), str(out_dir / "ttshared.mlir"), "--verify-each", "-o", str(out)])
    return out


def _emit_ttshared_main(example: str, verify_out: Path, out_dir: Path) -> Path:
    out = out_dir / "ttshared-main.mlir"
    text = verify_out.read_text()
    if "func.func @main() -> i32" in text:
        out.write_text(text)
        return out
    text = text.rstrip()
    if not text.endswith("}"):
        raise SystemExit(f"{example}: unexpected module ending in {verify_out}")
    body = text[:-1].rstrip()
    snippet = MAIN_SNIPPETS[example]
    out.write_text(f"{body}\n\n{snippet}\n}}\n")
    return out


def _emit_bufferized(ttshared_main: Path, out_dir: Path) -> Path:
    out = out_dir / "linalg_bufferized_no_vec_no_loops.mlir"
    _run_cmd(
        [
            _buddy_opt(),
            str(ttshared_main),
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=allow-return-allocs-from-loops=true",
            "-o",
            str(out),
        ]
    )
    return out


def _cleanup_root(out_root: Path, examples: list[str]) -> None:
    for ex in examples:
        d = out_root / ex
        for f in d.glob("*"):
            if f.is_file() and f.name not in KEEP_FILES:
                f.unlink()
    for f in out_root.glob("*"):
        if f.is_file():
            f.unlink()


def _validate_main_boundary(out_root: Path, examples: list[str]) -> None:
    for ex in examples:
        d = out_root / ex
        if "func.func @main() -> i32" in (d / "ttshared.mlir").read_text():
            raise SystemExit(f"{ex}: ttshared.mlir unexpectedly contains main")
        if "func.func @main() -> i32" in (d / "_ttshared_verify_out.mlir").read_text():
            raise SystemExit(f"{ex}: _ttshared_verify_out.mlir unexpectedly contains main")
        if "func.func @main() -> i32" not in (d / "ttshared-main.mlir").read_text():
            raise SystemExit(f"{ex}: ttshared-main.mlir missing main")
        if (
            "func.func @main() -> i32"
            not in (d / "linalg_bufferized_no_vec_no_loops.mlir").read_text()
        ):
            raise SystemExit(f"{ex}: linalg_bufferized_no_vec_no_loops.mlir missing main")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild experiments IR assets.")
    parser.add_argument(
        "--examples",
        nargs="+",
        default=ALL_EXAMPLES,
        choices=ALL_EXAMPLES,
        help="Examples to regenerate.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not apply whitelist cleanup.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    examples = args.examples
    out_root = Path(
        os.environ.get("TRITON_EXPERIMENT_OUT_ROOT", str(REPO_ROOT / "experiments"))
    )

    _require_env("TRITON_SHARED_OPT_PATH")
    _require_env("LLVM_BINARY_DIR")
    _require_env("BUDDY_MLIR_BINARY_DIR")
    if not shutil.which("python3"):
        raise SystemExit("python3 is required")

    _setup_python_path()
    _set_stable_threads()
    _select_cpu_backend()

    for ex in examples:
        out_dir = out_root / ex
        _dump_one(ex, out_dir)
        _verify_dump_exists(ex, out_dir)
        verify_out = _emit_verify_out(ex, out_dir)
        ttshared_main = _emit_ttshared_main(ex, verify_out, out_dir)
        _emit_bufferized(ttshared_main, out_dir)

    if not args.no_clean:
        _cleanup_root(out_root, examples)
    _validate_main_boundary(out_root, examples)

    print("Regenerated examples:", ", ".join(examples))
    for ex in examples:
        d = out_root / ex
        files = sorted([p.name for p in d.glob("*") if p.is_file()])
        print(f"[{ex}] " + ", ".join(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
