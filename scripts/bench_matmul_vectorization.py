import argparse
import os
import re
import statistics
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_latest_llir(triton_home: Path) -> Path | None:
    cache_dir = triton_home / ".triton" / "cache"
    if not cache_dir.is_dir():
        return None
    llirs = list(cache_dir.rglob("matmul_kernel.llir"))
    if not llirs:
        return None
    llirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return llirs[0]


def _vectorization_signals(llir_path: Path) -> dict:
    text = llir_path.read_text(errors="replace")
    vec_types = len(re.findall(r"<\s*\d+\s+x\s+float\s*>", text))
    fmuladd_vec = len(re.findall(r"@llvm\.fmuladd\.v\d+f32", text))
    shuf = len(re.findall(r"shufflevector", text))
    return {"vec_types": vec_types, "fmuladd_vec": fmuladd_vec, "shufflevector": shuf}


def _bench(fn, repeats: int) -> dict:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "avg": statistics.fmean(times),
        "min": min(times),
        "max": max(times),
        "std": statistics.pstdev(times) if len(times) > 1 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=int, default=[256, 384, 512, 640, 768])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    # Keep variance down and make runs more comparable.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    triton_home = Path(os.environ.get("TRITON_HOME", "")).expanduser()
    if not triton_home:
        print("ERROR: TRITON_HOME is not set (use a writable path, e.g. /tmp/exp_matmul_vec_on).")
        return 2

    examples_dir = _repo_root() / "python" / "examples"
    sys.path.insert(0, str(examples_dir))

    import torch  # noqa: E402

    # Ensure PyTorch uses a stable number of threads.
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

    import benchmark  # noqa: E402

    benchmark.select_cpu_backend()

    # Load the existing matmul implementation without running its __main__.
    import runpy  # noqa: E402

    ns = runpy.run_path(str(examples_dir / "test_matmul.py"))
    matmul = ns["matmul"]

    print("=== Config ===")
    print(f"TRITON_HOME={triton_home}")
    print(
        "TRITON_RISCV_DISABLE_MATMUL_VECTORIZATION="
        + os.environ.get("TRITON_RISCV_DISABLE_MATMUL_VECTORIZATION", "")
    )
    print(f"TORCH_NUM_THREADS={torch.get_num_threads()}")
    print(f"sizes={args.sizes} warmup={args.warmup} repeats={args.repeats}")

    for n in args.sizes:
        a = torch.randn((n, n), device="cpu", dtype=torch.float32)
        b = torch.randn((n, n), device="cpu", dtype=torch.float32)

        # Warmup: compile + run, then execute a few steady-state runs.
        for _ in range(args.warmup):
            matmul(a, b)

        stats = _bench(lambda: matmul(a, b), args.repeats)
        print(
            f"N={n}: Wall avg={stats['avg']:.6f}s min={stats['min']:.6f}s "
            f"std={stats['std']:.6f}s max={stats['max']:.6f}s"
        )

    llir = _find_latest_llir(triton_home)
    if llir is None:
        print("Could not find matmul_kernel.llir under TRITON_HOME/.triton/cache (did compilation run?).")
        return 1

    sig = _vectorization_signals(llir)
    print("=== Artifacts ===")
    print(f"latest_llir={llir}")
    print(f"vectorization_signals={sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
