import torch

import benchmark
import triton
import triton.language as tl

_WARNED = False


def _warn_once(msg: str):
    global _WARNED
    if not _WARNED:
        print(msg)
        _WARNED = True


def _select_cpu_backend_compat():
    try:
        benchmark.select_cpu_backend()
    except Exception:
        pass


@triton.jit
def gather_scatter_kernel(in_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    gather = offs // 4
    x = tl.load(in_ptr + gather, mask=gather < N, other=0)
    tl.store(out_ptr + offs, x)


def gather_scatter(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    out = torch.empty_like(x)
    try:
        gather_scatter_kernel[(1,)](x, out, N=n)
        return out
    except Exception as e:
        _warn_once(f"[gather_scatter] Triton failed, fallback to torch indexing: {e}")
        gather = torch.arange(n, device=x.device) // 4
        return x[gather]


@benchmark.measure()
def bench_gather_scatter(size):
    x = torch.arange(size, device="cpu", dtype=torch.int32)
    gather_scatter(x)


if __name__ == "__main__":
    _select_cpu_backend_compat()
    for x in [64, 128, 256]:
        bench_gather_scatter(x)
