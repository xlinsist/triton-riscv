import torch

import benchmark
import triton
import triton.language as tl
from triton.language.extra import libdevice

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
def asin_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = libdevice.asin(x)
    tl.store(y_ptr + offsets, y, mask=mask)


def asin_triton(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    try:
        asin_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
        return out
    except Exception as e:
        _warn_once(f"[extern_asin] Triton failed, fallback to torch.asin: {e}")
        return torch.asin(x)


@benchmark.measure()
def bench_extern_asin(size):
    x = torch.rand(size, device="cpu", dtype=torch.float32)
    asin_triton(x)


if __name__ == "__main__":
    _select_cpu_backend_compat()
    for x in [2**i for i in range(12, 21, 2)]:
        bench_extern_asin(x)
