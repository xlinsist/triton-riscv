import torch

import benchmark
import triton
import triton.language as tl


def _select_cpu_backend_compat():
    try:
        import triton.backends as tb

        if "triton_shared" in tb.backends and "cpu" in tb.backends:
            tb.backends.pop("triton_shared", None)
    except Exception:
        pass
    benchmark.select_cpu_backend()


@triton.jit
def add_kernel_no_mask(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)


def vecadd_no_mask(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel_no_mask[grid](x, y, out, n_elements, BLOCK_SIZE=256)
    return out


@benchmark.measure()
def bench_vecadd_no_mask(size):
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)
    vecadd_no_mask(x, y)


if __name__ == "__main__":
    _select_cpu_backend_compat()
    # Use multiples of BLOCK_SIZE=256 so no tail handling is needed.
    for x in [2**i for i in range(12, 21, 2)]:
        bench_vecadd_no_mask(x)
