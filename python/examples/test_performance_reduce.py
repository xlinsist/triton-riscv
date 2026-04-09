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
def reduce_sum_rows_kernel(x_ptr, out_ptr, stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    ptr = x_ptr + row * stride + offs
    x = tl.load(ptr, mask=offs < n_cols, other=0.0)
    y = tl.sum(x, axis=0)
    tl.store(out_ptr + row, y)


def reduce_sum_rows(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty((rows,), device=x.device, dtype=x.dtype)
    block = triton.next_power_of_2(cols)
    reduce_sum_rows_kernel[(rows,)](x, out, x.stride(0), cols, BLOCK_SIZE=block)
    return out


@benchmark.measure(repeats=20, warmup=5)
def bench_reduce(rows, cols):
    x = torch.randn((rows, cols), device="cpu", dtype=torch.float32)
    reduce_sum_rows(x)


if __name__ == "__main__":
    _select_cpu_backend_compat()
    for x in [256, 512, 1024]:
        bench_reduce(x, 256)
