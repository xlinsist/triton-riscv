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
def matvec_kernel(
    w_ptr,
    x_ptr,
    out_ptr,
    M,
    N,
    stride_wm,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    row_ptr = w_ptr + row * stride_wm
    acc = 0.0
    full_blocks = N // BLOCK_SIZE_N

    # Split full tiles from the tail so full-width loads avoid masked staging.
    for block in range(0, full_blocks):
        offs = block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        w = tl.load(row_ptr + offs).to(tl.float32)
        x = tl.load(x_ptr + offs).to(tl.float32)
        acc += tl.sum(w * x, axis=0)

    tail_start = full_blocks * BLOCK_SIZE_N
    if tail_start < N:
        offs = tail_start + tl.arange(0, BLOCK_SIZE_N)
        mask = offs < N
        w = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(w * x, axis=0)

    tl.store(out_ptr + row, acc)


def matvec(weight: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    assert weight.dim() == 2
    assert vec.dim() == 1
    assert weight.shape[1] == vec.shape[0]
    assert weight.is_contiguous()
    assert vec.is_contiguous()

    m, n = weight.shape
    out = torch.empty((m,), device=weight.device, dtype=torch.float32)
    block_size_n = min(triton.next_power_of_2(n), 1024)
    matvec_kernel[(m,)](weight, vec, out, m, n, weight.stride(0), BLOCK_SIZE_N=block_size_n)
    return out


@benchmark.measure()
def bench_matvec(m, n):
    weight = torch.randn((m, n), device="cpu", dtype=torch.float32)
    vec = torch.randn((n,), device="cpu", dtype=torch.float32)
    matvec(weight, vec)


if __name__ == "__main__":
    _select_cpu_backend_compat()
    for x in [256, 512, 1024]:
        bench_matvec(x, x)
