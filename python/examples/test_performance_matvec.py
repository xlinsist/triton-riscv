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
def matvec_kernel_aligned(
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
        block_start = block * BLOCK_SIZE_N
        w_block_ptr = tl.make_block_ptr(
            base=row_ptr,
            shape=(N,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE_N,),
            order=(0,),
        )
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(N,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE_N,),
            order=(0,),
        )
        w = tl.load(w_block_ptr).to(tl.float32)
        x = tl.load(x_block_ptr).to(tl.float32)
        acc += tl.sum(w * x, axis=0)

    tl.store(out_ptr + row, acc)


@triton.jit
def matvec_kernel_generic(
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

    for n in range(0, N, BLOCK_SIZE_N):
        offs = n + tl.arange(0, BLOCK_SIZE_N)
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
    kernel = matvec_kernel_aligned if n % block_size_n == 0 else matvec_kernel_generic
    kernel[(m,)](weight, vec, out, m, n, weight.stride(0), BLOCK_SIZE_N=block_size_n)
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
