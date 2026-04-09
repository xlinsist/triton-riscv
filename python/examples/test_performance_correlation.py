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
def correlation_kernel(
    src0_ptr,
    src1_ptr,
    out_ptr,
    out_channel,
    in_channel,
    height,
    width,
    hw,
    out_shift,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)

    w_idx = pid_x * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    h_idx = pid_y * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    bound = ((h_idx[:, None] < height) & (w_idx[None, :] < width)) & (w_idx[None, :] >= pid_z)
    offsets = h_idx[:, None] * width + w_idx[None, :]

    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.int16)
    src0_ptrs = src0_ptr + offsets
    src1_ptrs = src1_ptr + offsets

    for _ in range(in_channel):
        a = tl.load(src0_ptrs, mask=bound, other=0)
        b = tl.load(src1_ptrs - pid_z, mask=bound, other=0)
        acc += a * b
        src0_ptrs += hw
        src1_ptrs += hw

    out_idx = pid_z * hw + offsets
    out_val = (acc >> out_shift).to(tl.int8)
    tl.store(out_ptr + out_idx, out_val, mask=bound)


def correlation(src0: torch.Tensor, src1: torch.Tensor, out: torch.Tensor, out_shift: int):
    in_c, h, w = src0.shape
    out_c = out.shape[0]
    grid = lambda meta: (
        triton.cdiv(w, meta["BLOCK_SIZE_W"]),
        triton.cdiv(h, meta["BLOCK_SIZE_H"]),
        out_c,
    )
    try:
        correlation_kernel[grid](
            src0,
            src1,
            out,
            out_c,
            in_c,
            h,
            w,
            h * w,
            out_shift,
            BLOCK_SIZE_H=2,
            BLOCK_SIZE_W=8,
        )
        return out
    except Exception as e:
        _warn_once(f"[correlation] Triton failed, fallback to torch implementation: {e}")
        for oc in range(out_c):
            acc = torch.zeros((h, w), dtype=torch.int16, device=src0.device)
            for ic in range(in_c):
                a = src0[ic].to(torch.int16)
                b = torch.zeros_like(a)
                if oc == 0:
                    b[:, :] = src1[ic].to(torch.int16)
                else:
                    b[:, oc:] = src1[ic, :, :-oc].to(torch.int16)
                acc += a * b
            out[oc] = (acc >> out_shift).to(torch.int8)
        return out


@benchmark.measure()
def bench_correlation(in_c, out_c, h, w):
    src0 = torch.randint(0, 16, (in_c, h, w), device="cpu", dtype=torch.int8)
    src1 = torch.randint(0, 35, (in_c, h, w), device="cpu", dtype=torch.int8)
    out = torch.zeros((out_c, h, w), device="cpu", dtype=torch.int8)
    correlation(src0, src1, out, 0)


if __name__ == "__main__":
    _select_cpu_backend_compat()
    for x in [64, 96]:
        bench_correlation(32, 4, x, x)
