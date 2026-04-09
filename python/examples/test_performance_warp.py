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
def warp_kernel(src_ptr, offset_ptr, out_ptr, height, width, BLOCK_SIZE_W: tl.constexpr):
    pid_h = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    for off in range(0, width, BLOCK_SIZE_W):
        w_idx = off + tl.arange(0, BLOCK_SIZE_W)
        mask = w_idx < width

        offset_idx = pid_h * width + w_idx
        offset_val = tl.load(offset_ptr + offset_idx, mask=mask, other=0).to(tl.int16)
        offset_int = (offset_val >> 8).to(tl.int8)
        offset_frac = ((offset_val << 8) >> 8).to(tl.int8)

        right_idx = (w_idx.to(tl.int8) - offset_int).to(tl.int8)
        left_idx = (right_idx - 1).to(tl.int8)

        src_base = pid_c * height * width + pid_h * width
        right = tl.load(src_ptr + src_base + right_idx, mask=mask, other=0).to(tl.int8)
        left = tl.load(src_ptr + src_base + left_idx, mask=mask, other=0).to(tl.int8)

        right = tl.where(right_idx < 0, 0, right)
        left = tl.where(left_idx < 0, 0, left)

        out = (right.to(tl.int16) << 8)
        out += (left - right).to(tl.int16) * offset_frac.to(tl.int16)
        out = (out >> 8).to(tl.int8)

        tl.store(out_ptr + src_base + w_idx, out, mask=mask)


def warp(src: torch.Tensor, offset: torch.Tensor, out: torch.Tensor):
    c, h, w = src.shape
    try:
        warp_kernel[(h, c, 1)](src, offset, out, h, w, BLOCK_SIZE_W=16)
        return out
    except Exception as e:
        _warn_once(f"[warp] Triton failed, fallback to torch implementation: {e}")
        idx = torch.arange(w, device=src.device, dtype=torch.int16)[None, :].expand(h, w)
        offset_int = (offset >> 8).to(torch.int16)
        offset_frac = ((offset << 8) >> 8).to(torch.int16)
        right_idx = idx - offset_int
        left_idx = right_idx - 1

        for ch in range(c):
            row = src[ch].to(torch.int16)
            right = torch.where((right_idx >= 0) & (right_idx < w), row.gather(1, right_idx.clamp(0, w - 1).to(torch.int64)), 0)
            left = torch.where((left_idx >= 0) & (left_idx < w), row.gather(1, left_idx.clamp(0, w - 1).to(torch.int64)), 0)
            out_ch = ((right << 8) + (left - right) * offset_frac) >> 8
            out[ch] = out_ch.to(torch.int8)
        return out


@benchmark.measure()
def bench_warp(c, h, w):
    src = torch.randint(-64, 64, (c, h, w), device="cpu", dtype=torch.int8)
    offset = torch.zeros((h, w), device="cpu", dtype=torch.int16)
    out = torch.empty((c, h, w), device="cpu", dtype=torch.int8)
    warp(src, offset, out)


if __name__ == "__main__":
    _select_cpu_backend_compat()
    for x in [64, 128, 256]:
        bench_warp(3, x, x)
