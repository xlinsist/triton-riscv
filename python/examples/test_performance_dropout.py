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
def dropout_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    for i in range(0, tl.cdiv(BLOCK_SIZE, TILE_SIZE)):
        offsets = block_start + i * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        rnd = tl.rand(seed, offsets)
        keep = rnd > p
        out = tl.where(keep, x / (1 - p), 0.0)
        tl.store(out_ptr + offsets, out, mask=mask)


def seeded_dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    try:
        dropout_kernel[grid](x, out, n_elements, p, seed, BLOCK_SIZE=1024, TILE_SIZE=16)
        return out
    except Exception as e:
        _warn_once(f"[dropout] Triton failed, fallback to torch rand mask: {e}")
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        keep = torch.rand(x.shape, generator=g, device=x.device) > p
        return torch.where(keep, x / (1 - p), torch.zeros_like(x))


@benchmark.measure()
def bench_dropout(size, p):
    x = torch.randn((size,), device="cpu", dtype=torch.float32)
    seeded_dropout(x, p=p, seed=123)


if __name__ == "__main__":
    _select_cpu_backend_compat()
    for x in [2**i for i in range(14, 21, 2)]:
        bench_dropout(x, 0.5)
