import torch

import benchmark
from test_einsum_qhmd_hmpd_to_qhmp import (
    einsum_qhmd_hmpd_to_qhmp,
    select_cpu_backend_compat,
)


@benchmark.measure()
def bench_einsum(q, h, m, p, d):
    a = torch.randn((q, h, m, d), dtype=torch.float32, device="cpu")
    b = torch.randn((h, m, p, d), dtype=torch.float32, device="cpu")
    einsum_qhmd_hmpd_to_qhmp(a, b, BLOCK_QHM=4, BLOCK_P=8)


if __name__ == "__main__":
    select_cpu_backend_compat()
    for x in [8, 16, 32]:
        bench_einsum(2, 4, x, x, 32)
