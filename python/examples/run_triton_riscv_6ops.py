import benchmark
import triton.backends as tb

if "triton_shared" in tb.backends and "cpu" in tb.backends:
    tb.backends.pop("triton_shared", None)
benchmark.select_cpu_backend()

import test_performance_matmul as matmul
import test_performance_softmax as softmax
import test_performance_layernorm as layernorm
import test_performance_matvec as matvec
import test_performance_reduce as reduce_mod
import test_performance_vecadd as vecadd

print("=== OP: matmul ===")
for x in [512, 640, 768, 896, 1024]:
    matmul.bench_matmul(x, x, x)

print("=== OP: softmax ===")
for x in [1024, 2048, 4096]:
    softmax.bench_softmax(x)

print("=== OP: layernorm ===")
for x in [64, 128, 256]:
    layernorm.bench_layernorm(x, "triton")

print("=== OP: matvec ===")
for x in [256, 512, 1024]:
    matvec.bench_matvec(x, x)

print("=== OP: reduce ===")
for x in [256, 512, 1024]:
    reduce_mod.bench_reduce(x, 256)

print("=== OP: vecadd ===")
for x in [2**i for i in range(12, 21, 2)]:
    vecadd.bench_vecadd(x)
