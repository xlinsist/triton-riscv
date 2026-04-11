#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
}

RISCV_EXTRA_ENV = {
    "TRITON_CPU_BACKEND": "triton-riscv",
    "TRITON_RISCV_STRUCTURED_LDST_MODE": "tensor_first_vector_cpu",
}

PASSTHROUGH_ENV_KEYS = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "PATH",
    "TMPDIR",
    "TRITON_SHARED_OPT_PATH",
    "LLVM_BINARY_DIR",
    "BUDDY_MLIR_BINARY_DIR",
    "USER",
)


@dataclass(frozen=True)
class RepoSpec:
    name: str
    root: Path
    python: Path
    module_root: str


@dataclass(frozen=True)
class OpSpec:
    name: str
    input_desc: str
    config: dict


OPS = [
    OpSpec("softmax", "size=2048", {"size": 2048}),
    OpSpec("layernorm", "size=128", {"size": 128}),
    OpSpec("matvec", "M=N=2048", {"m": 2048, "n": 2048}),
    OpSpec("reduce", "rows=32768, cols=128", {"rows": 32768, "cols": 128}),
    OpSpec("vecadd", "size=4194304", {"size": 4194304}),
]


HELPER = r"""
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch

repo_root = Path(os.environ["REPO_ROOT"])
module_root = repo_root / os.environ["MODULE_ROOT"]
sys.path.insert(0, str(module_root))
sys.path.insert(0, str(repo_root))

import benchmark

op = os.environ["OP_NAME"]
cfg = json.loads(os.environ["OP_CONFIG"])
warmup = int(os.environ["WARMUP"])
repeats = int(os.environ["REPEATS"])

torch.set_num_threads(1)
torch.manual_seed(0)

if op == "softmax":
    mod = importlib.import_module("test_performance_softmax")
    benchmark.select_cpu_backend()
    size = int(cfg["size"])
    x = torch.randn(size, size, device="cpu")
    fn = lambda: mod.softmax(x)
elif op == "layernorm":
    mod = importlib.import_module("test_performance_layernorm")
    benchmark.select_cpu_backend()
    size = int(cfg["size"])
    device = "cpu"
    eps = 1e-5
    dtype = torch.float16
    x_shape = (size, size)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    fn = lambda: mod.LayerNorm.apply(x, w_shape, weight, bias, eps, device)
elif op == "matvec":
    mod = importlib.import_module("test_performance_matvec")
    if hasattr(mod, "_select_cpu_backend_compat"):
        mod._select_cpu_backend_compat()
    else:
        benchmark.select_cpu_backend()
    m = int(cfg["m"])
    n = int(cfg["n"])
    weight = torch.randn((m, n), device="cpu", dtype=torch.float32)
    vec = torch.randn((n,), device="cpu", dtype=torch.float32)
    fn = lambda: mod.matvec(weight, vec)
elif op == "reduce":
    mod = importlib.import_module("test_performance_reduce")
    if hasattr(mod, "_select_cpu_backend_compat"):
        mod._select_cpu_backend_compat()
    else:
        benchmark.select_cpu_backend()
    rows = int(cfg["rows"])
    cols = int(cfg["cols"])
    x = torch.randn((rows, cols), device="cpu", dtype=torch.float32)
    fn = lambda: mod.reduce_sum_rows(x)
elif op == "vecadd":
    mod = importlib.import_module("test_performance_vecadd")
    if hasattr(mod, "_select_cpu_backend_compat"):
        mod._select_cpu_backend_compat()
    else:
        benchmark.select_cpu_backend()
    size = int(cfg["size"])
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)
    fn = lambda: mod.vecadd(x, y)
else:
    raise RuntimeError(f"unsupported op: {op}")

for _ in range(warmup):
    fn()

times = []
for _ in range(repeats):
    start = time.perf_counter()
    fn()
    times.append(time.perf_counter() - start)

result = {
    "op": op,
    "wall_avg_s": sum(times) / len(times),
    "wall_min_s": min(times),
    "wall_max_s": max(times),
    "warmup": warmup,
    "repeats": repeats,
}
print(json.dumps(result, indent=2))
"""


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=args.timeout,
    )


def git_rev_parse(repo: Path) -> str:
    return run(["git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.strip()


def record_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_child_env() -> dict[str, str]:
    env = {}
    for key in PASSTHROUGH_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def build_env(repo: RepoSpec, out_dir: Path, op: OpSpec, extra_env: dict[str, str]) -> dict[str, str]:
    env = make_child_env()
    env.update(THREAD_ENV)
    env.update(extra_env)
    env["REPO_ROOT"] = str(repo.root)
    env["MODULE_ROOT"] = repo.module_root
    env["OP_NAME"] = op.name
    env["OP_CONFIG"] = json.dumps(op.config)
    env["WARMUP"] = str(args.warmup)
    env["REPEATS"] = str(args.repeats)
    env["TRITON_CACHE_DIR"] = str(out_dir / "cache" / repo.name / op.name)
    env["TRITON_SHARED_DUMP_PATH"] = str(out_dir / "ir" / repo.name / op.name)
    return env


def run_one(repo: RepoSpec, out_dir: Path, op: OpSpec, extra_env: dict[str, str]) -> dict:
    env = build_env(repo, out_dir, op, extra_env)
    proc = run([str(repo.python), "-c", HELPER], cwd=repo.root, env=env)
    log_path = out_dir / "logs" / repo.name / f"{op.name}.log"
    record_text(log_path, proc.stdout)
    result = json.loads(proc.stdout)
    result["input"] = op.input_desc
    result["repo"] = repo.name
    result["repo_root"] = str(repo.root)
    json_path = out_dir / "logs" / repo.name / f"{op.name}.json"
    record_text(json_path, json.dumps(result, indent=2) + "\n")
    return result


def write_results(out_dir: Path, cpu_rows: dict[str, dict], riscv_rows: dict[str, dict]) -> None:
    lines = [
        "# SG2044 Five Ops Report (Tensor-First RISC-V)",
        "",
        f"- timestamp: `{out_dir.name}`",
        "- mode: `benchmark`",
        "- metric: `speedup = Triton CPU Wall Avg / Triton RISC-V Wall Avg`",
        "- riscv forced env: `TRITON_RISCV_STRUCTURED_LDST_MODE=tensor_first_vector_cpu`",
        f"- warmup/repeats: `{args.warmup}/{args.repeats}`",
        "- single-thread env: `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/TORCH=1`",
        "",
        "| op | input | triton-cpu wall(s) | triton-riscv wall(s) | speedup |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    plot_rows = []
    for op in OPS:
        cpu_row = cpu_rows[op.name]
        riscv_row = riscv_rows[op.name]
        speedup = cpu_row["wall_avg_s"] / riscv_row["wall_avg_s"]
        lines.append(
            f"| {op.name} | `{op.input_desc}` | `{cpu_row['wall_avg_s']:.6f}` | `{riscv_row['wall_avg_s']:.6f}` | `{speedup:.6f}x` |"
        )
        plot_rows.append(
            {
                "op": op.name,
                "input": op.input_desc,
                "cpu_wall_avg_s": cpu_row["wall_avg_s"],
                "riscv_wall_avg_s": riscv_row["wall_avg_s"],
                "speedup": speedup,
            }
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- logs: `logs/cpu/*.log`, `logs/riscv/*.log`, `logs/cpu/*.json`, `logs/riscv/*.json`",
            "- ir: `ir/cpu/`, `ir/riscv/`",
            "- cache: `cache/cpu/`, `cache/riscv/`",
            "- meta: `meta/commits.txt`, `meta/python.txt`, `meta/env.txt`, `meta/sizes.txt`",
        ]
    )
    record_text(out_dir / "RESULTS.md", "\n".join(lines) + "\n")
    record_text(out_dir / "plot_data.json", json.dumps(plot_rows, indent=2) + "\n")


parser = argparse.ArgumentParser(description="Run SG2044 five-op benchmark with tensor-first RISC-V lowering.")
parser.add_argument(
    "--output-root",
    default="analysis/sg2044-5ops-seconds-tensorfirst",
    help="Directory under the repo where timestamped results are written.",
)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--repeats", type=int, default=20)
parser.add_argument("--timeout", type=int, default=1800, help="Per-subprocess timeout in seconds.")
args = parser.parse_args()

repo_root = Path(__file__).resolve().parents[1]
output_root = Path(args.output_root)
if output_root.is_absolute() or ".." in output_root.parts:
    raise SystemExit("--output-root must be a relative path under the repo root")
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out_dir = repo_root / output_root / timestamp
out_dir.mkdir(parents=True, exist_ok=False)

riscv_repo = RepoSpec(
    name="riscv",
    root=repo_root,
    python=repo_root / ".venv" / "bin" / "python",
    module_root="python/examples",
)
cpu_repo = RepoSpec(
    name="cpu",
    root=Path("/home/zhouxulin/intern/triton-cpu"),
    python=Path("/home/zhouxulin/intern/triton-cpu/.venv/bin/python"),
    module_root=".",
)

record_text(
    out_dir / "meta" / "commits.txt",
    "\n".join(
        [
            f"REMOTE_CPU_COMMIT={git_rev_parse(cpu_repo.root)}",
            f"REMOTE_PLUGIN_COMMIT={git_rev_parse(riscv_repo.root)}",
        ]
    )
    + "\n",
)
record_text(
    out_dir / "meta" / "python.txt",
    "\n".join(
        [
            f"REMOTE_CPU_PYTHON={cpu_repo.python}",
            f"REMOTE_RISCV_PYTHON={riscv_repo.python}",
        ]
    )
    + "\n",
)
record_text(
    out_dir / "meta" / "env.txt",
    "\n".join(
        [
            "THREAD_ENV=" + " ".join(f"{k}={v}" for k, v in THREAD_ENV.items()),
            f"WARMUP={args.warmup}",
            f"REPEATS={args.repeats}",
            "TRITON_RISCV_STRUCTURED_LDST_MODE=tensor_first_vector_cpu",
            f"TRITON_SHARED_OPT_PATH={os.environ.get('TRITON_SHARED_OPT_PATH', '')}",
            f"LLVM_BINARY_DIR={os.environ.get('LLVM_BINARY_DIR', '')}",
            f"BUDDY_MLIR_BINARY_DIR={os.environ.get('BUDDY_MLIR_BINARY_DIR', '')}",
        ]
    )
    + "\n",
)
record_text(
    out_dir / "meta" / "sizes.txt",
    "\n".join(f"{op.name}|{op.input_desc}" for op in OPS) + "\n",
)

cpu_rows = {}
riscv_rows = {}
for op in OPS:
    cpu_rows[op.name] = run_one(cpu_repo, out_dir, op, {})
    riscv_rows[op.name] = run_one(riscv_repo, out_dir, op, RISCV_EXTRA_ENV)

record_text(out_dir / "logs" / "cpu" / "results.json", json.dumps(cpu_rows, indent=2) + "\n")
record_text(out_dir / "logs" / "riscv" / "results.json", json.dumps(riscv_rows, indent=2) + "\n")
write_results(out_dir, cpu_rows, riscv_rows)

print(out_dir)
