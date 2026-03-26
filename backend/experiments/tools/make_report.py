#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def _read_summary(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _read_ir(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _case_sort_key(case_name: str) -> tuple:
    nums = []
    token = ""
    for ch in case_name:
        if ch.isdigit():
            token += ch
        else:
            if token:
                nums.append(int(token))
                token = ""
    if token:
        nums.append(int(token))
    return tuple(nums) if nums else (case_name,)


def _median(values: list[float]) -> float:
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        return math.nan
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _group_ir(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped = {}
    for r in rows:
        key = (r["mode"], r["workload"])
        grouped.setdefault(key, []).append(r)
    return grouped


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build experiment markdown report")
    p.add_argument("run_dir", help="Path like backend/experiments/ab-vir-vs-loops/<run_id>")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    raw_dir = run_dir / "raw"
    reports_dir = run_dir / "reports"
    cfg_dir = run_dir / "configs"

    summary_rows = _read_summary(raw_dir / "summary.csv")
    ir_rows = _read_ir(reports_dir / "ir_signals.csv")
    ir_grouped = _group_ir(ir_rows)

    cfg = {}
    cfg_path = cfg_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())

    summary_by_key = {}
    workloads = set()
    cases_by_workload = {}
    for r in summary_rows:
        key = (r["mode"], r["workload"], r["case_name"])
        summary_by_key[key] = r
        workloads.add(r["workload"])
        cases_by_workload.setdefault(r["workload"], set()).add(r["case_name"])

    lines = []
    lines.append("# EXPER-PERFORMANCE")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Run dir: `{run_dir}`")
    if cfg:
        lines.append(
            "- Config: independent_runs={} warmup={} repeats={} threads={}".format(
                cfg.get("independent_runs", "?"),
                cfg.get("warmup", "?"),
                cfg.get("repeats", "?"),
                cfg.get("threads", "?"),
            )
        )
    lines.append("- A path: `TRITON_RISCV_LOWERING_MODE=vir_vector`")
    lines.append("- B path: `TRITON_RISCV_LOWERING_MODE=linalg_loops`")
    lines.append("")

    for workload in sorted(workloads):
        lines.append(f"## {workload}")
        lines.append("")
        lines.append("| case | vir_vector avg_of_avg (s) | linalg_loops avg_of_avg (s) | speedup (loops/vir) |")
        lines.append("|---|---:|---:|---:|")
        speedups = []
        for case in sorted(cases_by_workload.get(workload, []), key=_case_sort_key):
            va = summary_by_key.get(("vir_vector", workload, case))
            lb = summary_by_key.get(("linalg_loops", workload, case))
            if not va or not lb:
                continue
            vir = float(va["avg_of_avg_s"])
            loops = float(lb["avg_of_avg_s"])
            sp = loops / vir if vir > 0 else math.nan
            speedups.append(sp)
            lines.append(f"| {case} | {vir:.6f} | {loops:.6f} | {sp:.3f}x |")
        lines.append("")
        if speedups:
            lines.append(
                f"- Median speedup (loops/vir): `{_median(speedups):.3f}x`"
            )
        else:
            lines.append("- No complete A/B rows for this workload.")

        vir_ir = ir_grouped.get(("vir_vector", workload), [])
        loop_ir = ir_grouped.get(("linalg_loops", workload), [])
        if vir_ir or loop_ir:
            def avg(key: str, rows: list[dict]) -> float:
                if not rows:
                    return math.nan
                return sum(float(x[key]) for x in rows) / len(rows)

            lines.append("- IR evidence (avg over collected runs):")
            lines.append(
                "  - vir_vector: vir_in_args={:.2f}, loops_in_args={:.2f}, replay_vir={:.2f}, replay_vector={:.2f}, llir_vec_fma={:.2f}".format(
                    avg("vir_in_args", vir_ir),
                    avg("loops_in_args", vir_ir),
                    avg("replay_vir_count", vir_ir),
                    avg("replay_vector_count", vir_ir),
                    avg("llir_vec_fma", vir_ir),
                )
            )
            lines.append(
                "  - linalg_loops: vir_in_args={:.2f}, loops_in_args={:.2f}, replay_vir={:.2f}, replay_vector={:.2f}, llir_vec_fma={:.2f}".format(
                    avg("vir_in_args", loop_ir),
                    avg("loops_in_args", loop_ir),
                    avg("replay_vir_count", loop_ir),
                    avg("replay_vector_count", loop_ir),
                    avg("llir_vec_fma", loop_ir),
                )
            )
        lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Raw rows: `{raw_dir / 'measurements.csv'}`")
    lines.append(f"- Summary: `{raw_dir / 'summary.csv'}`")
    lines.append(f"- IR signals: `{reports_dir / 'ir_signals.csv'}`")
    lines.append(f"- IR replay: `{reports_dir / 'ir-replay'}`")
    lines.append(f"- Copied cache artifacts: `{run_dir / 'artifacts'}`")
    lines.append(f"- Per-run TRITON_HOME snapshots: `{run_dir / 'homes'}`")
    lines.append("")

    out = reports_dir / "EXPER-PERFORMANCE.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
