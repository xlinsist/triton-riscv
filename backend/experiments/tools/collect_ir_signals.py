#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
import subprocess
from pathlib import Path


def _count_re(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


def _safe_read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def _replay_vir_counts(buddy_opt: str, ttsharedir: Path, out_dir: Path) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    l2v = out_dir / "after_lower_linalg_to_vir.mlir"
    v2vec = out_dir / "after_lower_vir_to_vector.mlir"

    vir_count = -1
    vec_count = -1

    try:
        subprocess.check_call(
            [buddy_opt, str(ttsharedir), "-lower-linalg-to-vir", "-o", str(l2v)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        vir_count = _count_re(r"\\bvir\\.", _safe_read(l2v))
    except Exception:
        vir_count = -1

    try:
        subprocess.check_call(
            [
                buddy_opt,
                str(ttsharedir),
                "-lower-linalg-to-vir",
                '-lower-vir-to-vector=vector-width=4',
                "-o",
                str(v2vec),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        vec_count = _count_re(r"\\bvector\\.", _safe_read(v2vec))
    except Exception:
        vec_count = -1

    return vir_count, vec_count


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect IR evidence from A/B run artifacts")
    p.add_argument("run_dir", help="Path like backend/experiments/ab-vir-vs-loops/<run_id>")
    p.add_argument(
        "--buddy-opt",
        default="",
        help="Override buddy-opt path. Default uses $BUDDY_MLIR_BINARY_DIR/buddy-opt",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    artifacts = run_dir / "artifacts"
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    if args.buddy_opt:
        buddy_opt = args.buddy_opt
    else:
        import os

        buddy_bin = os.environ.get("BUDDY_MLIR_BINARY_DIR", "")
        if not buddy_bin:
            raise SystemExit("Need --buddy-opt or BUDDY_MLIR_BINARY_DIR in env")
        buddy_opt = str(Path(buddy_bin) / "buddy-opt")

    rows = []
    for ttshared in sorted(artifacts.rglob("*.ttsharedir")):
        run_leaf = ttshared.parent
        rel = run_leaf.relative_to(artifacts)
        parts = rel.parts
        if len(parts) < 4:
            continue
        mode, workload, case_name, run_idx = parts[0], parts[1], parts[2], parts[3]

        llir = next(run_leaf.glob("*.llir"), None)
        if llir is None:
            continue

        args_file = run_leaf / "buddy-opt.args.txt"
        args_text = _safe_read(args_file)
        llir_text = _safe_read(llir)
        vir_in_args = "--lower-linalg-to-vir" in args_text if args_text else (mode == "vir_vector")
        loops_in_args = (
            "--convert-linalg-to-affine-loops" in args_text
            if args_text
            else (mode == "linalg_loops")
        )

        llvm_vec_types = _count_re(r"<\\s*\\d+\\s+x\\s+(?:float|half)\\s*>", llir_text)
        llvm_vec_fma = _count_re(r"@llvm\\.fmuladd\\.v\\d+f(?:16|32)", llir_text)
        llvm_shufflevector = _count_re(r"\\bshufflevector\\b", llir_text)

        replay_dir = reports / "ir-replay" / mode / workload / case_name / run_idx
        vir_count, vector_count = _replay_vir_counts(buddy_opt, ttshared, replay_dir)

        rows.append(
            {
                "mode": mode,
                "workload": workload,
                "case_name": case_name,
                "run_idx": run_idx,
                "vir_in_args": int(vir_in_args),
                "loops_in_args": int(loops_in_args),
                "replay_vir_count": vir_count,
                "replay_vector_count": vector_count,
                "llir_vec_types": llvm_vec_types,
                "llir_vec_fma": llvm_vec_fma,
                "llir_shufflevector": llvm_shufflevector,
                "artifact_dir": str(run_leaf),
            }
        )

    csv_path = reports / "ir_signals.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    grouped = {}
    for r in rows:
        key = (r["mode"], r["workload"])
        grouped.setdefault(key, []).append(r)

    md_path = reports / "ir-evidence.md"
    lines = ["# IR Evidence", ""]
    if not rows:
        lines.append("No rows found. Did benchmark copy artifacts correctly?")
    else:
        lines.append(f"Collected rows: {len(rows)}")
        lines.append("")
        lines.append("| mode | workload | vir_in_args(avg) | loops_in_args(avg) | replay_vir(avg) | replay_vector(avg) | llir_vec_types(avg) | llir_vec_fma(avg) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for key in sorted(grouped.keys()):
            rs = grouped[key]
            lines.append(
                "| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
                    key[0],
                    key[1],
                    statistics.fmean([x["vir_in_args"] for x in rs]),
                    statistics.fmean([x["loops_in_args"] for x in rs]),
                    statistics.fmean([x["replay_vir_count"] for x in rs]),
                    statistics.fmean([x["replay_vector_count"] for x in rs]),
                    statistics.fmean([x["llir_vec_types"] for x in rs]),
                    statistics.fmean([x["llir_vec_fma"] for x in rs]),
                )
            )

    md_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
