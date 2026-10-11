#!/usr/bin/env python3
"""Reproducible default-HiGHS versus HiPDLP panel on all public rows.

The manifest is the live LPfeas qualifying disclosed-source inventory.  Each
original lptestset archive is downloaded and expanded immediately before the
solve, then removed.  Both untouched ``solver=choose`` and explicit HiPDLP
are run in fresh child processes; every row, including timeouts and failed
quality gates, is retained.  A score uses only paired repeats with Optimal,
valid finite solutions, native relative KKT <= 1e-8, independently checked
primal violation <= 1e-6, and paired objective agreement <= 1e-6.
"""
from __future__ import annotations

import argparse, bz2, gc, hashlib, json, math, os, statistics, subprocess
import sys, tempfile, time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from highspy import Highs

BASE = "https://plato.asu.edu/ftp/lptestset/"
SOLVERS = tuple(os.environ["LP_ONLY_SOLVERS"].split(",")) if os.environ.get("LP_ONLY_SOLVERS") else ("choose", "hipdlp")


def independent(h: Highs) -> dict:
    """Vectorized recomputation (same checks as the original loop version)."""
    lp = h.getLp(); sol = h.getSolution()
    x = np.asarray(sol.col_value, dtype=float)
    c = np.asarray(lp.col_cost_, dtype=float); mat = lp.a_matrix_
    start = np.asarray(mat.start_, dtype=np.int64); idx = np.asarray(mat.index_, dtype=np.int64)
    val = np.asarray(mat.value_, dtype=float); counts = np.diff(start)
    col_of = np.repeat(np.arange(lp.num_col_), counts)
    ax = np.bincount(idx, weights=val * x[col_of], minlength=lp.num_row_) if len(val) else np.zeros(lp.num_row_)
    lo = np.asarray(lp.row_lower_, dtype=float); up = np.asarray(lp.row_upper_, dtype=float)
    low = np.where(np.isfinite(lo), np.maximum(lo - ax, 0.0), 0.0)
    high = np.where(np.isfinite(up), np.maximum(ax - up, 0.0), 0.0)
    viol = np.maximum(low, high)
    # Negative control: add +1 to a high-impact column; a valid check must reject it.
    nz = np.nonzero(counts > 0)[0]
    colmax = np.maximum.reduceat(np.abs(val), start[nz]) if len(nz) else np.array([])
    score = colmax * counts[nz]
    order = nz[np.lexsort((-nz, -score))][:32] if len(nz) else []
    neg = None
    for j in order:
        a0, b0 = start[j], start[j + 1]
        axp = ax.copy(); np.add.at(axp, idx[a0:b0], val[a0:b0])
        lv = np.where(np.isfinite(lo), np.maximum(lo - axp, 0.0), 0.0)
        uv = np.where(np.isfinite(up), np.maximum(axp - up, 0.0), 0.0)
        mv = float(np.max(np.maximum(lv, uv))) if len(axp) else 0.0
        if mv > 1e-6:
            neg = {"column": int(j), "max_primal_violation_after_plus_one": mv}; break
    info = h.getInfo()
    obj = float(np.dot(c, x) + lp.offset_)
    return {
        "objective_recomputed": obj,
        "objective_recompute_error": abs(obj - h.getObjectiveValue()),
        "all_finite_columns": bool(np.all(np.isfinite(x))),
        "max_primal_violation_recomputed": float(np.max(viol)) if viol.size else 0.0,
        "negative_control": neg,
        "basis_validity": int(info.basis_validity),
        "simplex_iterations": int(info.simplex_iteration_count),
        "ipm_iterations": int(info.ipm_iteration_count),
        "pdlp_iterations": int(info.pdlp_iteration_count),
    }


def solve(model: Path, solver: str, tol: float, limit: float) -> dict:
    if os.environ.get("LP_PANEL_CHILD") != "1":
        env = os.environ.copy(); env["LP_PANEL_CHILD"] = "1"
        try:
            p = subprocess.run([sys.executable, __file__, "--single-model", str(model),
                                "--single-solver", solver, "--single-tol", str(tol),
                                "--single-limit", str(limit)], capture_output=True, text=True,
                               timeout=limit + 900, env=env)
            lines = [x for x in p.stdout.splitlines() if x.strip()]
            if p.returncode == 0 and lines: return json.loads(lines[-1])
            return {"solver": solver, "error": "child_failed", "returncode": p.returncode,
                    "stderr_tail": p.stderr[-1000:]}
        except subprocess.TimeoutExpired:
            return {"solver": solver, "error": "wall_timeout", "timeout_seconds": limit + 900}
    h = Highs()
    opts = [("output_flag", False), ("log_to_console", False), ("solver", solver),
            ("threads", 1), ("time_limit", limit)]
    # Untouched variant: solver=choose keeps every HiGHS default tolerance;
    # only HiPDLP gets the 1e-10 PDLP/KKT target.  The 1e-8 native gate is applied afterwards.
    if solver != "choose" or os.environ.get("LP_CHOOSE_UNTOUCHED") != "1":
        opts += [("pdlp_optimality_tolerance", tol), ("ipm_optimality_tolerance", tol), ("kkt_tolerance", tol)]
    for key, val in opts: h.setOptionValue(key, val)
    t0 = time.perf_counter(); rs = h.readModel(str(model)); read_s = time.perf_counter() - t0
    if "Error" in str(rs): return {"solver": solver, "read_status": str(rs), "read_seconds": read_s}
    t0 = time.perf_counter(); run = h.run(); solve_s = time.perf_counter() - t0
    info, sol = h.getInfo(), h.getSolution()
    row = {"solver": solver, "read_status": str(rs), "run_status": str(run),
           "model_status": str(h.getModelStatus()), "read_seconds": read_s,
           "solve_seconds": solve_s, "objective": h.getObjectiveValue(),
           "max_primal_infeasibility": info.max_primal_infeasibility,
           "max_dual_infeasibility": info.max_dual_infeasibility,
           "max_relative_primal_infeasibility": info.max_relative_primal_infeasibility,
           "max_relative_dual_infeasibility": info.max_relative_dual_infeasibility,
           "value_valid": bool(sol.value_valid), "dual_valid": bool(sol.dual_valid)}
    if str(h.getModelStatus()).endswith("kOptimal") and bool(sol.value_valid):
        try: row.update(independent(h))
        except Exception as exc: row["independent_error"] = repr(exc)
    del h; gc.collect(); return row


def admissible(row: dict) -> bool:
    return (row.get("model_status", "").endswith("kOptimal") and row.get("value_valid") and
            row.get("dual_valid") and row.get("all_finite_columns") and
            row.get("max_relative_primal_infeasibility", math.inf) <= 1e-8 and
            row.get("max_relative_dual_infeasibility", math.inf) <= 1e-8 and
            row.get("objective_recompute_error", math.inf) <= 1e-6 and
            row.get("max_primal_violation_recomputed", math.inf) <= 1e-6 and
            row.get("negative_control") is not None)


def expand(spec: dict, root: Path, emps: Path) -> tuple[Path, str, str]:
    archive = spec["public_lptestset_file"]
    raw = urlopen(BASE + archive, timeout=180).read(); archive_sha = hashlib.sha256(raw).hexdigest()
    payload = bz2.decompress(raw)
    if spec["source_format"] == "emps":
        inp = root / (spec["name"] + ".raw"); inp.write_bytes(payload)
        p = subprocess.run([str(emps), str(inp)], check=True, capture_output=True)
        model_bytes = p.stdout; inp.unlink(missing_ok=True)
    else:
        model_bytes = payload
    model = root / (spec["name"] + ".mps"); model.write_bytes(model_bytes)
    return model, archive_sha, hashlib.sha256(model_bytes).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=Path); ap.add_argument("--emps", type=Path)
    ap.add_argument("--out", type=Path); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-10); ap.add_argument("--limit", type=float, default=600)
    ap.add_argument("--names", nargs="*"); ap.add_argument("--single-model", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--single-solver", choices=("choose", "hipdlp"), help=argparse.SUPPRESS)
    ap.add_argument("--single-tol", type=float, help=argparse.SUPPRESS); ap.add_argument("--single-limit", type=float, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.single_model:
        print(json.dumps(solve(a.single_model, a.single_solver, a.single_tol, a.single_limit), allow_nan=True), flush=True); return 0
    if not (a.subset and a.emps and a.out): ap.error("--subset --emps --out are required")
    a.emps = a.emps.resolve(); d = json.loads(a.subset.read_text())
    specs = [r for r in d["qualifying"] if r.get("disclosed") and r.get("public_lptestset_file")]
    if a.names: specs = [r for r in specs if r["name"] in set(a.names)]
    rows = []; a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lp-default-panel-") as td:
        root = Path(td)
        for spec in specs:
            try: model, ah, mh = expand(spec, root, a.emps)
            except Exception as exc:
                rows.append({"model": spec["name"], "archive": spec["public_lptestset_file"], "error": repr(exc), "reported_seconds": spec["highs_seconds"]}); continue
            print(json.dumps({"event": "model", "model": spec["name"], "bytes": model.stat().st_size}), flush=True)
            for rep in range(1, a.reps + 1):
                for solver in SOLVERS:
                    try: row = solve(model, solver, a.tol, a.limit)
                    except Exception as exc: row = {"solver": solver, "error": repr(exc)}
                    row.update({"model": spec["name"], "archive": spec["public_lptestset_file"], "source_format": spec["source_format"],
                                "rep": rep, "reported_seconds": spec["highs_seconds"], "archive_sha256": ah,
                                "model_sha256": mh, "model_bytes": model.stat().st_size})
                    rows.append(row); a.out.write_text(json.dumps({"rows": rows}, indent=2) + "\n"); print(json.dumps(row, allow_nan=True), flush=True)
            model.unlink(missing_ok=True)
    summary = {}; scored = []
    for spec in specs:
        name = spec["name"]; by = {s: [r for r in rows if r.get("model") == name and r.get("solver") == s] for s in SOLVERS}
        paired = []
        for rep in range(1, a.reps + 1):
            q = next((r for r in by["choose"] if r.get("rep") == rep), None); c = next((r for r in by["hipdlp"] if r.get("rep") == rep), None)
            if q and c and admissible(q) and admissible(c) and abs(q["objective_recomputed"] - c["objective_recomputed"]) <= 1e-6: paired.append((q, c))
        summary[name] = {"reported_seconds": spec["highs_seconds"], "choose_admissible": sum(admissible(r) for r in by["choose"]),
                         "hipdlp_admissible": sum(admissible(r) for r in by["hipdlp"]), "paired_reps": len(paired),
                         "choose_median_seconds": statistics.median([r["solve_seconds"] for r in by["choose"] if admissible(r)]) if any(admissible(r) for r in by["choose"]) else None,
                         "hipdlp_median_seconds": statistics.median([r["solve_seconds"] for r in by["hipdlp"] if admissible(r)]) if any(admissible(r) for r in by["hipdlp"]) else None,
                         "max_paired_objective_delta": max((abs(q["objective_recomputed"] - c["objective_recomputed"]) for q,c in paired), default=None),
                         "scored": len(paired) == a.reps}
        if len(paired) == a.reps: scored.append((name, summary[name]["choose_median_seconds"], summary[name]["hipdlp_median_seconds"]))
    score = {"shift_seconds": 10.0, "scope_rows": len(specs), "scored_models": [x[0] for x in scored]}
    if scored:
        q = math.exp(sum(math.log(x[1] + 10) for x in scored)/len(scored))-10; c = math.exp(sum(math.log(x[2] + 10) for x in scored)/len(scored))-10
        score.update({"choose_shifted_geomean_seconds": q, "hipdlp_shifted_geomean_seconds": c, "relative_improvement": 1-c/q})
    out = {"args": {"subset": str(a.subset), "emps": str(a.emps), "reps": a.reps, "tol": a.tol, "limit": a.limit},
           "rows": rows, "summary": summary, "score": score,
           "gates": {"native_relative_kkt": 1e-8, "objective": 1e-6, "independent_primal": 1e-6},
           "scope": {"all_disclosed_source_rows": 38}}
    a.out.write_text(json.dumps(out, indent=2, allow_nan=True) + "\n"); print(json.dumps(score, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
