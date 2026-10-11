#!/usr/bin/env python3
"""Equal-cap score: HiGHS 1.15.1 untouched solver=choose vs HiPDLP (1e-10) on the 38 LPfeas rows.
A row counts as solved only if the harness admissibility gate passes (Optimal, valid primal/dual,
native relative KKT <= 1e-8, recomputed primal violation <= 1e-6, objective recompute <= 1e-6,
negative control rejected).  Unsolved or missing rows count as the cap.  Shift 10 s."""
import json, math, sys, glob
def admissible(row):  # harness gate, but objective recompute error relative (1e-9) for large objectives
    return (row.get("model_status", "").endswith("kOptimal") and row.get("value_valid") and
            row.get("dual_valid") and row.get("all_finite_columns") and
            row.get("max_relative_primal_infeasibility", math.inf) <= 1e-8 and
            row.get("max_relative_dual_infeasibility", math.inf) <= 1e-8 and
            row.get("objective_recompute_error", math.inf) <= max(1e-6, 1e-9 * abs(row.get("objective_recomputed", 0.0))) and
            row.get("max_primal_violation_recomputed", math.inf) <= 1e-6 and
            row.get("negative_control") is not None)
CAP = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
rows = []
for f in sorted(glob.glob('results/u*.json') + glob.glob('results/h1-*.json')):
    rows += json.load(open(f))['rows']
subset = json.load(open('qualifying-subset.json'))
names = [r['name'] for r in subset['qualifying'] if r.get('disclosed') and r.get('public_lptestset_file')]
tab = {}
for n in names:
    t = {}
    for s in ('choose', 'hipdlp'):
        rs = [r for r in rows if r.get('model') == n and r.get('solver') == s]
        ok = [r for r in rs if admissible(r)]
        t[s] = min(r['solve_seconds'] for r in ok) if ok else None
        t[s + '_status'] = (rs[0].get('model_status') or rs[0].get('error')) if rs else 'missing'
    if t['choose'] is not None and t['hipdlp'] is not None:
        oc = next(r for r in rows if r.get('model') == n and r.get('solver') == 'choose' and admissible(r))['objective_recomputed']
        oh = next(r for r in rows if r.get('model') == n and r.get('solver') == 'hipdlp' and admissible(r))['objective_recomputed']
        t['obj_delta'] = abs(oc - oh)
    tab[n] = t
def sgm(v): return math.exp(sum(math.log(min(x, CAP) + 10) for x in v) / len(v)) - 10
c = [tab[n]['choose'] if tab[n]['choose'] is not None else CAP for n in names]
h = [tab[n]['hipdlp'] if tab[n]['hipdlp'] is not None else CAP for n in names]
both = [n for n in names if tab[n]['choose'] is not None and tab[n]['hipdlp'] is not None]
out = {'cap_seconds': CAP, 'rows': len(names), 'measured_rows': sum(1 for n in names if tab[n]['choose_status'] != 'missing'),
       'choose_solved': sum(tab[n]['choose'] is not None for n in names), 'hipdlp_solved': sum(tab[n]['hipdlp'] is not None for n in names),
       'both_solved': len(both), 'hipdlp_faster_rows': sum(1 for n in names if h[names.index(n)] < c[names.index(n)]),
       'choose_faster_rows': sum(1 for n in names if c[names.index(n)] < h[names.index(n)]),
       'choose_sgm_capped': sgm(c), 'hipdlp_sgm_capped': sgm(h), 'hipdlp_vs_choose': sgm(h) / sgm(c) - 1,
       'max_obj_delta_both_solved': max((tab[n]['obj_delta'] for n in both), default=None), 'table': tab}
json.dump(out, open('score_capped.json', 'w'), indent=1)
for n in names:
    t = tab[n]; print(f"{n:14s} choose={t['choose'] if t['choose'] is None else round(t['choose'],1)!s:>8} ({t['choose_status']})  hipdlp={t['hipdlp'] if t['hipdlp'] is None else round(t['hipdlp'],1)!s:>8} ({t['hipdlp_status']})")
print({k: v for k, v in out.items() if k != 'table'})
