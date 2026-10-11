# HiGHS 1.15.1: HiPDLP vs default `solver=choose` on the 38 public LPfeas rows (single thread)

Same-host comparison of HiGHS 1.15.1 (PyPI `highspy==1.15.1`) with **untouched defaults**
(`solver=choose`, default tolerances) against **explicit `solver=hipdlp`** with
`pdlp_optimality_tolerance = kkt_tolerance = 1e-10`, `threads=1`, on the 38 rows of Hans
Mittelmann's LPfeas benchmark that have a public source file in the lptestset tree.
4 vCPU / 8 GB Daytona sandboxes, one solve at a time, each in a fresh child process.

A solve counts only if it passes the same gate for both solvers: Optimal, valid primal and dual
values, native max relative primal and dual infeasibility <= 1e-8, primal violation recomputed from
the original MPS <= 1e-6, objective recomputed to within max(1e-6, 1e-9|obj|), and a +1 perturbation
negative control that the checker must reject (`src/default_vs_hipdlp_untouched.py`,
`score_capped.py`).

## Protocol

1. All 38 rows, both solvers, 600 s cap (`results/u1..u6.json`).
2. Rows where exactly one solver passed within 600 s were re-run for the other solver with a
   3,600 s cap (`results/h1-*.json`).
3. The 18 rows that neither solver passed within 600 s were not extended; they count as 3,600 s for
   both solvers.

## Result

| | default `choose` | HiPDLP |
|---|---:|---:|
| rows passed within 3,600 s | 16 | 13 |
| rows faster than the other solver | 10 | 10 |
| shifted geomean (10 s shift), all 38 rows | 1,055.4 s | 928.4 s (**-12.0%**) |
| shifted geomean, the 20 rows either solver passed | 345.3 s | 269.1 s (-22.0%) |

Per-row times (s) for the 20 rows at least one solver passed (— = did not pass within 3,600 s):

| row | default | HiPDLP |
|---|---:|---:|
| L1_sixm | 2,647.8 | 316.8 |
| a2864 | — | 27.2 |
| cont1 | 150.9 | 2,467.0 |
| datt256 | — | 4.0 |
| ex10 | 69.0 | 5.1 |
| fome13 | 69.4 | 85.3 |
| graph40-40 | — | 4.1 |
| irish-e | 135.4 | — |
| neos | 261.5 | — |
| neos5052403 | 565.7 | 275.2 |
| nug08-3rd | 219.4 | 0.4 |
| pds-100 | 49.3 | — |
| qap15 | 812.7 | 13.0 |
| rmine15 | 485.9 | — |
| s250r10 | 73.0 | — |
| savsched1 | 492.0 | — |
| square41 | 51.0 | — |
| storm_1000 | 39.4 | 1,042.1 |
| support10 | 132.1 | 51.1 |
| woodlands09 | — | 17.2 |

HiPDLP passes four rows that default does not finish in an hour (a2864, datt256, graph40-40,
woodlands09) and is much faster on nug08-3rd (0.4 vs 219 s), qap15 (13 vs 813 s) and L1_sixm
(317 vs 2,648 s). Default passes seven rows HiPDLP does not (irish-e, neos, pds-100, rmine15,
s250r10, savsched1, square41; on pds-100 and rmine15 HiPDLP reports Optimal but its recomputed
primal violation is 1.5e-4 and 1.4e-6). Where both pass, objectives agree to within 2.3e-10 relative (largest gap: support10).
The two solvers are complementary rather than one dominating.

Scope: single thread, one repetition, one host type; HiPDLP returns no basis, so it is not a drop-in
replacement where a basis is needed. Mittelmann's own LPfeas runs use different hardware and
settings, so absolute times are not comparable with that table.
