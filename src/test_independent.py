"""Check the vectorized independent() equals the original loop version on random LPs."""
import importlib.util, sys, numpy as np
from highspy import Highs, HighsLp, ObjSense
def load(p, n):
    spec = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old = load("src/default_vs_hipdlp_panel.py", "old"); new = load("src/default_vs_hipdlp_untouched.py", "new")
rng = np.random.default_rng(0)
for t in range(5):
    m, n = 40, 60
    A = (rng.random((m, n)) < 0.15) * rng.integers(1, 5, (m, n)).astype(float)
    h = Highs(); h.setOptionValue("output_flag", False)
    lp = HighsLp(); lp.num_col_ = n; lp.num_row_ = m
    lp.col_cost_ = rng.random(n); lp.col_lower_ = np.zeros(n); lp.col_upper_ = np.full(n, 10.0)
    lp.row_lower_ = np.full(m, 1.0); lp.row_upper_ = np.full(m, 1e30)
    starts = [0]; index = []; value = []
    for j in range(n):
        for i in range(m):
            if A[i, j]: index.append(i); value.append(A[i, j])
        starts.append(len(index))
    lp.a_matrix_.format_ = lp.a_matrix_.format_.__class__(1); lp.a_matrix_.start_ = starts; lp.a_matrix_.index_ = index; lp.a_matrix_.value_ = value
    h.passModel(lp); h.run()
    a, b = old.independent(h), new.independent(h)
    for k in a:
        if isinstance(a[k], float): assert abs(a[k] - b[k]) <= 1e-9 * max(1, abs(a[k])), (k, a[k], b[k])
        else: assert a[k] == b[k], (k, a[k], b[k])
print("independent() vectorized == loop on 5 random LPs")
