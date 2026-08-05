#!/usr/bin/env python3
"""
T3 for the paper: AIC verdict vs worst-case price, on the SAME reduction.

Reduction under test: the 7-coefficient drop set = the greedy set PLUS
c36HpsiL. c36HpsiL costs only 0.72 dropped alone, but the joint set costs
~50 -- the drops interact. Question: does AIC catch that?

AIC accepts the reduced model when chi2_red - chi2_full < 2*(n dropped) = 14.
We generate pseudodata with the truth at the origin (SM) and at the worst box
point for this set, fit both models properly (multi-seed Gauss-Newton; the
full 19-parameter fit is degenerate, so the winning reduced solution is also
used as a full-fit seed, which enforces chi2_full <= chi2_red), and compare.
No conclusions are printed that the numbers do not compute.
"""
import json
import numpy as np
import battery_lib as bl
import smeftgeo as sg

wall = sg.nda_wall(5.0)
NREP = 200

# ---- realcov-rebinned DY (same construction as dy_reduce_realcov) ---------
dy = sg.load_dy()
names = dy["names"]
ctr = 0.5*(dy["binlo"] + dy["binhi"])
meas = json.load(open("/tmp/meas.json"))
rows = meas["values"]
lo = np.array([float(r["x"][0]["low"]) for r in rows])
hi = np.array([float(r["x"][0]["high"]) for r in rows])
xs = np.array([float(r["y"][0]["value"]) for r in rows])
cms_cov = np.load("/tmp/cms_cov.npy")
win = np.where((lo >= 90) & (hi <= 610))[0]
Cw = cms_cov[np.ix_(win, win)]
dw = np.sqrt(np.diag(Cw))
relunc = dw/np.abs(xs[win])
R = Cw/np.outer(dw, dw)
A = np.zeros((len(win), 19)); Q = np.zeros((len(win), 19, 19))
SM = np.zeros(len(win)); nmap = np.zeros(len(win), int)
for b in range(len(ctr)):
    k = np.where((ctr[b] >= lo[win]) & (ctr[b] < hi[win]))[0]
    if len(k):
        k = k[0]
        A[k] += dy["A"][b]; Q[k] += dy["Q"][b]; SM[k] += dy["NSM"][b]
        nmap[k] += 1
keepb = nmap > 0
A, Q, SM = A[keepb], Q[keepb], SM[keepb]
relunc, R = relunc[keepb], R[np.ix_(keepb, keepb)]
Aw, Qw = bl.prep(A/SM[:, None], Q/SM[:, None, None],
                 cov=R*np.outer(relunc, relunc))
nO = Aw.shape[0]
box = np.full(19, wall)

DROP = ["cHB", "cHW", "cHd6", "cHu6", "cLL", "c16Hψq", "c36HψL"]
dropset = [names.index(x) for x in DROP]
keepc = [j for j in range(19) if j not in dropset]
npen = 2*len(dropset)


def gn_fit(Y, keep_idx, seeds, iters=25):
    """multi-seed MONOTONE Gauss-Newton fit with only keep_idx free (others 0).
    A step is only accepted if it lowers chi2 (backtracking 1, 1/2, 1/4);
    otherwise that replica keeps its current point. Guarantees the returned
    chi2 is never worse than the best seed. Returns per-replica best chi2 and
    the winning parameters."""
    cb = box[keep_idx]

    def chi_of(th):
        c = np.zeros((len(Y), 19)); c[:, keep_idx] = th
        return ((bl.mu(Aw, Qw, c) - Y)**2).sum(1)

    best_chi = np.full(len(Y), np.inf)
    best_th = np.zeros((len(Y), len(keep_idx)))
    for s in seeds:
        th = np.clip(s.copy(), -cb, cb)
        chi = chi_of(th)
        for _ in range(iters):
            c = np.zeros((len(Y), 19)); c[:, keep_idx] = th
            r = bl.mu(Aw, Qw, c) - Y
            Jc = Aw[None] + np.einsum('oij,nj->noi', Qw, c)
            Jt = Jc[:, :, keep_idx]
            JTJ = np.einsum('noa,nob->nab', Jt, Jt)
            lam = 1e-8*np.einsum('naa->n', JTJ)/len(keep_idx) + 1e-12
            JTJ = JTJ + lam[:, None, None]*np.eye(len(keep_idx))
            JTr = np.einsum('noa,no->na', Jt, r)
            step = np.linalg.solve(JTJ, JTr[..., None])[..., 0]
            improved = np.zeros(len(Y), bool)
            for alpha in (1.0, 0.5, 0.25):
                trial = np.clip(th - alpha*step, -cb, cb)
                chit = chi_of(trial)
                take = (~improved) & (chit < chi - 1e-12)
                th[take] = trial[take]; chi[take] = chit[take]
                improved |= take
        upd = chi < best_chi
        best_chi[upd] = chi[upd]; best_th[upd] = th[upd]
    return best_chi, best_th


# ---- worst box point for this drop set + its price ------------------------
B = bl._boxsamples(box, 800, seed=123)
d2 = bl._proj_chi2(Aw, Qw, bl.mu(Aw, Qw, B), B[:, keepc], keepc, None, -1,
                   clampbox=box[keepc])
einf = float(d2.max())
worst_pt = B[int(d2.argmax())]
print(f"drop set ({len(DROP)}): {DROP}")
print(f"worst-case price of the set (E_inf) = {einf:.1f}"
      f"  -> E_inf verdict: {'REJECT' if einf > 4 else 'ACCEPT'} the reduction"
      f" (threshold 4 = 2 sigma), independent of where the truth sits\n")

rng = np.random.default_rng(5)
print(f"{'truth at':<20}{'<chi2_red - chi2_full>':>24}{'AIC penalty':>13}"
      f"{'AIC verdict':>20}")
for tag, truth in [("origin (SM)", np.zeros(19)),
                   ("worst box point", worst_pt)]:
    Y = bl.mu(Aw, Qw, truth[None, :]) + rng.standard_normal((NREP, nO))
    tile = lambda v: np.tile(v, (NREP, 1))
    chi_red, th_red = gn_fit(Y, keepc, [tile(truth[keepc]),
                                        tile(np.zeros(len(keepc)))])
    # full fit seeded from truth, origin, AND the reduced winner (embedded);
    # the last seed enforces chi_full <= chi_red up to GN convergence
    emb = np.zeros((NREP, 19)); emb[:, keepc] = th_red
    chi_full, _ = gn_fit(Y, list(range(19)),
                         [tile(truth), tile(np.zeros(19)), emb])
    nested_ok = np.mean(chi_full <= chi_red + 1e-6)
    dchi = float(np.mean(chi_red - chi_full))
    verdict = "ACCEPT" if dchi < npen else "REJECT"
    print(f"{tag:<20}{dchi:>24.1f}{npen:>13}{verdict:>20}"
          f"   (nested check: {nested_ok*100:.0f}% ok)")
