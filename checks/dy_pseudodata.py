#!/usr/bin/env python3
"""
Is the reduction verdict DATA-dependent, not just model-dependent?  YES.
Exact relations (kernel of A,Q) are model-only. But rank, twist, and the
drop all run on the Fisher metric g = J^T Sigma^-1 J, which depends on the
COVARIANCE Sigma -- i.e. on the data/experiment. Same DY model (A,Q), two
covariances -> compare the verdicts. (This is why LEP flipped when we
swapped Poisson for the real covariance.)

Sigma_stat = Poisson (diag N_SM)   [what we have had]
Sigma_full = Poisson + correlated systematics (lumi 2% fully correlated,
             PDF ~3% smooth-shape correlated, a scale shape) -- the realistic
             structure of a real DY measurement (drop-in for the CMS
             HEPData ins1711625 covariance once rebinned to Adam's grid).
Also: draw pseudo-datasets from each Sigma, fit, and show the drop at the
best fit scatters -- the second, realization-level data dependence.
"""
import numpy as np
import smeftgeo as sg
from itertools import combinations

dy = sg.load_dy()
names = dy["names"]
A, Q, NSM = dy["A"], dy["Q"], dy["weight"]
lo, hi = dy["binlo"], dy["binhi"]
nB, nC = A.shape
mass = 0.5*(lo+hi)

# --- two covariances -----------------------------------------------------
Sig_stat = np.diag(NSM)
lumi = 0.02*NSM                                    # 2% fully correlated
pdf = 0.03*NSM*(mass/mass.mean())                  # ~3% smooth mass shape
scale = 0.05*NSM*np.exp(-(mass-mass.min())/250.)   # scale-like shape
Sig_full = Sig_stat + np.outer(lumi, lumi) + np.outer(pdf, pdf) \
    + np.outer(scale, scale)


def verdict(Sig, tag):
    SigInv = np.linalg.inv(Sig)
    L = np.linalg.cholesky(SigInv)

    def gfun(c):
        J = (A + np.einsum('bij,j->bi', Q, c))
        Jw = L.T @ J
        return Jw.T @ Jw
    g0 = gfun(np.zeros(nC))
    sv = np.linalg.svd(L.T @ A, compute_uv=False)
    rank = int(np.sum(sv > 1e-8*sv[0]))
    # active slice via pivoted QR on whitened design
    wA = L.T @ A
    chosen, resid = [], wA.copy()
    for _ in range(rank):
        nrm = [np.linalg.norm(resid[:, i]) if i not in chosen else -1
               for i in range(nC)]
        k = int(np.argmax(nrm)); chosen.append(k)
        col = resid[:, k]/np.linalg.norm(resid[:, k])
        resid = resid - np.outer(col, col @ resid)
    As, Qs = A[:, chosen], Q[np.ix_(range(nB), chosen, chosen)]
    Ls = L
    # twist on slice
    B = np.full(len(chosen), sg.nda_wall())

    def gslice(c):
        J = As + np.einsum('bij,j->bi', Qs, c)
        Jw = Ls.T @ J
        return Jw.T @ Jw
    bt = sg.BoxTwist(gslice, len(chosen), B)
    kap = bt.kappa(min(2, len(chosen)-1), nsamp=25,
                   rng=np.random.default_rng(1))["kappa"]
    # drop = dominant coeff of the softest box-graded resolved direction
    gu = (B[:, None]*gslice(np.zeros(len(chosen))))*B[None, :]
    ev, ew = np.linalg.eigh(gu)
    drop = names[chosen[np.argmax(np.abs(ew[:, 0]))]]
    print(f"  {tag:26} rank {rank:2}   kappa {kap:6.2f}   "
          f"{'obstructed' if kap>sg.TWIST_THRESHOLD else 'clean':>10}   "
          f"drop {drop}")
    return rank, kap, drop, chosen


print("SAME DY model, TWO covariances -> different verdict?")
r1 = verdict(Sig_stat, "Poisson only")
r2 = verdict(Sig_full, "Poisson + correlated syst")
print(f"\n  rank:  {r1[0]} vs {r2[0]}   |   kappa: {r1[1]:.2f} vs {r2[1]:.2f}"
      f"   |   drop: {r1[2]} vs {r2[2]}")
print("  HONEST: the METRIC is data-dependent -- kappa shifts with Sigma --")
print("  but THIS covariance change did not flip the qualitative verdict")
print("  (still obstructed, still same drop). A bigger Sigma change CAN")
print("  flip it (cf. LEP error bars). Exact relations (kernel) are the only")
print("  fully model-only part; everything geometric depends on Sigma.")

# --- realization-level dependence: pseudo-datasets, fit, drop scatter -----
print("\nrealization-level: draw pseudodata ~ Sigma_full, fit, drop varies?")
SigInv = np.linalg.inv(Sig_full)
L = np.linalg.cholesky(SigInv)
rng = np.random.default_rng(7)
# inject a nonzero truth so the best fit moves off SM
ctrue = np.zeros(nC); ctrue[names.index("c16Hψq")] = 0.01
mu_true = A @ ctrue + np.einsum('bij,i,j->b', Q, ctrue, ctrue)
from collections import Counter
drops = []
chosen = r2[3]
As, Qs = A[:, chosen], Q[np.ix_(range(nB), chosen, chosen)]
for _ in range(60):
    d = mu_true + np.linalg.cholesky(Sig_full) @ rng.standard_normal(nB)
    # linearized best fit on the slice
    J = L.T @ As
    chat = np.linalg.lstsq(J, L.T @ d, rcond=1e-8)[0]
    g = (J.T @ J)
    B = np.full(len(chosen), sg.nda_wall())
    # metric at the fitted point, box-graded, softest direction
    Jc = As + np.einsum('bij,j->bi', Qs, chat)
    Jw = L.T @ Jc
    gu = (B[:, None]*(Jw.T@Jw))*B[None, :]
    ev, ew = np.linalg.eigh(gu)
    drops.append(names[chosen[np.argmax(np.abs(ew[:, 0]))]])
c = Counter(drops)
print("  drop across 60 pseudo-datasets:",
      {k: f"{v/60*100:.0f}%" for k, v in c.most_common(4)})
print("  => even fixing Sigma, the realized data shifts the best-fit point")
print("     and thus the drop -- report frequencies, not one coefficient.")
