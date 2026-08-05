#!/usr/bin/env python3
"""
Does the method survive REALISTIC inputs?  In a FeynRules -> MadGraph ->
binned-events pipeline you do not get symbolic A, Q; you get MC-estimated
coefficients with statistical noise, and exact zeros/relations are not
certifiable.  Here we inject MC-scale noise into the (clean, symbolic) DY
A, Q and check that the GEOMETRIC verdicts are stable and that the exact
cHB-cHW relation degrades gracefully into 'blind within noise'.

Noise model (per bin b): every coefficient gets
  hat A_bi = A_bi + eps * scale_b * G ,   scale_b = rms of |A_b.| in that bin
(so structurally-zero entries acquire an O(eps) MC value -- the honest case
where you cannot certify a zero). Same for Q with its own per-bin scale.
"""
import json
import numpy as np
import smeftgeo as sg
from itertools import combinations

dy = sg.load_dy()
names = dy["names"]
A0, Q0, w = dy["A"], dy["Q"], dy["weight"]
nB, nC = A0.shape
sqw = np.sqrt(w)
iHB, iHW = names.index("cHB"), names.index("cHW")


def add_noise(eps, rng, mode="stat"):
    """mode='stat': multiplicative MC noise on the COMPUTED (nonzero)
       entries only -- structural zeros (no diagram) stay zero. Faithful to
       a MadGraph pipeline.
    mode='all': additive noise everywhere at the per-bin scale -- the
       pessimistic case where you cannot certify any zero."""
    if mode == "stat":
        An = A0 * (1 + eps * rng.standard_normal(A0.shape))
        Qn = Q0 * (1 + eps * rng.standard_normal(Q0.shape))
    else:
        Asc = np.sqrt((A0**2).mean(1))
        Qsc = np.sqrt((Q0**2).mean((1, 2)))
        An = A0 + eps * Asc[:, None] * rng.standard_normal(A0.shape)
        Qn = Q0 + eps * Qsc[:, None, None] * rng.standard_normal(Q0.shape)
    Qn = 0.5 * (Qn + Qn.transpose(0, 2, 1))
    return An, Qn


def noise_floor(A, eps):
    """singular-value scale that MC noise alone produces in the whitened
    design -- the honest rank/blind threshold."""
    Asc = np.sqrt((A0**2).mean(1))
    return eps * np.linalg.norm(Asc / sqw) / np.sqrt(nC) * np.sqrt(nB)


def kappa_slice(A, Q, m=2, nsamp=25, seed=2):
    chosen, _ = sg.active_slice(A, w)
    As, Qs = A[:, chosen], Q[np.ix_(range(nB), chosen, chosen)]
    ns = len(chosen)
    B = np.full(ns, sg.nda_wall())

    def gfun(c):
        return sg.fisher_metric(As, Qs, w, c)
    bt = sg.BoxTwist(gfun, ns, B)
    return bt.kappa(min(m, ns-1), nsamp=nsamp,
                    rng=np.random.default_rng(seed))["kappa"], chosen


from collections import Counter
u_rel = np.zeros(nC); u_rel[iHB] = 1; u_rel[iHW] = -1
u_rel /= np.linalg.norm(u_rel)

for mode in ("stat", "all"):
    print(f"\n===== noise model = '{mode}' "
          f"({'MC stat on nonzeros; structure preserved' if mode=='stat' else 'additive everywhere; no zero certifiable'}) =====")
    print(f"{'eps':>6} {'rank':>7} {'kappa':>9} {'obstr?':>7} "
          f"{'top drop':>10} {'top-3 stable?':>26} {'cHB-cHW/floor':>14}")
    for eps in [0.0, 0.01, 0.03, 0.1, 0.3]:
        ranks, kaps, topdrop, relfloor = [], [], [], []
        for t in range(6 if eps > 0 else 1):
            rng = np.random.default_rng(100 + t)
            A, Q = add_noise(eps, rng, mode) if eps > 0 else (A0, Q0)
            wA = A / sqw[:, None]
            sv = np.linalg.svd(wA, compute_uv=False)
            floor = max(noise_floor(A0, eps) if mode == "all"
                        else eps * sv[0] / np.sqrt(nB), 1e-8*sv[0])
            ranks.append(int(np.sum(sv > floor)))
            g0 = sg.fisher_metric(A, Q, w)
            B = np.full(nC, sg.nda_wall())
            ev, ew = np.linalg.eigh((B[:, None]*g0)*B[None, :])
            res = ew[:, ev > floor**2]
            soft = res[:, 0] if res.shape[1] else ew[:, 0]
            topdrop.append(int(np.argmax(np.abs(soft))))
            relfloor.append(np.linalg.norm((A @ u_rel)/sqw)/floor)
            if t == 0 and eps in (0.0, 0.1):
                kaps.append(kappa_slice(A, Q)[0])
        rk = f"{min(ranks)}-{max(ranks)}" if len(set(ranks)) > 1 else str(ranks[0])
        kap = f"{kaps[0]:.2f}" if kaps else "-"
        obstr = ("yes" if kaps[0] > sg.TWIST_THRESHOLD else "no") if kaps else "-"
        cnt = Counter(topdrop)
        top = cnt.most_common(1)[0]
        frac = top[1]/len(topdrop)
        print(f"{eps:>6} {rk:>7} {kap:>9} {obstr:>7} {names[top[0]]:>10} "
              f"{f'{names[top[0]]} in {int(frac*100)}% of trials':>26} "
              f"{np.mean(relfloor):>14.2f}")

print("\nHONEST reading:")
print("  ROBUST: the twist verdict (kappa >> threshold -> obstructed,")
print("    drops-only) is stable under MC noise in both models.")
print("  ROBUST: the exact relation is handled uniformly -- protected in the")
print("    faithful model (structural zero), below the floor in the")
print("    pessimistic one; no separate analytic step needed either way.")
print("  NOT ROBUST: the specific top-drop coefficient flips with noise, and")
print("    the rank count is threshold-sensitive (the naive floor here")
print("    inflates rank in 'stat', collapses it in 'all').")
print("  CONCLUSION: the general/trustworthy output is the CERTIFICATE (which")
print("    KIND of reduction), not the coefficient-level decision. On real MC")
print("    inputs, drops must be reported as bootstrap stability fractions")
print("    over MC replicas, with a principled noise-aware rank threshold")
print("    (Marchenko-Pastur / bootstrapped singular values) -- an open item.")
