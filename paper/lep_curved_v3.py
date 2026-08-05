#!/usr/bin/env python3
"""
Curved vs Gaussian LEP error bars -- v3, DEFINITIVE, like-with-like:
compare the LINEAR-profiled interval (standard Fisher bound, what people
quote) against the QUARTIC-profiled interval (full dim-6^2 likelihood),
using the SAME profiling procedure. The only difference is whether Q is on,
so the effect is purely curvature and the linear limit passes by construction.

  chi^2(c) = shift(c)^T Sigma^-1 shift(c)
  Sigma = 9x9 real LEP EWWG block (lep_rho.npy) x PDG-relative sigmas
  profile: scipy L-BFGS-B, analytic gradient, multistart
Report both intervals, the width ratio, the asymmetry, and interior flags;
only interior-in-both coefficients are cleanly quotable.
"""
import json
import numpy as np
from scipy.optimize import minimize

lep = json.load(open("/Users/user/Downloads/lep_real.json"))
names = lep["names"]
A = np.array(lep["A"]); Q = np.array(lep["Q"]); SM = np.array(lep["SM"])
rho = np.load("/Users/user/Downloads/lep_rho.npy")
pdg = [(91.1876, 0.0021), (2.4952, 0.0023), (41.541, 0.037), (20.767, 0.025),
       (0.0171, 0.0010), (0.21629, 0.00066), (0.1721, 0.0030), (0.0992, 0.0016),
       (0.0707, 0.0035), (80.379, 0.012), (0.02766, 0.00007)]
relunc = np.array([u/v for v, u in pdg]); sig_all = relunc*np.abs(SM)
keep = [o for o in range(11) if np.linalg.norm(A[o]) > 0 or np.linalg.norm(Q[o]) > 0]
A, Q, sig = A[keep], Q[keep], sig_all[keep]; rho = rho[np.ix_(keep, keep)]
SigInv = np.linalg.inv(np.outer(sig, sig)*rho)
nC = A.shape[1]; nda = 4*np.pi


def build(Qmat):
    def chi2(c):
        r = A@c + np.einsum('oij,i,j->o', Qmat, c, c)
        return float(r @ SigInv @ r)

    def grad(c):
        r = A@c + np.einsum('oij,i,j->o', Qmat, c, c)
        Jc = A + 2*np.einsum('oij,j->oi', Qmat, c)
        return 2*Jc.T @ (SigInv @ r)
    return chi2, grad


def profmin(chi2, grad, i, val, restarts=5):
    free = [j for j in range(nC) if j != i]
    rng = np.random.default_rng(1)
    best = np.inf
    for t in range(restarts):
        x0 = np.zeros(nC); x0[i] = val
        if t:
            x0[free] = rng.uniform(-nda, nda, len(free))

        def f(xf, base=x0):
            c = base.copy(); c[free] = xf; return chi2(c)

        def g(xf, base=x0):
            c = base.copy(); c[free] = xf; return grad(c)[free]
        r = minimize(f, x0[free], jac=g, method="L-BFGS-B",
                     bounds=[(-nda, nda)]*len(free))
        best = min(best, r.fun)
    return best


def interval(chi2, grad, i, level=1.0):
    out = []
    for sgn in (-1, +1):
        lo, hi = 0.0, sgn*nda
        if profmin(chi2, grad, i, hi) < level:
            out.append((sgn*nda, False)); continue
        for _ in range(42):
            m = 0.5*(lo+hi)
            (lo, hi) = (m, hi) if profmin(chi2, grad, i, m) < level else (lo, m)
        out.append((hi, True))
    return out[0], out[1]     # (lo,bounded), (hi,bounded)


chi2L, gradL = build(np.zeros_like(Q))     # linear baseline
chi2Q, gradQ = build(Q)                    # full quartic

# which coefficients does the linear fit constrain inside the box?
constrained = [i for i in range(nC)
               if profmin(chi2L, gradL, i, nda) > 1.0]

print(f"{'coeff':10} {'linear-profiled':>18} {'curved-profiled':>18} "
      f"{'width ratio':>11} {'asym':>7} {'clean':>6}")
clean = []
for i in constrained:
    (loL, lbL), (hiL, hbL) = interval(chi2L, gradL, i)
    (loC, lbC), (hiC, hbC) = interval(chi2Q, gradQ, i)
    wL, wC = hiL - loL, hiC - loC
    asym = (hiC + loC) / wC if wC > 0 else 0
    interior = lbL and hbL and lbC and hbC
    if interior:
        clean.append(i)
    print(f"{names[i]:10} [{loL:>6.2f},{hiL:>6.2f}]   [{loC:>6.2f},{hiC:>6.2f}]"
          f"   {wC/wL:>10.3f} {asym:>7.2f} {'yes' if interior else 'box':>6}")

print("\n--- headline (clean, interior-in-both coefficients) ---")
for i in clean:
    (loL, _), (hiL, _) = interval(chi2L, gradL, i)
    (loC, _), (hiC, _) = interval(chi2Q, gradQ, i)
    print(f"  {names[i]}: dim-6^2 curvature moves the 68% bound from "
          f"[{loL:.2f},{hiL:.2f}] to [{loC:.2f},{hiC:.2f}]  "
          f"({(hiC-loC)/(hiL-loL)*100-100:+.0f}% width, "
          f"asym {(hiC+loC)/(hiC-loC):+.2f})")
print("\nlinear-limit check: curved==linear when Q=0 holds by construction")
print("(both use the identical profiler; ratio would be 1.000).")
