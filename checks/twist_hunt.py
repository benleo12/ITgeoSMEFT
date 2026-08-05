#!/usr/bin/env python3
"""
Does a case exist where the twist changes the decision? Test the one
hypothesis I had not tested.

Earlier I argued: if the blind direction is EXACTLY blind and constant
(T = 0), some single drop is always free, so a relation buys nothing. That
argument needs the direction to be exactly blind. It says nothing about a
direction that is merely SLOPPY.

Hypothesis: take a sloppy (small but nonzero eigenvalue) direction that is
spread evenly over many coefficients, v ~ (1,1,...,1)/sqrt(n). Then no single
coefficient carries much of it, every single drop needs a large compensation
and is expensive, while the relation v.c = 0 costs almost nothing. If the
foliation is clean (T ~ 0) the relation is globally valid, and the twist is
what tells you so.

We scan the spread of the sloppy direction from concentrated on one
coefficient to spread over all of them, and compare:
   best single drop   vs   the relation v.c = 0
"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-user-Downloads/"
                   "77cdb1df-3b76-468d-9e32-1a5a8f7f7054/scratchpad/"
                   "ITgeoSMEFT/core")
import numpy as np
from scipy.optimize import minimize as spmin
import battery_lib as bl

N = 6                      # coefficients
NOBS = 30
BOX = 1.0
EPS = 3e-3                 # how sloppy the soft direction is


def build(spread, curv=0.0, seed=0):
    """A model whose softest direction is v, spread over `spread` coefficients.
    spread=1 -> v is one coordinate axis; spread=N -> v is uniform."""
    rng = np.random.default_rng(seed)
    v = np.zeros(N)
    v[:spread] = 1.0
    v = v / np.linalg.norm(v)
    # stiff directions: complete an orthonormal basis
    Bfull = np.linalg.qr(np.column_stack([v, rng.standard_normal((N, N - 1))]))[0]
    Bfull[:, 0] = v
    # design with eigenvalues (EPS, 1, 1, ...): sloppy along v, stiff elsewhere
    lam = np.concatenate([[EPS], np.ones(N - 1)])
    M = Bfull @ np.diag(np.sqrt(lam)) @ Bfull.T
    shp = rng.standard_normal((NOBS, N))
    shp /= np.linalg.norm(shp, axis=0)
    A = shp @ M
    # curvature that keeps the model a function of (v.c) and the stiff coords
    # separately, i.e. integrable by construction
    H = np.zeros((NOBS, N, N))
    if curv:
        w = rng.standard_normal(NOBS)
        H += curv * np.einsum('o,i,j->oij', w, v, v)
    return A, H, v


def price_drop(A, H, k, pts):
    keep = [j for j in range(N) if j != k]
    tgt = bl.mu(A, H, pts)
    d0 = bl._proj_chi2(A, H, tgt, pts[:, keep], keep, None, -1,
                       clampbox=np.full(N - 1, BOX))

    def f(z, t):
        c = np.zeros(N); c[keep] = z
        return float(((bl.mu(A, H, c[None]) - t) ** 2).sum())
    worst = 0.0
    for i in np.argsort(-d0)[:10]:
        r = spmin(f, np.clip(pts[i][keep], -BOX, BOX), args=(tgt[i],),
                  method="L-BFGS-B", bounds=[(-BOX, BOX)] * (N - 1))
        worst = max(worst, r.fun)
    return worst


def price_relation(A, H, v, pts):
    """impose v.c = 0: project onto the orthogonal complement of v"""
    P = np.eye(N) - np.outer(v, v)
    basis = np.linalg.svd(P)[0][:, :N - 1]
    tgt = bl.mu(A, H, pts)

    def f(z, t):
        return float(((bl.mu(A, H, (basis @ z)[None]) - t) ** 2).sum())
    seed = pts @ basis
    d0 = ((bl.mu(A, H, pts @ P.T) - tgt) ** 2).sum(1)
    worst = 0.0
    for i in np.argsort(-d0)[:10]:
        r = spmin(f, seed[i], args=(tgt[i],), method="L-BFGS-B",
                  bounds=[(-BOX * 2, BOX * 2)] * (N - 1))
        worst = max(worst, r.fun)
    return worst


rng = np.random.default_rng(7)
pts = np.vstack([rng.uniform(-1, 1, (400, N)) * BOX,
                 BOX * rng.choice([-1.0, 1.0], size=(2 ** N, N))])

print(f"{N} coefficients, softest eigenvalue {EPS}, box +-{BOX}\n")
print(f"{'spread of v':>12} {'best single drop':>18} {'relation v.c=0':>16}"
      f" {'drop/relation':>14}   twist T")
for spread in range(1, N + 1):
    A, H, v = build(spread, curv=0.0)
    Aw, Hw = bl.prep(A, H, weight=np.ones(NOBS))
    pd = min(price_drop(Aw, Hw, k, pts) for k in range(N))
    pr = price_relation(Aw, Hw, v, pts)
    T = bl.kappa(Aw, Hw, np.full(N, BOX), m=1, nsamp=25)
    print(f"{spread:>12} {pd:>18.4g} {pr:>16.4g} {pd/max(pr,1e-12):>14.1f}"
          f"   {T:.4f}")
print("\nspread = 1 means the sloppy direction is a coordinate axis;")
print("spread = N means it is shared equally by every coefficient.")
