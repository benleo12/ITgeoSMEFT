#!/usr/bin/env python3
"""
Fix the two gates before writing the paper.

CONVENTION (from extract_lep.wl lines 42-50): LEP predictions are
  mu = SM + A.c + c.Q.c
so the Hessian is H = 2Q. Earlier battery runs fed Q in as H, which
understated LEP curvature by 2. Everything here uses H = 2Q.
DY is unaffected (its file stores the Hessian directly).

GATE 1 (T4): the two blind directions of the LEP linear design, with names,
and the price of forcing both relations (zero the blind parts of c):
  price = worst prediction change over the box, ||mu(c) - mu(Pc)||^2.
GATE 2 (numbers check): corrected LEP kappa, blind-direction rotation, and
shift cost with H = 2Q -- do the paper numbers move?
"""
import numpy as np
import battery_lib as bl
import smeftgeo as sg

lep = sg.load_lep()
names = lep["names"]
rho = np.load("lep_rho.npy")
cov = rho * np.outer(lep["sigma"], lep["sigma"])
H = 2.0 * lep["Q"]                      # true Hessian
Aw, Hw = bl.prep(lep["A"], H, cov=cov)
nC = len(names)

# ---------- the two blind directions, named ----------
u_, s, vt = np.linalg.svd(Aw)
null = vt[8:]                            # 2 x 10
print("blind directions of the LEP linear design (kernel of A):")
for k, u in enumerate(null):
    u = u / np.abs(u).max()
    parts = [f"{u[i]:+.3f} {names[i]}" for i in np.argsort(-np.abs(u))
             if abs(u[i]) > 0.02]
    print(f"  u{k+1}: " + "  ".join(parts))

# ---------- GATE 1: price of forcing both relations ----------
P = np.eye(nC) - null.T @ null           # zero the blind parts
print("\nprice of forcing both relations (worst Delta chi^2 over the box):")
rng = np.random.default_rng(0)
for Lam in [5.0, 2.0, 1.0]:
    wall = sg.nda_wall(Lam)
    C = rng.uniform(-wall, wall, (4000, nC))
    d = bl.mu(Aw, Hw, C) - bl.mu(Aw, Hw, C @ P.T)
    price = float((d**2).sum(1).max())
    print(f"  Lambda = {Lam:3.0f} TeV box (|c| <= {wall:.3f}):  "
          f"price = {price:.3g}")

# ---------- GATE 2: corrected LEP numbers (H = 2Q vs the old Q) ----------
print("\ncorrected LEP numbers with H = 2Q (old runs used Q):")
wall = sg.nda_wall(5.0)
for tag, Hin in [("old (Q as H)", lep["Q"]), ("corrected (2Q)", H)]:
    Aw2, Hw2 = bl.prep(lep["A"], Hin, cov=cov)
    cc = bl.active_coords(Aw2)
    kap = bl.kappa(Aw2, Hw2, np.full(nC, wall), m=1, nsamp=25, coords=cc)
    # rotation of the blind combination + cost of a fixed shift along it
    As, Hs = Aw2[:, cc], Hw2[np.ix_(range(Hw2.shape[0]), cc, cc)]
    n = len(cc)

    def metric(c):
        J = As + np.einsum('oij,j->oi', Hs, c)
        return J.T @ J
    v0 = np.linalg.eigh(metric(np.zeros(n)))[1][:, 0]
    rr = np.random.default_rng(1)
    ang = [np.degrees(np.arccos(np.clip(abs(
        np.linalg.eigh(metric(rr.uniform(-.95, .95, n)*wall))[1][:, 0] @ v0),
        0, 1))) for _ in range(200)]
    step = 0.5*wall*v0

    def cost(c):
        d = bl.mu(As, Hs, (c+step)[None]) - bl.mu(As, Hs, c[None])
        return float((d**2).sum())
    worst = max(cost(rr.uniform(-.45, .45, n)*wall) for _ in range(200))
    print(f"  {tag:16}: kappa {kap:.3f} | rotation median "
          f"{np.median(ang):4.1f} deg | shift cost worst {worst:.2e}")
