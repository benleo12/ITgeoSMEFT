#!/usr/bin/env python3
"""
Audit finding 1 (critical): the LEP export substitutes x -> x0 = (v/Lambda)^2,
so its fit variables are Lambda-explicit Warsaw coefficients C with NDA range
|C| <= 4pi. Our LEP runs used the absorbed-normalization wall 0.030 instead,
a box ~413x too small. DY is in the absorbed normalization, so DY was right.

This script settles it and recomputes every LEP number on the correct box:
 1. normalization check: median per-unit-coefficient response, LEP vs DY
 2. exact sparsified kernel vectors u1, u2 (audit says -0.456 and -0.388)
 3. kappa (m=1, resolved slice) and blind-direction rotation on |C| <= 4pi
 4. price of forcing both relations on |C| <= 4pi, with all 2^10 corners and
    L-BFGS-B polish (the monotone GN stalls on LEP, audit finding 3)
 5. price versus Lambda done correctly: rescale A by (x_L/x0) and Q by
    (x_L/x0)^2 at fixed box, never shrink the box
 6. LEP single-drop ladder on the correct box, polished
"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-user-Downloads/"
                   "77cdb1df-3b76-468d-9e32-1a5a8f7f7054/scratchpad/"
                   "ITgeoSMEFT/core")
import numpy as np
from scipy.optimize import minimize as spmin
from fractions import Fraction
import battery_lib as bl
import smeftgeo as sg

lep = sg.load_lep()
names = lep["names"]
rho = np.load("/Users/user/Downloads/lep_rho.npy")
cov = rho * np.outer(lep["sigma"], lep["sigma"])
H = 2.0 * lep["Q"]
Aw, Hw = bl.prep(lep["A"], H, cov=cov)
nC = len(names)
BOX = 4 * np.pi                       # correct NDA range for Lambda-explicit C
x0 = (0.246 / 5.0) ** 2

# ---- 1. normalization check ------------------------------------------------
perunit_lep = np.median(np.abs(lep["A"]) / np.abs(lep["SM"])[:, None]
                        [np.abs(lep["A"]) > 0].reshape(-1, 1) if False else
                        np.abs(lep["A"] / lep["SM"][:, None])[np.abs(lep["A"]) > 0])
dy = sg.load_dy()
perunit_dy = np.median((np.abs(dy["A"]) / dy["NSM"][:, None])[np.abs(dy["A"]) > 0])
print(f"1. per-unit-coefficient relative response: LEP {perunit_lep:.2e} "
      f"(x0 = {x0:.2e} -> variables are C), DY {perunit_dy:.2e} "
      f"(O(1) -> absorbed normalization, DY box 0.030 was correct)")

# ---- 2. exact sparsified kernel vectors ------------------------------------
u_, s, vt = np.linalg.svd(Aw)
null = vt[8:]                                       # 2 x 10

def sparsify(target_idx):
    """combination of the two null vectors with a 1 in target_idx and the
    smallest other support"""
    M = null[:, :]
    # solve for coefficients a: a . null has value 1 at target and minimal norm
    T = null[:, target_idx]
    a = np.linalg.lstsq(T[None, :].T @ T[None, :] + 0*np.eye(2), T, rcond=None)[0] \
        if False else np.linalg.solve(null[:, [target_idx]].T @ null[:, [target_idx]] + 1e-30*np.eye(1),
                                      np.array([1.0]))  # placeholder
    return None

# direct: u1 pinned by cHD=1 and dGF6=0; u2 pinned by dGF6=1 and cHD=0
iHD, iGF = names.index("cHD"), names.index("δGF6")
M = np.array([[null[0][iHD], null[1][iHD]], [null[0][iGF], null[1][iGF]]])
a1 = np.linalg.solve(M.T, np.array([1.0, 0.0]))
a2 = np.linalg.solve(M.T, np.array([0.0, 1.0]))
u1 = a1 @ null; u2 = a2 @ null
print("\n2. sparsified kernel vectors (residual = |A u| / |A| |u|):")
for tag, u in [("u1", u1), ("u2", u2)]:
    r = np.linalg.norm(Aw @ u) / (np.linalg.norm(Aw) * np.linalg.norm(u))
    parts = [f"{u[i]:+.4f} {names[i]}" for i in np.argsort(-np.abs(u))
             if abs(u[i]) > 1e-3]
    print(f"   {tag} (residual {r:.1e}): " + "  ".join(parts))
u1n = u1 / np.linalg.norm(u1); u2v = u2 - (u2 @ u1n) * u1n
u2n = u2v / np.linalg.norm(u2v)
null_on = np.stack([u1n, u2n])

# ---- 3. kappa and rotation on the correct box ------------------------------
cc = bl.active_coords(Aw)
kap = bl.kappa(Aw, Hw, np.full(nC, BOX), m=1, nsamp=30, coords=cc)
As = Aw[:, cc]; Hs = Hw[np.ix_(range(Hw.shape[0]), cc, cc)]
n = len(cc)

def metric(c):
    J = As + np.einsum('oij,j->oi', Hs, c)
    return J.T @ J
v0 = np.linalg.eigh(metric(np.zeros(n)))[1][:, 0]
rng = np.random.default_rng(0)
ang = [np.degrees(np.arccos(np.clip(abs(
    np.linalg.eigh(metric(rng.uniform(-.95, .95, n) * BOX))[1][:, 0] @ v0),
    0, 1))) for _ in range(300)]
step = 0.5 * BOX * v0

def shift_cost(c):
    d = bl.mu(As, Hs, (c + step)[None]) - bl.mu(As, Hs, c[None])
    return float((d ** 2).sum())
worst_shift = max(shift_cost(rng.uniform(-.45, .45, n) * BOX)
                  for _ in range(300))
print(f"\n3. correct box |C| <= 4pi: kappa(m=1) = {kap:.2f}  "
      f"rotation median {np.median(ang):.1f} deg (max {max(ang):.1f})  "
      f"shift cost origin {shift_cost(np.zeros(n)):.3g} -> worst "
      f"{worst_shift:.3g}")

# ---- 4./5. price of forcing both relations, polished, vs Lambda ------------
P = np.eye(nC) - null_on.T @ null_on          # projector onto u1,u2 = 0
basis = np.linalg.svd(null_on, full_matrices=True)[2][2:]   # 8-dim family

def price_relations(scaleA, scaleH, nsamp=1500, seed=1):
    Awl, Hwl = Aw * scaleA, Hw * scaleH
    rngl = np.random.default_rng(seed)
    pts = rngl.uniform(-1, 1, (nsamp, nC)) * BOX
    corners = BOX * (2 * ((np.arange(2 ** 10)[:, None] >>
                           np.arange(10)) & 1) - 1)          # all 1024 corners
    allpts = np.vstack([pts, corners])
    tgt = bl.mu(Awl, Hwl, allpts)

    def resid2(z, t):
        c = z @ basis
        return float(((bl.mu(Awl, Hwl, c[None]) - t) ** 2).sum())
    worst = 0.0; worst_c = None
    # cheap first pass: projection point as seed, no polish
    d0 = ((bl.mu(Awl, Hwl, allpts @ P.T) - tgt) ** 2).sum(1)
    order = np.argsort(-d0)[:40]                  # polish the 40 worst
    for i in order:
        z0 = (allpts[i] @ P.T) @ basis.T
        r = spmin(resid2, z0, args=(tgt[i],), method="L-BFGS-B",
                  bounds=[(-BOX * 3.2, BOX * 3.2)] * 8)
        if r.fun > worst:
            worst = r.fun
    return worst

print("\n4./5. price of forcing u1 = u2 = 0 (polished, corners included):")
for Lam in [5.0, 10.0, 20.0]:
    xl = (0.246 / Lam) ** 2
    p = price_relations(xl / x0, (xl / x0) ** 2)
    print(f"   Lambda = {Lam:4.0f} TeV: worst-case Delta chi^2 = {p:.3g}")

# ---- 6. LEP single-drop ladder on the correct box, polished ----------------
print("\n6. single-drop ladder on |C| <= 4pi (polished):")

def price_drop(k, nsamp=800, seed=2):
    keep = [j for j in range(nC) if j != k]
    rngl = np.random.default_rng(seed)
    pts = rngl.uniform(-1, 1, (nsamp, nC)) * BOX
    corners = BOX * (2 * ((np.arange(2 ** 10)[:, None] >>
                           np.arange(10)) & 1) - 1)
    allpts = np.vstack([pts, corners])
    tgt = bl.mu(Aw, Hw, allpts)

    def resid2(z, t):
        c = np.zeros(nC); c[keep] = z
        return float(((bl.mu(Aw, Hw, c[None]) - t) ** 2).sum())
    d0 = bl._proj_chi2(Aw, Hw, tgt, allpts[:, keep], keep, None, -1,
                       clampbox=np.full(nC - 1, BOX))
    worst = 0.0
    for i in np.argsort(-d0)[:30]:
        r = spmin(resid2, allpts[i][keep], args=(tgt[i],), method="L-BFGS-B",
                  bounds=[(-BOX, BOX)] * (nC - 1))
        worst = max(worst, r.fun)
    return worst

ladder = {names[k]: price_drop(k) for k in range(nC)}
for nm in sorted(ladder, key=ladder.get):
    print(f"   {nm:10} {ladder[nm]:12.3g}")
