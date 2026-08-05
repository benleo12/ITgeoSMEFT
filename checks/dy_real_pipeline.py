#!/usr/bin/env python3
"""
Real-data run (95-600 GeV, 101 bins, 19 dim-6 coefficients, dim-8 zeroed):
  C1c  active slice + width cliff
  C1d  twist certificate kappa (toy-calibrated threshold)
  C6   frozen vs curved GeoDrop battery on the nondegenerate slice

Machinery identical to the validated c1a/c1cd/c6 Python pipeline
(projector-based twist, box-graded reference metric, correct gradient
dmu = A + H.c, geodesics on the slice only). Toy calibration constants
from c1cd_dy_kappa.py run of 2026-07-02:
  kappa_clean(Model X) = 3.851e-3, kappa_obstructed(Model O) = 8.173e-1,
  working threshold (geometric midpoint) = 5.610e-2.
"""
import json
import numpy as np
from itertools import combinations

KAPPA_THRESHOLD = 5.610e-2

with open('/Users/user/Downloads/dy_real_95_600.json') as f:
    data = json.load(f)
names = data["names"]
NSM = np.array(data["NSM"], float)
A = np.array(data["A"], float)
H = np.array(data["H"], float)
nBins, nC = A.shape
print(f"REAL DATA: {nBins} bins ({data['binlo'][0]}-{data['binhi'][-1]} GeV), "
      f"{nC} coefficients")

# ==================================================================== #
# C1c: decoupled / lifted, width spectrum, greedy slice, cliff          #
# ==================================================================== #
print("\n" + "=" * 60)
print("C1c (real data): active slice and width cliff")
print("=" * 60)

decoupled = [i for i in range(nC)
             if np.abs(A[:, i]).max() == 0 and np.abs(H[:, i, :]).max() == 0]
linflat = [i for i in range(nC)
           if np.abs(A[:, i]).max() == 0 and i not in decoupled]
print("exactly decoupled:", [names[i] for i in decoupled] or "none")
print("flat in A, lifted at dim6^2:", [names[i] for i in linflat] or "none")

whitA = A / np.sqrt(NSM)[:, None]
sv = np.linalg.svd(whitA, compute_uv=False)
print("\nwidth spectrum:")
for k, s in enumerate(sv[:12]):
    print(f"  {k+1}: {s:.3e}")
rankLin = int(np.sum(sv > 1e-8 * sv[0]))
print(f"linear rank: {rankLin} / {nC}")

chosen = []
resid = whitA.copy()
for _ in range(rankLin):
    norms = np.array([np.linalg.norm(resid[:, i])
                      if (i not in chosen and i not in decoupled) else -1
                      for i in range(nC)])
    best = int(np.argmax(norms))
    if norms[best] <= 1e-8 * sv[0]:
        break
    chosen.append(best)
    col = resid[:, best] / np.linalg.norm(resid[:, best])
    resid = resid - np.outer(col, col @ resid)
print(f"slice ({len(chosen)}): {[names[i] for i in chosen]}")

vev, Lam0 = 0.246, 5.0
thresh = 4 * np.pi * (vev / Lam0) ** 2
ns = len(chosen)
Asl = A[:, chosen]
Hsl = H[np.ix_(range(nBins), chosen, chosen)]
sqN = np.sqrt(NSM)


def gslice(c):
    dm = (Asl + np.einsum('bij,j->bi', Hsl, c)) / sqN[:, None]
    return dm.T @ dm


g0 = gslice(np.zeros(ns))
B = np.full(ns, thresh)
gu0 = (B[:, None] * g0) * B[None, :]
evu, ewu = np.linalg.eigh(gu0)
widths = np.sqrt(np.abs(evu))[::-1]
print("slice widths (box-graded):", [f"{w:.2e}" for w in widths])
ratios = widths[:-1] / widths[1:]
print("cliff ratios:", [f"{r:.1f}" for r in ratios])
kSoft = len(widths) - (int(np.argmax(ratios)) + 1)
print(f"soft-block size k = {kSoft}")

ginv = np.linalg.inv(g0)
bdata = np.minimum(2 * np.sqrt(np.abs(np.diag(ginv))), 1.0)
print("B_data proxy (2-sigma widths, capped 1):")
for k, i in enumerate(chosen):
    print(f"    {names[i]}: +-{bdata[k]:.4f}")

# ==================================================================== #
# C1d: kappa (identical BoxTwist machinery as c1cd_dy_kappa.py)         #
# ==================================================================== #
print("\n" + "=" * 60)
print("C1d (real data): kappa Step-0")
print("=" * 60)


class BoxTwist:
    def __init__(self, gfun, n, box):
        self.gfun, self.n, self.B = gfun, n, np.asarray(box, float)

    def gu(self, u):
        c = self.B * u
        return (self.B[:, None] * self.gfun(c)) * self.B[None, :]

    def proj(self, u, m):
        G = self.gu(u)
        G = 0.5 * (G + G.T)
        ev, ew = np.linalg.eigh(G)
        soft, stiff = ew[:, :m].T, ew[:, m:].T
        return soft.T @ soft, stiff.T @ stiff, stiff

    def twist(self, u, m, fd=1e-4):
        Psf, Pst, stiff = self.proj(u, m)
        dP = np.zeros((self.n, self.n, self.n))
        for d in range(self.n):
            e = np.zeros(self.n)
            e[d] = fd
            dP[d] = (self.proj(u + e, m)[1] - self.proj(u - e, m)[1]) / (2*fd)
        best = 0.0
        for a, b in combinations(range(len(stiff)), 2):
            X, Y = stiff[a], stiff[b]
            brkt = np.einsum('d,dij,j->i', X, dP, Y) \
                 - np.einsum('d,dij,j->i', Y, dP, X)
            best = max(best, float(np.linalg.norm(Psf @ brkt)))
        return best

    def kappa(self, m, nsamp=60, label="", rng=None):
        rng = rng or np.random.default_rng(0)
        ts = [self.twist(rng.uniform(-0.95, 0.95, self.n), m)
              for _ in range(nsamp)]
        ts = np.array(ts)
        kap = float(np.median(ts)) * 2.0 * np.sqrt(self.n)
        print(f"  {label}: ||T|| med = {np.median(ts):.3e} "
              f"(90% {np.quantile(ts, .9):.2e}, max {ts.max():.2e})   "
              f"kappa_med = {kap:.3e}")
        return kap


for m in sorted({kSoft, 1, 2}):
    print(f"\n--- kappa, soft-block m = {m} ---")
    kNDA = BoxTwist(gslice, ns, np.full(ns, thresh)).kappa(
        m, label="B_NDA(5 TeV)", rng=np.random.default_rng(2))
    kDAT = BoxTwist(gslice, ns, bdata).kappa(
        m, label="B_data proxy", rng=np.random.default_rng(2))
    for Lam in [1.0, 2.0]:
        BoxTwist(gslice, ns, np.full(ns, 4*np.pi*(vev/Lam)**2)).kappa(
            m, label=f"Lambda={Lam} TeV", rng=np.random.default_rng(3))
    print(f"  VERDICT (m={m}): kappa(B_data) = {kDAT:.3e} vs threshold "
          f"{KAPPA_THRESHOLD:.3e} -> "
          f"{'OBSTRUCTED (drop regime)' if kDAT >= KAPPA_THRESHOLD else 'CLEAN RELATION EXISTS (relation regime)'}")

# ==================================================================== #
# C6: frozen vs curved battery on the slice                            #
# ==================================================================== #
print("\n" + "=" * 60)
print("C6 (real data): frozen vs curved GeoDrop on the slice")
print("=" * 60)


def dmu(c):
    return Asl + np.einsum('bij,j->bi', Hsl, c)


def acc_curved(c, v):
    vHv = np.einsum('i,bij,j->b', v, Hsl, v)
    S = (vHv / NSM) @ dmu(c)
    return -np.linalg.solve(gslice(c), S)


def acc_frozen(c, v):
    vHv = np.einsum('i,bij,j->b', v, Hsl, v)
    S = (vHv / NSM) @ Asl
    return -np.linalg.solve(g0, S)


def shoot(v0, mode, h=1e-4, maxsteps=400000):
    acc = acc_curved if mode == "curved" else acc_frozen
    c, v = np.zeros(ns), v0.copy()
    E0 = float(v @ gslice(c) @ v)
    properL, s, Edrift = 0.0, 0.0, 0.0
    for step in range(maxsteps):
        k1c, k1v = v, acc(c, v)
        k2c, k2v = v + 0.5*h*k1v, acc(c + 0.5*h*k1c, v + 0.5*h*k1v)
        k3c, k3v = v + 0.5*h*k2v, acc(c + 0.5*h*k2c, v + 0.5*h*k2v)
        k4c, k4v = v + h*k3v, acc(c + h*k3c, v + h*k3v)
        cn = c + (h/6)*(k1c + 2*k2c + 2*k3c + k4c)
        vn = v + (h/6)*(k1v + 2*k2v + 2*k3v + k4v)
        gm = gslice(0.5*(c+cn))
        properL += float(np.sqrt(abs((0.5*(v+vn)) @ gm @ (0.5*(v+vn))))) * h
        c, v, s = cn, vn, s + h
        if step % 100 == 0:
            E = float(v @ gslice(c) @ v)
            Edrift = max(Edrift, abs(E - E0)/max(abs(E0), 1e-30))
        wall = np.where(np.abs(c) >= thresh)[0]
        if len(wall):
            i = wall[np.argmax(np.abs(c[wall]))]
            return {"wall": names[chosen[i]], "L": properL, "Edrift": Edrift}
    return {"wall": None, "L": properL, "Edrift": Edrift}


soft1 = B * ewu[:, 0]
soft2 = B * ewu[:, 1]
soft1 /= np.linalg.norm(soft1)
soft2 /= np.linalg.norm(soft2)
for tag, v in (("soft1", soft1), ("soft2", soft2)):
    top = np.argsort(np.abs(v))[::-1][:3]
    print(f"  {tag}: "
          + ", ".join(f"{names[chosen[i]]}={v[i]:+.2f}" for i in top))

print(f"\n{'dir':6} {'sgn':4} {'h':8} {'frozen->':12} {'L_f':11} "
      f"{'curved->':12} {'L_c':11} {'Edrift':9}")
flips, runs = 0, 0
for tag, vdir in (("soft1", soft1), ("soft2", soft2)):
    for sgn in (+1, -1):
        for hstep in (1e-4, 5e-5):
            fz = shoot(sgn*vdir, "frozen", h=hstep)
            cv = shoot(sgn*vdir, "curved", h=hstep)
            runs += 1
            flip = fz["wall"] != cv["wall"]
            flips += flip
            print(f"{tag:6} {sgn:+4d} {hstep:8.0e} "
                  f"{str(fz['wall']):12} {fz['L']:<11.4g} "
                  f"{str(cv['wall']):12} {cv['L']:<11.4g} "
                  f"{cv['Edrift']:<9.2e}" + ("   FLIP" if flip else ""))

print(f"\nflips: {flips}/{runs}")
print("\n========== REAL-DATA VERDICTS ==========")
print(f"rank: {rankLin}/19; slice: {[names[i] for i in chosen]}")
print(f"kappa(B_data, m={kSoft}) vs 0.056 threshold: see above")
print(f"flip: {flips}/{runs} -> "
      f"{'curvature changes the drop' if flips else 'decision is curvature-robust'}")
