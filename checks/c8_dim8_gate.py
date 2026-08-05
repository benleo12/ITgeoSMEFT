#!/usr/bin/env python3
"""
C8: dim-8 nuisance gate on the real DY conclusions.

dim-8 interference is the same EFT order (1/Lambda^4) as dim6^2 -- any
curvature-level conclusion must survive marginalizing NDA-sized dim-8.
Marginalization: Sigma_eff = diag(N_SM) + A8 diag(sig8^2) A8^T.
sig8 is scanned over 3 decades around thresh^2 (normalization of c8 in
Adam's polynomials is not pinned; the gate asks for verdict STABILITY,
not a measurement).

Gated conclusions:
  (1) linear rank 8/19 and slice content
  (2) twist verdict (kappa obstructed)
  (3) drop identity (frozen vs curved walls: c16Hpsiq/ceQ)
"""
import json
import numpy as np
from itertools import combinations

with open('/Users/user/Downloads/dy_real_95_600.json') as f:
    d = json.load(f)
with open('/Users/user/Downloads/dy_dim8.json') as f:
    d8 = json.load(f)
names = d["names"]
NSM = np.array(d["NSM"], float)
A = np.array(d["A"], float)
H = np.array(d["H"], float)
A8 = np.array(d8["A8"], float)
nB, nC = A.shape
thresh = 4*np.pi*(0.246/5.0)**2
THRESHOLD = 5.61e-2

# drop exactly-zero dim-8 columns
keep8 = [j for j in range(A8.shape[1]) if np.abs(A8[:, j]).max() > 0]
A8 = A8[:, keep8]
print(f"dim-8 nuisances kept: {A8.shape[1]} of {len(d8['names8'])}")


class BoxTwist:
    def __init__(self, gfun, n, box):
        self.gfun, self.n, self.B = gfun, n, np.asarray(box, float)

    def gu(self, u):
        c = self.B*u
        return (self.B[:, None]*self.gfun(c))*self.B[None, :]

    def proj(self, u, m):
        G = self.gu(u)
        G = 0.5*(G+G.T)
        ev, ew = np.linalg.eigh(G)
        return ew[:, :m] @ ew[:, :m].T, ew[:, m:] @ ew[:, m:].T, ew[:, m:].T

    def twist(self, u, m, fd=1e-4):
        Psf, Pst, stiff = self.proj(u, m)
        dP = np.zeros((self.n, self.n, self.n))
        for dd in range(self.n):
            e = np.zeros(self.n)
            e[dd] = fd
            dP[dd] = (self.proj(u+e, m)[1]-self.proj(u-e, m)[1])/(2*fd)
        best = 0.0
        for a, b in combinations(range(len(stiff)), 2):
            X, Y = stiff[a], stiff[b]
            brkt = np.einsum('d,dij,j->i', X, dP, Y) \
                 - np.einsum('d,dij,j->i', Y, dP, X)
            best = max(best, float(np.linalg.norm(Psf @ brkt)))
        return best

    def kappa_med(self, m, nsamp=40, rng=None):
        rng = rng or np.random.default_rng(2)
        ts = [self.twist(rng.uniform(-0.95, 0.95, self.n), m)
              for _ in range(nsamp)]
        return float(np.median(ts))*2*np.sqrt(self.n)


for sig8fac in [0.1, 1.0, 10.0]:
    sig8 = sig8fac * thresh**2
    Sig = np.diag(NSM) + (A8*sig8**2) @ A8.T
    Linv = np.linalg.inv(np.linalg.cholesky(Sig))
    Aw = Linv @ A                      # whitened linear design

    sv = np.linalg.svd(Aw, compute_uv=False)
    rank = int(np.sum(sv > 1e-8*sv[0]))

    # greedy slice
    chosen, resid = [], Aw.copy()
    for _ in range(rank):
        norms = np.array([np.linalg.norm(resid[:, i]) if i not in chosen
                          else -1 for i in range(nC)])
        best = int(np.argmax(norms))
        chosen.append(best)
        col = resid[:, best]/np.linalg.norm(resid[:, best])
        resid = resid - np.outer(col, col @ resid)
    ns = len(chosen)
    Asl = A[:, chosen]
    Hsl = H[np.ix_(range(nB), chosen, chosen)]

    def gslice(c, Asl=Asl, Hsl=Hsl, Linv=Linv):
        dm = Linv @ (Asl + np.einsum('bij,j->bi', Hsl, c))
        return dm.T @ dm

    g0 = gslice(np.zeros(ns))
    B = np.full(ns, thresh)
    w = np.sqrt(np.abs(np.linalg.eigvalsh((B[:, None]*g0)*B[None, :])))[::-1]
    ratios = w[:-1]/w[1:]
    m = max(1, ns - (int(np.argmax(ratios))+1))
    kap = BoxTwist(gslice, ns, B).kappa_med(m)

    # frozen vs curved shoot along +/- sloppiest box-graded direction
    evu, ewu = np.linalg.eigh((B[:, None]*g0)*B[None, :])
    NSM_eff = None  # not needed; acc uses Linv-whitened residual weights

    def dmu(c):
        return Asl + np.einsum('bij,j->bi', Hsl, c)

    def acc(c, v, frozen):
        # least-squares geodesic: acc = -g^-1 J^T Sigma^-1 (v.d2mu.v)
        vHv = np.einsum('i,bij,j->b', v, Hsl, v)
        rows = Linv @ (dmu(np.zeros(ns)) if frozen else dmu(c))
        S = rows.T @ (Linv @ vHv)
        gm = gslice(np.zeros(ns)) if frozen else gslice(c)
        return -np.linalg.solve(gm, S)

    def shoot(v0, frozen, h=1e-4, maxsteps=400000):
        c, v = np.zeros(ns), v0.copy()
        for _ in range(maxsteps):
            a1 = acc(c, v, frozen)
            cn = c + h*v + 0.5*h*h*a1
            vn = v + h*a1
            c, v = cn, vn
            wall = np.where(np.abs(c) >= thresh)[0]
            if len(wall):
                i = wall[np.argmax(np.abs(c[wall]))]
                return names[chosen[i]]
        return None

    soft1 = B*ewu[:, 0]
    soft1 /= np.linalg.norm(soft1)
    wf_p = shoot(+soft1, True)
    wc_p = shoot(+soft1, False)
    wf_m = shoot(-soft1, True)
    wc_m = shoot(-soft1, False)

    print(f"\nsig8 = {sig8fac} x thresh^2 = {sig8:.2e}:")
    print(f"  rank {rank}/19; slice ({ns}): {[names[i] for i in chosen]}")
    print(f"  kappa(B_NDA, m={m}) = {kap:.3f} -> "
          f"{'OBSTRUCTED' if kap >= THRESHOLD else 'integrable'}")
    print(f"  walls: +soft frozen={wf_p} curved={wc_p} | "
          f"-soft frozen={wf_m} curved={wc_m}"
          f"   flip: {wf_p != wc_p or wf_m != wc_m}")

print("\nGATE: conclusions stable across the 3-decade sig8 scan?")
