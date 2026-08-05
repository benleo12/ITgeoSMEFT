"""
smeftgeo -- information geometry + MBAM toolkit for SMEFT model reduction.

Consolidates the validated pieces of the campaign into one clean library:
  data loading (DY, LEP)               load_dy, load_lep
  Fisher metric and spectrum           fisher_metric, spectrum
  named null-space relations           null_relations
  twist certificate (drop vs relation) box_twist
  geodesics to EFT walls (GeoDrop)     geodesic
  MBAM ride (no-boundary demo)         mbam_ride
  reduction price (E_inf, worst-case)  price_worstcase
  curved vs Gaussian intervals         curved_intervals

Conventions (established and validated over the campaign):
  predictions per bin/observable:  mu_b(c) = SM_b + A[b].c + c.Q[b].c
  gradient:                        dmu_b   = A[b] + Q[b].c        (Q = Hessian)
  Fisher metric (Gaussian data):   g(c) = sum_b dmu_b outer dmu_b / weight_b
  reference metric for the twist:  the NDA/box grading metric, NOT identity
  geodesics: only on the resolved (nondegenerate) slice, never full-space pinv
"""
import json
import numpy as np
from itertools import combinations

VEV = 0.246          # TeV
LAM_DEFAULT = 5.0    # TeV


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_dy(path="/Users/user/Downloads/dy_real_95_600.json"):
    """Adam's EFT-valid Drell-Yan: 101 bins x 19 dim-6 coeffs, dim-8 zeroed."""
    d = json.load(open(path))
    return dict(
        names=d["names"],
        NSM=np.array(d["NSM"], float),
        A=np.array(d["A"], float),          # (nBins, nC)
        Q=np.array(d["H"], float),          # (nBins, nC, nC) Hessian
        binlo=np.array(d["binlo"], float),
        binhi=np.array(d["binhi"], float),
        weight=np.array(d["NSM"], float),   # Poisson weight
    )


def load_lep(path="/Users/user/Downloads/lep_real.json"):
    """Adam's SMEFT_LEP.nb: 11 observables x 10 flavor-universal coeffs."""
    d = json.load(open(path))
    # LEP EWWG / PDG uncertainties, order per SMEFT_LEP.nb cell 3:
    # mZ, GammaZ, sigma_had, R0l, AFBl, R0b, R0c, AFBb, AFBc, mW2, dalpha
    sig = np.array([0.0021, 0.0023, 0.037, 0.025, 0.0010,
                    0.00066, 0.0030, 0.0016, 0.0035, 2*80.379*0.012, 0.0001])
    return dict(
        names=d["names"],
        SM=np.array(d["SM"], float),
        A=np.array(d["A"], float),          # (nObs, nC)
        Q=np.array(d["Q"], float),          # (nObs, nC, nC)
        sigma=sig,
        weight=sig**2,
    )


# --------------------------------------------------------------------------
# Fisher metric and spectrum
# --------------------------------------------------------------------------
def gradient(A, Q, c):
    """dmu[b] = A[b] + Q[b].c  (per bin/observable)."""
    return A + np.einsum('bij,j->bi', Q, c)


def fisher_metric(A, Q, weight, c=None):
    """g(c) = sum_b dmu_b outer dmu_b / weight_b."""
    if c is None:
        c = np.zeros(A.shape[1])
    dmu = gradient(A, Q, c) / np.sqrt(weight)[:, None]
    return dmu.T @ dmu


def spectrum(g, tol=1e-8):
    """eigenvalues (descending), rank, eigenvectors."""
    vals, vecs = np.linalg.eigh(g)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    rank = int(np.sum(vals > tol * vals.max()))
    return vals, vecs, rank


def nda_wall(Lam=LAM_DEFAULT):
    """NDA validity half-width: |c_i| <= 4 pi (v/Lambda)^2."""
    return 4 * np.pi * (VEV / Lam) ** 2


# --------------------------------------------------------------------------
# exact all-order relations -- OPTIONAL cross-check, NOT the general pipeline
# --------------------------------------------------------------------------
# WARNING: this reads exact zeros from A, Q and is only meaningful when they
# are known SYMBOLICALLY / to high precision (as in Adam's model file). In a
# realistic FeynRules->MadGraph pipeline A, Q are MC-estimated with noise, no
# exact kernel exists, and blind directions must be found geometrically with
# a NOISE-CALIBRATED threshold (see noise_robustness.py). Use this only to
# validate; never as a required step.
def exact_relations(A, Q, tol=1e-8):
    """Combinations u with A.u = 0 AND Q[b].u = 0 for every bin: then
    mu(c + t u) == mu(c) identically -- an exact symmetry. Analytic; found
    cHB - cHW in the DY file. ONLY valid for symbolic/high-precision A, Q.
    Returns (basis vectors, stacked singular values)."""
    stack = np.vstack([A] + [Q[b] for b in range(Q.shape[0])])
    u_, s, vt = np.linalg.svd(stack)
    rank = int(np.sum(s > tol * s[0]))
    return vt[rank:], s


# --------------------------------------------------------------------------
# named null-space relations
# --------------------------------------------------------------------------
def null_relations(A, tol=1e-8, maxden=12, height=3):
    """exact blind directions (kernel of the linear design), sparsified to
    small-integer combinations. Returns list of (vector, sparsity)."""
    from fractions import Fraction
    u, s, vt = np.linalg.svd(A)
    nz = int(np.sum(s > tol * s[0]))
    null = vt[nz:]                       # rows span the kernel
    if len(null) == 0:
        return []
    out, seen = [], set()
    from itertools import product
    for coeffs in product(range(-height, height + 1), repeat=len(null)):
        if all(x == 0 for x in coeffs):
            continue
        v = sum(x * null[i] for i, x in enumerate(coeffs))
        if np.abs(v).max() < 1e-9:
            continue
        v = v / np.abs(v).max()
        key = tuple(np.round(v, 3))
        if key in seen:
            continue
        seen.add(key)
        nnz = int(np.sum(np.abs(v) > 0.02))
        out.append((v, nnz))
    out.sort(key=lambda t: t[1])
    return out


# --------------------------------------------------------------------------
# twist certificate  (drop vs relation)
# --------------------------------------------------------------------------
class BoxTwist:
    """Frobenius twist of the sloppy m-plane over the validity box, measured
    in the box-graded (NDA) reference metric. Median ||T|| across the box;
    small -> a clean global relation exists, large -> only drops."""

    def __init__(self, gfun, n, box):
        self.gfun, self.n, self.B = gfun, n, np.asarray(box, float)

    def _gu(self, u):                    # metric in box-graded coordinates
        c = self.B * u
        return (self.B[:, None] * self.gfun(c)) * self.B[None, :]

    def _proj(self, u, m):
        G = 0.5 * (self._gu(u) + self._gu(u).T)
        ev, ew = np.linalg.eigh(G)
        soft, stiff = ew[:, :m].T, ew[:, m:].T
        return soft.T @ soft, stiff.T @ stiff, stiff

    def twist_at(self, u, m, fd=1e-4):
        Psoft, _, stiff = self._proj(u, m)
        dP = np.zeros((self.n, self.n, self.n))
        for d in range(self.n):
            e = np.zeros(self.n); e[d] = fd
            dP[d] = (self._proj(u + e, m)[1] - self._proj(u - e, m)[1]) / (2*fd)
        best = 0.0
        for a, b in combinations(range(len(stiff)), 2):
            X, Y = stiff[a], stiff[b]
            brk = np.einsum('d,dij,j->i', X, dP, Y) \
                - np.einsum('d,dij,j->i', Y, dP, X)
            best = max(best, float(np.linalg.norm(Psoft @ brk)))
        return best

    def kappa(self, m, nsamp=60, rng=None):
        rng = rng or np.random.default_rng(2)
        ts = np.array([self.twist_at(rng.uniform(-0.95, 0.95, self.n), m)
                       for _ in range(nsamp)])
        diam = 2 * np.sqrt(self.n)
        return dict(kappa=float(np.median(ts)) * diam,
                    tmed=float(np.median(ts)),
                    t90=float(np.quantile(ts, 0.9)),
                    tmax=float(ts.max()))


TWIST_THRESHOLD = 5.61e-2   # toy-calibrated (clean 0.0039 vs obstructed 0.82)


# --------------------------------------------------------------------------
# active slice (resolved, nondegenerate subspace) -- geodesics live here
# --------------------------------------------------------------------------
def active_slice(A, weight, tol=1e-8):
    """greedy pivoted selection of a coordinate subset spanning the row space
    -> a well-conditioned Warsaw-coordinate slice for geodesics."""
    nC = A.shape[1]
    wA = A / np.sqrt(weight)[:, None]
    sv = np.linalg.svd(wA, compute_uv=False)
    rank = int(np.sum(sv > tol * sv[0]))
    chosen, resid = [], wA.copy()
    for _ in range(rank):
        norms = np.array([np.linalg.norm(resid[:, i]) if i not in chosen else -1
                          for i in range(nC)])
        best = int(np.argmax(norms))
        if norms[best] <= tol * sv[0]:
            break
        chosen.append(best)
        col = resid[:, best] / np.linalg.norm(resid[:, best])
        resid = resid - np.outer(col, col @ resid)
    return chosen, rank


# --------------------------------------------------------------------------
# geodesics (GeoDrop) and the MBAM ride
# --------------------------------------------------------------------------
def _accel(A, Q, weight, c, v, frozen_g0=None):
    """least-squares geodesic acceleration -g^-1 J^T Sigma^-1 (v.d2mu.v)."""
    dmu = gradient(A, Q, c)
    vQv = np.einsum('i,bij,j->b', v, Q, v)
    rhs = (vQv / weight) @ dmu
    g = frozen_g0 if frozen_g0 is not None else \
        (dmu / np.sqrt(weight)[:, None]).T @ (dmu / np.sqrt(weight)[:, None])
    return -np.linalg.solve(g, rhs)


def geodesic(A, Q, weight, v0, wall, h=1e-4, maxsteps=400000, frozen=False):
    """RK4 geodesic from the origin to |c_i| = wall on the given slice.
    Returns dict(wall_index, length, drift, endpoint). frozen=True holds the
    connection at the origin (the flat/eigenvalue answer)."""
    n = A.shape[1]
    g0 = (gradient(A, Q, np.zeros(n)) / np.sqrt(weight)[:, None])
    g0 = g0.T @ g0
    fz = g0 if frozen else None

    def gmat(c):
        dm = gradient(A, Q, c) / np.sqrt(weight)[:, None]
        return dm.T @ dm

    c, v = np.zeros(n), np.array(v0, float) / np.linalg.norm(v0)
    E0 = float(v @ gmat(c) @ v)
    length, drift = 0.0, 0.0
    for step in range(maxsteps):
        k1c, k1v = v, _accel(A, Q, weight, c, v, fz)
        k2c, k2v = v + .5*h*k1v, _accel(A, Q, weight, c + .5*h*k1c, v + .5*h*k1v, fz)
        k3c, k3v = v + .5*h*k2v, _accel(A, Q, weight, c + .5*h*k2c, v + .5*h*k2v, fz)
        k4c, k4v = v + h*k3v, _accel(A, Q, weight, c + h*k3c, v + h*k3v, fz)
        cn = c + (h/6)*(k1c + 2*k2c + 2*k3c + k4c)
        vn = v + (h/6)*(k1v + 2*k2v + 2*k3v + k4v)
        gm = gmat(0.5*(c + cn))
        length += float(np.sqrt(abs((0.5*(v+vn)) @ gm @ (0.5*(v+vn))))) * h
        c, v = cn, vn
        if step % 100 == 0:
            drift = max(drift, abs(float(v @ gmat(c) @ v) - E0)/max(abs(E0), 1e-30))
        hit = np.where(np.abs(c) >= wall)[0]
        if len(hit):
            i = hit[np.argmax(np.abs(c[hit]))]
            return dict(wall=int(i), length=length, drift=drift, c=c)
    return dict(wall=None, length=length, drift=drift, c=c)


def mbam_ride(A, Q, weight, v0, max_frac=1e4, maxsteps=400000):
    """ride the sloppy geodesic with NO wall; record how far |c| gets and
    whether the smallest metric eigenvalue collapses (it does not, in a
    truncated EFT -> MBAM has no boundary)."""
    n = A.shape[1]
    def gmat(c):
        dm = gradient(A, Q, c) / np.sqrt(weight)[:, None]
        return dm.T @ dm
    lam0 = np.linalg.eigvalsh(gmat(np.zeros(n)))[0]
    c, v = np.zeros(n), np.array(v0, float) / np.linalg.norm(v0)
    length, track = 0.0, []
    h = 1e-3
    for step in range(maxsteps):
        a1 = _accel(A, Q, weight, c, v)
        h = min(max(1e-6, 0.1/(np.linalg.norm(a1) + 1e-9)), 0.05)
        cn = c + h*v + 0.5*h*h*a1
        vn = v + h*_accel(A, Q, weight, c, v)
        gm = gmat(0.5*(c+cn))
        length += float(np.sqrt(abs((0.5*(v+vn)) @ gm @ (0.5*(v+vn)))))*h
        c, v = cn, vn
        if step % (20 if step < 2000 else 200) == 0:
            track.append((float(np.abs(c).max()), length,
                          float(np.linalg.eigvalsh(gmat(c))[0]/lam0)))
        if np.abs(c).max() > max_frac:
            break
    return np.array(track)   # columns: max|c|, length, lam_min/lam0


# --------------------------------------------------------------------------
# reduction price (E_inf, worst-case over the box)  and curved intervals
# --------------------------------------------------------------------------
def predict(A, Q, c):
    return A @ c + np.einsum('oij,i,j->o', Q, c, c)


def price_worstcase(A, Q, sigma, relation_family, box, ngrid=5):
    """sup over the box of squared Fisher distance from the full model to the
    reduced family = worst-case Delta chi^2 of imposing the reduction."""
    grid = np.linspace(-box, box, ngrid)
    corners = np.array(np.meshgrid(*[grid]*A.shape[1])).reshape(A.shape[1], -1).T \
        if A.shape[1] <= 3 else None
    worst = 0.0
    pts = corners if corners is not None else \
        (np.random.default_rng(0).uniform(-box, box, (2000, A.shape[1])))
    Rpred = np.array([predict(A, Q, r)/sigma for r in relation_family])
    for p in pts:
        fp = predict(A, Q, p)/sigma
        worst = max(worst, float(((Rpred - fp)**2).sum(1).min()))
    return worst


def curved_intervals(A, Q, sigma, nda, level=1.0):
    """Gaussian (Fisher) vs true (profiled quartic chi^2) 68% intervals for
    each coefficient. Returns per-coefficient dict."""
    nO, nC = A.shape

    def shifts(c):
        return A @ c + np.einsum('oij,i,j->o', Q, c, c)

    def chi2(c):
        return float(np.sum((shifts(c)/sigma)**2))

    def resJ(c):
        return (A + 2*np.einsum('oij,j->oi', Q, c)) / sigma[:, None]

    Aw = A / sigma[:, None]
    F = Aw.T @ Aw
    Finv = np.linalg.pinv(F, rcond=1e-12)
    gauss = np.sqrt(np.abs(np.diag(Finv)))
    constrained = gauss < 0.5 * nda

    def profile(fix_i, fix_val):
        c = np.zeros(nC); c[fix_i] = fix_val
        free = [j for j in range(nC) if j != fix_i]
        for _ in range(60):
            r = shifts(c)/sigma
            step = np.linalg.lstsq(resJ(c)[:, free], -r, rcond=1e-10)[0]
            for a in (1.0, 0.5, 0.25):
                trial = c.copy()
                trial[free] = np.clip(c[free] + a*step, -nda, nda)
                if chi2(trial) <= chi2(c) + 1e-9:
                    c = trial; break
            else:
                break
        return chi2(c)

    out = {}
    for i in range(nC):
        if not constrained[i]:
            out[i] = dict(gauss=gauss[i], curved=None, flat=True)
            continue
        edges = {}
        for sgn in (+1, -1):
            lo, hi = 0.0, sgn*nda
            if profile(i, hi) - level < 0:
                edges[sgn] = (sgn*nda, False)
                continue
            for _ in range(40):
                mid = 0.5*(lo+hi)
                if profile(i, mid) - level < 0:
                    lo = mid
                else:
                    hi = mid
            edges[sgn] = (hi, True)
        out[i] = dict(gauss=gauss[i], curved=(edges[-1][0], edges[+1][0]),
                      bounded=edges[+1][1] and edges[-1][1], flat=False)
    return out
