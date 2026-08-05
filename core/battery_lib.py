"""
battery_lib -- validated engine for the comprehensive twist/reduction battery.

Sigma-whitened internally: prep() folds the covariance (diagonal sigma^2 OR a
full correlated covariance) into (Aw, Qw) so every downstream quantity uses the
identity metric and E_inf is literally worst-case Delta chi^2. General-n: runs
on toys AND real DY/LEP with real covariances.

CONVENTION (matches smeftgeo / real DY Hessian): mu = A.c + 1/2 c.Q.c,
gradient J = A + Q.c (Q = Hessian, NO factor 2 -- a factor-2 here caused a
spurious flip before, see memory).

Quantities for a whitened model (Aw, Qw, box):
  rank_of(Aw)                linear Fisher rank (gap-aware SVD)
  kappa(Aw,Qw,box,m,coords)  Frobenius twist of the sloppy m-plane, box-graded.
                             MUST pass coords=active_coords(Aw) for real
                             rank-deficient data, else the projector derivative
                             explodes across the blind subspace.
  best_drop / einf_drop      worst-case Delta chi^2 of a single-coeff drop
  einf_relation              worst-case Delta chi^2 of the best curved relation
  exact_kernel_dim(A,Q)      dim of the all-orders exact-blind subspace
Invariants (checked in the smoke test): flat Q=0 -> kappa=0; E_rel <= E_drop;
DY on its slice -> kappa obstructed; LEP with real cov -> kappa obstructed.
"""
import numpy as np
from scipy.optimize import minimize
import smeftgeo as sg


# ---------- predictions, whitening ---------------------------------------
def mu(A, Q, c):
    c = np.atleast_2d(c)
    return c @ A.T + 0.5*np.einsum('oij,ni,nj->no', Q, c, c)


def prep(A, Q, weight=None, cov=None):
    """Sigma-whiten. weight = diagonal sigma^2, OR cov = full covariance.
    Returns (Aw, Qw) with Aw^T Aw = A^T Sigma^-1 A."""
    if cov is not None:
        P = np.linalg.pinv(cov, rcond=1e-12)
        ev, U = np.linalg.eigh(0.5*(P + P.T))
        L = U @ np.diag(np.sqrt(np.clip(ev, 0, None))) @ U.T   # P = L L^T
        return L.T @ A, np.einsum('oO,Oij->oij', L.T, Q)
    s = np.sqrt(weight)
    return A/s[:, None], Q/s[:, None, None]


def rank_of(Aw, tol=1e-8):
    sv = np.linalg.svd(Aw, compute_uv=False)
    return int(np.sum(sv > tol*sv[0])), sv


def active_coords(Aw, tol=1e-8):
    """resolved Warsaw-coordinate subset (pivoted QR on the whitened design)."""
    return sg.active_slice(Aw, np.ones(Aw.shape[0]), tol)[0]


# ---------- kappa (twist) -------------------------------------------------
def kappa(Aw, Qw, box, m=1, nsamp=40, seed=1, coords=None):
    if coords is not None:
        Aw = Aw[:, coords]
        Qw = Qw[np.ix_(range(Qw.shape[0]), coords, coords)]
        box = np.asarray(box)[coords]

    def gfun(c):
        J = Aw + np.einsum('oij,j->oi', Qw, c)
        return J.T @ J
    return sg.BoxTwist(gfun, Aw.shape[1], box).kappa(
        m, nsamp=nsamp, rng=np.random.default_rng(seed))["kappa"]


# ---------- vectorized GN projection -> E_inf -----------------------------
def _proj_chi2(Aw, Qw, targets, seed_th, keep, hp, k, iters=15, clampbox=None):
    n, na = Aw.shape[1], len(keep)

    def assemble(th):
        N = th.shape[0]
        c = np.zeros((N, n)); dc = np.zeros((N, n, na))
        for a, j in enumerate(keep):
            c[:, j] = th[:, a]; dc[:, j, a] = 1.0
        if hp is not None:
            a0, b0, p2, q2, s2 = hp
            ci, cj = th[:, 0], th[:, 1]
            c[:, k] = a0*ci + b0*cj + p2*ci**2 + q2*ci*cj + s2*cj**2
            dc[:, k, 0] = a0 + 2*p2*ci + q2*cj
            dc[:, k, 1] = b0 + q2*ci + 2*s2*cj
        return c, dc
    def chi_of(th):
        c, _ = assemble(th)
        return ((mu(Aw, Qw, c) - targets)**2).sum(1)

    def clamp(th):
        return np.clip(th, -clampbox, clampbox) if clampbox is not None else th

    # MONOTONE Gauss-Newton: a step is accepted only where it lowers chi2
    # (backtracking 1, 1/2, 1/4). Without this, stalled projections inflate
    # distances -- and sup-based prices (E_inf) grab exactly those failures.
    th = clamp(seed_th.copy())
    chi = chi_of(th)
    for _ in range(iters):
        c, dc = assemble(th)
        r = mu(Aw, Qw, c) - targets
        Jc = Aw[None] + np.einsum('oij,nj->noi', Qw, c)
        Jt = np.einsum('noi,nia->noa', Jc, dc)
        JTJ = np.einsum('noa,nob->nab', Jt, Jt)
        lam = 1e-8*np.einsum('naa->n', JTJ)/na + 1e-12
        JTJ = JTJ + lam[:, None, None]*np.eye(na)
        JTr = np.einsum('noa,no->na', Jt, r)
        step = np.linalg.solve(JTJ, JTr[..., None])[..., 0]
        improved = np.zeros(len(chi), bool)
        for alpha in (1.0, 0.5, 0.25):
            trial = clamp(th - alpha*step)
            chit = chi_of(trial)
            take = (~improved) & (chit < chi - 1e-12)
            th[take] = trial[take]; chi[take] = chit[take]
            improved |= take
    return chi


def _boxsamples(box, nsamp, seed):
    return np.random.default_rng(seed).uniform(-1, 1, (nsamp, len(box)))*box


def einf_drop(Aw, Qw, box, k, nbox=1500, seed=0, ret_pt=False):
    B = _boxsamples(box, nbox, seed)
    keep = [j for j in range(Aw.shape[1]) if j != k]
    cb = np.asarray(box)[keep]
    d2 = _proj_chi2(Aw, Qw, mu(Aw, Qw, B), B[:, keep], keep, None, k, clampbox=cb)
    if ret_pt:
        return float(d2.max()), B[int(d2.argmax())]
    return float(d2.max())


def best_drop(Aw, Qw, box, nbox=1500, seed=0):
    ed = [einf_drop(Aw, Qw, box, k, nbox, seed) for k in range(Aw.shape[1])]
    return int(np.argmin(ed)), min(ed), ed


def einf_relation(Aw, Qw, box, k, free2, nbox=1500, seed=0):
    B = _boxsamples(box, nbox, seed)
    keep = list(free2) + [j for j in range(Aw.shape[1])
                          if j != k and j not in free2]
    cb = np.asarray(box)[keep]
    tgt = mu(Aw, Qw, B)
    base = float(_proj_chi2(Aw, Qw, tgt, B[:, keep], keep, None, k,
                            clampbox=cb).max())
    best, bp = base, np.zeros(5)
    for sd in [np.zeros(5), np.array([-1, 0, 0, 0, 0.])]:
        r = minimize(lambda p: float(_proj_chi2(Aw, Qw, tgt, B[:, keep],
                     keep, p, k, clampbox=cb).max()), sd, method='Nelder-Mead',
                     options=dict(xatol=3e-3, fatol=1e-6, maxiter=150))
        if r.fun < best:
            best, bp = float(r.fun), r.x
    return best, bp


# ---------- exact relations (kernel) -- the division of labor -------------
def exact_kernel_dim(A, Q, tol=1e-8):
    v, s = sg.exact_relations(A, Q, tol)
    return len(v), s


# ---------- toy model builder --------------------------------------------
def toy(n=3, nobs=10, curv=0.6, indep=1.0, align=0.5, seed=0):
    """rank-(n-1) misaligned design + curvature carrying an `indep`-independent
    kinematic profile (indep=0 absorbable -> kappa~0; indep=1 fresh shape ->
    fires). curv scales Q; align sets how off-axis the blind direction is."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.4, 1.6, nobs)
    shapes = np.stack([t**p for p in np.linspace(1, 3, n-1)], 1)
    A = np.zeros((nobs, n))
    A[:, :n-1] = shapes
    A[:, n-1] = shapes @ (align*rng.uniform(0.5, 1.5, n-1))
    fresh = np.sin(3*t) + 0.5*np.cos(5*t)
    gsh = indep*fresh + (1-indep)*t
    Q = np.zeros((nobs, n, n))
    for i in range(n):
        for j in range(i, n):
            w = curv*rng.uniform(-1, 1)
            Q[:, i, j] += w*gsh; Q[:, j, i] = Q[:, i, j]
    return A, Q, np.ones(nobs)


def rotate_basis(A, Q, seed=0):
    n = A.shape[1]
    R = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))[0]
    return A @ R, np.einsum('ai,oij,jb->oab', R.T, Q, R), R


def add_mc_noise(A, Q, rel=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return (A*(1 + rel*rng.standard_normal(A.shape)*(A != 0)),
            Q*(1 + rel*rng.standard_normal(Q.shape)*(Q != 0)))
