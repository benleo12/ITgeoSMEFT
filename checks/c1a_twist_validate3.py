#!/usr/bin/env python3
"""
C1a: N-dimensional bracket-twist tool -- validation on the 3-param toys.

Python port of c1a_twist_validate3.wl (Wolfram license down).

Generalizes the scalar Frobenius twist tau = |v . curl v| of
insensitivity_foliation.wl to arbitrary dimension and soft-block size m:
  soft  = span of m sloppiest eigenvectors of M(c) = h^-1 g(c)
  stiff = h-orthogonal complement (leaf tangent of the candidate reduction)
  T(X,Y) = P_soft [X, Y]  for X,Y a frame of the stiff distribution
Clean global relation exists iff T = 0 (Frobenius).

Validation: N=3, m=1 on Models O and X where the scalar tau verdicts are
known from insensitivity_foliation.wl:
  Model X (crossing, alpha=5):  LOW twist  (clean reduction exists)
  Model O (original, eps=5):    HIGH twist (obstructed, ~280x higher)
PASS: (a) pointwise correlation between |T| and tau, (b) same LOW/HIGH
verdicts, (c) verdict stable under h = I vs h = g(0).
"""
import numpy as np

bins1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5]); sigma1 = 0.05
bins2 = np.array([0.5, 0.6, 0.7, 0.8, 0.9]); sigma2 = 0.05
nP = 3


# --- models: predictions and analytic Jacobians -------------------------
# Model O: mu1 = c1 t + c3 t^2 + ep c1 c3 t^2 ; mu2 = c2 t^2 + c3 t + ep c2 c3 t^2
# Model X: mu1 = c1 t + c3 t^2 + al c1^2 t^2  ; mu2 = c2 t^2 + c3 t + al c2^2 t
def jac_O(c, ep=5.0):
    c1, c2, c3 = c
    J1 = np.stack([bins1 + ep*c3*bins1**2,
                   np.zeros_like(bins1),
                   bins1**2 + ep*c1*bins1**2], axis=1)
    J2 = np.stack([np.zeros_like(bins2),
                   bins2**2 + ep*c3*bins2**2,
                   bins2 + ep*c2*bins2**2], axis=1)
    return J1, J2


def jac_X(c, al=5.0):
    c1, c2, c3 = c
    J1 = np.stack([bins1 + 2*al*c1*bins1**2,
                   np.zeros_like(bins1),
                   bins1**2], axis=1)
    J2 = np.stack([np.zeros_like(bins2),
                   bins2**2 + 2*al*c2*bins2,
                   bins2], axis=1)
    return J1, J2


def fisher(jac, c):
    J1, J2 = jac(c)
    return J1.T @ J1 / sigma1**2 + J2.T @ J2 / sigma2**2


# --- soft/stiff split relative to reference metric h ---------------------
def make_whitener(h):
    L = np.linalg.cholesky(h)          # h = L L^T
    Linv = np.linalg.inv(L)
    return Linv, Linv.T                # Linv, LinvT


def soft_stiff(g, Linv, LinvT, m):
    Msym = Linv @ g @ LinvT
    Msym = 0.5*(Msym + Msym.T)
    ev, ew = np.linalg.eigh(Msym)      # ascending
    vecs = (LinvT @ ew).T              # rows = h-orthonormal eigvecs in c-space
    return ev, vecs[:m], vecs[m:]      # evals, soft, stiff


def projectors(g, Linv, LinvT, h, m):
    """h-orthogonal projectors onto soft / stiff subspaces.
    Built from the SUBSPACE (sum over block eigenvectors), so smooth across
    intra-block eigenvalue crossings -- the Riesz-projector property. The
    projector in c-coordinates: P = sum_k v_k (v_k^T h), h-self-adjoint."""
    _, soft, stiff = soft_stiff(g, Linv, LinvT, m)
    Psoft = sum(np.outer(v, h @ v) for v in soft)
    Pstiff = sum(np.outer(v, h @ v) for v in stiff)
    return Psoft, Pstiff


def align(frame, ref):
    """order/sign-align frame rows to reference rows by greedy overlap"""
    n = len(frame)
    used, out = set(), []
    for a in range(n):
        best, bval, sgn = -1, -1.0, 1.0
        for k in range(n):
            if k in used:
                continue
            ov = float(ref[a] @ frame[k])
            if abs(ov) > bval:
                bval, best, sgn = abs(ov), k, np.sign(ov) or 1.0
        used.add(best)
        out.append(sgn * frame[best])
    return np.array(out)


# --- analysis on a grid ---------------------------------------------------
gridN, gridM = 8, 0.9


def analyze(label, jac, hchoice):
    axis = np.linspace(-gridM, gridM, gridN + 1)
    hstep = axis[1] - axis[0]
    npts = gridN + 1

    g0 = fisher(jac, np.zeros(3))
    h = np.eye(3) if hchoice == "identity" else g0
    Linv, LinvT = make_whitener(h)

    _, softRef, stiffRef = soft_stiff(g0, Linv, LinvT, 1)

    soft = np.zeros((npts, npts, npts, 1, 3))
    stiff = np.zeros((npts, npts, npts, 2, 3))
    Pst = np.zeros((npts, npts, npts, 3, 3))    # stiff projector (smooth)
    Psf = np.zeros((npts, npts, npts, 3, 3))    # soft projector
    for i in range(npts):
        for j in range(npts):
            for k in range(npts):
                c = np.array([axis[i], axis[j], axis[k]])
                g = fisher(jac, c)
                _, s, st = soft_stiff(g, Linv, LinvT, 1)
                soft[i, j, k] = align(s, softRef)
                stiff[i, j, k] = align(st, stiffRef)
                Psf[i, j, k], Pst[i, j, k] = projectors(g, Linv, LinvT, h, 1)

    tau_list, T_list = [], []
    for i in range(1, npts - 1):
        for j in range(1, npts - 1):
            for k in range(1, npts - 1):
                # scalar tau = |v . curl v| with Euclidean-normalized v
                vn = lambda ii, jj, kk: soft[ii, jj, kk, 0] / \
                    np.linalg.norm(soft[ii, jj, kk, 0])
                dVx = (vn(i+1, j, k) - vn(i-1, j, k)) / (2*hstep)
                dVy = (vn(i, j+1, k) - vn(i, j-1, k)) / (2*hstep)
                dVz = (vn(i, j, k+1) - vn(i, j, k-1)) / (2*hstep)
                crl = np.array([dVy[2]-dVz[1], dVz[0]-dVx[2], dVx[1]-dVy[0]])
                tau_list.append(abs(float(vn(i, j, k) @ crl)))

                # PROJECTOR-BASED bracket tensor (smooth across intra-block
                # crossings): vector fields X = P(c) a with a fixed at the
                # central-point stiff frame. Then
                #   [X,Y] = (Pa . grad P) b - (Pb . grad P) a
                # and T(a,b) = P_soft [X,Y]. Frame fields themselves are
                # never differentiated -- only the projector matrix P(c).
                dP = np.stack([
                    (Pst[i+1, j, k]-Pst[i-1, j, k]) / (2*hstep),
                    (Pst[i, j+1, k]-Pst[i, j-1, k]) / (2*hstep),
                    (Pst[i, j, k+1]-Pst[i, j, k-1]) / (2*hstep)])
                a, b = stiff[i, j, k, 0], stiff[i, j, k, 1]
                Xc = Pst[i, j, k] @ a      # = a (already stiff)
                Yc = Pst[i, j, k] @ b
                gradP_alongX = np.einsum('d,dij->ij', Xc, dP)
                gradP_alongY = np.einsum('d,dij->ij', Yc, dP)
                brkt = gradP_alongX @ Yc - gradP_alongY @ Xc
                Tvec = Psf[i, j, k] @ brkt
                # h-norm of the escape component
                T_list.append(float(np.sqrt(abs(Tvec @ h @ Tvec))))

    tau = np.array(tau_list); T = np.array(T_list)
    char = 1.0 / (2*gridM)
    corr = np.corrcoef(tau, T)[0, 1]
    vt = "LOW (clean relation exists)" if np.median(tau) < 0.15*char \
        else "HIGH (obstructed)"
    vT = "LOW (clean relation exists)" if np.median(T) < 0.15*char \
        else "HIGH (obstructed)"

    print(f"=== {label}   [h = {hchoice}] ===")
    print(f"  tau (old scalar): median = {np.median(tau):.3e}   "
          f"max = {tau.max():.3e}   verdict: {vt}")
    print(f"  |T| (bracket):    median = {np.median(T):.3e}   "
          f"max = {T.max():.3e}   verdict: {vT}")
    print(f"  pointwise correlation(tau, |T|) = {corr:.3f}")
    print(f"  median ratio |T|/tau = {np.median(T/(tau+1e-30)):.3f}\n")
    return np.median(tau), np.median(T), vt, vT


print("#" * 64)
print("  C1a: bracket-tensor twist vs scalar twist, 3-param toys")
print("#" * 64 + "\n")

tXI, TXI, vtXI, vTXI = analyze("MODEL X (crossing, alpha=5)", jac_X, "identity")
tOI, TOI, vtOI, vTOI = analyze("MODEL O (original, eps=5)", jac_O, "identity")
tXG, TXG, vtXG, vTXG = analyze("MODEL X (crossing, alpha=5)", jac_X, "g0")
tOG, TOG, vtOG, vTOG = analyze("MODEL O (original, eps=5)", jac_O, "g0")

print("#" * 64)
print("  C1a VERDICT")
print("#" * 64)
print(f"Model X: tau = {vtXI} | T(h=I) = {vTXI} | T(h=g0) = {vTXG}")
print(f"Model O: tau = {vtOI} | T(h=I) = {vTOI} | T(h=g0) = {vTOG}")
print(f"O/X twist ratio: tau = {tOI/max(tXI,1e-30):.1f}   "
      f"T(h=I) = {TOI/max(TXI,1e-30):.1f}   T(h=g0) = {TOG/max(TXG,1e-30):.1f}")
print("PASS criteria: X=LOW & O=HIGH everywhere; correlation > 0.7; "
      "O/X ratio >> 1")
