#!/usr/bin/env python3
"""
C3 + C4: exact-optimum battery on a dial family, and the phase map.

Dial family (n=2 coefficients, 2 observables + curvature probe):
  mu(c) = A c + kap_c * (c1*c2) * e_q
  A = diag(sqrt(lam)) R(theta)^T  with lam = (1, rho^2):
    rho   = width ratio (sloppy/stiff), 0 < rho <= 1
    theta = angle of the SLOPPY eigenvector from the c2-axis
            (theta = 0: sloppy = c2 -> dropping c2 is optimal;
             theta = 45 deg: maximally ambiguous)
  e_q = extra observable direction carrying the quadratic term
        (kap_c = 0: exactly flat).

Objective for every method: E_inf(R) = sup_{c in B} min_{r in R}
  ||mu(c) - mu(r)||^2,  B = [-1,1]^2  (Sigma = identity; A carries weights).

Methods scored:
  drop-c1, drop-c2         (L0, exact scan)
  eigen-drop               (drop = largest |component| of sloppy eigvec)
  GeoDrop                  (drop = wall hit by geodesic along sloppy dir)
  best rational relation   (u.c = 0, |u_i| <= 5 integers)     (L1)
  best curved relation     (c_i = a c_j + b c_j^2, optimized)  (L2)
  PCA / optimal subspace   (continuous 1-dim subspace optimum) (L3 floor)

Flat-case closed-form anchors (kap_c = 0):
  E(drop k)     = Schur complement S_kk * 1          (worst |c_k| = 1)
  E(optimal L3) = lam_soft * (|v_s1| + |v_s2|)^2     (worst box corner)

Noncentrality closure: E[chi2_red - chi2_full] = 1 + d^2(c*, R) for
nested reductions with 1 constraint, verified by pseudodata replicas.
"""
import numpy as np
import json
from itertools import product

rng_global = np.random.default_rng(7)


def make_model(theta, rho, kapc):
    """returns mu(c) -> vector of 3 observables, and A (2x2 linear part)"""
    ct, st = np.cos(theta), np.sin(theta)
    # sloppy eigenvector at angle theta from c2-axis: v_s = (st, ct)
    # stiff: (ct, -st). R columns = eigvecs: [stiff, sloppy]
    R = np.array([[ct, st], [-st, ct]])
    A2 = np.diag([1.0, rho]) @ R.T            # 2x2: rows = observables
    A = np.vstack([A2, np.zeros(2)])          # 3rd observable: quadratic only

    def mu(C):
        """C: (...,2) -> (...,3)"""
        C = np.asarray(C, float)
        lin = C @ A2.T
        quad = kapc * (C[..., 0]*C[..., 1])[..., None]
        return np.concatenate([lin, quad], axis=-1)
    return mu, A2


def einf_family(mu, Rpoints, boxgrid):
    """E_inf: sup over boxgrid of min over Rpoints of ||mu(c)-mu(r)||^2"""
    MU_B = mu(boxgrid)                      # (NB,3)
    MU_R = mu(Rpoints)                      # (NR,3)
    d2 = ((MU_B[:, None, :] - MU_R[None, :, :])**2).sum(-1)   # (NB,NR)
    return float(d2.min(axis=1).max())


def boxgrid2(n=41):
    x = np.linspace(-1, 1, n)
    return np.stack(np.meshgrid(x, x), -1).reshape(-1, 2)


BOX = boxgrid2(41)
TLINE = np.linspace(-1, 1, 401)


def R_drop(k):
    """drop coefficient k: family = {c_k = 0}"""
    P = np.zeros((len(TLINE), 2))
    P[:, 1-k] = TLINE
    return P


def R_linear(u):
    """relation u.c = 0: the line through origin orthogonal to u"""
    w = np.array([-u[1], u[0]], float)
    w = w/np.linalg.norm(w)
    return TLINE[:, None]*w[None, :]


def R_curved(a, b, j):
    """c_i = a c_j + b c_j^2 (i = 1-j)"""
    P = np.zeros((len(TLINE), 2))
    P[:, j] = TLINE
    P[:, 1-j] = a*TLINE + b*TLINE**2
    return P


def sloppy_vec(A2):
    g = A2.T @ A2
    ev, ew = np.linalg.eigh(g)
    return ew[:, 0], ev            # sloppy eigvec, eigenvalues ascending


def geodrop_decision(mu, A2, kapc, theta, rho):
    """shoot geodesic along +/- sloppy dir to |c_k|=1 wall; drop = wall
    of the shorter proper length (min over signs)."""
    # metric and Christoffel from mu numerically
    def g(c):
        eps = 1e-6
        J = np.zeros((3, 2))
        for i in range(2):
            e = np.zeros(2)
            e[i] = eps
            J[:, i] = (mu(c+e)-mu(c-e))/(2*eps)
        return J.T @ J

    def acc(c, v):
        eps = 1e-5
        # acc = -g^-1 sum_o (v.d2mu_o.v) dmu_o  (least-squares geodesic)
        J = np.zeros((3, 2))
        Hvv = np.zeros(3)
        for i in range(2):
            e = np.zeros(2)
            e[i] = eps
            J[:, i] = (mu(c+e)-mu(c-e))/(2*eps)
        # d2mu contracted with v twice via directional second difference
        Hvv = (mu(c+eps*v) - 2*mu(c) + mu(c-eps*v))/eps**2
        return -np.linalg.solve(g(c), J.T @ Hvv)

    vs, _ = sloppy_vec(A2)
    best = (None, np.inf)
    for sgn in (+1, -1):
        c, v = np.zeros(2), sgn*vs.copy()
        L, h = 0.0, 5e-3
        for _ in range(4000):
            a1 = acc(c, v)
            cn = c + h*v + 0.5*h*h*a1
            vn = v + h*a1
            gm = g(0.5*(c+cn))
            L += float(np.sqrt(max((0.5*(v+vn)) @ gm @ (0.5*(v+vn)), 0)))*h
            c, v = cn, vn
            if np.abs(c).max() >= 1.0:
                k = int(np.argmax(np.abs(c)))
                if L < best[1]:
                    best = (k, L)
                break
    return best[0]


def run_cell(theta, rho, kapc):
    mu, A2 = make_model(theta, rho, kapc)
    vs, ev = sloppy_vec(A2)
    g = A2.T @ A2

    out = {}
    # L0 drops
    e0 = einf_family(mu, R_drop(0), BOX)
    e1 = einf_family(mu, R_drop(1), BOX)
    out["drop_c1"], out["drop_c2"] = e0, e1
    out["drop_best"] = min(e0, e1)
    out["drop_best_k"] = int(np.argmin([e0, e1]))
    # eigen decision
    k_eig = int(np.argmax(np.abs(vs)))
    out["eigen_drop_k"] = k_eig
    out["eigen_drop"] = [e0, e1][k_eig]
    # GeoDrop decision
    k_geo = geodrop_decision(mu, A2, kapc, theta, rho)
    out["geodrop_k"] = k_geo
    out["geodrop"] = [e0, e1][k_geo] if k_geo is not None else np.nan
    # L1 rational relations
    bestL1 = np.inf
    bestu = None
    for u1, u2 in product(range(0, 6), range(-5, 6)):
        if u1 == 0 and u2 <= 0:
            continue
        if np.gcd(u1, abs(u2)) > 1:
            continue
        e = einf_family(mu, R_linear((u1, u2)), BOX)
        if e < bestL1:
            bestL1, bestu = e, (u1, u2)
    out["L1_rational"], out["L1_u"] = bestL1, bestu
    # L3 continuous subspace optimum (scan direction angle)
    bestL3 = np.inf
    for phi in np.linspace(0, np.pi, 181):
        w = np.array([np.cos(phi), np.sin(phi)])
        e = einf_family(mu, TLINE[:, None]*w[None, :], BOX)
        bestL3 = min(bestL3, e)
    out["L3_subspace"] = bestL3
    # L2 curved relation (only meaningful with curvature): coarse-to-fine
    j = 1 - out["drop_best_k"]          # keep the better-kept coordinate
    bestL2, bestab = np.inf, (0, 0)
    for a in np.linspace(-1, 1, 21):
        for b in np.linspace(-1, 1, 21):
            e = einf_family(mu, R_curved(a, b, j), BOX)
            if e < bestL2:
                bestL2, bestab = e, (a, b)
    a0, b0 = bestab
    for a in np.linspace(a0-0.1, a0+0.1, 11):
        for b in np.linspace(b0-0.1, b0+0.1, 11):
            e = einf_family(mu, R_curved(a, b, j), BOX)
            if e < bestL2:
                bestL2, bestab = e, (a, b)
    out["L2_curved"], out["L2_ab"] = bestL2, bestab

    # flat-case closed-form anchors
    if kapc == 0:
        S = [g[0, 0]-g[0, 1]**2/g[1, 1], g[1, 1]-g[0, 1]**2/g[0, 0]]
        out["anchor_drop"] = [S[0], S[1]]
        corner = (np.abs(vs[0])+np.abs(vs[1]))**2
        out["anchor_L3"] = ev[0]*corner
    return out


# ==================================================================== #
# PHASE 1: flat verification + theta_crit boundary                     #
# ==================================================================== #
print("=" * 70)
print("PHASE 1: flat battery (kap_c = 0), closed-form anchors, theorems")
print("=" * 70)
thetas = np.radians([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
rhos = [0.02, 0.05, 0.1, 0.2, 0.5]
epsilon = 0.10
flat = {}
anchor_fail = 0
geo_neq_opt = 0
for th in thetas:
    for rho in rhos:
        cell = run_cell(th, rho, 0.0)
        flat[(round(np.degrees(th)), rho)] = cell
        # anchor checks: closed form assumes UNCONSTRAINED reduced parameter,
        # so it is a lower bound on the boxed-family numeric value
        lb_ok = (cell["drop_c1"] >= cell["anchor_drop"][0]*(1-0.02) and
                 cell["drop_c2"] >= cell["anchor_drop"][1]*(1-0.02))
        if not lb_ok:
            anchor_fail += 1
        if cell["geodrop_k"] != cell["drop_best_k"]:
            geo_neq_opt += 1

print(f"closed-form drop anchors (lower bound respected): "
      f"{len(thetas)*len(rhos)-anchor_fail}/{len(thetas)*len(rhos)}")
print(f"GeoDrop = optimal drop in flat case: "
      f"{len(thetas)*len(rhos)-geo_neq_opt}/{len(thetas)*len(rhos)}")

print(f"\ntheta_crit test (drop within {epsilon:.0%} of L3 floor):")
print(f"{'rho':6} {'empirical theta_crit':22} {'arctan(sqrt(eps))':18}")
for rho in rhos:
    tc = None
    for th in thetas:
        cell = flat[(round(np.degrees(th)), rho)]
        if cell["drop_best"] > (1+epsilon)*max(cell["L3_subspace"], 1e-15):
            tc = np.degrees(th)
            break
    print(f"{rho:<6} {str(tc)+' deg':22} "
          f"{np.degrees(np.arctan(np.sqrt(epsilon))):.1f} deg")

print("\nflat lattice collapse check (L1 rational ~ L3 floor):")
for rho in [0.02, 0.2]:
    for thd in [0, 20, 45]:
        cell = flat[(thd, rho)]
        r = cell["L1_rational"]/max(cell["L3_subspace"], 1e-15)
        print(f"  theta={thd:2d} rho={rho}: L1/L3 = {r:.3f}  (u={cell['L1_u']})")

# ==================================================================== #
# PHASE 2: noncentrality closure                                       #
# ==================================================================== #
print("\n" + "=" * 70)
print("PHASE 2: pseudodata noncentrality closure")
print("=" * 70)
# Flat case: exact unconstrained least-squares fits (the theorem's setting:
# E[chi2_red - chi2_full] = Delta_dof + d^2 = 1 + d^2 for nested linear fits)
for (thd, rho) in [(20, 0.1), (35, 0.2), (40, 0.05)]:
    mu, A2 = make_model(np.radians(thd), rho, 0.0)
    A3 = np.vstack([A2, np.zeros(2)])          # 3-obs linear map
    cstar = np.array([0.9, -0.95])             # near corner: larger d^2
    a1 = A3[:, 0]                              # reduced model: c2 = 0
    # exact d^2: unconstrained projections
    Pfull = A3 @ np.linalg.pinv(A3)
    Pred = np.outer(a1, a1)/(a1 @ a1)
    mustar = mu(cstar)
    d2 = float(mustar @ (Pfull - Pred) @ mustar)
    nrep = 4000
    rng = np.random.default_rng(11)
    Y = mustar[None, :] + rng.normal(size=(nrep, 3))
    chif = ((Y - Y @ Pfull.T)**2).sum(-1)
    chir = ((Y - Y @ Pred.T)**2).sum(-1)
    m = float(np.mean(chir - chif))
    se = float(np.std(chir - chif)/np.sqrt(nrep))
    ok = abs(m - (1 + d2)) < 3*se + 0.02*(1 + d2)
    print(f"  theta={thd}, rho={rho} (flat, exact fits): predicted 1+d^2 = "
          f"{1+d2:.3f}   measured = {m:.3f} +- {se:.3f}   "
          f"({'PASS' if ok else 'FAIL'})")

# ==================================================================== #
# PHASE 3: curvature panels                                            #
# ==================================================================== #
print("\n" + "=" * 70)
print("PHASE 3: curvature panels (kap_c > 0)")
print("=" * 70)
kap_panels = [0.0, 0.5, 1.0, 2.0]
curved = {}
print(f"{'kapc':5} {'theta':6} {'rho':5} | {'dropB':9} {'L1':9} "
      f"{'L2':9} {'L3':9} | {'geo=eig':8} {'L2 gain':8}")
for kapc in kap_panels:
    for thd in [10, 30, 45]:
        for rho in [0.05, 0.2]:
            cell = run_cell(np.radians(thd), rho, kapc)
            curved[(kapc, thd, rho)] = cell
            gain = cell["L1_rational"]/max(cell["L2_curved"], 1e-15)
            print(f"{kapc:<5} {thd:<6} {rho:<5} | "
                  f"{cell['drop_best']:<9.3g} {cell['L1_rational']:<9.3g} "
                  f"{cell['L2_curved']:<9.3g} {cell['L3_subspace']:<9.3g} | "
                  f"{str(cell['geodrop_k']==cell['eigen_drop_k']):8} "
                  f"{gain:<8.2f}")

# ==================================================================== #
# PHASE 4: the phase map (which lattice level suffices, per cell)      #
# ==================================================================== #
print("\n" + "=" * 70)
print("PHASE 4: phase map -- lowest lattice level within 10% of L3 floor")
print("  D = drop suffices, R = linear relation needed, C = curved needed")
print("=" * 70)
maps = {}
for kapc in [0.0, 1.0]:
    print(f"\nkap_c = {kapc}:   theta ->")
    hdr = "rho    " + " ".join(f"{int(round(np.degrees(t))):>4d}" for t in thetas)
    print(hdr)
    for rho in rhos:
        row = []
        for th in thetas:
            key = (round(np.degrees(th)), rho)
            cell = flat[key] if kapc == 0.0 else run_cell(th, rho, kapc)
            floor = max(min(cell["L3_subspace"], cell["L2_curved"]), 1e-15)
            if cell["drop_best"] <= (1+epsilon)*floor:
                tag = "D"
            elif cell["L1_rational"] <= (1+epsilon)*floor:
                tag = "R"
            else:
                tag = "C"
            row.append(tag)
            maps[(kapc, round(np.degrees(th)), rho)] = tag
        print(f"{rho:<6} " + " ".join(f"{t:>4}" for t in row))

# place the real datasets
print("\nreal-data placement (theta = angle of sloppy dir from nearest axis,")
print(" rho = soft width ratio within the soft pair):")
print("  DY 95-600 : sloppy max|comp| = 0.67 -> theta ~ 48 deg;"
      " widths 2.9e-4/1.6e-2 -> rho ~ 0.018 -> deep R/C zone BUT")
print("    kappa-obstructed -> relations unavailable -> drop with price tag")
print("  LEP       : kernel exact (rho -> 0 at linear order): relations exact")
print("    by construction; theta undefined for exact kernel (any basis)")

json.dump({str(k): v for k, v in maps.items()},
          open("/Users/user/Downloads/c4_phasemap.json", "w"))
print("\nphase map saved -> c4_phasemap.json")
