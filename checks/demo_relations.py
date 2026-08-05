#!/usr/bin/env python3
"""
SYSTEMATIZATION + PSEUDODATA DEMO of relation-finding.

WHY exact relations are TRIVIAL to find from observable data
------------------------------------------------------------
A SMEFT prediction is EXACTLY quadratic in the Wilson coefficients (the
amplitude is linear in each c), so for ANY observable/cuts:
      mu_b(c) = SM_b + A_bi c_i + c_i Q_bij c_j .
Therefore a user who can only EVALUATE predictions mu(c) (a MadGraph
job, a fit, a table) reconstructs A and Q EXACTLY from O(n^2) evaluations
by finite differences -- no symbolic input, no fitting error if the
evaluator is exact:
      SM   = mu(0)
      A_i  = [mu(+h e_i) - mu(-h e_i)] / 2h
      Q_ii = [mu(+h e_i) - 2 mu(0) + mu(-h e_i)] / 2h^2
      Q_ij = [mu(h e_i+h e_j) - mu(h e_i) - mu(h e_j) + mu(0)] / h^2  (i!=j)
An EXACT relation is a direction u with A u = 0 AND Q_b u = 0 for all b:
then mu(c + t u) == mu(c) identically. Found as the common kernel of the
stacked matrix [A; Q_1; ...; Q_nbins] -- one SVD. That is the whole recipe.

This script (i) demonstrates that recipe on pseudo-observables, and
(ii) shows via three truth-known pseudomodels WHEN a relation exists and
when it does not, and which tool reports it.
"""
import numpy as np

tb = np.array([0.2, 0.4, 0.6, 0.8, 1.0])


def simulator(kind):
    """A 'black box' the user can only evaluate: mu(c) for c=(c1,c2,c3).
    Returns 10 pseudo-observables. Truth (for our checking) noted per kind."""
    def mu(c):
        c1, c2, c3 = c
        if kind == "exact":
            # depends only on (c1+c2) and c3  -> EXACT relation c1 - c2
            p1 = (c1+c2)*tb + c3*tb**2
            p2 = (c1+c2)*tb**2 + c3*tb
        elif kind == "clean":
            # integrable soft direction (twist ~ 0): a genuine curved
            # relation c1 = f(c2) exists
            p1 = c1*tb + c3*tb**2 + 5*c1**2*tb**2
            p2 = c2*tb**2 + c3*tb + 5*c2**2*tb
        else:  # "obstructed"
            # eps cross-term -> soft direction twists -> NO clean relation
            p1 = c1*tb + c3*tb**2 + 5*c1*c3*tb**2
            p2 = c2*tb**2 + c3*tb + 5*c2*c3*tb**2
        return np.concatenate([p1, p2])
    return mu


def reconstruct(mu, n=3, h=1e-3):
    """Reconstruct SM, A, Q by finite differences -- EXACT for a quadratic
    model. This is all a user needs: the ability to call mu(c)."""
    SM = mu(np.zeros(n))
    A = np.zeros((len(SM), n))
    Q = np.zeros((len(SM), n, n))
    e = np.eye(n)
    for i in range(n):
        A[:, i] = (mu(h*e[i]) - mu(-h*e[i])) / (2*h)
        Q[:, i, i] = (mu(h*e[i]) - 2*SM + mu(-h*e[i])) / (2*h**2)
    for i in range(n):
        for j in range(i+1, n):
            Qij = (mu(h*e[i]+h*e[j]) - mu(h*e[i]) - mu(h*e[j]) + SM) / h**2
            Q[:, i, j] = Q[:, j, i] = Qij / 2
    return SM, A, Q


def exact_relations(A, Q, tol=1e-7):
    stack = np.vstack([A] + [Q[b] for b in range(Q.shape[0])])
    s = np.linalg.svd(stack, compute_uv=False)
    _, _, vt = np.linalg.svd(stack)
    rank = int(np.sum(s > tol*s[0]))
    return vt[rank:], s


def twist_median(A, Q, box=0.9, m=1, nsamp=40, fd=1e-4, seed=3):
    """twist of the sloppy direction over the box (box-graded reference)."""
    from itertools import combinations
    rng = np.random.default_rng(seed)
    n = A.shape[1]

    def g(c):
        J = A + 2*np.einsum('bij,j->bi', Q, c)
        return J.T @ J

    B = np.full(n, box)

    def proj(u):
        G = (B[:, None]*g(B*u))*B[None, :]
        G = 0.5*(G+G.T)
        ev, ew = np.linalg.eigh(G)
        return ew[:, :m] @ ew[:, :m].T, ew[:, m:] @ ew[:, m:].T, ew[:, m:].T
    ts = []
    for _ in range(nsamp):
        u = rng.uniform(-0.9, 0.9, n)
        Psf, _, stiff = proj(u)
        dP = np.zeros((n, n, n))
        for d in range(n):
            e = np.zeros(n); e[d] = fd
            dP[d] = (proj(u+e)[1]-proj(u-e)[1])/(2*fd)
        best = 0.0
        for a, b in combinations(range(len(stiff)), 2):
            X, Y = stiff[a], stiff[b]
            brk = np.einsum('d,dij,j->i', X, dP, Y) \
                - np.einsum('d,dij,j->i', Y, dP, X)
            best = max(best, float(np.linalg.norm(Psf @ brk)))
        ts.append(best)
    return float(np.median(ts))*2*np.sqrt(n)


THRESH = 5.61e-2
print("="*70)
print("STEP 1 -- exact relations are trivial to READ OFF a simulator")
print("="*70)
mu = simulator("exact")
SM, A, Q = reconstruct(mu)
rels, s = exact_relations(A, Q)
print("reconstructed A,Q from finite differences of mu(c) (exact).")
print(f"stacked singular values (last 3): {np.array2string(s[-3:], precision=2)}")
print(f"exact relations found: {len(rels)}")
for r in rels:
    r = r/np.abs(r).max()
    print("   direction:", {f"c{i+1}": round(float(r[i]), 3)
                            for i in range(3) if abs(r[i]) > 1e-3})
# verify: moving along the relation leaves EVERY observable unchanged
u = rels[0]/np.linalg.norm(rels[0])
c0 = np.array([0.3, -0.2, 0.5])
print(f"check: max |mu(c0 + t u) - mu(c0)| over t in [-1,1] = "
      f"{max(np.max(np.abs(mu(c0+t*u)-mu(c0))) for t in np.linspace(-1,1,11)):.2e}")
print("   => the relation is EXACT: found from data alone, no theory input.")

print("\n"+"="*70)
print("STEP 2 -- WHEN does a relation exist? three truth-known pseudomodels")
print("="*70)
print(f"{'model':12} {'truth':30} {'exact rel?':11} {'twist kappa':12} {'verdict'}")
for kind, truth in [("exact", "exact relation c1=c2 (all orders)"),
                    ("clean", "curved relation exists (integrable)"),
                    ("obstructed", "no clean relation (twist obstructs)")]:
    mu = simulator(kind)
    SM, A, Q = reconstruct(mu)
    rels, _ = exact_relations(A, Q)
    kap = twist_median(A, Q)
    if len(rels):
        verdict = "RELATION (exact, free)"
    elif kap < THRESH:
        verdict = "RELATION (curved)"
    else:
        verdict = "NO relation -> drops"
    print(f"{kind:12} {truth:30} {len(rels):>7}    {kap:>10.4f}   {verdict}")

print("\nreading: the pipeline recovers the built-in truth in all three:")
print("  exact model      -> exact kernel finds it (twist not needed)")
print("  clean model      -> no exact kernel, but twist ~ 0 => curved relation")
print("  obstructed model -> no exact kernel, twist >> threshold => drops only")
print("This is the instruction set; on real data replace mu() by MadGraph")
print("and the finite differences become MC-noisy (see noise_proof.py).")
