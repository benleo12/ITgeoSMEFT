#!/usr/bin/env python3
"""
Does kappa detect a degeneracy in a FLAT model?  No -- and that is correct.

User's model:  mu_b = c1 t + c2 t^2 + (c3+c4) t^3   (linear in every c_i => flat)
It has an exact hidden relation: c3 and c4 enter only as (c3+c4), so c3-c4 is
blind. Question: shouldn't kappa measure that?

Division of labor being tested:
  RANK / kernel of A  -> DETECTS the degeneracy (returns c3-c4).  Works flat.
  kappa (twist)       -> asks whether the blind direction knits into a GLOBAL
                         relation or only local drops. It is the c-dependence
                         of the blind subspace, so for a FLAT (constant) metric
                         it is IDENTICALLY 0 -- there is no knob that changes
                         that. kappa=0 is the correct verdict "relation is
                         clean & global", which a flat linear kernel always is.
Then: turn on ONE curvature knob (eps*c3*c4*t^3) and watch kappa leave 0.
"""
import numpy as np
import smeftgeo as sg

nb = 8
t = np.linspace(0.4, 1.6, nb)

# ---- FLAT model: A has columns [t, t^2, t^3, t^3]; Q = 0 -----------------
A = np.stack([t, t**2, t**3, t**3], axis=1)          # (8,4)
Q0 = np.zeros((nb, 4, 4))

vals, vecs, rank = sg.spectrum(A.T @ A)
print("FLAT model  mu = c1 t + c2 t^2 + (c3+c4) t^3")
print(f"  Fisher rank = {rank}/4   (one blind direction)")
print("  degeneracy from the KERNEL of A (null_relations):")
for v, nnz in sg.null_relations(A)[:2]:
    print(f"    {np.round(v, 3)}   -> this is c3 - c4  (blind)")


def kappa(Q):
    def gfun(c):
        J = A + 2*np.einsum('bij,j->bi', Q, c)
        return J.T @ J
    return sg.BoxTwist(gfun, 4, np.ones(4)).kappa(
        1, nsamp=40, rng=np.random.default_rng(1))["kappa"]


print(f"  kappa (twist)      = {kappa(Q0):.6e}   <- IDENTICALLY 0 for flat")
print("  => kappa did NOT find the relation. The KERNEL did. kappa=0 just")
print("     certifies the relation is clean/global (it is: exact & linear).")

# ---- curvature with the SAME shape (t^3) -> STILL integrable, kappa=0 -----
print("\ncurvature SAME shape  mu += eps * c3 c4 t^3  (factors through one")
print("  combo W=(c3+c4)+eps*c3c4 -> curved relation survives):")
for eps in [0.0, 1.0, 4.0]:
    Q = np.zeros((nb, 4, 4)); Q[:, 2, 3] = Q[:, 3, 2] = 0.5*eps*t**3
    print(f"  eps={eps:4.1f}  kappa={kappa(Q):.6f}   (integrable -> 0)")

# ---- curvature with an INDEPENDENT shape g=sin(3t): NON-integrable --------
# (shape must NOT be t, t^2 or t^3 -- those get absorbed into the linear terms)
g = np.sin(3*t)
print("\ncurvature INDEP shape  mu += eps * c1(c3-c4) sin(3t)  (a kinematic")
print("  profile independent of the linear terms -> cannot be absorbed):")
for eps in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
    Q = np.zeros((nb, 4, 4))
    Q[:, 0, 2] = Q[:, 2, 0] = 0.5*eps*g             # +c1 c3, shape sin(3t)
    Q[:, 0, 3] = Q[:, 3, 0] = -0.5*eps*g            # -c1 c4, shape sin(3t)
    print(f"  eps={eps:4.1f}  kappa={kappa(Q):.6f}")
print("  => kappa fires ONLY when curvature enters with kinematics that break")
print("     the factorization. THAT is twist: not 'is there a degeneracy'")
print("     (rank/kernel does that), but 'does the degeneracy survive as ONE")
print("     global relation once curvature bends it, or fall apart to drops'.")
