#!/usr/bin/env python3
"""
When does kappa FIRE?  Well-conditioned 3-coefficient example (no exact
degeneracy, so the twist is numerically clean).

mu measured in bins of a kinematic variable t; three coefficients with three
DISTINCT linear shapes (full rank, one direction softest). The softest
direction is what twist watches. Whether it ROTATES as you move -- and whether
that rotation is integrable -- is set by the dim-6^2 term:

  no curvature                       -> softest dir fixed        -> kappa = 0
  curvature, shape in {t, t^1.5, t^2}-> absorbable by redefining -> kappa small
  curvature, shape = sin(3t)         -> a profile the linear terms can't mimic,
                                        coupling all 3 coeffs     -> kappa FIRES
"""
import numpy as np
import battery_lib as bl

nb = 10
t = np.linspace(0.4, 1.6, nb)
A = np.stack([t, t**1.5, t**2], axis=1)         # 3 distinct shapes -> full rank


def Qcouple(shape, eps=0.8):
    """symmetric dim-6^2 coupling of all three coefficients with profile
    shape(t):  eps * shape * (c1 c2 + c2 c3 + c1 c3)."""
    Q = np.zeros((nb, 3, 3))
    for i in range(3):
        for j in range(i+1, 3):
            Q[:, i, j] = Q[:, j, i] = 0.5*eps*shape
    return Q


box = np.ones(3)
cases = [("flat (Q=0)", np.zeros((nb, 3, 3))),
         ("curvature, shape = t^1.5 (a linear shape, absorbable)", Qcouple(t**1.5)),
         ("curvature, shape = sin(3t) (independent profile)", Qcouple(np.sin(3*t)))]
for name, Q in cases:
    Aw, Qw = bl.prep(A, Q, weight=np.ones(nb))
    r = bl.rank_of(Aw)[0]
    kap = bl.kappa(Aw, Qw, box, m=1, nsamp=60)
    print(f"  kappa = {kap:6.3f}   (rank {r}/3)   {name}")

print("\nSo kappa fires on the LAST one: dim-6^2 curvature that (a) is nonzero,")
print("(b) couples >=3 coefficients, and (c) carries a kinematic profile the")
print("linear terms cannot reproduce -- so the softest direction genuinely")
print("rotates and cannot be straightened by any redefinition of coefficients.")
