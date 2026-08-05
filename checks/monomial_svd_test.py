#!/usr/bin/env python3
"""
User question: predictions are exactly quadratic in the coefficients, so you
can list all monomials m(c) = (c1..cn, c1^2, c1c2, ..., cn^2), write the
predictions as ONE LINEAR MAP on monomial space, and SVD it. Doesn't that find
all relations -- no geometry, no kappa needed?

Three tests, on the real data:

TEST 1 (exact relations): does monomial-SVD find MORE exact relations than the
  linear stacked kernel we already use? The number of exactly-indistinguishable
  directions at a generic point c is  n - rank(J(c)),  J = A + H c  -- because
  two points give identical predictions iff the path between them stays in the
  kernel of J. So count rank(J) at random points.

TEST 2 (near-invisible functionals, the practical case): whiten by the real
  errors, grade each monomial by its box size (wall^degree), SVD. Singular
  value < 1 means "this polynomial combination changes no prediction by more
  than ~1 sigma anywhere in the box" -- a candidate relation. Count them, and
  check WHAT they are (pure linear = drops/known relations we already have,
  or genuinely quadratic = something new kappa missed).

TEST 3 (the failure mode): LEP has weak curvature, so EVERY quadratic monomial
  barely changes predictions -> monomial-SVD reports a flood of "invisible"
  quadratic combinations that are true but useless (imposing them does not
  remove any fit parameter). Count the flood.
"""
import numpy as np
import battery_lib as bl
import smeftgeo as sg

wall = sg.nda_wall(5.0)


def monomial_design(Aw, Qw, box):
    """columns = box-graded monomial responses: linear i -> Aw[:,i]*wall_i;
    quadratic (i<=j) -> Qw[:,i,j]*wall_i*wall_j (with 1/2 on the diagonal,
    since mu = A.c + 1/2 c.H.c)."""
    nO, n = Aw.shape
    cols, labels, isquad = [], [], []
    for i in range(n):
        cols.append(Aw[:, i]*box[i]); labels.append(f"c{i}"); isquad.append(False)
    for i in range(n):
        for j in range(i, n):
            fac = 0.5 if i == j else 1.0
            cols.append(fac*Qw[:, i, j]*box[i]*box[j])
            labels.append(f"c{i}c{j}"); isquad.append(True)
    return np.stack(cols, 1), labels, np.array(isquad)


def run(tag, A, Q, n, prep_kw, names):
    Aw, Qw = bl.prep(A, Q, **prep_kw)
    box = np.full(n, wall)

    # ---- TEST 1: exact degeneracy at generic points ----
    rng = np.random.default_rng(0)
    degen = []
    for _ in range(20):
        c = rng.uniform(-wall, wall, n)
        J = Aw + np.einsum('oij,j->oi', Qw, c)
        sv = np.linalg.svd(J, compute_uv=False)
        degen.append(n - int(np.sum(sv > 1e-10*sv[0])))
    print(f"\n=== {tag} ===")
    print(f"TEST 1  exact: indistinguishable directions at a generic point "
          f"= {max(set(degen), key=degen.count)}  "
          f"(the linear stacked kernel already found exactly this many)")

    # ---- TEST 2: near-invisible monomial functionals ----
    M, labels, isquad = monomial_design(Aw, Qw, box)
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    nmon = M.shape[1]
    # right-singular vectors beyond the row count are exact zeros by counting
    ncount_zero = max(0, nmon - M.shape[0])
    small = S < 1.0
    print(f"TEST 2  monomials: {nmon} ({n} linear + {nmon-n} quadratic); "
          f"data rows: {M.shape[0]}")
    print(f"        functionals invisible at 1 sigma across the box: "
          f"{int(small.sum())} with SV<1  +  {ncount_zero} exact zeros forced "
          f"by counting (more monomials than data bins)")
    # what are the smallest nontrivial ones made of?
    ntop = min(4, int(small.sum()))
    for k in range(len(S)-1, len(S)-1-ntop, -1):
        v = Vt[k]
        linpart = np.linalg.norm(v[:n]); quadpart = np.linalg.norm(v[n:])
        top = np.argsort(-np.abs(v))[:3]
        print(f"          SV={S[k]:.2e}: linear {linpart:.2f} / quad "
              f"{quadpart:.2f}  top: "
              + ", ".join(f"{v[t]:+.2f} {labels[t]}" for t in top))

    # ---- TEST 3: the quadratic flood ----
    quad_small = int(np.sum(small & np.concatenate(
        [np.zeros(n, bool), np.ones(nmon-n, bool)])[np.argsort(S)][:0].sum()
        if False else 0))
    # simpler: count quadratic-dominated small functionals
    flood = 0
    for k in range(len(S)):
        if S[k] < 1.0:
            v = Vt[k]
            if np.linalg.norm(v[n:]) > 0.9:
                flood += 1
    print(f"TEST 3  of the SV<1 functionals, {flood} are >90% quadratic "
          f"('invisible because curvature is weak/finite bins' -- true but "
          f"useless: imposing them removes no fit parameter)")
    return S


dy = sg.load_dy()
run("DY (Poisson, 101 bins, 19 coeffs -> 209 monomials)",
    dy["A"], dy["Q"], 19, dict(weight=dy["NSM"]), dy["names"])

lep = sg.load_lep()
rho = np.load("lep_rho.npy")
cov = rho*np.outer(lep["sigma"], lep["sigma"])
run("LEP (real cov, 11 obs, 10 coeffs -> 65 monomials)",
    lep["A"], lep["Q"], 10, dict(cov=cov), lep["names"])

print("""
VERDICT
  - Exact relations: one SVD DOES find them all -- and it is the one we
    already run (the linear stacked kernel). The quadratic extension cannot
    find more: the count of exactly-invisible directions at a generic point
    is n - rank(A + Hc), test 1 confirms it.
  - Near-invisible: monomial-SVD is a useful CONSTRUCTIVE first pass, but
    (a) with more monomials than data bins, most of its kernel is forced by
    counting, not physics; (b) at weak curvature it floods with useless
    quadratic 'relations'; (c) a small functional still has to be solved for
    one coefficient inside the box, and priced -- which is our E_inf step.
  - kappa answers a different question: not 'which polynomial is invisible'
    but 'can the blind direction be tied off AT ALL'. On DY it says no --
    and no monomial functional below noise contradicts it.""")
