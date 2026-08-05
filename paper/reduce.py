#!/usr/bin/env python3
"""
The reduction procedure, one step at a time.

INPUT CONTRACT. The procedure sees exactly four things:

    A     (nobs, n)        linear response of each measurement to each coeff
    H     (nobs, n, n)     quadratic response (the Hessian)
    Sigma (nobs, nobs)     covariance of the measurements
    box   (n,)             validity range of each coefficient

It never sees what the measurements are, what cuts were applied, which
process produced them, or how many of them there are. Bins, angular
observables, cross sections, asymmetries, one experiment or ten stacked
together all enter the same way. Everything the procedure decides is a
function of (A, H, Sigma, box) alone, and every step below is global: it
quantifies over the whole validity range, never at a single point.

The steps:

    1  whiten by Sigma            put everything in chi^2 units
    2  exact relations            common kernel of [A; H_b], imposed free
    3  resolved rank              how many combinations the data see
    4  twist number T             is a relation available, or only drops
    5  price each single drop     worst case over the validity range
    6  greedy joint reduction     add drops while the joint price allows
    7  report                     kept, dropped, price, certificate

Run:  python reduce.py           (applies it to Drell-Yan and to LEP)
"""
import numpy as np
from scipy.optimize import minimize as spmin
import battery_lib as bl
import smeftgeo as sg


def reduce_model(A, H, Sigma, box, names=None, max_price=4.0, seed=0,
                 nsamp=800, ncorner=4096, npolish=15, verbose=True):
    """Run the full procedure. Returns a dict; prints each step if verbose."""
    n = A.shape[1]
    names = names or [f"c{i}" for i in range(n)]
    box = np.asarray(box, float)
    say = print if verbose else (lambda *a, **k: None)

    # ---------------------------------------------------------------- 1
    # Whiten by the covariance. After this the squared distance between two
    # predictions IS their chi^2 separation, so every price below is read
    # directly in chi^2 units. This is the only place the data enters.
    Aw, Hw = bl.prep(A, H, cov=Sigma)
    say(f"\n  step 1  whitened by Sigma: {Aw.shape[0]} measurements, "
        f"{n} coefficients")

    # ---------------------------------------------------------------- 2
    # Exact relations. A direction u with A u = 0 and H_b u = 0 for every
    # measurement changes no prediction from any starting point, so it can be
    # imposed for free. One SVD of the stacked response finds all of them.
    stack = np.vstack([Aw] + [Hw[b] for b in range(Hw.shape[0])])
    sv = np.linalg.svd(stack, compute_uv=False)
    U, S, Vt = np.linalg.svd(stack)
    nexact = int(np.sum(S <= 1e-10 * S[0]))
    exact = Vt[len(S) - nexact:] if nexact else np.zeros((0, n))
    say(f"  step 2  exact relations: {nexact}")
    for u in exact:
        u = u / np.abs(u).max()
        terms = "  ".join(f"{u[i]:+.3f} {names[i]}"
                          for i in np.argsort(-np.abs(u)) if abs(u[i]) > 1e-3)
        say(f"            {terms}   (free)")

    # ---------------------------------------------------------------- 3
    # Resolved rank: how many coefficient combinations the data actually
    # constrain, and which coordinates carry them.
    rank, _ = bl.rank_of(Aw)
    resolved = bl.active_coords(Aw)
    say(f"  step 3  resolved rank: {rank} of {n}")

    # ---------------------------------------------------------------- 4
    # Twist number, on the resolved slice, over the whole validity range.
    # Small T means the blind combination is the same everywhere and one
    # relation removes it. Large T means it rotates and only drops are
    # honest. This is a verdict about the whole range, not about a point.
    T = bl.kappa(Aw, Hw, box, m=1, nsamp=40, coords=resolved)
    verdict = "relation available" if T < sg.TWIST_THRESHOLD else "drops only"
    say(f"  step 4  twist number T = {T:.3f}  ->  {verdict}"
        f"   (threshold {sg.TWIST_THRESHOLD:.3f})")

    # ---------------------------------------------------------------- 5
    # Price each single drop. The price is the largest chi^2 the reduction
    # can cause ANYWHERE in the validity range, so we sample the interior,
    # include every sign corner, and polish the worst candidates. A local
    # estimate at the best-fit point would not be a bound.
    rng = np.random.default_rng(seed)
    pts = np.vstack([rng.uniform(-1, 1, (nsamp, n)) * box,
                     box * rng.choice([-1.0, 1.0], size=(ncorner, n))])

    def price(dropset):
        keep = [j for j in range(n) if j not in dropset]
        if not keep:
            return float("inf")
        tgt = bl.mu(Aw, Hw, pts)
        rough = bl._proj_chi2(Aw, Hw, tgt, pts[:, keep], keep, None, -1,
                              clampbox=box[keep])

        def chi2(z, t):
            c = np.zeros(n)
            c[keep] = z
            return float(((bl.mu(Aw, Hw, c[None]) - t) ** 2).sum())
        worst = 0.0
        for i in np.argsort(-rough)[:npolish]:
            r = spmin(chi2, np.clip(pts[i][keep], -box[keep], box[keep]),
                      args=(tgt[i],), method="L-BFGS-B",
                      bounds=list(zip(-box[keep], box[keep])))
            worst = max(worst, r.fun)
        return worst

    singles = {k: price([k]) for k in range(n)}
    order = sorted(range(n), key=lambda k: singles[k])
    say("  step 5  price of each single drop (worst case over the range):")
    for k in order:
        mark = "free" if singles[k] < 1e-3 else (
            "cheap" if singles[k] < 1 else
            "borderline" if singles[k] < max_price else "expensive")
        say(f"            {names[k]:12} {singles[k]:12.3g}   {mark}")

    # ---------------------------------------------------------------- 6
    # Greedy joint reduction. Prices do not add, so each candidate is judged
    # by the price of the whole set, not by its own.
    say(f"  step 6  greedy joint reduction (stop when joint price > "
        f"{max_price}):")
    dropped, joint = [], 0.0
    for k in order:
        p = price(dropped + [k])
        if p <= max_price:
            dropped, joint = dropped + [k], p
            say(f"            + {names[k]:12} joint {p:10.3g}   keep going")
        else:
            say(f"            + {names[k]:12} joint {p:10.3g}   stop")
            break

    # ---------------------------------------------------------------- 7
    kept = [names[j] for j in range(n) if j not in dropped]
    say(f"  step 7  result: {n} -> {len(kept)} coefficients, "
        f"joint worst-case price {joint:.3g} "
        f"({np.sqrt(max(joint, 0)):.1f} sigma)")
    say(f"            dropped: {[names[k] for k in dropped]}")
    say(f"            certified by T = {T:.3f} ({verdict})")
    return dict(n_exact=nexact, exact=exact, rank=rank, twist=T,
                singles={names[k]: singles[k] for k in range(n)},
                dropped=[names[k] for k in dropped], kept=kept, joint=joint)


# ======================================================================
# The same function, no per-dataset logic, applied to two different
# experiments with different processes, observables, cuts and covariances.
# ======================================================================
if __name__ == "__main__":
    import json

    # ---- Drell-Yan: binned dilepton spectrum, measured CMS covariance ----
    dy = sg.load_dy()
    ctr = 0.5 * (dy["binlo"] + dy["binhi"])
    meas = json.load(open("/tmp/meas.json"))
    rows = meas["values"]
    lo = np.array([float(r["x"][0]["low"]) for r in rows])
    hi = np.array([float(r["x"][0]["high"]) for r in rows])
    xs = np.array([float(r["y"][0]["value"]) for r in rows])
    cms = np.load("/tmp/cms_cov.npy")
    win = np.where((lo >= 90) & (hi <= 610))[0]
    Cw = cms[np.ix_(win, win)]
    dw = np.sqrt(np.diag(Cw))
    rel = dw / np.abs(xs[win])
    R = Cw / np.outer(dw, dw)
    A = np.zeros((len(win), 19)); Hm = np.zeros((len(win), 19, 19))
    SM = np.zeros(len(win)); nm = np.zeros(len(win), int)
    for b in range(len(ctr)):
        k = np.where((ctr[b] >= lo[win]) & (ctr[b] < hi[win]))[0]
        if len(k):
            k = k[0]
            A[k] += dy["A"][b]; Hm[k] += dy["Q"][b]
            SM[k] += dy["NSM"][b]; nm[k] += 1
    m = nm > 0
    print("=" * 66)
    print("DRELL-YAN: dilepton mass spectrum, measured CMS covariance")
    print("=" * 66)
    reduce_model(A[m] / SM[m, None], Hm[m] / SM[m, None, None],
                 (R * np.outer(rel, rel))[np.ix_(m, m)],
                 np.full(19, sg.nda_wall(5.0)), dy["names"])

    # ---- LEP: Z-pole observables, measured LEP EWWG covariance ----------
    lep = sg.load_lep()
    cov = np.load("/Users/user/Downloads/lep_rho.npy") * \
        np.outer(lep["sigma"], lep["sigma"])
    print("\n" + "=" * 66)
    print("LEP EWPO: Z-pole observables, measured LEP EWWG covariance")
    print("=" * 66)
    reduce_model(lep["A"], 2.0 * lep["Q"], cov,
                 np.full(10, 4 * np.pi), lep["names"])
