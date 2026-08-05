#!/usr/bin/env python3
"""
DECISIVE utility test for kappa: does ANY subsector of the real SMEFT/LHC data
admit a clean (integrable) relation, even though the full fit does not?

For every small subset S of Wilson coefficients (size 2-4) we compute, on real
DY and LEP:
  rank_S      -- is there a blind direction within S?
  kappa_S     -- is that blind direction integrable (clean, relatable) or not?
  kernel_S    -- is the blind direction already an EXACT (linear) kernel vector
                 (then it's trivial, found without kappa) or genuinely CURVED?
A "win" for kappa = a subset with a blind direction that is CURVED (not exact
kernel) AND kappa_S below threshold => a nontrivial relation kappa alone
certifies. We report the full distribution honestly, win or lose.
"""
import numpy as np
import itertools as it
import battery_lib as bl
import smeftgeo as sg

THR = sg.TWIST_THRESHOLD
wall = sg.nda_wall(5.0)


def verify_win(As, Qs):
    """confirm a curved relation actually beats the drop: E_drop vs E_relation
    on this sub-block (whitened). Returns (E_drop, E_relation)."""
    k = As.shape[1]
    kd, ed, _ = bl.best_drop(As, Qs, np.full(k, wall), nbox=500)
    free2 = [j for j in range(k) if j != kd][:2]
    er, _ = bl.einf_relation(As, Qs, np.full(k, wall), kd, free2, nbox=500)
    return ed, er


def scan(name, A, Q, prep_kw, names, coords_restrict=None, sizes=(4, 5, 6)):
    Aw, Qw = bl.prep(A, Q, **prep_kw)
    pool = coords_restrict if coords_restrict is not None else range(A.shape[1])
    pool = list(pool)
    rows = []
    for k in sizes:
        for S in it.combinations(pool, k):
            S = list(S)
            As, Qs = Aw[:, S], Qw[np.ix_(range(Qw.shape[0]), S, S)]
            r, sv = bl.rank_of(As)
            if r < 3 or r == k:                # need >=3 resolved (meaningful
                continue                       # twist) AND a blind direction
            cc = bl.active_coords(As)
            if len(cc) < 3:                    # twist trivially 0 in <3 dims
                continue
            exdim, _ = bl.exact_kernel_dim(As, Qs)
            kap = bl.kappa(As, Qs, np.full(k, wall), m=1, nsamp=20, coords=cc)
            rows.append(dict(S=[names[i] for i in S], rank=r, k=k,
                             exact=exdim, kappa=float(kap), idx=S))
    return rows, Aw, Qw


def report(name, rows, Aw, Qw):
    ks = np.array([r["kappa"] for r in rows])
    print(f"\n=== {name}: {len(rows)} subsets (resolved>=3, has blind dir) ===")
    if not len(ks):
        print("  none"); return
    print(f"  kappa: min {ks.min():.3f} median {np.median(ks):.3f} "
          f"max {ks.max():.3f}  (threshold {THR:.3f})")
    clean = [r for r in rows if r["kappa"] < THR]
    wins = [r for r in clean if r["exact"] == 0]      # curved (non-kernel)
    print(f"  clean (kappa<thr): {len(clean)}/{len(rows)};  of those CURVED "
          f"(kappa WIN candidates): {len(wins)}")
    # VERIFY the top curved-clean candidates with an actual E_relation vs E_drop
    print("  verifying curved-clean candidates (E_drop should >> E_relation~0):")
    for r in sorted(wins, key=lambda x: x["kappa"])[:4]:
        S = r["idx"]
        ed, er = verify_win(Aw[:, S], Qw[np.ix_(range(Qw.shape[0]), S, S)])
        tag = "REAL curved relation" if (er < 0.2*ed and ed > 1e-6) else \
              "not better than drop (spurious)" if ed > 1e-6 else \
              "drop already free (linear, not a kappa win)"
        print(f"    kappa={r['kappa']:.3f} E_drop={ed:.4g} E_rel={er:.4g}  "
              f"[{tag}]  {r['S']}")


dy = sg.load_dy()
Awp = bl.prep(dy["A"], dy["Q"], weight=dy["NSM"])[0]
dyslice = bl.active_coords(Awp)                        # 8 resolved coeffs
rows, Aw, Qw = scan("DY", dy["A"], dy["Q"], dict(weight=dy["NSM"]),
                    dy["names"], coords_restrict=dyslice)
report("DY (Poisson), within resolved 8-slice", rows, Aw, Qw)

lep = sg.load_lep()
rho = np.load("lep_rho.npy"); sig = lep["sigma"]
rows, Aw, Qw = scan("LEP", lep["A"], lep["Q"], dict(cov=rho*np.outer(sig, sig)),
                    lep["names"])
report("LEP (real cov)", rows, Aw, Qw)

print("\nVERDICT: a REAL kappa win = curved-clean subset where a curved relation")
print("actually beats the best drop (E_rel << E_drop). Those are reducible")
print("sub-models kappa finds that neither linear-kernel nor drop analysis gives.")
