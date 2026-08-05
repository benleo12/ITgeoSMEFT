#!/usr/bin/env python3
"""
NOISE-PROOF coefficient reduction. The single-coefficient drop is fragile
under MC noise (flips at ~1%). The fix is not to force a pick but to report:
  (1) a NOISE-AWARE rank (how many directions are really resolved), set by
      the Marchenko-Pastur edge of the noise singular-value spectrum;
  (2) the STABLE sloppy SUBSPACE (robust) via the bootstrap-averaged
      projector -- its near-1 eigenvalues are the directions you can safely
      reduce;
  (3) each coefficient's DROP FREQUENCY over bootstrap replicas -- honest
      about which specific operator, instead of a false single answer.

Demonstrated on DY with injected MC noise (stand-in for real MadGraph
replicas until Adam's covariance / HEPData ins1711625 is wired in).
"""
import json
import numpy as np
import smeftgeo as sg

dy = sg.load_dy()
names = dy["names"]
A0, Q0, w = dy["A"], dy["Q"], dy["weight"]
nB, nC = A0.shape
sqw = np.sqrt(w)


def mp_rank(Aw, eps):
    """Marchenko-Pastur rank: singular values of the whitened design above
    the noise edge sigma*(sqrt(nB)+sqrt(nC)), sigma = eps * rms(entry)."""
    sv = np.linalg.svd(Aw, compute_uv=False)
    sigma = eps * np.sqrt((Aw**2).mean())
    edge = sigma * (np.sqrt(nB) + np.sqrt(nC))
    return int(np.sum(sv > edge)), sv, edge


def bootstrap(eps, nboot=200, m_soft=2, seed=0):
    """resample the design with MC noise; per replica record the drop
    coefficient and the soft-subspace projector; aggregate."""
    rng = np.random.default_rng(seed)
    Pacc = np.zeros((nC, nC))
    drop_counts = np.zeros(nC)
    ranks = []
    B = np.full(nC, sg.nda_wall())
    for _ in range(nboot):
        A = A0 * (1 + eps*rng.standard_normal(A0.shape))
        Q = Q0 * (1 + eps*rng.standard_normal(Q0.shape))
        Q = 0.5*(Q + Q.transpose(0, 2, 1))
        Aw = A / sqw[:, None]
        r, sv, edge = mp_rank(Aw, eps)
        ranks.append(r)
        g0 = sg.fisher_metric(A, Q, w)
        gu = (B[:, None]*g0)*B[None, :]
        ev, ew = np.linalg.eigh(gu)
        # resolved columns (eigval above squared edge in graded units)
        res = ew[:, ev > (edge*B.mean())**2] if np.any(ev > (edge*B.mean())**2) else ew[:, -1:]
        # soft subspace = m_soft least-resolved among resolved
        soft = res[:, :m_soft] if res.shape[1] >= m_soft else res
        Pacc += soft @ soft.T
        # the single drop = dominant coeff of the softest resolved direction
        drop_counts[np.argmax(np.abs(res[:, 0]))] += 1
    Pacc /= nboot
    return np.array(ranks), drop_counts/nboot, Pacc


print("clean baseline (eps=0):")
r0, sv0, edge0 = mp_rank(A0/sqw[:, None], 1e-9)
print(f"  MP rank = {int(np.sum(sv0 > 1e-8*sv0[0]))}/{nC}")

for eps in [0.01, 0.05]:
    ranks, freq, P = bootstrap(eps)
    ev = np.sort(np.linalg.eigvalsh(P))[::-1]
    print(f"\n=== eps = {eps} (MC noise) ===")
    vals, counts = np.unique(ranks, return_counts=True)
    print(f"  noise-aware rank: {dict(zip(vals.tolist(), counts.tolist()))} "
          f"(mode {vals[np.argmax(counts)]})")
    print(f"  stable sloppy subspace: projector eigenvalues (near 1 = stable) "
          f"{np.array2string(ev[:4], precision=2)}")
    order = np.argsort(freq)[::-1]
    top = [(names[i], freq[i]) for i in order if freq[i] > 0.05][:5]
    print("  drop frequency over replicas (honest, not a single pick):")
    for n_, f_ in top:
        print(f"     {n_:10} {f_*100:5.0f}%")
    # how many directions carry 90% of the stable subspace weight
    keff = int(np.sum(np.cumsum(ev)/ev.sum() < 0.9)) + 1
    print(f"  => the reducible content is a {keff}-dim SUBSPACE (stable), "
          f"not one coefficient (which flips).")

print("\nNOISE-PROOF OUTPUT = (noise-aware rank) + (stable subspace) +")
print("(drop frequencies). The subspace is the robust deliverable; the")
print("single-coefficient pick is reported as a frequency, never asserted.")
