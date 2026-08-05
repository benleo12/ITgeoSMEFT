#!/usr/bin/env python3
"""
INDICATIVE DY analysis with the REAL CMS covariance (HEPData ins1711625,
Table 10), vs Poisson. Answers: does the real measurement covariance move
the DY verdict, the way it flipped LEP's error bars?

Method (unit-free, relative):
  - rebin Adam's 5-GeV theory bins into the 22 CMS mass bins inside
    95-600 GeV (center assignment; cross sections add);
  - relative design J_rel[b,i] = A_CMS[b,i] / SM_CMS[b];
  - relative covariance Sigma_rel = R (CMS correlations) x
    (rel-unc outer rel-unc), vs Poisson diag(1/N_SM);
  - Fisher g = J_rel^T Sigma_rel^-1 J_rel; compare rank, twist, drop.
CAVEAT: center-assignment rebin is APPROXIMATE; exact needs Adam's
measurement-comparison setup. This is indicative, not a publication number.
"""
import json
import numpy as np
import smeftgeo as sg

# --- Adam's DY model on his 5-GeV grid ---
dy = sg.load_dy()
names = dy["names"]
A_fine, Q_fine, NSM_fine = dy["A"], dy["Q"], dy["weight"]
lo_f, hi_f = dy["binlo"], dy["binhi"]
ctr_f = 0.5*(lo_f+hi_f)
nC = A_fine.shape[1]

# --- CMS bins + covariance, restricted to 95-600 GeV window ---
cms_bins = np.load("/tmp/cms_bins.npy")        # (43,2) low,high
cms_cov = np.load("/tmp/cms_cov.npy")          # (43,43)
win = np.where((cms_bins[:, 0] >= 90) & (cms_bins[:, 1] <= 610))[0]
cb = cms_bins[win]
Cw = cms_cov[np.ix_(win, win)]
dw = np.sqrt(np.diag(Cw))
# relative uncertainty per CMS bin needs the measured value; use HEPData meas
meas = json.load(open("/tmp/meas.json"))
xs_all = np.array([float(r["y"][0]["value"]) for r in meas["values"]])
xs = xs_all[win]
relunc = dw / np.abs(xs)
R = Cw / np.outer(dw, dw)                        # correlation matrix
nCMS = len(win)
print(f"CMS bins in window: {nCMS}   median rel-unc {np.median(relunc):.3f}   "
      f"mean|rho| {np.mean(np.abs(R[~np.eye(nCMS, dtype=bool)])):.3f}")

# --- rebin Adam's fine bins into CMS bins (center assignment) ---
A = np.zeros((nCMS, nC)); Q = np.zeros((nCMS, nC, nC)); SM = np.zeros(nCMS)
NSM = np.zeros(nCMS); nmap = np.zeros(nCMS, int)
for b in range(len(ctr_f)):
    k = np.where((ctr_f[b] >= cb[:, 0]) & (ctr_f[b] < cb[:, 1]))[0]
    if len(k):
        k = k[0]
        A[k] += A_fine[b]; Q[k] += Q_fine[b]; NSM[k] += NSM_fine[b]
        nmap[k] += 1
SM = NSM.copy()                                  # SM cross-section proxy
keep = nmap > 0
A, Q, SM, NSM, relunc, R = (A[keep], Q[keep], SM[keep], NSM[keep],
                            relunc[keep], R[np.ix_(keep, keep)])
n = keep.sum()
print(f"CMS bins populated by Adam's grid: {n}  "
      f"(fine bins per CMS bin: {nmap[keep].min()}-{nmap[keep].max()})")


def analyze(Sig_rel, tag):
    Jrel = A / SM[:, None]                        # relative linear design
    SigInv = np.linalg.pinv(Sig_rel, rcond=1e-10)
    L = np.linalg.cholesky(SigInv + 1e-12*np.eye(n)) \
        if np.all(np.linalg.eigvalsh(SigInv) > 0) else None

    def gfun(c):
        J = (A + np.einsum('bij,j->bi', Q, c)) / SM[:, None]
        return J.T @ SigInv @ J
    g0 = gfun(np.zeros(nC))
    sv = np.linalg.svd(Jrel, compute_uv=False)  # unweighted rank proxy
    # rank from whitened
    w, V = np.linalg.eigh(g0)
    rank = int(np.sum(w > 1e-8*w.max()))
    # active slice
    chosen = list(np.argsort(-np.abs(np.diag(g0)))[:rank])
    # twist on slice (box-graded), reuse fisher via closure
    As, Qs = A[:, chosen], Q[np.ix_(range(n), chosen, chosen)]
    B = np.full(len(chosen), sg.nda_wall())

    def gslice(c):
        J = (As + np.einsum('bij,j->bi', Qs, c)) / SM[:, None]
        return J.T @ SigInv @ J
    kap = sg.BoxTwist(gslice, len(chosen), B).kappa(
        min(2, len(chosen)-1), nsamp=25, rng=np.random.default_rng(1))["kappa"]
    gu = (B[:, None]*gslice(np.zeros(len(chosen))))*B[None, :]
    ev, ew = np.linalg.eigh(gu)
    drop = names[chosen[np.argmax(np.abs(ew[:, 0]))]]
    print(f"  {tag:22} rank {rank:2}  kappa {kap:6.2f}  "
          f"{'obstructed' if kap > sg.TWIST_THRESHOLD else 'clean':>10}  "
          f"drop {drop}")
    return rank, kap, drop


print("\n=== DY verdict: real CMS covariance vs Poisson (indicative) ===")
Sig_pois = np.diag(1.0/np.maximum(NSM, 1.0))       # Poisson relative
Sig_cms = R * np.outer(relunc, relunc)             # CMS relative covariance
rp = analyze(Sig_pois, "Poisson")
rc = analyze(Sig_cms, "real CMS covariance")
print(f"\n  Poisson: rank {rp[0]}, kappa {rp[1]:.2f}, drop {rp[2]}")
print(f"  CMS    : rank {rc[0]}, kappa {rc[1]:.2f}, drop {rc[2]}")
print(f"  MOVED? rank {'yes' if rp[0]!=rc[0] else 'no'}, "
      f"drop {'yes' if rp[2]!=rc[2] else 'no'}, "
      f"kappa ratio {rc[1]/max(rp[1],1e-9):.2f}x")
