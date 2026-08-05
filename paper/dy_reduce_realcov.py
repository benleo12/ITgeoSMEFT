#!/usr/bin/env python3
"""
Same DY reduction table, but with the REAL CMS covariance (HEPData ins1711625,
Table 10) instead of Poisson. These are the physical prices: the real
measurement has a ~6% systematics floor, so it is far less precise than raw
Poisson counting, and drops get much cheaper.

Then the bottom line for the paper: greedy reduction -- keep dropping the
cheapest coefficient until the JOINT worst-case cost passes 4 (= 2 sigma).
Report how far you get. (Rebin: Adam's 5-GeV bins summed into the CMS mass
bins, same as dy_real_covariance.py; approximate center-assignment, flagged.)
"""
import json
import sys
import numpy as np
import battery_lib as bl
import smeftgeo as sg

wall = sg.nda_wall(5.0)
NBOX = 800

# ---- rebuild the rebinned model + real covariance (as in dy_real_covariance)
dy = sg.load_dy()
names = dy["names"]
ctr = 0.5*(dy["binlo"] + dy["binhi"])
nC = 19

meas = json.load(open("/tmp/meas.json"))
rows = meas["values"]
lo = np.array([float(r["x"][0]["low"]) for r in rows])
hi = np.array([float(r["x"][0]["high"]) for r in rows])
xs = np.array([float(r["y"][0]["value"]) for r in rows])
cms_cov = np.load("/tmp/cms_cov.npy")

win = np.where((lo >= 90) & (hi <= 610))[0]
cb_lo, cb_hi = lo[win], hi[win]
Cw = cms_cov[np.ix_(win, win)]
dw = np.sqrt(np.diag(Cw))
relunc = dw/np.abs(xs[win])
R = Cw/np.outer(dw, dw)

A = np.zeros((len(win), nC)); Q = np.zeros((len(win), nC, nC))
SM = np.zeros(len(win)); nmap = np.zeros(len(win), int)
for b in range(len(ctr)):
    k = np.where((ctr[b] >= cb_lo) & (ctr[b] < cb_hi))[0]
    if len(k):
        k = k[0]
        A[k] += dy["A"][b]; Q[k] += dy["Q"][b]; SM[k] += dy["NSM"][b]
        nmap[k] += 1
keep = nmap > 0
A, Q, SM = A[keep], Q[keep], SM[keep]
relunc, R = relunc[keep], R[np.ix_(keep, keep)]
# relative units: response/SM, covariance = correlations x relative uncertainty
Arel = A/SM[:, None]
Qrel = Q/SM[:, None, None]
cov_rel = R*np.outer(relunc, relunc)
Aw, Qw = bl.prep(Arel, Qrel, cov=cov_rel)
box = np.full(nC, wall)
print(f"rebinned to {keep.sum()} CMS bins; median relative error "
      f"{np.median(relunc)*100:.1f}%  (vs Poisson ~0.1%)")


def einf_dropset(dropset, seed=0, nbox=NBOX):
    kp = [j for j in range(nC) if j not in dropset]
    B = bl._boxsamples(box, nbox, seed)
    d2 = bl._proj_chi2(Aw, Qw, bl.mu(Aw, Qw, B), B[:, kp], kp, None, -1,
                       clampbox=box[kp])
    return float(d2.max())


print("\nsingle-drop prices with the REAL covariance "
      "(worst-case Delta chi^2 over the box):")
prices = {}
for k in range(nC):
    prices[k] = einf_dropset([k], seed=k)
    sys.stdout.flush()
order = sorted(prices, key=prices.get)
for k in order:
    p = prices[k]
    verdict = ("cheap (< 1)" if p < 1 else "borderline (< 4)" if p < 4
               else "expensive")
    print(f"   {names[k]:12} {p:12.3g}   {verdict}")

# ---- greedy: drop cheapest first until the joint cost passes 4 (2 sigma) --
print("\ngreedy reduction (add cheapest drop while joint cost <= 4):")
dropped = []
joint = 0.0
for k in order:
    trial = dropped + [k]
    j = einf_dropset(trial, seed=123)
    if j <= 4.0:
        dropped = trial
        joint = j
        print(f"   drop {names[k]:12} -> joint worst-case {j:8.3g}  (keep going)")
    else:
        print(f"   drop {names[k]:12} -> joint worst-case {j:8.3g}  STOP "
              f"(> 4)")
        break
kept = [names[k] for k in range(nC) if k not in dropped]
print(f"\nRESULT with real errors: 19 coefficients -> {len(kept)} kept "
      f"at total worst-case cost {joint:.2g} (= {np.sqrt(max(joint,0)):.1f} sigma)")
print(f"   dropped: {[names[k] for k in dropped]}")
print(f"   kept   : {kept}")
print("\nCAVEAT: center-assignment rebin (approximate); exact numbers need")
print("Adam's binning setup. Poisson-vs-real comparison is the honest point.")
