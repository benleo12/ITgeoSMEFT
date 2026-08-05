#!/usr/bin/env python3
"""Audit finding 4: DY prices were sup over 800 random interior points, a
lower bound. Re-estimate the two headline joint prices (greedy set 0.72,
plus C_Hl3 10.3) with 4096 random sign-corners + interior + L-BFGS-B polish."""
import sys, json
sys.path.insert(0, "/private/tmp/claude-501/-Users-user-Downloads/"
                   "77cdb1df-3b76-468d-9e32-1a5a8f7f7054/scratchpad/ITgeoSMEFT/core")
import numpy as np
from scipy.optimize import minimize as spmin
import battery_lib as bl, smeftgeo as sg

wall = sg.nda_wall(5.0)
dy = sg.load_dy(); names = dy["names"]
ctr = 0.5*(dy["binlo"]+dy["binhi"])
meas = json.load(open("/tmp/meas.json")); rows = meas["values"]
lo = np.array([float(r["x"][0]["low"]) for r in rows])
hi = np.array([float(r["x"][0]["high"]) for r in rows])
xs = np.array([float(r["y"][0]["value"]) for r in rows])
cc = np.load("/tmp/cms_cov.npy")
win = np.where((lo >= 90) & (hi <= 610))[0]
Cw = cc[np.ix_(win, win)]; dw = np.sqrt(np.diag(Cw))
relu = dw/np.abs(xs[win]); R = Cw/np.outer(dw, dw)
A = np.zeros((len(win), 19)); Q = np.zeros((len(win), 19, 19))
SM = np.zeros(len(win)); nm = np.zeros(len(win), int)
for b in range(len(ctr)):
    k = np.where((ctr[b] >= lo[win]) & (ctr[b] < hi[win]))[0]
    if len(k): k = k[0]; A[k] += dy["A"][b]; Q[k] += dy["Q"][b]; SM[k] += dy["NSM"][b]; nm[k] += 1
keep = nm > 0
Aw, Qw = bl.prep(A[keep]/SM[keep][:, None], Q[keep]/SM[keep][:, None, None],
                 cov=(R*np.outer(relu, relu))[np.ix_(keep, keep)])
box = np.full(19, wall)
rng = np.random.default_rng(9)

def joint_price(dropset, tag):
    kp = [j for j in range(19) if j not in dropset]
    pts = rng.uniform(-1, 1, (800, 19))*wall
    corners = wall*rng.choice([-1., 1.], size=(4096, 19))
    allp = np.vstack([pts, corners])
    tgt = bl.mu(Aw, Qw, allp)
    d0 = bl._proj_chi2(Aw, Qw, tgt, allp[:, kp], kp, None, -1,
                       clampbox=box[kp])
    def r2(z, t):
        c = np.zeros(19); c[kp] = z
        return float(((bl.mu(Aw, Qw, c[None]) - t)**2).sum())
    worst = 0.0
    for i in np.argsort(-d0)[:30]:
        r = spmin(r2, np.clip(allp[i][kp], -wall, wall), args=(tgt[i],),
                  method="L-BFGS-B", bounds=[(-wall, wall)]*len(kp))
        worst = max(worst, r.fun)
    print(f"{tag}: corner+polish worst-case = {worst:.3g}")
    return worst

six = [names.index(x) for x in ["cHB","cHW","cHd6","cHu6","cLL","c16Hψq"]]
joint_price(six, "greedy 6-set (was 0.72)")
joint_price(six + [names.index("c36HψL")], "7-set (was 10.3)")
