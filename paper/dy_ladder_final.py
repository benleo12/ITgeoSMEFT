#!/usr/bin/env python3
"""Definitive DY ladder + greedy: rebinned CMS grid, BOTH error models on the
same grid, 800 interior + 4096 sign-corner points, L-BFGS-B polish of the
worst 15 per price. Greedy joint set with threshold 4 (2 sigma)."""
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
ccov = np.load("/tmp/cms_cov.npy")
win = np.where((lo >= 90) & (hi <= 610))[0]
Cw = ccov[np.ix_(win, win)]; dw = np.sqrt(np.diag(Cw))
relu = dw/np.abs(xs[win]); R = Cw/np.outer(dw, dw)
A = np.zeros((len(win), 19)); Q = np.zeros((len(win), 19, 19))
SM = np.zeros(len(win)); nm = np.zeros(len(win), int)
for b in range(len(ctr)):
    k = np.where((ctr[b] >= lo[win]) & (ctr[b] < hi[win]))[0]
    if len(k): k = k[0]; A[k] += dy["A"][b]; Q[k] += dy["Q"][b]; SM[k] += dy["NSM"][b]; nm[k] += 1
kp = nm > 0
A, Q, SM = A[kp], Q[kp], SM[kp]
relu, R = relu[kp], R[np.ix_(kp, kp)]
CMS = bl.prep(A/SM[:, None], Q/SM[:, None, None], cov=R*np.outer(relu, relu))
POIS = bl.prep(A/SM[:, None], Q/SM[:, None, None],
               cov=np.diag(1.0/np.maximum(SM, 1.0)))
box = np.full(19, wall)
rng = np.random.default_rng(11)
PTS = np.vstack([rng.uniform(-1, 1, (800, 19))*wall,
                 wall*rng.choice([-1., 1.], size=(4096, 19))])

def price(AwQw, dropset, seed_polish=15):
    Aw, Qw = AwQw
    keep = [j for j in range(19) if j not in dropset]
    tgt = bl.mu(Aw, Qw, PTS)
    d0 = bl._proj_chi2(Aw, Qw, tgt, PTS[:, keep], keep, None, -1,
                       clampbox=box[keep])
    def r2(z, t):
        c = np.zeros(19); c[keep] = z
        return float(((bl.mu(Aw, Qw, c[None]) - t)**2).sum())
    worst = 0.0
    for i in np.argsort(-d0)[:seed_polish]:
        r = spmin(r2, np.clip(PTS[i][keep], -wall, wall), args=(tgt[i],),
                  method="L-BFGS-B", bounds=[(-wall, wall)]*len(keep))
        worst = max(worst, r.fun)
    return worst

out = {"singles": {}}
for k in range(19):
    out["singles"][names[k]] = {"cms": price(CMS, [k]),
                                "pois": price(POIS, [k])}
    print(names[k], out["singles"][names[k]], flush=True)

order = sorted(range(19), key=lambda k: out["singles"][names[k]]["cms"])
dropped, greedy = [], []
for k in order:
    trial = dropped + [k]
    j = price(CMS, trial)
    greedy.append({"add": names[k], "joint": j})
    print("greedy try", names[k], j, flush=True)
    if j <= 4.0:
        dropped = trial
    else:
        break
out["greedy"] = greedy
out["final_dropped"] = [names[k] for k in dropped]
out["final_joint"] = price(CMS, dropped) if dropped else 0.0
print("FINAL:", out["final_dropped"], out["final_joint"])
json.dump(out, open("dy_ladder_final.json", "w"), indent=1)
