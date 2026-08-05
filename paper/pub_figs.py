#!/usr/bin/env python3
"""
Publication figure set for the paper. Every figure recomputes its data from
the validated pipeline (battery_lib + smeftgeo) except the price ladder,
whose numbers come from the monotone-engine runs of dy_reduce_realcov.py and
dy_reduce_end2end.py (provenance in comments).

Figures:
  pub_spectrum     (a) DY, (b) LEP whitened Fisher spectra + rank
  pub_kernel       DY: dim-6^2 lift of the 11 linear-blind directions
  pub_rotation     (a) rotation of the blind combination, (b) cost of a fixed
                   shift along it -- the twist figure, DY vs LEP
  pub_energy       DY: dim-6 vs dim-6^2 whitened response vs m_ll
  pub_ladder       DY: worst-case price of dropping each coefficient,
                   real CMS covariance vs Poisson
  pub_mbam         MBAM ride: no boundary before the EFT wall
  pub_sigma        kappa vs where the covariance puts its Fisher weight
"""
import numpy as np
import matplotlib.pyplot as plt
import pubstyle as ps
import battery_lib as bl
import smeftgeo as sg

ps.set_style()
WALL = sg.nda_wall(5.0)


# ========================================================================
def pub_spectrum():
    dy, lep = sg.load_dy(), sg.load_lep()
    fig, axes = plt.subplots(1, 2, figsize=(ps.DOUBLE, 2.7))
    for ax, tag, (A, Q, w), n in [
            (axes[0], "a", (dy["A"], dy["Q"], dy["weight"]), 19),
            (axes[1], "b", (lep["A"], lep["Q"], lep["sigma"]**2), 10)]:
        Aw = A / np.sqrt(w)[:, None]
        vals = np.sort(np.linalg.eigvalsh(Aw.T @ Aw))[::-1]
        sv = np.linalg.svd(Aw, compute_uv=False)
        rank = int(np.sum(sv > 1e-8 * sv[0]))
        idx = np.arange(1, n + 1)
        ax.semilogy(idx[:rank], np.abs(vals[:rank]), "o", ms=4.5,
                    color=ps.VERMIL, label="resolved")
        ax.semilogy(idx[rank:], np.maximum(np.abs(vals[rank:]), 1e-14), "o",
                    ms=4, mfc="white", color=ps.GRAY, label="blind")
        ax.axvline(rank + 0.5, ls="--", lw=0.8, color="0.45")
        ax.set_xlabel("eigenvalue index")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.text(0.95, 0.90, rf"rank $={rank}/{n}$", transform=ax.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
        ps.panel_tag(ax, tag, outside=True)
        ax.legend(loc="lower left")
    axes[0].set_ylabel(r"eigenvalue of $g=J^{\rm T}\Sigma^{-1}J$")
    axes[0].text(0.62, 0.96, "Drell--Yan", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=9)
    axes[1].text(0.5, 0.96, "LEP EWPO", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=9)
    fig.tight_layout()
    ps.save(fig, "pub_spectrum")


# ========================================================================
def pub_kernel():
    dy = sg.load_dy()
    A, H, names = dy["A"], dy["Q"], dy["names"]
    # basis of the 11-dim linear kernel, rotated so the exact all-orders
    # direction (stacked kernel) is the first element
    u_, s, vt = np.linalg.svd(A / np.sqrt(dy["weight"])[:, None])
    null = vt[8:]                                    # 11 x 19
    exact = sg.exact_relations(A, H)[0][0]           # cHB - cHW
    basis = [exact]
    for v in null:
        w = v - sum((v @ b) * b for b in basis)
        if np.linalg.norm(w) > 1e-8:
            basis.append(w / np.linalg.norm(w))
    basis = basis[:11]
    lifts, labels, seen = [], [], set()
    for u in basis:
        lifts.append(max(np.linalg.norm(H[b] @ u) for b in range(H.shape[0])))
        top = np.argsort(np.abs(u))[::-1]
        lab = ",".join(ps.tex(names[t]) for t in top[:2])
        if lab in seen:                       # Gram-Schmidt vectors can share
            lab = ",".join(ps.tex(names[t]) for t in top[:3])  # their top-2
        seen.add(lab)
        labels.append(lab)
    labels[0] = ps.tex("cHB") + r"$-$" + ps.tex("cHW")

    order = np.argsort(lifts)[::-1]
    fig, ax = plt.subplots(figsize=(ps.SINGLE, 2.9))
    y = np.arange(11)
    vals = [max(lifts[i], 1e-10) for i in order]
    cols = [ps.GREEN if lifts[i] < 1 else ps.PURPLE for i in order]
    ax.barh(y, vals, color=cols, height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"dim-6$^2$ lift of blind direction, "
                  r"$\max_b |H_b\,u|$")
    ax.text(vals[-1] * 30, 10.0, "exact to all orders:\nzero-price relation",
            fontsize=7, color=ps.GREEN, va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ps.GREEN,
                      lw=0.6))
    fig.tight_layout()
    ps.save(fig, "pub_kernel")


# ========================================================================
def _blind_study(Aw, Qw, coords, wall, npts=300, seed=0):
    As = Aw[:, coords]
    Qs = Qw[np.ix_(range(Qw.shape[0]), coords, coords)]
    n = len(coords)

    def metric(c):
        J = As + np.einsum('oij,j->oi', Qs, c)
        return J.T @ J
    v0 = np.linalg.eigh(metric(np.zeros(n)))[1][:, 0]
    rng = np.random.default_rng(seed)
    ang = []
    for _ in range(npts):
        c = rng.uniform(-0.95, 0.95, n) * wall
        v = np.linalg.eigh(metric(c))[1][:, 0]
        ang.append(np.degrees(np.arccos(np.clip(abs(v @ v0), 0, 1))))
    step = 0.5 * wall * v0

    def cost(c):
        d = bl.mu(As, Qs, (c + step)[None]) - bl.mu(As, Qs, c[None])
        return float((d ** 2).sum())
    worst = max(cost(rng.uniform(-0.45, 0.45, n) * wall) for _ in range(npts))
    return np.array(ang), cost(np.zeros(n)), worst


def pub_rotation():
    dy = sg.load_dy()
    Aw, Qw = bl.prep(dy["A"], dy["Q"], weight=dy["NSM"])
    cc = bl.active_coords(Aw)
    ang_dy, c0_dy, cw_dy = _blind_study(Aw, Qw, cc, WALL)
    kap_dy = bl.kappa(Aw, Qw, np.full(19, WALL), m=1, nsamp=25, coords=cc)

    lep = sg.load_lep()
    cov = np.load("lep_rho.npy") * np.outer(lep["sigma"], lep["sigma"])
    # LEP file stores Q with mu = A.c + c.Q.c (extract_lep.wl), so the
    # Hessian battery_lib expects is 2Q
    Awl, Qwl = bl.prep(lep["A"], 2.0 * lep["Q"], cov=cov)
    ccl = bl.active_coords(Awl)
    ang_lp, c0_lp, cw_lp = _blind_study(Awl, Qwl, ccl, WALL)
    kap_lp = bl.kappa(Awl, Qwl, np.full(10, WALL), m=1, nsamp=25, coords=ccl)

    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.DOUBLE, 2.7))
    rngj = np.random.default_rng(3)
    a.scatter(rngj.normal(1, .05, len(ang_dy)), ang_dy, s=9, color=ps.BLUE,
              alpha=0.3, edgecolor="none")
    a.scatter(rngj.normal(2, .05, len(ang_lp)), ang_lp, s=9, color=ps.ORANGE,
              alpha=0.3, edgecolor="none")
    for x, ang, kapv, col in [(1, ang_dy, kap_dy, ps.BLUE),
                              (2, ang_lp, kap_lp, ps.ORANGE)]:
        a.hlines(np.median(ang), x - .16, x + .16, color=col, lw=2.2)
        a.text(x, 97, rf"$\kappa={kapv:.2f}$", ha="center", fontsize=8,
               color=col)
    a.set_xticks([1, 2], ["Drell--Yan", "LEP EWPO"])
    a.set_xlim(0.5, 2.5); a.set_ylim(-3, 106)
    a.set_ylabel("rotation of blind direction [deg]")
    ps.panel_tag(a, "a")

    xs = np.array([1, 2])
    b.bar(xs - .16, [c0_dy, c0_lp], width=.3, color="white",
          edgecolor=[ps.BLUE, ps.ORANGE], lw=1.6, label="at origin")
    b.bar(xs + .16, [cw_dy, cw_lp], width=.3, color=[ps.BLUE, ps.ORANGE],
          label="worst point in validity range")
    b.set_yscale("log")
    b.axhline(1.0, color=ps.GRAY, ls="--", lw=1.0)
    b.text(2.42, 1.5, "detectable", color=ps.GRAY, fontsize=7, ha="right")
    for x, v in zip([1 - .16, 1 + .16, 2 - .16, 2 + .16],
                    [c0_dy, cw_dy, c0_lp, cw_lp]):
        b.text(x, v * 1.6, f"{v:.1e}" if v < 0.01 else f"{v:.1f}",
               ha="center", fontsize=6.5)
    b.set_xticks(xs, ["Drell--Yan", "LEP EWPO"])
    b.set_ylim(1e-6, 3e2)
    b.set_ylabel(r"$\Delta\chi^2$ of fixed shift along $v_0$")
    b.legend(loc="upper right")
    ps.panel_tag(b, "b")
    fig.tight_layout()
    ps.save(fig, "pub_rotation")


# ========================================================================
def pub_energy():
    dy = sg.load_dy()
    mll = 0.5 * (dy["binlo"] + dy["binhi"])
    s = np.sqrt(dy["NSM"])
    wA = np.linalg.norm(dy["A"], axis=1) / s
    wQ = np.linalg.norm(dy["Q"].reshape(len(mll), -1), axis=1) / s
    pA = np.polyfit(np.log(mll), np.log(wA), 1)[0]
    pQ = np.polyfit(np.log(mll), np.log(wQ), 1)[0]
    fig, ax = plt.subplots(figsize=(ps.SINGLE, 2.6))
    ax.loglog(mll, wA / wA[0], "-", color=ps.BLUE, lw=1.6,
              label=rf"dim-6 ($A_b$), slope ${pA:.1f}$")
    ax.loglog(mll, wQ / wQ[0], "--", color=ps.ORANGE, lw=1.6,
              label=rf"dim-6$^2$ ($H_b$), slope ${pQ:.1f}$")
    ax.set_xlabel(r"$m_{\ell\ell}$ [GeV]")
    ax.set_ylabel("whitened response (norm.)")
    ax.text(0.04, 0.06, "Drell--Yan", transform=ax.transAxes, fontsize=9)
    ax.legend(loc="upper right")
    fig.tight_layout()
    ps.save(fig, "pub_energy")


# ========================================================================
def pub_ladder():
    # numbers: monotone-engine runs of dy_reduce_realcov.py (real CMS
    # covariance, HEPData ins1711625 rebinned) and dy_reduce_end2end.py
    # (Poisson), Lambda = 5 TeV box, 800 box points
    real = {"cHB": 2.42e-06, "cHW": 3.14e-06, "cHd6": 6.24e-05,
            "cHu6": 0.0105, "cLL": 0.0635, "c16Hψq": 0.103,
            "c36HψL": 0.719, "cHD": 1.04, "cHe6": 2.01, "cLd": 2.03,
            "ceQ": 2.72, "c36Hψq": 2.85, "cLu": 3.4, "c16HψL": 3.97,
            "cHWB": 6.2, "ced": 10.4, "ceu": 49.4, "cLQ1": 72.8,
            "cLQ3": 376.0}
    pois = {"cHB": 0.00373, "cHW": 0.00488, "cHd6": 0.665, "cHu6": 12.8,
            "c16Hψq": 159.0, "cLd": 565.0, "cLL": 619.0, "ceQ": 844.0,
            "cHD": 2.02e3, "cLu": 3.39e3, "ced": 3.62e3, "c36HψL": 1.15e4,
            "ceu": 1.35e4, "cHe6": 1.38e4, "cLQ1": 1.55e4, "c36Hψq": 1.77e4,
            "cHWB": 1.85e4, "c16HψL": 6.09e4, "cLQ3": 1.52e5}
    dropped = {"cHB", "cHW", "cHd6", "cHu6", "cLL", "c16Hψq"}
    names = sorted(real, key=real.get)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(ps.SINGLE, 4.2))
    ax.barh(y, [max(real[n], 3e-7) for n in names], height=0.62,
            color=[ps.BLUE if n in dropped else "#B0C8DC" for n in names],
            label=None)
    ax.plot([pois[n] for n in names], y, "o", ms=3.6, mfc="white",
            mec=ps.VERMIL, mew=1.0, ls="none", label="Poisson errors")
    ax.set_xscale("log")
    ax.axvline(1.0, color=ps.GRAY, ls="--", lw=0.9)
    ax.axvline(4.0, color=ps.GRAY, ls=":", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([ps.tex(n) for n in names], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"worst-case $\Delta\chi^2$ of dropping the coefficient")
    # legend in the empty top-right region; thresholds explained IN the legend
    import matplotlib.patches as mpatches
    h = [mpatches.Patch(color=ps.BLUE, label="real CMS cov., dropped set"),
         mpatches.Patch(color="#B0C8DC", label="real CMS cov., kept"),
         plt.Line2D([], [], marker="o", mfc="white", mec=ps.VERMIL, ls="none",
                    label="Poisson errors"),
         plt.Line2D([], [], color=ps.GRAY, ls="--", label=r"$\Delta\chi^2=1$"),
         plt.Line2D([], [], color=ps.GRAY, ls=":",
                    label=r"$\Delta\chi^2=4\;(2\sigma)$")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=2, fontsize=6.8, frameon=False, columnspacing=1.2,
              handlelength=1.6)
    fig.tight_layout()
    ps.save(fig, "pub_ladder")


# ========================================================================
def pub_mbam():
    dy = sg.load_dy()
    chosen, _ = sg.active_slice(dy["A"], dy["weight"])
    A = dy["A"][:, chosen]
    Q = dy["Q"][np.ix_(range(dy["A"].shape[0]), chosen, chosen)]
    g0 = sg.fisher_metric(A, Q, dy["weight"])
    _, vecs, _ = sg.spectrum(g0)
    track = sg.mbam_ride(A, Q, dy["weight"], vecs[:, -1], max_frac=3e3)
    jump = np.abs(np.diff(np.log10(np.maximum(track[:, 2], 1e-12))))
    cut = len(track)
    bad = np.where(jump > np.log10(20))[0]
    if len(bad) and bad[-1] > 0.8 * len(track):
        cut = bad[np.argmax(bad > 0.8 * len(track))]
    cut = min(cut, int(0.97 * len(track)))   # also trim the terminal
    track = track[:cut]                      # adaptive-step spike
    # hard stop at 2000 x wall: the terminal steps are step-size artifacts,
    # and the no-boundary claim only needs three decades past the wall
    track = track[track[:, 0] / sg.nda_wall() <= 2e3]

    fig, ax = plt.subplots(figsize=(ps.SINGLE, 2.7))
    ax.loglog(track[:, 0] / sg.nda_wall(), track[:, 2], "-", lw=1.4,
              color=ps.VERMIL, label="DY slice (truncated EFT)")
    x = np.logspace(0, 3.5, 60)
    ax.loglog(x, np.exp(-x / 50), "--", color=ps.BLUE,
              label="saturating model (true boundary)")
    ax.axvline(1.0, ls="--", lw=0.9, color="0.4")
    ax.text(1.35, 1e-12, "EFT validity\nwall", fontsize=7, color="0.35")
    ax.axhline(1.0, ls=":", lw=0.8, color="0.6")
    ax.set_xlabel(r"$|c|_{\max}\,/\,$wall")
    ax.set_ylabel(r"$\lambda_{\min}(g)\,/\,\lambda_{\min}(g_0)$")
    ax.set_ylim(1e-28, 1e7)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    ps.save(fig, "pub_mbam")


# ========================================================================
def pub_sigma():
    dy = sg.load_dy()
    A, Q, N = dy["A"], dy["Q"], dy["NSM"]
    mll = 0.5 * (dy["binlo"] + dy["binhi"])
    mref = 200.0
    sigmas = {
        "Poisson": N,
        "flat 5% rel.": (0.05 * N) ** 2,
        "Poisson + 5% syst": N + (0.05 * N) ** 2,
        r"high-$m_{\ell\ell}$ precise": (0.05 * N * (mref / mll)) ** 2,
        r"low-$m_{\ell\ell}$ precise": (0.05 * N * (mll / mref)) ** 2,
    }
    cc = bl.active_coords(bl.prep(A, Q, weight=N)[0])
    pts = []
    for name, w in sigmas.items():
        Aw, Qw = bl.prep(A, Q, weight=w)
        kap = bl.kappa(Aw, Qw, np.full(19, WALL), m=2, nsamp=30, coords=cc)
        info = np.linalg.norm(Aw, axis=1) ** 2
        frac = info[mll > mref].sum() / info.sum()
        pts.append((name, frac * 100, kap))
    fig, ax = plt.subplots(figsize=(ps.SINGLE, 2.7))
    cols = [ps.BLUE, ps.ORANGE, ps.GREEN, ps.PURPLE, ps.GRAY]
    off = {"Poisson": (6, -3), "flat 5% rel.": (-6, 2),
           "Poisson + 5% syst": (-6, -10),
           r"high-$m_{\ell\ell}$ precise": (-6, -3),
           r"low-$m_{\ell\ell}$ precise": (-6, -12)}
    for (name, x, k), c in zip(pts, cols):
        ax.scatter(x, k, s=45, color=c, zorder=3, edgecolor="white", lw=0.7)
        ax.annotate(name, (x, k), textcoords="offset points",
                    xytext=off[name], fontsize=7, color=c,
                    ha="left" if name == "Poisson" else "right")
    ax.axhline(sg.TWIST_THRESHOLD, color=ps.GRAY, ls="--", lw=0.9)
    ax.text(30, sg.TWIST_THRESHOLD * 1.4, "clean/obstructed threshold",
            fontsize=7, color=ps.GRAY)
    ax.set_yscale("log")
    ax.set_xlim(15, 118)
    ax.set_ylim(0.03, 15)                    # headroom: no clipped labels
    ax.set_xlabel(r"Fisher weight above $200$ GeV [%]")
    ax.set_ylabel(r"twist $\kappa$")
    fig.tight_layout()
    ps.save(fig, "pub_sigma")


if __name__ == "__main__":
    pub_spectrum()
    pub_kernel()
    pub_energy()
    pub_ladder()
    pub_sigma()
    pub_rotation()
    pub_mbam()
    print("all publication figures written")
