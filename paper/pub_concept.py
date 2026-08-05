#!/usr/bin/env python3
"""
Two explanatory figures. Both are computed from models, not drawn by hand.

pub_concept_mbam: what MBAM does in a truncated EFT, on a two-coefficient
  slice of the real Drell-Yan model chosen for visible curvature. The
  geodesic launched along the sloppy direction bends, never meets a manifold
  boundary, and exits through the NDA wall, which is what names the dropped
  coefficient. The straight sloppy ray is shown for contrast: curvature
  bends the path but both leave through the same wall, which is the
  stability we find throughout. Panel (b) is the same slice in observable
  space, where the manifold is a bounded patch whose edge is the image of
  the wall rather than a place where the metric degenerates.

pub_concept_twist: what the twist number measures. Frobenius integrability
  says the planes orthogonal to the blind direction mesh into surfaces only
  if they close under the bracket. We test that directly by walking a closed
  rectangle with steps that always lie in the local stiff plane. If the
  planes are integrable the walk stays on one surface and closes. If not it
  drifts along the blind direction, by an amount growing as the square of
  the step size. The integrable model here is exactly integrable by
  construction, so the contrast is qualitative rather than a matter of
  degree.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import pubstyle as ps
import battery_lib as bl
import smeftgeo as sg

ps.set_style()
WALL = sg.nda_wall(5.0)
PAIR = ("c16HψL", "c36Hψq")     # visible curvature, sloppy direction off-axis


# ======================================================================
def pub_concept_mbam():
    dy = sg.load_dy()
    Aw, Qw = bl.prep(dy["A"], dy["Q"], weight=dy["NSM"])
    idx = [dy["names"].index(PAIR[0]), dy["names"].index(PAIR[1])]
    names = [ps.tex(n) for n in PAIR]
    A = Aw[:, idx]
    Q = Qw[np.ix_(range(Qw.shape[0]), idx, idx)]

    def metric(c):
        J = A + np.einsum('oij,j->oi', Q, c)
        return J.T @ J

    def ride(v0, curved=True, h=3e-4, nmax=400000):
        c = np.zeros(2)
        v = v0 / np.sqrt(v0 @ metric(c) @ v0)
        path = [c.copy()]
        for _ in range(nmax):
            if curved:
                J = A + np.einsum('oij,j->oi', Q, c)
                vQv = np.einsum('i,oij,j->o', v, Q, v)
                a = -np.linalg.solve(metric(c), J.T @ vQv)
            else:
                a = np.zeros(2)
            c = c + h * v + 0.5 * h * h * a
            v = v + h * a
            path.append(c.copy())
            if np.abs(c).max() >= WALL:
                break
        return np.array(path)

    g0 = metric(np.zeros(2))
    ev, evec = np.linalg.eigh(g0)
    vs = evec[:, 0]

    # a fan of launch directions: every one of them leaves through a wall,
    # so there is no manifold boundary to stop the ride anywhere
    angles = np.linspace(0, np.pi, 13, endpoint=False)
    fan = [ride(np.array([np.cos(t), np.sin(t)])) for t in angles]
    fan += [ride(-np.array([np.cos(t), np.sin(t)])) for t in angles]
    sloppy = [ride(vs), ride(-vs)]

    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.DOUBLE, 3.0))

    # ---------- (a) coefficient space ----------
    a.add_patch(Rectangle((-WALL, -WALL), 2 * WALL, 2 * WALL, fill=False,
                          ec=ps.GRAY, lw=1.3, ls="--", zorder=1))
    for p_ in fan:
        a.plot(p_[:, 0], p_[:, 1], "-", color="0.72", lw=0.9, zorder=2)
        a.plot(p_[-1, 0], p_[-1, 1], ".", color="0.6", ms=3.5, zorder=3)
    for p_ in sloppy:
        a.plot(p_[:, 0], p_[:, 1], "-", color=ps.VERMIL, lw=2.2, zorder=4)
        a.plot(p_[-1, 0], p_[-1, 1], "o", color=ps.VERMIL, ms=6.5, zorder=5)
    a.set_xlim(-1.5 * WALL, 1.5 * WALL)
    a.set_ylim(-1.5 * WALL, 1.5 * WALL)
    a.set_xlabel(names[0])
    a.set_ylabel(names[1])
    a.text(0.5, 0.055, "validity range", transform=a.transAxes, fontsize=6.8,
           color=ps.GRAY, ha="center")
    a.text(0.035, 0.90, "grey: other launch directions\norange: sloppiest",
           transform=a.transAxes, fontsize=6.6, color="0.35", va="top")
    a.annotate("every ride ends on a wall", xy=sloppy[1][-1],
               xytext=(-1.35 * WALL, -1.05 * WALL), fontsize=6.8,
               color=ps.VERMIL, ha="left",
               arrowprops=dict(arrowstyle="->", color=ps.VERMIL, lw=0.9))
    ps.panel_tag(a, "a", outside=True)

    # ---------- (b) observable space ----------
    gx = np.linspace(-WALL, WALL, 21)
    G1, G2 = np.meshgrid(gx, gx)
    C = np.stack([G1.ravel(), G2.ravel()], 1)
    MU = bl.mu(A, Q, C)
    mean = MU.mean(0)
    Vt = np.linalg.svd(MU - mean, full_matrices=False)[2]

    def proj(X):
        return (bl.mu(A, Q, X) - mean) @ Vt[:2].T
    P = (MU - mean) @ Vt[:2].T
    P1 = P[:, 0].reshape(G1.shape)
    P2 = P[:, 1].reshape(G1.shape)
    for i in range(0, 21, 4):
        b.plot(P1[i, :], P2[i, :], "-", color="0.88", lw=0.6, zorder=1)
        b.plot(P1[:, i], P2[:, i], "-", color="0.88", lw=0.6, zorder=1)
    for e in [(0, slice(None)), (-1, slice(None)),
              (slice(None), 0), (slice(None), -1)]:
        b.plot(P1[e], P2[e], "-", color=ps.GRAY, lw=1.8, zorder=2)
    for p_ in fan:
        M = proj(p_)
        b.plot(M[:, 0], M[:, 1], "-", color="0.72", lw=0.8, zorder=3)
    for p_ in sloppy:
        M = proj(p_)
        b.plot(M[:, 0], M[:, 1], "-", color=ps.VERMIL, lw=2.2, zorder=4)
        b.plot(M[-1, 0], M[-1, 1], "o", color=ps.VERMIL, ms=6.5, zorder=5)
    b.set_xlabel("leading observable direction")
    b.set_ylabel("second observable direction")
    b.text(0.03, 0.06, "edge is the image of the wall",
           transform=b.transAxes, fontsize=6.8, color=ps.GRAY, ha="left")
    ps.panel_tag(b, "b", outside=True)
    fig.tight_layout()
    ps.save(fig, "pub_concept_mbam")
    print(f"   {len(fan) + len(sloppy)} rides launched, all ended on a wall")


# ======================================================================
def _models(nb=12):
    """(A, Q) for an exactly integrable model and an obstructed one."""
    t = np.linspace(0.4, 1.6, nb)
    f1, f2 = 0.9 + 0.4 * t, 0.6 + 0.5 * t ** 2

    # integrable: predictions depend on c only through u = c1 + c2 and c3,
    # so the blind direction is exactly (1,-1,0) everywhere and T = 0 exactly
    A_int = np.stack([f1, f1, f2], axis=1)
    Q_int = np.zeros((nb, 3, 3))
    for i in (0, 1):
        for j in (0, 1):
            Q_int[:, i, j] += 0.8 * f1          # quadratic in u
    Q_int[:, 2, 2] += 0.8 * f2                  # quadratic in c3

    # obstructed: curvature carrying a kinematic shape the linear terms
    # cannot mimic, coupling all three coefficients
    A_obs = np.stack([t, t ** 1.5, t ** 2], axis=1)
    sh = np.sin(3 * t)
    Q_obs = np.zeros((nb, 3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            Q_obs[:, i, j] = Q_obs[:, j, i] = 0.5 * 0.8 * sh
    return (A_int, Q_int), (A_obs, Q_obs)


def _walk(A, Q, side, nsub=80):
    """Closed rectangle walked with steps that stay in the local stiff plane.
    Returns the drift along the blind direction at each step."""
    n = A.shape[1]
    c = np.zeros(n)
    ref = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]

    def frame(c):
        J = A + np.einsum('oij,j->oi', Q, c)
        w, V = np.linalg.eigh(J.T @ J)
        return V[:, 0], V[:, 1:]            # blind direction, stiff plane
    v0 = frame(c)[0]
    drift = [0.0]
    for leg, sgn in [(0, +1), (1, +1), (0, -1), (1, -1)]:
        for _ in range(nsub):
            _, S = frame(c)
            d = S @ (S.T @ ref[leg])
            nrm = np.linalg.norm(d)
            if nrm < 1e-14:
                break
            c = c + sgn * (side / nsub) * d / nrm
            drift.append(float(v0 @ c))     # component along the blind dir
    return np.array(drift), abs(float(v0 @ c))


def pub_concept_twist():
    (Ai, Qi), (Ao, Qo) = _models()
    Ti = bl.kappa(*bl.prep(Ai, Qi, weight=np.ones(Ai.shape[0])),
                  np.ones(3), m=1, nsamp=40)
    To = bl.kappa(*bl.prep(Ao, Qo, weight=np.ones(Ao.shape[0])),
                  np.ones(3), m=1, nsamp=40)
    lab_i = "integrable, $T < 10^{-9}$"
    lab_o = f"obstructed, $T={To:.2f}$"

    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.DOUBLE, 2.9))

    # ---------- (a) drift along the blind direction, around one loop -------
    side = 0.6
    for (A, Q), col, lab in [((Ai, Qi), ps.BLUE, lab_i),
                             ((Ao, Qo), ps.VERMIL, lab_o)]:
        dr, gap = _walk(A, Q, side)
        x = np.linspace(0, 1, len(dr))
        a.plot(x, dr, "-", color=col, lw=1.8, label=lab)
        a.plot([1], [dr[-1]], "o", color=col, ms=6, zorder=4)
    a.axhline(0, color="0.6", lw=0.8, ls=":")
    for frac in (0.25, 0.5, 0.75):
        a.axvline(frac, color="0.9", lw=0.7, zorder=0)
    a.set_xlabel("fraction of the way around the loop")
    a.set_ylabel("drift along the blind direction")
    a.legend(loc="upper left", fontsize=6.8)
    a.annotate("does not close", xy=(1.0, _walk(Ao, Qo, side)[0][-1]),
               xytext=(0.55, 0.42), textcoords="axes fraction", fontsize=6.8,
               color=ps.VERMIL,
               arrowprops=dict(arrowstyle="->", color=ps.VERMIL, lw=0.9))
    ps.panel_tag(a, "a", outside=True)

    # ---------- (b) failure to close versus loop size ----------
    sides = np.geomspace(0.1, 0.9, 7)
    for (A, Q), col, lab in [((Ai, Qi), ps.BLUE, lab_i),
                             ((Ao, Qo), ps.VERMIL, lab_o)]:
        gaps = [max(_walk(A, Q, s)[1], 1e-18) for s in sides]
        b.loglog(sides, gaps, "o-", color=col, ms=4.5, lw=1.5, label=lab)
        if col == ps.VERMIL:
            ref = gaps[-1] * (sides / sides[-1]) ** 2
            b.loglog(sides, ref, "--", color=ps.GRAY, lw=1.0)
    b.text(0.62, 0.72, "slope 2", transform=b.transAxes, fontsize=6.8,
           color=ps.GRAY, rotation=26)
    b.text(0.36, 0.11, "machine precision", transform=b.transAxes,
           fontsize=6.8, color=ps.BLUE)
    b.set_xlabel("loop side")
    b.set_ylabel("failure to close")
    b.legend(loc="upper left", fontsize=6.8)
    ps.panel_tag(b, "b", outside=True)
    fig.tight_layout()
    ps.save(fig, "pub_concept_twist")
    print(f"   T integrable = {Ti:.2e}, T obstructed = {To:.3f}")


if __name__ == "__main__":
    pub_concept_mbam()
    pub_concept_twist()
