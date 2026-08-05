#!/usr/bin/env python3
"""
Two conceptual illustrations for the paper. These are schematics, drawn to
explain the geometry, not measurements.

fig_schematic_mbam: why model reduction in a truncated EFT needs a wall.
  (a) The classical sloppy-model picture: the model manifold is a ribbon that
      narrows to an edge, the metric degenerates there, and that edge is the
      reduced model MBAM rides to.
  (b) A truncated EFT: the manifold keeps going, nothing degenerates, and the
      ride is stopped by the validity wall instead.
  (c) Back in coefficient space, the wall the geodesic leaves through names
      the coefficient that is set to zero.

fig_schematic_twist: what integrability means for a blind direction.
  (a) The planes orthogonal to the blind direction stack into surfaces. The
      blind direction is normal to all of them, each surface is a level set
      of one function, and that function is the relation.
  (b) The planes rotate as one moves, so no surface stays tangent to them.
      A loop taken inside the planes returns displaced along the blind
      direction, and there is no relation to write down.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform
import pubstyle as ps

ps.set_style()
INK = "#2A2E34"
SURF = "#B9C6D2"
WALLC = "#8A8A8A"


class Arrow3D(FancyArrowPatch):
    """A 3D arrow that survives the projection."""

    def __init__(self, xs, ys, zs, *args, **kw):
        super().__init__((0, 0), (0, 0), *args, **kw)
        self._xyz = (xs, ys, zs)

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._xyz
        xp, yp, _ = proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xp[0], yp[0]), (xp[1], yp[1]))
        return float(np.min(zs))


def _bare(ax):
    ax.set_axis_off()
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)


# ======================================================================
def fig_schematic_mbam():
    fig = plt.figure(figsize=(ps.DOUBLE, 2.5))

    # ---------- (a) classical: the ribbon narrows to a real edge ----------
    axa = fig.add_subplot(1, 3, 1, projection="3d")
    u = np.linspace(0, 1, 60)
    v = np.linspace(-1, 1, 40)
    U, V = np.meshgrid(u, v)
    width = (1 - U) ** 0.9                      # narrows to zero at u = 1
    X, Y = U, V * width
    Z = 0.30 * U ** 2 - 0.25 * (V * width) ** 2
    axa.plot_surface(X, Y, Z, color=SURF, alpha=0.85, linewidth=0,
                     rstride=2, cstride=2, shade=True, zorder=1)
    axa.plot(u, 0 * u, 0.30 * u ** 2, color=ps.VERMIL, lw=2.2, zorder=6)
    axa.add_artist(Arrow3D([0.80, 0.99], [0, 0], [0.30 * .8 ** 2, 0.30 * .99 ** 2],
                           mutation_scale=9, lw=2.2, arrowstyle="-|>",
                           color=ps.VERMIL, zorder=7))
    axa.scatter([0], [0], [0], color=INK, s=16, zorder=8)
    axa.text(0.02, 0.28, 0.30, "best fit", fontsize=6.6, color=INK)
    axa.text(0.62, -0.05, 0.62, "boundary:\nthe metric\ndegenerates",
             fontsize=6.6, color=ps.VERMIL, ha="center")
    axa.view_init(elev=26, azim=-58)
    _bare(axa)
    axa.set_title("(a)  classical sloppy model", fontsize=7.6, pad=-2,
                  loc="left")

    # ---------- (b) EFT: nothing degenerates, a wall cuts it ----------
    axb = fig.add_subplot(1, 3, 2, projection="3d")
    U, V = np.meshgrid(np.linspace(0, 1.25, 70), np.linspace(-1, 1, 40))
    X, Y = U, V * 0.85
    Z = 0.30 * U ** 2 - 0.25 * (V * 0.85) ** 2
    keep = U <= 0.82
    Xa, Ya, Za = np.where(keep, X, np.nan), np.where(keep, Y, np.nan), np.where(keep, Z, np.nan)
    Xb, Yb, Zb = np.where(~keep, X, np.nan), np.where(~keep, Y, np.nan), np.where(~keep, Z, np.nan)
    axb.plot_surface(Xa, Ya, Za, color=SURF, alpha=0.85, linewidth=0,
                     rstride=2, cstride=2, shade=True, zorder=1)
    axb.plot_surface(Xb, Yb, Zb, color=SURF, alpha=0.22, linewidth=0,
                     rstride=2, cstride=2, shade=False, zorder=1)
    vv = np.linspace(-1, 1, 30) * 0.85
    axb.plot(0.82 + 0 * vv, vv, 0.30 * 0.82 ** 2 - 0.25 * vv ** 2,
             color=WALLC, lw=2.0, ls="--", zorder=6)
    uu = np.linspace(0, 0.82, 40)
    axb.plot(uu, 0 * uu, 0.30 * uu ** 2, color=ps.VERMIL, lw=2.2, zorder=6)
    axb.add_artist(Arrow3D([0.66, 0.815], [0, 0],
                           [0.30 * .66 ** 2, 0.30 * .815 ** 2],
                           mutation_scale=9, lw=2.2, arrowstyle="-|>",
                           color=ps.VERMIL, zorder=7))
    axb.scatter([0], [0], [0], color=INK, s=16, zorder=8)
    axb.text(0.72, 0.02, 0.52, "validity wall", fontsize=6.6, color=WALLC,
             ha="center")
    axb.text(1.02, -0.02, 0.18, "manifold\ncontinues", fontsize=6.6,
             color="0.55", ha="center")
    axb.view_init(elev=26, azim=-58)
    _bare(axb)
    axb.set_title("(b)  truncated EFT", fontsize=7.6, pad=-2, loc="left")

    # ---------- (c) coefficient space: the exit names the drop ----------
    axc = fig.add_subplot(1, 3, 3)
    axc.add_patch(Rectangle((-1, -1), 2, 2, fill=False, ec=WALLC, lw=1.3,
                            ls="--"))
    s = np.linspace(0, 1, 200)
    px = 1.0 * s
    py = -0.55 * s + 0.22 * s ** 2
    axc.plot(px, py, "-", color=ps.VERMIL, lw=2.2)
    axc.add_patch(FancyArrowPatch((px[-12], py[-12]), (px[-1], py[-1]),
                                  arrowstyle="-|>", mutation_scale=9,
                                  color=ps.VERMIL, lw=2.2))
    axc.plot([0], [0], "o", color=INK, ms=4.5)
    axc.plot([1], [py[-1]], "o", color=ps.VERMIL, ms=6)
    axc.annotate("", xy=(1, py[-1]), xytext=(0, py[-1]),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.0,
                                 ls=":"))
    axc.text(0.45, py[-1] + 0.10, "set this one to zero", fontsize=6.6,
             color=INK, ha="center")
    axc.text(0, -1.28, "$c_1$", fontsize=8, ha="center")
    axc.text(-1.30, 0, "$c_2$", fontsize=8, va="center")
    axc.text(0, 1.10, "validity range", fontsize=6.6, color=WALLC,
             ha="center")
    axc.set_xlim(-1.45, 1.45)
    axc.set_ylim(-1.45, 1.45)
    axc.set_aspect("equal")
    axc.set_axis_off()
    axc.set_title("(c)  the exit names the drop", fontsize=7.6, pad=2,
                  loc="left")

    fig.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.02,
                        wspace=0.02)
    ps.save(fig, "fig_schematic_mbam")


# ======================================================================
def _plane(ax, centre, normal, size=0.42, color=SURF, alpha=0.75, lw=0.6):
    """Draw a small parallelogram with the given unit normal."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    tmp = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    c = np.asarray(centre, float)
    pts = [c + size * (a * e1 + b * e2) for a, b in
           [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    poly = Poly3DCollection([pts], facecolor=color, edgecolor=INK,
                            alpha=alpha, linewidths=lw)
    ax.add_collection3d(poly)
    return e1, e2


def fig_schematic_twist():
    fig = plt.figure(figsize=(ps.DOUBLE, 2.7))

    # ---------- (a) integrable: the planes stack into surfaces ----------
    axa = fig.add_subplot(1, 2, 1, projection="3d")
    xs = np.linspace(-1, 1, 24)
    X, Y = np.meshgrid(xs, xs)
    for k, z in enumerate([-0.62, 0.0, 0.62]):
        Z = z + 0.10 * X ** 2 - 0.06 * Y ** 2       # gently curved sheets
        axa.plot_surface(X, Y, Z, color=SURF, alpha=0.55, linewidth=0,
                         rstride=3, cstride=3, shade=False, zorder=1)
    for (px, py) in [(-0.55, -0.5), (0.5, 0.45), (0.0, -0.1)]:
        z0 = 0.10 * px ** 2 - 0.06 * py ** 2
        axa.add_artist(Arrow3D([px, px], [py, py], [z0, z0 + 0.52],
                               mutation_scale=8, lw=1.7, arrowstyle="-|>",
                               color=ps.VERMIL, zorder=9))
    axa.text(-0.15, -1.05, 1.24, "blind direction", fontsize=6.8,
             color=ps.VERMIL)
    axa.text(0.95, 0.9, -0.30, r"$\phi = $ const", fontsize=7.2, color=INK)
    axa.set_xlim(-1.1, 1.1); axa.set_ylim(-1.1, 1.1); axa.set_zlim(-1.0, 1.3)
    axa.view_init(elev=18, azim=-62)
    _bare(axa)
    axa.set_title("(a)  $T = 0$: the surfaces exist, and their label is the "
                  "relation", fontsize=7.4, pad=-4, loc="left")

    # ---------- (b) obstructed: the planes twist, no surface fits --------
    axb = fig.add_subplot(1, 2, 2, projection="3d")
    for k, z in enumerate(np.linspace(-0.75, 0.75, 5)):
        ang = 0.85 * z                              # the planes rotate with z
        nrm = [np.sin(ang), -np.sin(0.45 * ang), np.cos(ang)]
        _plane(axb, (0, 0, z), nrm, size=0.62, alpha=0.62)
        axb.add_artist(Arrow3D([0, 0.42 * nrm[0]], [0, 0.42 * nrm[1]],
                               [z, z + 0.42 * nrm[2]], mutation_scale=8,
                               lw=1.7, arrowstyle="-|>", color=ps.VERMIL,
                               zorder=9))
    # the loop that fails to close, drawn on the side
    th = np.linspace(0, 2 * np.pi, 120)
    lx, ly = 0.30 * np.cos(th) + 0.95, 0.30 * np.sin(th) - 0.35
    lz = -0.75 + 0.34 * th / (2 * np.pi)            # drifts upward, no closure
    axb.plot(lx, ly, lz, color=INK, lw=1.5, zorder=10)
    axb.add_artist(Arrow3D([lx[-6], lx[-1]], [ly[-6], ly[-1]],
                           [lz[-6], lz[-1]], mutation_scale=8, lw=1.5,
                           arrowstyle="-|>", color=INK, zorder=11))
    axb.plot([lx[0]], [ly[0]], [lz[0]], "o", color=INK, ms=3.5, mfc="white",
             zorder=11)
    axb.text(1.30, -0.35, -0.30, "a loop inside\nthe planes\ndoes not close",
             fontsize=6.6, color=INK, ha="center")
    axb.text(-0.20, -1.05, 1.16, "blind direction", fontsize=6.8,
             color=ps.VERMIL)
    axb.set_xlim(-1.1, 1.4); axb.set_ylim(-1.1, 1.1); axb.set_zlim(-1.0, 1.3)
    axb.view_init(elev=18, azim=-62)
    _bare(axb)
    axb.set_title("(b)  $T > 0$: the planes twist, so no surface fits them",
                  fontsize=7.4, pad=-4, loc="left")

    fig.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.0,
                        wspace=0.02)
    ps.save(fig, "fig_schematic_twist")


if __name__ == "__main__":
    fig_schematic_mbam()
    fig_schematic_twist()
