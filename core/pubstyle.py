"""
Shared publication style for all paper figures (REVTeX two-column).

Usage:
    import pubstyle as ps
    ps.set_style()
    fig, ax = plt.subplots(figsize=(ps.SINGLE, 2.6))
    ...
    ps.save(fig, "pub_spectrum")

Conventions:
  - Computer Modern look via mathtext (no system latex needed).
  - No figure titles: captions carry the message; panels get (a)/(b) tags
    and informative axis labels / in-panel annotations so each figure is
    readable on its own.
  - Okabe-Ito colorblind-safe colors.
  - Warsaw-basis operator names via TEX[...] (data files use Adam's
    internal names).
"""
import matplotlib as mpl

SINGLE = 3.375        # REVTeX single-column width in inches
DOUBLE = 7.0          # full-width figure*

# Okabe-Ito
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
VERMIL = "#D55E00"
GRAY = "#7F7F7F"

# Adam's internal coefficient names -> Warsaw-basis TeX
TEX = {
    "c36HψL": r"$C_{H\ell}^{(3)}$",
    "c36Hψq": r"$C_{Hq}^{(3)}$",
    "c16HψL": r"$C_{H\ell}^{(1)}$",
    "c16Hψq": r"$C_{Hq}^{(1)}$",
    "cLL":    r"$C_{\ell\ell}$",
    "cHD":    r"$C_{HD}$",
    "cHu6":   r"$C_{Hu}$",
    "cHd6":   r"$C_{Hd}$",
    "cHe6":   r"$C_{He}$",
    "cHWB":   r"$C_{HWB}$",
    "cHW":    r"$C_{HW}$",
    "cHB":    r"$C_{HB}$",
    "cLQ3":   r"$C_{\ell q}^{(3)}$",
    "cLQ1":   r"$C_{\ell q}^{(1)}$",
    "ced":    r"$C_{ed}$",
    "ceu":    r"$C_{eu}$",
    "ceQ":    r"$C_{qe}$",
    "cLu":    r"$C_{\ell u}$",
    "cLd":    r"$C_{\ell d}$",
    "δGF6":   r"$\delta G_F$",
}


def tex(name):
    return TEX.get(name, name)


def set_style():
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })


def panel_tag(ax, letter, x=0.02, y=0.96, outside=False):
    """(a)/(b) tag. outside=True puts it above the top-left corner so it can
    never collide with data."""
    if outside:
        ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=9, fontweight="bold")
    else:
        ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="left",
                va="top", fontsize=9, fontweight="bold")


def save(fig, name, outdir="/Users/user/Downloads"):
    fig.savefig(f"{outdir}/{name}.pdf")
    fig.savefig(f"{outdir}/{name}.png")
    print(f"wrote {name}.pdf/.png")
