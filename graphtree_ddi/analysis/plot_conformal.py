"""Conformal referral results figure (three panels), drawn from saved tables only.

Reads conformal_transductive.csv and conformal_inductive.csv and writes
fig_conformal.png (300 dpi) and fig_conformal.pdf into ``--out_dir``. Exits
non-zero if any text artist leaves the figure or two text artists overlap.

    python -m graphtree_ddi.analysis.plot_conformal \
        --in_dir results/analysis \
        --out_dir results/analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "graphtree_ddi" / "paths.py").is_file():
            return p
    raise RuntimeError("cannot locate repository root (graphtree_ddi/paths.py)")


_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FUS = "#2D6E69"
GREY = "#A6A6A6"
OK = {"blue": "#0072B2", "orange": "#E69F00", "purple": "#CC79A7", "sky": "#56B4E9", "verm": "#D55E00"}
ALPHAS = (0.005, 0.01, 0.02, 0.05)
CAL_DRUG = "drug-level CRC"
CAL_POOLED = "pooled (catalogued-pair) calibration"
DPI = 300
W_CM, H_CM = 16.0, 6.6

SHORT = {
    "Exp-5": "GraphTree-DDI", "Exp-3": "R-GCN (hybrid KG)", "Exp-2": "R-GCN (DDI-only)",
    "DistMult": "DistMult", "MLP": "MLP", "ComplEx": "ComplEx", "DDIMDL": "DDIMDL-style",
    "DrugPrior": "Drug prior (B3)", "Exp-1": "XGBoost", "DeepDDI_SSP": "DeepDDI-style", "Exp-4": "R-GAT",
}


def pick(df, metric, alpha, beta, **eq):
    m = (df["metric"] == metric) & np.isclose(df["alpha"], alpha) & np.isclose(df["beta"], beta)
    for k, v in eq.items():
        m &= df[k] == v
    sub = df[m]
    if len(sub) != 1:
        raise SystemExit(f"expected exactly one row for {metric} a={alpha} b={beta} {eq}, got {len(sub)}")
    r = sub.iloc[0]
    return float(r["mean"]), float(r["sd"])


def curve(df, metric, beta, **eq):
    vals = [pick(df, metric, a, beta, **eq) for a in ALPHAS]
    return np.array([v[0] for v in vals]), np.array([v[1] for v in vals])


def log_err(y, s):
    lo = np.minimum(s, y * 0.9)  # keep lower whisker positive on log axes
    return np.vstack([lo, s])


def pct_fmt(v, _=None):
    if v <= 0:
        return ""
    p = v * 100
    return f"{p:g}%"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in_dir", default="", help="directory containing the conformal CSV tables")
    parser.add_argument(
        "--transductive", default="",
        help="conformal_transductive.csv (default: <in_dir>/conformal_transductive.csv)",
    )
    parser.add_argument(
        "--inductive", default="",
        help="conformal_inductive.csv (default: <in_dir>/conformal_inductive.csv)",
    )
    parser.add_argument("--out_dir", required=True, help="directory for fig_conformal.png and fig_conformal.pdf")
    parser.add_argument(
        "--method_names", default="",
        help="JSON display-name map (default: resources/method_names_11.json)",
    )
    return parser.parse_args(argv)


def _table_paths(args) -> tuple[Path, Path]:
    in_dir = Path(args.in_dir) if str(args.in_dir).strip() else None
    trans = Path(args.transductive) if str(args.transductive).strip() else None
    inductive = Path(args.inductive) if str(args.inductive).strip() else None
    if trans is None or inductive is None:
        if in_dir is None:
            raise SystemExit("pass --in_dir or both --transductive and --inductive")
        trans = trans or (in_dir / "conformal_transductive.csv")
        inductive = inductive or (in_dir / "conformal_inductive.csv")
    for path in (trans, inductive):
        if not path.is_file():
            raise SystemExit(f"missing table: {path}")
    return trans, inductive


def main(argv=None) -> int:
    args = parse_args(argv)
    trans_path, ind_path = _table_paths(args)
    fig_dir = Path(args.out_dir)
    names_path = (
        Path(args.method_names)
        if str(args.method_names).strip()
        else (_REPO_ROOT / "resources" / "method_names_11.json")
    )
    if not names_path.is_file():
        raise SystemExit(f"method name map not found: {names_path}")
    names = json.loads(names_path.read_text(encoding="utf-8"))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6,
    })
    dt = pd.read_csv(trans_path)
    di = pd.read_csv(ind_path)
    missing = [k for k in SHORT if k not in names]
    if missing:
        raise SystemExit(f"methods missing from method_names_11.json: {missing}")

    fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54))
    cm = 1.0 / 2.54

    def add_axes(x0, y0, w, h):  # all in cm from bottom-left
        return fig.add_axes([x0 / W_CM, y0 / H_CM, w / W_CM, h / H_CM])

    top = H_CM - 0.75           # top edge of axes (title strip above)
    bot_common = 3.02           # bottom edge of axes (b, c)
    axA = add_axes(2.66, 0.95, 2.86, top - 0.95)
    axB = add_axes(6.55, bot_common, 3.55, top - bot_common)
    axC = add_axes(11.55, bot_common, 3.95, top - bot_common)

    # ---------------- (a) catalogued route -----------------
    rows = []
    for m in SHORT:
        mu, sd = pick(dt, "review_frac", 0.01, 0.01, method=m)
        rows.append((m, mu * 100, sd * 100))
    rows.sort(key=lambda r: r[1])
    n = len(rows)
    ypos = np.arange(n)[::-1]  # smallest (GraphTree-DDI) at top
    xlo, xhi = 0.03, 450.0
    axA.set_xscale("log")
    axA.set_xlim(xlo, xhi)
    for (m, mu, sd), y in zip(rows, ypos):
        col = FUS if m == "Exp-5" else GREY
        axA.barh(y, mu - xlo, left=xlo, height=0.62, color=col, edgecolor="none", zorder=2)
        err = log_err(np.array([mu]), np.array([sd]))
        axA.errorbar([mu], [y], xerr=err, color="#222222", elinewidth=0.6, capsize=1.3, capthick=0.6,
                     linestyle="none", zorder=3)
        axA.text((mu + sd) * 1.18, y, f"{mu:.2f}%" if mu < 10 else f"{mu:.1f}%", ha="left", va="center",
                 fontsize=7, fontweight="bold" if m == "Exp-5" else "normal",
                 color=FUS if m == "Exp-5" else "#333333", zorder=4)
    axA.set_yticks(ypos)
    axA.set_yticklabels([SHORT[m] for m, _, _ in rows])
    for lab, (m, _, _) in zip(axA.get_yticklabels(), rows):
        if m == "Exp-5":
            lab.set_fontweight("bold")
            lab.set_color(FUS)
    axA.set_ylim(-0.6, n - 0.4)
    axA.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    axA.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    axA.xaxis.set_minor_locator(NullLocator())
    axA.spines["bottom"].set_bounds(xlo, 100)
    axA.set_xlabel("Pharmacist review (% of pairs)")

    # ---------------- (b) uncatalogued route -----------------
    xs = np.array(ALPHAS)
    spec_b = (
        ("Exp-1", "XGBoost (routed branch)", OK["verm"], "o", "-"),
        ("MLP", "MLP", OK["blue"], "s", "--"),
        ("DDIMDL", "DDIMDL-style", OK["orange"], "D", "-."),
        ("DistMult", "DistMult", OK["purple"], "^", ":"),
        ("Exp-5", "GraphTree-DDI (unrouted)", FUS, "v", "-"),
    )
    for m, lab, col, mk, ls in spec_b:
        y, s = curve(di, "review_frac", 0.01, method=m, partition="S1", calibration=CAL_DRUG)
        axB.errorbar(xs, y, yerr=s, color=col, marker=mk, linestyle=ls, linewidth=0.9, markersize=3.0,
                     capsize=1.4, elinewidth=0.6, capthick=0.6, label=lab)
    axB.set_xscale("log")
    axB.set_ylim(0, 1)
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    axB.set_xlabel(r"Severe-miss budget $\alpha$")
    axB.set_ylabel("Pharmacist review fraction")

    # ---------------- (c) severe-miss control -----------------
    spec_c = (
        (dt, {"method": "Exp-5"}, "miss_sev", "GraphTree-DDI, catalogued (pairs)", FUS, "o", "-"),
        (di, {"method": "Exp-1", "partition": "S1", "calibration": CAL_DRUG}, "drug_miss_sev",
         "XGBoost, S1 drug-level CRC", OK["verm"], "s", "-"),
        (di, {"method": "Exp-1", "partition": "S1", "calibration": CAL_POOLED}, "drug_miss_sev",
         "XGBoost, S1 catalogued thresholds", OK["verm"], "^", "--"),
        (di, {"method": "Exp-5", "partition": "S1", "calibration": CAL_POOLED}, "drug_miss_sev",
         "GraphTree-DDI (unrouted), S1\ncatalogued thresholds", FUS, "D", "--"),
    )
    axC.set_xscale("log")
    axC.set_yscale("log")
    for frame, eq, metric, lab, col, mk, ls in spec_c:
        y, s = curve(frame, metric, 0.01, **eq)
        axC.errorbar(xs, y, yerr=log_err(y, s), color=col, marker=mk, linestyle=ls, linewidth=0.9,
                     markersize=3.0, capsize=1.4, elinewidth=0.6, capthick=0.6, label=lab, zorder=3)
    axC.plot([0.003, 2.0], [0.003, 2.0], color="#4d4d4d", linestyle="--", linewidth=0.7, zorder=1,
             label=r"Nominal ($y = x$)")
    axC.set_ylim(0.002, 1.6)
    axC.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0]))
    axC.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    axC.yaxis.set_minor_locator(NullLocator())
    axC.spines["left"].set_bounds(0.002, 1.0)
    axC.set_xlabel(r"Severe-miss budget $\alpha$")
    axC.set_ylabel("Realized severe-miss rate")

    for ax in (axB, axC):
        ax.set_xlim(0.0038, 0.07)
        ax.xaxis.set_major_locator(FixedLocator(list(ALPHAS)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}%"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    for ax in (axA, axB, axC):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2.2, width=0.5, pad=1.2)

    # legends below the axes (figure-anchored so they never touch data)
    def legend(ax, x_cm, ncol, **kw):
        h, l = ax.get_legend_handles_labels()
        nom = [i for i, t in enumerate(l) if t.startswith("Nominal")]
        order = [i for i in range(len(l)) if i not in nom] + nom
        h, l = [h[i] for i in order], [l[i] for i in order]
        nom = [i for i, t in enumerate(l) if t.startswith('Nominal')]
        order = [i for i in range(len(l)) if i not in nom] + nom
        h, l = [h[i] for i in order], [l[i] for i in order]
        return ax.legend(h, l, loc="upper left", bbox_to_anchor=(x_cm / W_CM, (bot_common - 1.0) / H_CM),
                         bbox_transform=fig.transFigure, ncol=ncol, frameon=False, fontsize=7,
                         handlelength=1.9, columnspacing=0.8, labelspacing=0.28, borderaxespad=0.0,
                         handletextpad=0.4, borderpad=0.0, **kw)
    legend(axB, 6.55, 1)  # left edge = axB left edge (clear gap from panel (a), which ends at 5.52 cm)
    legend(axC, 10.55, 1)

    # panel tags + titles in figure coordinates (cm)
    ty = (H_CM - 0.62) / H_CM
    def tag(x_cm, t, title):
        fig.text(x_cm / W_CM, ty, t, fontweight="bold", ha="left", va="bottom", fontsize=7)
        fig.text((x_cm + 0.62) / W_CM, ty, title, ha="left", va="bottom", fontsize=7)
    tag(0.12, "(a)", "Catalogued route (α = β = 1%)")
    tag(5.55, "(b)", "Uncatalogued route (S1)")
    tag(10.55, "(c)", "Severe-miss control")

    fig_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = fig_dir / "fig_conformal.png", fig_dir / "fig_conformal.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)

    outside, overlaps = audit(fig, (axA, axB, axC))
    w_cm, h_cm = (v * 2.54 for v in fig.get_size_inches())
    plt.close(fig)
    print(f"[figure] {w_cm:.2f} x {h_cm:.2f} cm; text outside={len(outside)} overlaps={len(overlaps)}")
    for o in outside:
        print("[figure] outside:", o)
    for o in overlaps:
        print("[figure] overlap:", o)
    print(f"[write] {png}\n[write] {pdf}")
    return 0 if not outside and not overlaps else 1


def audit(fig, axes):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fb = fig.bbox
    arts = list(fig.texts)
    for ax in axes:
        arts += [ax.title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        arts += ax.get_xticklabels() + ax.get_yticklabels()
        lg = ax.get_legend()
        if lg is not None:
            arts += lg.get_texts()
    items = []
    for t in arts:
        s = t.get_text()
        if not t.get_visible() or s is None or not str(s).strip():
            continue
        bb = t.get_window_extent(rend)
        if bb.width <= 0 and bb.height <= 0:
            continue
        items.append((str(s).replace("\n", " / "), bb))
    outside = [(s, tuple(round(v, 1) for v in (b.x0, b.y0, b.x1, b.y1))) for s, b in items
               if b.x0 < fb.x0 - 1 or b.y0 < fb.y0 - 1 or b.x1 > fb.x1 + 1 or b.y1 > fb.y1 + 1]
    overlaps = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            ix = min(a.x1, b.x1) - max(a.x0, b.x0)
            iy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ix > 0.8 and iy > 0.8:
                overlaps.append((items[i][0], items[j][0], round(ix, 1), round(iy, 1)))
    # no text may sit on top of a legend of another panel / the axes of the next panel
    for ax in axes:
        lg = ax.get_legend()
        if lg is None:
            continue
        lb = lg.get_window_extent(rend)
        if lb.x0 < fb.x0 - 1 or lb.x1 > fb.x1 + 1 or lb.y0 < fb.y0 - 1:
            outside.append(("legend", tuple(round(v, 1) for v in (lb.x0, lb.y0, lb.x1, lb.y1))))
        for ax2 in axes:
            ab = ax2.get_window_extent(rend)
            if min(lb.x1, ab.x1) - max(lb.x0, ab.x0) > 0 and min(lb.y1, ab.y1) - max(lb.y0, ab.y0) > 0.5:
                overlaps.append(("legend", "axes", 0, 0))
    # a legend must keep a horizontal gap (>= 0.3 cm) from every text artist and the axes box of the
    # OTHER panels it overlaps vertically (bar value labels, tick labels, axis labels, other legend)
    min_gap = 0.3 / 2.54 * fig.dpi
    for ax in axes:
        lg = ax.get_legend()
        if lg is None:
            continue
        lb = lg.get_window_extent(rend)
        for ax2 in axes:
            if ax2 is ax:
                continue
            others = [("axes box", ax2.get_window_extent(rend))]
            cand = [ax2.title, ax2.xaxis.label, ax2.yaxis.label] + list(ax2.texts)
            cand += ax2.get_xticklabels() + ax2.get_yticklabels()
            lg2 = ax2.get_legend()
            if lg2 is not None:
                cand += lg2.get_texts()
            for t in cand:
                s = t.get_text()
                if t.get_visible() and s is not None and str(s).strip():
                    others.append((repr(str(s).replace("\n", " / ")), t.get_window_extent(rend)))
            for name, ob in others:
                if min(lb.y1, ob.y1) - max(lb.y0, ob.y0) <= 0:
                    continue  # no vertical overlap
                gap = max(lb.x0, ob.x0) - min(lb.x1, ob.x1)  # negative = horizontal overlap
                if gap < min_gap:
                    overlaps.append(("legend gap", f"{name} of other panel", round(gap, 1),
                                     f"< {min_gap:.1f}px"))
    # legends of neighbouring panels must not collide
    lgs = [ax.get_legend().get_window_extent(rend) for ax in axes if ax.get_legend() is not None]
    for i in range(len(lgs)):
        for j in range(i + 1, len(lgs)):
            if min(lgs[i].x1, lgs[j].x1) - max(lgs[i].x0, lgs[j].x0) > 0.5:
                overlaps.append(("legend", "legend", 0, 0))
    return outside, overlaps


if __name__ == "__main__":
    sys.exit(main())
