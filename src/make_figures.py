"""C1-R / ICICT 2027 manuscript figures.

Read-only consumer of the four published hash-pinned G5 aggregate files. No label, score or
feature file is opened. Every plotted number is either (a) read from a G5 JSON
field, or (b) an arithmetic function of such fields that follows the frozen
definitions (five-run mean, min, max; pair weight w_u; count bound B_u).

Usage:
    python make_figures.py <g5_results_dir> <figure_output_dir> [<data_dump.json>]

Outputs (vector PDF, fonts embedded as TrueType): optional_fig_cohorts.pdf,
fig2_yearly_auc.pdf, fig3_full_unseen.pdf, plus PNG previews for visual QA and a
JSON dump of every plotted value.
"""
import hashlib
import json
import os
import sys
from fractions import Fraction as Fr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EXPECTED_SHA256 = {
    "G5_per_run_metrics.json": "b4616e70c70a992580147aa7372fde1367c7afef45a0c0074980884954f8004b",
    "G5_model_summary.json": "58d4708a5280109a5558e82f1cec74ae0d9b490d7f799bc7eb42e0682965d8f0",
    "G5_model_independent_counts.json": "97b1c4bec9fd655fd06d65db91e99dae6d49c04cfac9be75a80262575b616080",
    "G5_E3.json": "4d66548b4aecc0bf21bf00f1bfb3ddff0cfe625ca5a29730716ce8ce8a1bb947",
}

UNITS = ["IID_2013", "IID_2014", "NEAR_2016", "NEAR_2017"] + ["FAR_%d" % y for y in range(2018, 2026)]
TICK = {"IID_2013": "Dec'13", "IID_2014": "Aug'14"}  # last observed month of each training year
MODELS = ["svm", "lgbm", "xgb", "mlp"]
LABEL = {"svm": "SVM", "lgbm": "LightGBM", "xgb": "XGBoost", "mlp": "MLP"}
# Colour is redundant with marker shape and dash pattern (grayscale-safe).
STYLE = {
    "svm": dict(color="#184f95", marker="o", ls="-"),
    "lgbm": dict(color="#d95926", marker="s", ls="--"),
    "xgb": dict(color="#138a5f", marker="^", ls="-."),
    "mlp": dict(color="#6250d6", marker="D", ls=":"),
}
RUNS = ["1", "2", "3", "4", "5"]
INK = "#1a1a1a"
MUTED = "#5f5f5f"
GRID = "#d9d9d9"
W_IN = 4.80  # 122 mm text width of svproc.cls


def load_verified(g5dir):
    data = {}
    for name, want in EXPECTED_SHA256.items():
        raw = open(os.path.join(g5dir, name), "rb").read()
        got = hashlib.sha256(raw).hexdigest()
        if got != want:
            raise SystemExit("SHA-256 MISMATCH for %s: %s != %s -- STOP" % (name, got, want))
        data[name] = json.loads(raw)
    return data


def setup():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
        "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.1, "lines.markersize": 4.2,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


def unit_stats(per_run, unit, key):
    """Five-run mean, min, max of a per-unit quantity (exact rational arithmetic)."""
    out = {}
    for m in MODELS:
        v = [Fr(per_run[m][r]["units"][unit][key]["exact"]) for r in RUNS]
        out[m] = (float(sum(v) / 5), float(min(v)), float(max(v)))
    return out


def regime_decor(ax, y_text, fontsize=6.8):
    """Vertical separators and regime labels; x positions are unit indices."""
    for x in (1.5, 3.5):
        ax.axvline(x, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=0)
    tr = ax.get_xaxis_transform()
    for x, s in ((0.5, "IID"), (2.5, "NEAR (secondary)"), (7.5, "FAR (primary horizon)")):
        ax.text(x, y_text, s, transform=tr, ha="center", va="bottom", fontsize=fontsize, color=MUTED)


# ------------------------------------------------------------------ Figure 1
def fig1(outdir, counts):
    fig = plt.figure(figsize=(W_IN, 0.80))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 16.8)
    ax.axis("off")
    fs = 6.6
    # --- timeline row
    years = [2013, 2014, 2016, 2017] + list(range(2018, 2026))
    x0, bw, gap, y0, bh = 2.0, 7.05, 0.75, 3.9, 4.8
    xs = {}
    x = x0
    for y in years:
        if y == 2016:
            x += 2.2  # 2015 is absent from the release
        xs[y] = x
        if y in (2013, 2014):
            ax.add_patch(Rectangle((x, y0), bw * 0.72, bh, fc="#c9c9c9", ec=INK, lw=0.6))
            ax.add_patch(Rectangle((x + bw * 0.72, y0), bw * 0.28, bh, fc="white", ec=INK, lw=0.6, hatch="////"))
        elif y in (2016, 2017):
            ax.add_patch(Rectangle((x, y0), bw, bh, fc="white", ec=INK, lw=0.6, hatch="...."))
        else:
            ax.add_patch(Rectangle((x, y0), bw, bh, fc="white", ec=INK, lw=0.6))
        ax.text(x + bw / 2, y0 - 1.0, str(y), ha="center", va="top", fontsize=fs, color=INK)
        x += bw + gap
    ax.text(xs[2016] - 1.45, y0 + bh / 2, "//", ha="center", va="center", fontsize=fs, color=MUTED)

    def brace(xa, xb, text, ha="center"):
        yb = y0 + bh + 0.9
        ax.plot([xa, xa, xb, xb], [yb, yb + 0.9, yb + 0.9, yb], color=INK, lw=0.6, solid_capstyle="butt")
        xt = xa if ha == "left" else (xa + xb) / 2
        ax.text(xt, yb + 1.4, text, ha=ha, va="bottom", fontsize=fs, linespacing=1.2)

    n_train = 172328
    n_eval = sum(counts[u]["F"]["n"] for u in UNITS)
    brace(xs[2013], xs[2014] + bw, "TRAIN (gray)\nIID (hatched)", ha="left")
    brace(xs[2016], xs[2017] + bw, "NEAR (secondary)")
    brace(xs[2018], xs[2025] + bw, "FAR: eight yearly units (primary horizon)")

    fig.savefig(os.path.join(outdir, "optional_fig_cohorts.pdf"), metadata={"CreationDate": None, "Creator": "matplotlib"})
    fig.savefig(os.path.join(outdir, "optional_fig_cohorts.png"))
    plt.close(fig)
    return {"train_rows": n_train, "evaluation_rows": n_eval}


# ------------------------------------------------------------------ Figure 2
def fig2(outdir, per_run, counts, e3):
    fig, ax = plt.subplots(figsize=(W_IN, 2.1))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.895, bottom=0.245)
    xi = list(range(len(UNITS)))
    dump = {}
    stats = {u: unit_stats(per_run, u, "A_F") for u in UNITS}
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    ax.text(-0.40, 0.507, "chance", ha="left", va="bottom", fontsize=6.3, color=MUTED)
    bound = [e3[u]["AUC_max"]["value"] for u in UNITS]
    ax.plot(xi, bound, color=INK, lw=0.9, ls="-", marker="_", ms=5, zorder=2,
            label=r"$\mathrm{AUC}_{\max}$ bound")
    dodge = {"svm": -0.09, "lgbm": -0.03, "xgb": 0.03, "mlp": 0.09}
    for m in MODELS:
        mean = [stats[u][m][0] for u in UNITS]
        lo = [stats[u][m][1] for u in UNITS]
        hi = [stats[u][m][2] for u in UNITS]
        st = STYLE[m]
        ax.plot(xi, mean, color=st["color"], ls=st["ls"], marker=st["marker"], mfc="white",
                mec=st["color"], mew=0.9, zorder=4, label=LABEL[m])
        if m in ("lgbm", "mlp"):  # the two models whose five runs differ
            for i in xi:
                ax.plot([i + dodge[m]] * 2, [lo[i], hi[i]], color=st["color"], lw=1.6, alpha=0.55,
                        solid_capstyle="butt", zorder=3)
        dump[m] = {u: {"mean": stats[u][m][0], "min": stats[u][m][1], "max": stats[u][m][2]} for u in UNITS}
    dump["AUC_max"] = dict(zip(UNITS, bound))
    ax.set_ylim(0.46, 1.03)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlim(-0.45, 11.45)
    ax.set_ylabel("Full-arm ROC-AUC, $A(u,F)$")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    labels = []
    for u in UNITS:
        n1 = counts[u]["F"]["malware"]
        labels.append("%s\n%s" % (TICK.get(u, u.split("_")[1]), format(n1, ",")))
    ax.set_xticks(xi)
    ax.set_xticklabels(labels)
    ax.text(-0.118, -0.155, "unit\nmalware", transform=ax.transAxes, ha="left", va="center",
            fontsize=6.0, color=MUTED, linespacing=1.3)
    regime_decor(ax, 1.01)
    ax.legend(loc="lower center", bbox_to_anchor=(0.635, 0.075), ncol=2, frameon=True, framealpha=0.92,
              edgecolor=GRID, handlelength=2.6, columnspacing=1.1, borderpad=0.35, labelspacing=0.3)
    fig.savefig(os.path.join(outdir, "fig2_yearly_auc.pdf"), metadata={"CreationDate": None, "Creator": "matplotlib"})
    fig.savefig(os.path.join(outdir, "fig2_yearly_auc.png"))
    plt.close(fig)
    return dump


# ------------------------------------------------------------------ Figure 3
def fig3(outdir, per_run, counts):
    # V3c (F10): figure height 2.3 -> 2.55 in; panel sizes and margins are kept, the added 0.25 in hold the legend
    H0, H1 = 2.3, 2.55
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W_IN, H1), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.16})
    fig.subplots_adjust(left=0.115, right=0.985, top=1 - 0.085 * H0 / H1, bottom=0.125 * H0 / H1 + (H1 - H0) / H1)
    xi = list(range(len(UNITS)))
    dump = {"Delta_u": {}, "B_u": {}, "max_abs_Delta_u": {}}
    stats = {u: unit_stats(per_run, u, "Delta_u") for u in UNITS}
    dodge = {"svm": -0.24, "lgbm": -0.08, "xgb": 0.08, "mlp": 0.24}
    a1.axhline(0.0, color=INK, lw=0.7, zorder=1)
    for m in MODELS:
        st = STYLE[m]
        x = [i + dodge[m] for i in xi]
        mean = [1e3 * stats[u][m][0] for u in UNITS]
        lo = [1e3 * stats[u][m][1] for u in UNITS]
        hi = [1e3 * stats[u][m][2] for u in UNITS]
        if m in ("lgbm", "mlp"):
            for xx, l, h in zip(x, lo, hi):
                a1.plot([xx, xx], [l, h], color=st["color"], lw=1.6, alpha=0.55, solid_capstyle="butt", zorder=2)
        a1.plot(x, mean, ls="none", marker=st["marker"], mfc="white", mec=st["color"], mew=0.9,
                zorder=3, label=LABEL[m])
        dump["Delta_u"][m] = {u: {"mean": stats[u][m][0], "min": stats[u][m][1], "max": stats[u][m][2]} for u in UNITS}
    a1.set_ylim(-7.5, 7.5)
    a1.set_yticks([-6, -3, 0, 3, 6])
    a1.set_ylabel(r"$\Delta_u$ ($\times 10^{-3}$)")
    a1.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    a1.legend(loc="upper right", ncol=4, frameon=True, framealpha=0.92, edgecolor=GRID,
              handletextpad=0.2, columnspacing=0.9, borderpad=0.3)
    a1.text(0.008, 0.93, "a", transform=a1.transAxes, fontsize=8, fontweight="bold", va="top")
    regime_decor(a1, 1.02)

    B, mx = [], []
    for u in UNITS:
        F, U = counts[u]["F"], counts[u]["U"]
        b = 1 - Fr(U["malware"] * U["benign"], F["malware"] * F["benign"])
        B.append(float(b))
        v = max(abs(Fr(per_run[m][r]["units"][u]["Delta_u"]["exact"])) for m in MODELS for r in RUNS)
        mx.append(float(v))
        dump["B_u"][u] = float(b)
        dump["max_abs_Delta_u"][u] = float(v)
    floor = 1e-5
    a2.bar(xi, [b - floor for b in B], bottom=floor, width=0.56, color="#bdbdbd", edgecolor=MUTED, lw=0.5,
           zorder=2, label="count bound $B_u$ (before training)")
    a2.plot(xi, mx, ls="none", marker="o", ms=4.4, color=INK, zorder=3,
            label=r"largest observed $|\Delta_u|$ (4 models, 5 runs)")
    a2.set_yscale("log")
    a2.set_ylim(floor, 6.0)
    a2.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    a2.set_ylabel("AUC difference (log)")
    a2.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    # V3c (finding F10): legend moved below the axes so that it no longer overlaps the bars (data unchanged)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=True, framealpha=0.92,
              edgecolor=GRID, borderpad=0.3, columnspacing=1.2, handletextpad=0.5)
    a2.text(0.008, 0.93, "b", transform=a2.transAxes, fontsize=8, fontweight="bold", va="top")
    for x in (1.5, 3.5):
        a2.axvline(x, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=0)
    a2.set_xticks(xi)
    a2.set_xticklabels([TICK.get(u, u.split("_")[1]) for u in UNITS])
    a2.set_xlim(-0.5, 11.5)
    for ax in (a1, a2):
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(outdir, "fig3_full_unseen.pdf"), metadata={"CreationDate": None, "Creator": "matplotlib"})
    fig.savefig(os.path.join(outdir, "fig3_full_unseen.png"))
    plt.close(fig)
    return dump


def main():
    g5dir, outdir = sys.argv[1], sys.argv[2]
    dump_path = sys.argv[3] if len(sys.argv) > 3 else None
    os.makedirs(outdir, exist_ok=True)
    d = load_verified(g5dir)
    setup()
    per_run = d["G5_per_run_metrics.json"]
    counts = d["G5_model_independent_counts.json"]
    dump = {
        "source_sha256": EXPECTED_SHA256,
        "fig1": fig1(outdir, counts),
        "fig2": fig2(outdir, per_run, counts, d["G5_E3.json"]),
        "fig3": fig3(outdir, per_run, counts),
    }
    if dump_path:
        with open(dump_path, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, indent=1, sort_keys=True)
    print("figures written to", outdir)


if __name__ == "__main__":
    main()
