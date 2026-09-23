# -*- coding: utf-8 -*-
"""Generate the seven manuscript figures from the bundled processed results.

The figures use a colour-blind-friendly palette for the online article while
retaining redundant hatches, markers, and line styles so that the information
remains distinguishable after greyscale conversion for print.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# Okabe-Ito-inspired colour-blind-friendly palette.  Hatches/markers/line styles
# provide redundant encodings for monochrome reproduction.
METHOD_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
STAT_COLORS = ["#0072B2", "#E69F00", "#009E73"]
HATCHES = ["", "//", "xx", ".."]
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]
EDGE_COLOR = "black"
EDGE_WIDTH = 0.8


def _bar(ax, x, values, *, colors, hatches, width=0.8, label=None):
    bars = ax.bar(
        x,
        values,
        width=width,
        color=colors,
        edgecolor=EDGE_COLOR,
        linewidth=EDGE_WIDTH,
        label=label,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    return bars


def fig01_static_exact():
    df = pd.read_csv(RESULTS / "exact_calibration" / "static_summary.csv")
    order = ["WEIGHT", "RATIO", "STRONG4", "STRONG8"]
    df = df.set_index("method").loc[order].reset_index()
    x = np.arange(len(order))
    mean = 100 * df["mean_ratio_to_OPT"].to_numpy()
    p05 = 100 * df["p05_ratio_to_OPT"].to_numpy()
    hit = 100 * df["optimal_hit_rate"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))

    _bar(
        axes[0], x, mean,
        colors=METHOD_COLORS,
        hatches=HATCHES,
    )
    # The original exploratory plot used (mean-p05) as a symmetric y-error,
    # which creates a meaningless upper whisker above the feasible 100% bound.
    # Here the lower-tail statistic is shown directly: each whisker runs only
    # downward from the mean to the empirical 5th percentile.
    axes[0].vlines(x, p05, mean, color="black", linewidth=1.35, zorder=4)
    axes[0].scatter(x, p05, marker="_", s=105, color="black", zorder=5)
    axes[0].set_xticks(x, order)
    axes[0].set_ylabel("Solution quality (% of OPT)")
    axes[0].set_ylim(max(0, float(np.min(p05)) - 5), 101)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].text(-0.13, 1.03, "(a)", transform=axes[0].transAxes, fontweight="bold")

    _bar(
        axes[1], x, hit,
        colors=METHOD_COLORS,
        hatches=HATCHES,
    )
    axes[1].set_xticks(x, order)
    axes[1].set_ylabel("Optimal-hit rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].text(-0.13, 1.03, "(b)", transform=axes[1].transAxes, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig01_static_exact_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def fig02_dynamic_exact():
    df = pd.read_csv(RESULTS / "safe_refresh" / "summary.csv").set_index("method")
    methods = ["LOCAL", "SAFE010", "SAFE005"]
    stats = [
        ("mean_ratio_to_OPT", "Mean"),
        ("p05_ratio_to_OPT", "5th percentile"),
        ("p01_ratio_to_OPT", "1st percentile"),
    ]
    x = np.arange(len(methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, (col, label) in enumerate(stats):
        vals = [100 * df.loc[m, col] for m in methods]
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=label,
            color=STAT_COLORS[i],
            edgecolor=EDGE_COLOR,
            linewidth=EDGE_WIDTH,
        )
        for bar in bars:
            bar.set_hatch(HATCHES[i])
    ax.set_xticks(x, methods)
    ax.set_ylabel("Quality (% of OPT)")
    ax.set_ylim(80, 101)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02_dynamic_exact_quality.pdf", bbox_inches="tight")
    plt.close(fig)


def fig03_source_quality():
    res = pd.read_csv(RESULTS / "red2pack" / "results.csv")
    methods = ["LOCAL", "SAFE010", "SAFE005", "STRONG8"]
    x = np.arange(len(methods))
    width = 0.36
    syn = [100 * res[res.source == "synthetic"][f"{m}_ratio_to_extended_BKS"].mean() for m in methods]
    real = [100 * res[res.source == "real"][f"{m}_ratio_to_extended_BKS"].mean() for m in methods]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    b1 = ax.bar(x - width / 2, syn, width, label="Synthetic", color="#0072B2",
                edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    b2 = ax.bar(x + width / 2, real, width, label="Real", color="#E69F00",
                edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    for bar in b1:
        bar.set_hatch("//")
    for bar in b2:
        bar.set_hatch("..")
    ax.set_xticks(x, methods)
    ax.set_ylabel("Mean quality (% of external reference)")
    ax.set_ylim(88, 100.5)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03_external_quality_by_source.pdf", bbox_inches="tight")
    plt.close(fig)


def fig04_tradeoff():
    res = pd.read_csv(RESULTS / "red2pack" / "results.csv")
    timing = pd.read_csv(RESULTS / "timing" / "task_timing.csv")
    sr = res[res.source == "synthetic"]
    st = timing[timing.source == "synthetic"]
    methods = ["LOCAL", "SAFE010", "SAFE005"]
    xs = [st[f"{m}_mean_time_ms"].mean() for m in methods]
    ys = [100 * sr[f"{m}_ratio_to_extended_BKS"].mean() for m in methods]
    r2p_x = 1000 * sr["R2P_total_time_s"].mean()
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    offsets = {"LOCAL": (6, 4), "SAFE010": (6, -12), "SAFE005": (6, 8)}
    for i, (x0, y0, lab) in enumerate(zip(xs, ys, methods)):
        ax.scatter(
            [x0], [y0],
            color=METHOD_COLORS[i],
            marker=MARKERS[i],
            s=55,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        ax.annotate(lab, (x0, y0), xytext=offsets[lab], textcoords="offset points")
    ax.scatter([r2p_x], [100.0], color="black", marker="D", s=52, zorder=3)
    ax.annotate("red2pack", (r2p_x, 100.0), xytext=(-58, -14), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Mean per-update / from-scratch time (ms, log scale)")
    ax.set_ylabel("Mean quality (% of external reference)")
    ax.set_ylim(89, 100.7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04_external_quality_time_synthetic.pdf", bbox_inches="tight")
    plt.close(fig)


def fig05_scaling():
    res = pd.read_csv(RESULTS / "red2pack" / "results.csv")
    timing = pd.read_csv(RESULTS / "timing" / "task_timing.csv")
    sr = res[res.source == "synthetic"]
    st = timing[timing.source == "synthetic"]
    ns = sorted(int(n) for n in sr.n.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for i, m in enumerate(["LOCAL", "SAFE010", "SAFE005", "STRONG8"]):
        ys = [100 * sr[sr.n == n][f"{m}_ratio_to_extended_BKS"].mean() for n in ns]
        axes[0].plot(
            ns, ys,
            marker=MARKERS[i],
            color=METHOD_COLORS[i],
            linestyle=LINE_STYLES[i],
            label=m,
            markeredgecolor="black",
            markeredgewidth=0.4,
        )
    axes[0].set_xscale("log")
    axes[0].set_xticks(ns, [str(n) for n in ns])
    axes[0].set_xlabel("Number of vertices")
    axes[0].set_ylabel("Mean quality (% of external reference)")
    axes[0].set_ylim(89, 95.5)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[0].text(-0.13, 1.03, "(a)", transform=axes[0].transAxes, fontweight="bold")

    for i, m in enumerate(["SAFE010", "SAFE005"]):
        j = i + 1
        ys = [st[st.n == n][f"{m}_speedup_vs_R2P_sameenv"].median() for n in ns]
        axes[1].plot(
            ns, ys,
            marker=MARKERS[j],
            color=METHOD_COLORS[j],
            linestyle=LINE_STYLES[j],
            label=m,
            markeredgecolor="black",
            markeredgewidth=0.4,
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks(ns, [str(n) for n in ns])
    axes[1].set_xlabel("Number of vertices")
    axes[1].set_ylabel("Median speedup vs. red2pack")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.25)
    axes[1].text(-0.13, 1.03, "(b)", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05_external_scaling_quality_speed.pdf", bbox_inches="tight")
    plt.close(fig)


def fig06_real_quality():
    res = pd.read_csv(RESULTS / "red2pack" / "results.csv")
    rr = res[res.source == "real"]
    graphs = ["ca-GrQc", "ca-HepTh", "p2p-Gnutella08", "p2p-Gnutella09"]
    methods = ["LOCAL", "SAFE010", "SAFE005", "STRONG8"]
    x = np.arange(len(graphs))
    width = 0.20
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, m in enumerate(methods):
        vals = [100 * rr[rr.graph_name == g][f"{m}_ratio_to_extended_BKS"].mean() for g in graphs]
        bars = ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=m,
            color=METHOD_COLORS[i],
            edgecolor=EDGE_COLOR,
            linewidth=EDGE_WIDTH,
        )
        for bar in bars:
            bar.set_hatch(HATCHES[i])
    ax.set_xticks(x, graphs, rotation=12)
    ax.set_ylabel("Mean quality (% of external reference)")
    ax.set_ylim(96.5, 100.2)
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06_external_real_quality.pdf", bbox_inches="tight")
    plt.close(fig)


def fig07_refresh():
    df = pd.read_csv(RESULTS / "safe_refresh" / "summary.csv").set_index("method")
    methods = ["SAFE020", "SAFE010", "SAFE005"]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))

    vals0 = [100 * df.loc[m, "refresh_accept_rate"] for m in methods]
    _bar(axes[0], x, vals0, colors=METHOD_COLORS[:3], hatches=HATCHES[:3])
    axes[0].set_xticks(x, methods)
    axes[0].set_ylabel("Refresh acceptance rate (%)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].text(-0.13, 1.03, "(a)", transform=axes[0].transAxes, fontweight="bold")

    vals1 = [100 * df.loc[m, "mean_accepted_relative_gain"] for m in methods]
    _bar(axes[1], x, vals1, colors=METHOD_COLORS[:3], hatches=HATCHES[:3])
    axes[1].set_xticks(x, methods)
    axes[1].set_ylabel("Mean gain when accepted (%)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].text(-0.13, 1.03, "(b)", transform=axes[1].transAxes, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig07_refresh_behavior.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig01_static_exact()
    fig02_dynamic_exact()
    fig03_source_quality()
    fig04_tradeoff()
    fig05_scaling()
    fig06_real_quality()
    fig07_refresh()
    print(f"Wrote seven colour-online / greyscale-safe figures to {FIGURES}")


if __name__ == "__main__":
    main()
