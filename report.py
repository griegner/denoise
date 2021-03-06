import warnings

warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from nilearn import plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection


def find_files(denoise, strategy):
    denoise.joinpath("group").mkdir(exist_ok=True)
    qcs = sorted(denoise.glob("sub-*/*_qc.tsv"))
    vectors_noise = sorted(denoise.glob("sub-*/*strat-none_vect.npy"))
    vectors_denoise = sorted(denoise.glob(f"sub-*/*strat-{strategy}_vect.npy"))
    plots = sorted(denoise.glob(f"sub-*/*strat-{strategy}_plot.png"))
    return qcs, vectors_noise, vectors_denoise, plots


def qc_means(qcs):
    columns = pd.read_csv(qcs[0], sep="\t").columns
    qc_means = [pd.read_csv(qc, sep="\t").mean(axis=0).to_list() for qc in qcs]
    return pd.DataFrame(qc_means, columns=columns)


def plot_atlas(atlas, ax):
    plotting.plot_roi(
        atlas,
        display_mode="xz",
        cut_coords=(0, 0),
        annotate=False,
        draw_cross=False,
        axes=ax,
    )


def plot_summary_dist(ax, vectors_noise, vectors_denoise, denoise, strategy):

    edges = []
    [edges.extend(np.load(vector)) for vector in vectors_noise]
    sns.kdeplot(edges, color="#B62E33", linewidth=1, ax=ax)
    np.save(denoise / "group/strat-none_vect.npy", edges)
    del edges

    edges = []
    means = []
    for vector in vectors_denoise:
        vector = np.load(vector)
        edges.extend(vector)
        means.append(np.mean(vector))
        sns.kdeplot(vector, color="#3C83BC", linewidth=0.08, ax=ax)
    np.save(denoise / f"group/strat-{strategy}_vect.npy", edges)
    del edges

    ax.axvline(x=0, c="k", alpha=0.3, linestyle="dashed")
    ax.plot(
        means,
        [-0.1] * len(means),
        color="#3C83BC",
        linestyle="none",
        marker="|",
        markersize=10,
        alpha=0.2,
    )

    ax.text(-0.1, 0.9, strategy, transform=ax.transAxes, weight="bold")
    ax.text(
        -0.1,
        -0.1,
        f"distribution of connectivity values (r) across {len(means)} functional runs",
        transform=ax.transAxes,
    )


def qc_fc(qc, npys, strategies):

    qc_fc_ = {}
    for npy, strategy in zip(npys, strategies):
        fc = np.load(npy).reshape(len(qc), -1)
        pvals_unc = [stats.pearsonr(qc, fc_col)[1] for fc_col in fc.T]
        pvals_corr = fdrcorrection(pvals_unc, alpha=0.05)[0]
        qc_fc_[strategy] = (pvals_corr.sum() / len(pvals_corr)) * 100

    return pd.Series(qc_fc_)


def compare(denoise, qc_means, ax):
    npys = sorted(denoise.glob("group/*vect.npy"))
    strategies = [re.search("strat-(.*)_", npy.stem).group(1) for npy in npys]

    # axis 0
    means = {}
    cmap = plt.get_cmap("binary")
    for npy, strategy in zip(npys, strategies):
        dist = np.load(npy)
        dist_mean = dist.mean()
        means[strategy] = dist_mean
        color = cmap(1 - dist_mean)
        if strategy == "none":
            color = "#B62E33"
        sns.kdeplot(
            dist,
            linewidth=0.2,
            ax=ax[0],
            color=color,
            fill=True,
            alpha=0.2,
            label=strategy,
        )

    means = sorted(means.items(), key=lambda x: x[1])
    h, l = ax[0].get_legend_handles_labels()
    handels = [h[l.index(mean[0])] for mean in means]
    labels = [f"{idx+1}. {mean[0]}" for idx, mean in enumerate(means)]

    ax[0].legend(
        handels, labels, prop={"size": 6}, loc="upper left", bbox_to_anchor=(-0.1, 0.85)
    )
    ax[0].axvline(x=0, c="k", alpha=0.3, linestyle="dashed")
    ax[0].text(-0.1, 0.9, "strategies (sorted by mean)", transform=ax[0].transAxes)
    ax[0].text(
        -0.1,
        -0.1,
        f"distribution of connectivity values (r) across {len(strategies)} strategies",
        transform=ax[0].transAxes,
    )

    # axis 1
    qc_fc_ = qc_fc(qc_means, npys, strategies).sort_values(ascending=False)
    colors = ["#B62E33" if strat == "none" else "k" for strat in qc_fc_.index]
    qc_fc_.plot(kind="barh", color=colors, alpha=0.2, ax=ax[1])
    sns.despine(ax=ax[1], left=True, bottom=True)
    ax[1].bar_label(ax[1].containers[0], fontsize=6, fmt="   %d%%")

    order = qc_fc_.index
    for idx, strat in enumerate(order):
        ax[1].text(
            0.5, idx, f"{len(order)-idx}. {strat}", fontsize=6, ha="left", va="center"
        )

    ax[1].text(
        -0.1,
        -0.1,
        "qc-fc: % of edges correlated with RMS motion (q < .05)",
        transform=ax[1].transAxes,
    )


def html_report(denoise, strategy, confounds, plots):

    with open(f"{denoise}/group/strat-{strategy}_report.html", "w") as f_out:
        print(
            f"<img src='{denoise}/group/strat-{strategy}_plot.png' style='float: left; width: 100%'>",
            "<hr>",
            sep="\n",
            file=f_out,
        )
        print(f"<small>{confounds}</small>", file=f_out)

        for idx, plot in enumerate(plots):
            if idx % 4 == 0:
                print("<hr>", file=f_out)
            print(
                f"<img src='{str(plot)}' style='float: left; width: 24%; margin-right: 1%'>",
                file=f_out,
            )
