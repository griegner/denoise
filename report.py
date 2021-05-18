import warnings; warnings.filterwarnings('ignore')

from pathlib import Path
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set_style('white')

def find_files(denoise, strategy):
    Path(f'{denoise}/group').mkdir(exist_ok=True)
    npys = sorted(Path(denoise).glob(f'sub-*/*strat-{strategy}_connMat.npy'))
    plots = sorted(Path(denoise).glob(f'sub-*/*strat-{strategy}_plot.png'))
    assert len(npys)==len(plots), 'missing connMat.npy or plot.png files'
    return npys, plots

def plot_atlas(atlas, ax):
    plotting.plot_roi(atlas, display_mode='xz', cut_coords=(0,0), annotate=False, draw_cross=False, axes=ax)

def plot_summary_dist(ax, npys, strategy):

    edges = []
    means = []
    for npy in npys:
        # upper triangle only
        npy = np.load(npy)
        npy[np.tril_indices(npy.shape[0], -1)] = np.nan
        npy = npy[~np.isnan(npy)].flatten()
        edges.extend(npy)
        means.append(np.mean(npy))
        sns.distplot(npy, color='#3C83BC', hist=False, kde_kws=dict(linewidth=.08), ax=ax)

    ax.axvline(x=0, c='k', alpha=.3, linestyle='dashed')
    ax.plot(means, [-0.1]*len(means), color='#3C83BC', linestyle='none', marker='|', markersize=10, alpha=.2)

    ax.text(-0.1, 0.9, strategy, transform=ax.transAxes, weight='bold')
    ax.text(-0.1, -0.1, f'distribution of connectivity values (r) across {len(means)} functional runs', transform=ax.transAxes)

    return edges

def compare(denoise, ax):
    denoise = Path(denoise)
    npys = sorted(denoise.glob('group/*connMat.npy'))
    strategies = [strategy.stem[9:-8] for strategy in npys]

    cmap = plt.get_cmap('binary')
    means = {}
    for npy, strategy in zip(npys, strategies):
        dist = np.load(npy)
        dist_mean = dist.mean()
        means[strategy] = dist_mean
        color = cmap(1-dist_mean)
        sns.kdeplot(dist, linewidth=0.2, ax=ax, color=color, fill=True, alpha=.3, label=strategy)

    means = sorted(means.items(), key=lambda x:x[1])
    h, l = ax.get_legend_handles_labels()
    handels = [h[l.index(mean[0])] for mean in means]
    labels = [mean[0] for mean in means]
    
    ax.legend(handels, labels, prop={'size': 6}, loc='upper left', bbox_to_anchor=(-0.1,.85))
    ax.axvline(x=0, c='k', alpha=.3, linestyle='dashed')
    ax.text(-0.1, 0.9, 'strategies (min-max)', transform=ax.transAxes)
    ax.text(-0.1, -0.1, f'distribution of connectivity values (r) across {len(strategies)} strategies', transform=ax.transAxes)
   
def html_report(denoise, strategy, confounds, plots):

    with open(f'{denoise}/group/strategy-{strategy}_report.html', 'w') as f_out:
        print(f"<img src='{denoise}/group/strategy-{strategy}_plot.png' style='float: left; width: 100%'>", '<hr>', sep='\n', file=f_out)
        print(f"<small>{confounds}</small>", file=f_out)
        
        for idx, plot in enumerate(plots):
            if idx%4==0: print('<hr>', file=f_out)
            print(f"<img src='{str(plot)}' style='float: left; width: 24%; margin-right: 1%'>", file=f_out)