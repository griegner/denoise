import warnings; warnings.filterwarnings('ignore')

import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('white')

def find_files(denoise, strategy):
    denoise.joinpath('group').mkdir(exist_ok=True)
    vectors = sorted(denoise.glob(f'sub-*/*strat-{strategy}_vect.npy'))
    plots = sorted(denoise.glob(f'sub-*/*strat-{strategy}_plot.png'))
    assert len(vectors)==len(plots), 'missing vect.npy or plot.png files'
    return vectors, plots

def plot_atlas(atlas, ax):
    plotting.plot_roi(atlas, display_mode='xz', cut_coords=(0,0), annotate=False, draw_cross=False, axes=ax)

def plot_summary_dist(ax, vectors, strategy):

    edges = []
    means = []
    for vector in vectors:
        vector = np.load(vector)
        edges.extend(vector)
        means.append(np.mean(vector))
        sns.kdeplot(vector, color='#3C83BC', linewidth=.08, ax=ax)

    ax.axvline(x=0, c='k', alpha=.3, linestyle='dashed')
    ax.plot(means, [-0.1]*len(means), color='#3C83BC', linestyle='none', marker='|', markersize=10, alpha=.2)

    ax.text(-0.1, 0.9, strategy, transform=ax.transAxes, weight='bold')
    ax.text(-0.1, -0.1, f'distribution of connectivity values (r) across {len(means)} functional runs', transform=ax.transAxes)

    return edges

def compare(denoise, ax):
    npys = sorted(denoise.glob('group/*vect.npy'))
    strategies = [strategy.stem[9:-8] for strategy in npys] # needs regex

    means = {}
    cmap = plt.get_cmap('binary')
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