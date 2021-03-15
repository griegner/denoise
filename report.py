from pathlib import Path
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('white')

import warnings
warnings.filterwarnings('ignore')


def find_files(denoise, pipeline):
    Path(f'{denoise}/group').mkdir(exist_ok=True)
    npys = sorted(Path(denoise).glob(f'**/*pipeline-{pipeline}_connMat.npy'))
    plots = sorted(Path(denoise).glob(f'**/*pipeline-{pipeline}_plot.png'))
    return npys, plots

def plot_atlas(ax):
    atlas = datasets.fetch_atlas_basc_multiscale_2015(version='asym')["scale122"]
    plotting.plot_roi(atlas, display_mode='xz', cut_coords=(0,0), annotate=False, draw_cross=False, axes=ax)

def plot_summary_dist(ax, npys, pipeline):

    means = []
    for npy in npys:
        npy = np.hstack(np.load(npy))
        means.append(np.nanmean(npy))
        sns.distplot(npy, color='#3C83BC', hist=False, kde_kws=dict(linewidth=.08), ax=ax)

    ax.axvline(x=0, c='k', alpha=.3, linestyle='dashed')
    ax.plot(means, [-0.1]*len(means), color='#3C83BC', linestyle='none', marker='|', markersize=10, alpha=.2)

    ax.text(-0.1, 0.9, pipeline, transform=ax.transAxes, weight='bold')
    ax.text(-0.1, -0.1, f'distribution of connectivity values (r) across {len(means)} functional runs', transform=ax.transAxes)
    
def html_report(denoise, pipeline, plots):

    with open(f'{denoise}/group/pipeline-{pipeline}_report.html', 'w') as f_out:
        print(f"<img src='{denoise}/group/pipeline-{pipeline}_plot.png' style='float: left; width: 100%'>", file=f_out)
        
        for idx, plot in enumerate(plots):
            if idx%4==0: print('<hr>', file=f_out)
            print(f"<img src='{str(plot)}' style='float: left; width: 24%; margin-right: 1%'>", file=f_out)