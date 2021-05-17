#!/Users/zeidanlab_pro/anaconda3/envs/neuroimg/bin/python

import warnings; warnings.filterwarnings('ignore')

from load_confounds import Confounds
from bids.layout import parse_file_entities
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('white')
import numpy as np
import pandas as pd
from nilearn import plotting, datasets, image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

from strategies import strategies
import report

def get_args(strategies):
    
    parser = argparse.ArgumentParser(description='denoise fMRI data')
    parser.add_argument('fmriprep', type=Path, help='path to fmriprep directory')
    parser.add_argument('strategy', type=str, help=' - '.join(strategies))
    parser.add_argument('--smooth_fwhm', type=float, default=6, help='choose smoothing kernel, default fwhm 6mm')
    parser.add_argument('--report_only', action='store_true', help='run group-level summary only')
    args = parser.parse_args()
    assert args.strategy in strategies, f'{args.strategy} not a valid strategy'
    assert args.fmriprep.is_dir(), 'fmriprep directory does not exist'
    return args

class Data:

    def __init__(self, fmriprep, strategy):
        print('indexing files: ', end='')
        self.fmriprep = fmriprep
        self.atlas = datasets.fetch_atlas_basc_multiscale_2015(version='asym')['scale122']
        self.masks = self.get_masks()
        self.preprocs = self.get_preprocs()
        self.confounds = self.get_confounds(strategy)
        self.motion_dfs = self.get_motion_dfs()
        assert len(self.masks)==len(self.preprocs)==len(self.confounds)==len(self.motion_dfs), \
            'missings fmriprep files'
        print('done')

    def get_masks(self):
        masks =  self.fmriprep.glob('**/sub-*_task-*_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz')
        return sorted([str(mask) for mask in masks])

    def get_preprocs(self):
        preprocs = self.fmriprep.glob('**/sub-*_task-*_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')
        return sorted([str(preproc) for preproc in preprocs])

    def get_confounds(self, strategy):
        return Confounds(**strategy).load(self.preprocs)

    def get_motion_dfs(self):
        dfs = sorted(self.fmriprep.glob('**/sub-*_task-*_desc-confounds_timeseries.tsv'))
        return [pd.read_csv(df, sep='\t', usecols=['framewise_displacement', 'rmsd']) for df in dfs]

def build_path(derivatives, sub, task, space, strategy):
    path = derivatives / f'denoise/sub-{sub}'
    path.mkdir(parents=True, exist_ok=True)
    # BIDS patter: sub-{subject}_task-{task}_space-{space}_strat-{strategy}_{suffix}.{extension}
    path_plot = path / f'sub-{sub}_task-{task}_space-{space}_strat-{strategy}_plot.png'
    path_matrix = path / f'sub-{sub}_task-{task}_space-{space}_strat-{strategy}_connMat.npy'
    return path_plot, path_matrix

def plot_dist(ax, matrix, matrix_clean):
    sns.distplot(matrix.flatten(), color='#B62E33', hist=False, ax=ax)
    sns.distplot(matrix_clean.flatten(), color='#3C83BC', hist=False, ax=ax)
    ax.axvline(x=0, c='k', alpha=.3, linestyle='dashed')

def plot_connectome(ax, matrix, matrix_clean, sub, task):
    ax.set_title(f'sub-{sub} task-{task}', loc='left', weight='bold', size='medium')
    plotting.plot_matrix(matrix, colorbar=False, tri='full', axes=ax)
    plotting.plot_matrix(matrix_clean, colorbar=False, tri='lower', axes=ax)

def plot_fd(ax, fd, fd_thresh):
    ax.set_title(
        f'FD   mean: {fd.mean():.2}   outliers > {fd_thresh}mm: {(fd > fd_thresh).sum():.2%}', 
        loc='left', 
        size='small'
        )
    ax.axhline(y=fd_thresh, color='r', linestyle=':', lw=.6)
    ax.plot(fd, color='k', lw=0.5)
    outliers = [idx for idx, val in enumerate(fd) if val>fd_thresh]
    ax.plot(outliers, [0]*len(outliers), color='r', linestyle='none', marker='|')

def plot_carpet(ax, preproc, preproc_clean, mask):
    plotting.plot_carpet(preproc, mask_img=mask, axes=ax[0])
    plotting.plot_carpet(preproc_clean, mask_img=mask, axes=ax[1])

def main():  

    args = get_args(strategies.keys())
    derivatives = args.fmriprep.parent
    fd_thresh = strategies[args.strategy]['fd_thresh'] 

    data = Data(args.fmriprep, strategies[args.strategy])
    
    if not args.report_only:

        for mask, preproc, confound, motion_df in zip(data.masks, data.preprocs, data.confounds, data.motion_dfs):

            file_entities = parse_file_entities(preproc)
            sub, task, space = file_entities['subject'], file_entities['task'], file_entities['space']

            print(f'sub-{sub} task-{task}')
            path_plot, path_matrix = build_path(derivatives, sub, task, space, args.strategy)

            mask, preproc = image.load_img(mask), image.load_img(preproc)
            preproc_clean = image.clean_img(preproc, detrend=False, standardize=False, confounds=confound, mask_img=mask)

            masker = NiftiLabelsMasker(data.atlas, mask_img=mask, smoothing_fwhm=args.smooth_fwhm, 
                                       memory=str(derivatives / '.denoise_cache'), memory_level=3)
            ts = masker.fit_transform(preproc)
            ts_clean = masker.fit_transform(preproc_clean)

            correlation = ConnectivityMeasure(kind='correlation', discard_diagonal=True)
            matrix = correlation.fit_transform([ts])[0]; np.fill_diagonal(matrix, np.nan)
            matrix_clean = correlation.fit_transform([ts_clean])[0]; np.fill_diagonal(matrix_clean, np.nan)
            np.save(path_matrix, matrix_clean)

            fig, axs = plt.subplots(5, figsize=(4,8), gridspec_kw={'height_ratios': [2, 10, 1, 2, 2]})
            for ax in axs: ax.axis('off'); ax.margins(0,.02)
            plot_dist(axs[0], matrix, matrix_clean)
            plot_connectome(axs[1], matrix, matrix_clean, sub, task)
            plot_fd(axs[2], motion_df.framewise_displacement, fd_thresh)
            plot_carpet(axs[3:], preproc, preproc_clean, mask)

            fig.savefig(path_plot, dpi=300, bbox_inches='tight')
            plt.close(fig)
            del fig, axs

    print(f'group-level summary: strategy-{args.strategy}_report.html')
    denoise = f'{derivatives}/denoise'
    npys, plots = report.find_files(denoise, args.strategy)
    fig, axs = plt.subplots(ncols=3, figsize=(16,2), gridspec_kw={'width_ratios': [2,2,2]})
    for ax in axs: ax.axis('off'); ax.margins(0,0)
    report.plot_atlas(axs[0])
    edges = report.plot_summary_dist(axs[1], npys, args.strategy)
    np.save(f'{denoise}/group/strategy-{args.strategy}_connMat.npy', edges)
    report.compare(denoise, axs[2])
    fig.savefig(f'{denoise}/group/strategy-{args.strategy}_plot.png', dpi=300)
    report.html_report(denoise, args.strategy, plots)
    plt.close(fig)
    del fig, axs

if __name__ == '__main__': main()