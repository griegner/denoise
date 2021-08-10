#!/Users/zeidanlab/miniconda3/envs/neuroimg/bin/python

from strategies import strategies
import report

import warnings; warnings.filterwarnings('ignore')
from load_confounds import Confounds
from bids.layout import parse_file_entities
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('white')
import numpy as np; import pandas as pd
from nilearn import plotting, datasets, image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

def get_args(strategies):
    parser = argparse.ArgumentParser(description='denoise fMRI data')
    parser.add_argument('fmriprep', type=Path, help='path to fmriprep directory')
    parser.add_argument('strategy', type=str, help=' - '.join(strategies))
    parser.add_argument('--smooth_fwhm', type=float, default=6, help='smoothing kernel, default fwhm 6mm')
    parser.add_argument('--report_only', action='store_true', help='run group-level summary only')
    args = parser.parse_args()
    assert args.strategy in strategies, f'{args.strategy} is not a valid strategy'
    assert args.fmriprep.is_dir(), 'fmriprep directory does not exist'
    return args

class Data:
    def __init__(self, fmriprep, strategy):
        print('\nindexing files... ', end='')
        self.fmriprep = fmriprep
        self.masks = self.get_masks()
        self.preprocs = self.get_preprocs()
        self.confounds = self.get_confounds(strategy)
        self.motion_dfs = self.get_motion_dfs()
        assert len(self.masks)==len(self.preprocs)==len(self.confounds)==len(self.motion_dfs), \
            'missings fmriprep files'
        print(f'found {len(self.preprocs)} images')

    def get_masks(self):
        masks =  self.fmriprep.glob('**/sub-*_task-n*_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz')
        return sorted([str(mask) for mask in masks])

    def get_preprocs(self):
        preprocs = self.fmriprep.glob('**/sub-*_task-n*_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')
        return sorted([str(preproc) for preproc in preprocs])

    def get_confounds(self, strategy):
        return Confounds(**strategy).load(self.preprocs)

    def get_motion_dfs(self):
        dfs = sorted(self.fmriprep.glob('**/sub-*_task-n*_desc-confounds_timeseries.tsv'))
        return [pd.read_csv(df, sep='\t', usecols=['framewise_displacement', 'std_dvars', 'rmsd']) for df in dfs]

def build_path(derivatives, sub, task, strategy):
    path = derivatives / f'denoise/sub-{sub}'
    path.mkdir(parents=True, exist_ok=True)
    # BIDS pattern: sub-{subject}_task-{task}_strat-{strategy}_{suffix}.{extension}
    path_plot = path / f'sub-{sub}_task-{task}_strat-{strategy}_plot.png'
    path_none = path / f'sub-{sub}_task-{task}_strat-none_vect.npy'
    path_vector = path / f'sub-{sub}_task-{task}_strat-{strategy}_vect.npy'
    path_qc = path / f'sub-{sub}_task-{task}_qc.tsv'
    return path_plot, path_none, path_vector, path_qc

def get_timeseries(preproc, atlas, mask, smooth_fwhm, derivatives):
    masker = NiftiLabelsMasker(
                atlas, 
                mask_img=mask, 
                smoothing_fwhm=smooth_fwhm, 
                strategy='mean',
                memory=str(derivatives / '.denoise_cache'), 
                memory_level=3
                )
    return masker.fit_transform(preproc[0]), masker.fit_transform(preproc[1])

def get_correlations(timeseries, path_none, path_vector, strategy):
    correlation_vector = ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)
    vector = correlation_vector.fit_transform(timeseries)
    np.save(path_none, vector[0]) # noise
    np.save(path_vector, vector[1]) # denoise
    correlation_matrix = ConnectivityMeasure(kind='correlation', vectorize=False)
    matrix = correlation_matrix.fit_transform(timeseries)
    np.fill_diagonal(matrix[0], np.nan); np.fill_diagonal(matrix[1], np.nan)
    return vector, matrix

def plot_dist(ax, vector):
    sns.kdeplot(vector[0], color='#B62E33', ax=ax)
    sns.kdeplot(vector[1], color='#3C83BC', ax=ax)
    ax.axvline(x=0, c='k', alpha=.3, linestyle='dashed')

def plot_connectome(ax, matrix, sub, task):
    ax.set_title(f'sub-{sub} task-{task}', loc='left', weight='bold', size='medium')
    plotting.plot_matrix(matrix[0], colorbar=False, tri='full', axes=ax)
    plotting.plot_matrix(matrix[1], colorbar=False, tri='lower', axes=ax)

def plot_outliers(ax, motion_df, fd_thresh, dvars_thresh):
    
    motion_df['fd_outliers'] = motion_df.framewise_displacement.gt(fd_thresh) 
    motion_df['dvars_outliers'] = motion_df.std_dvars.gt(dvars_thresh)
    motion_df['outliers'] = motion_df.fd_outliers + motion_df.dvars_outliers
    outliers = motion_df.index[motion_df.outliers].to_list()
    rows = motion_df.shape[0]
    p_outliers = len(outliers)/rows
    
    ax[0].set_title(
        f'FD mean={motion_df.framewise_displacement.mean():.2f} \
        DVARS mean={motion_df.std_dvars.mean():.2f} \
        outliers={p_outliers:.2%}', 
        loc='left', 
        size='small'
        )

    ax[0].axhline(y=fd_thresh, color='r', linestyle=':', lw=.6)
    ax[0].plot(motion_df.framewise_displacement, color='k', lw=0.5)

    ax[1].axhline(y=dvars_thresh, color='r', linestyle=':', lw=.6)
    ax[1].plot(motion_df.std_dvars, color='k', lw=0.5)
    ax[1].plot(outliers, [0]*len(outliers), color='r', linestyle='none', marker='|')

    return p_outliers>0.1 or motion_df.framewise_displacement.mean()>0.5 # exclusion criteria

def plot_carpet(ax, preproc, mask):
    plotting.plot_carpet(preproc[0], mask_img=mask, axes=ax[0])
    plotting.plot_carpet(preproc[1], mask_img=mask, axes=ax[1])

def run_summary(vector, matrix, sub, task, motion_df, fd_thresh, dvars_thresh, preproc, mask, paths):
    fig, axs = plt.subplots(6, figsize=(4,9), gridspec_kw={'height_ratios': [2, 10, 1, 1, 2, 2]})
    for ax in axs: ax.axis('off'); ax.margins(0,.02)
    plot_dist(axs[0], vector)
    plot_connectome(axs[1], matrix, sub, task)
    exclude = plot_outliers(axs[2:4], motion_df, fd_thresh, dvars_thresh)
    plot_carpet(axs[4:], preproc, mask)
    if exclude: 
        axs[0].set_title('exclude', color='r', fontweight='bold', loc='left')
        [path.unlink() for path in paths[1:]]
    fig.savefig(paths[0], dpi=300, bbox_inches='tight')
    plt.close(fig)

def group_summary(derivatives, strategy, atlas):
    print(f'group-level summary: strat-{strategy}_report.html')
    denoise = derivatives / 'denoise'
    qcs, vectors_noise, vectors_denoise, plots = report.find_files(denoise, strategy)
    qc_means = report.qc_means(qcs)
    fig, axs = plt.subplots(ncols=4, figsize=(20,2), gridspec_kw={'width_ratios': [1,2,2,2]})
    for ax in axs: ax.axis('off'); ax.margins(0,0)
    report.plot_atlas(atlas, axs[0])
    report.plot_summary_dist(axs[1], vectors_noise, vectors_denoise, denoise, strategy)
    report.compare(denoise, qc_means['rmsd'], axs[2:])
    fig.savefig(f'{denoise}/group/strat-{strategy}_plot.png', dpi=300)
    report.html_report(denoise, strategy, strategies[strategy], plots)
    plt.close(fig)
    del fig, axs


def main():  

    args = get_args(strategies.keys())
    derivatives = args.fmriprep.parent
    fd_thresh = strategies[args.strategy]['fd_thresh'] 
    dvars_thresh = strategies[args.strategy]['std_dvars_thresh'] 
    atlas = datasets.fetch_atlas_basc_multiscale_2015(version='asym')['scale122']
    
    if not args.report_only:

        data = Data(args.fmriprep, strategies[args.strategy])

        for mask, preproc, confound, motion_df in zip(data.masks, data.preprocs, data.confounds, data.motion_dfs):

            file_entities = parse_file_entities(preproc)
            sub, task= file_entities['subject'], file_entities['task']
            print(f'> sub-{sub} task-{task}')
            path_plot, path_none, path_vector, path_qc = build_path(derivatives, sub, task, args.strategy)
            motion_df.to_csv(path_qc, sep='\t', index=False)

            mask, preproc_noise = image.load_img(mask), image.load_img(preproc)
            assert mask.shape == preproc_noise.shape[:-1], 'mask and preproc images have different shapes'
            preproc_denoise = image.clean_img(preproc, detrend=False, standardize=False, confounds=confound, mask_img=mask)
            preproc = preproc_noise, preproc_denoise

            timeseries = get_timeseries(preproc, atlas, mask, args.smooth_fwhm, derivatives)

            vector, matrix = get_correlations(timeseries, path_none, path_vector, args.strategy)

            run_summary(vector, matrix, sub, task, motion_df, fd_thresh, dvars_thresh, preproc, mask, (path_plot, path_none, path_vector, path_qc))


    group_summary(derivatives, args.strategy, atlas)

if __name__ == '__main__': main()