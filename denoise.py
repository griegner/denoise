#!/Users/zeidanlab_pro/anaconda3/envs/neuroimg/bin/python

from bids import BIDSLayout
from pathlib import Path
import argparse
from joblib import memory
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting
from nilearn import datasets
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import warnings
warnings.filterwarnings('ignore')

import report

def get_args():
    parser = argparse.ArgumentParser(description='denoise fMRI data')
    parser.add_argument('derivatives', type=Path, help='path to derivatives, parent directory of fmriprep')
    parser.add_argument('pipeline', type=str, help='pipelines: <6,24>HMPCompCor<Separate,Combined>SpikeReg[GS]')
    parser.add_argument('--fd_threshold', type=float, default=0.9, help='choose FD threshold for SpikeReg, default 0.9mm')
    parser.add_argument('--smooth_fwhm', type=float, default=6, help='choose smoothing kernel, default fwhm 6mm')
    parser.add_argument('--report_only', action='store_true', help='run group-level summary only')
    args = parser.parse_args()
    assert args.derivatives.is_dir(), 'derivatives directory does not exist'
    return args

def get_data(derivatives):
    atlas = datasets.fetch_atlas_basc_multiscale_2015(version='asym')["scale122"]
    layout = BIDSLayout(derivatives + '/fmriprep', validate=False, index_metadata=False)
    preprocsICA = layout.get(space='MNI152NLin6Asym', suffix='bold', extension='.nii.gz')
    preprocs = [file for file in preprocsICA if 'desc-preproc' in file.filename]
    masks = layout.get(regex_search=True, task='.[1-4]', space='MNI152NLin6Asym', suffix='mask', extension='.nii.gz')
    dfs = layout.get(suffix='timeseries', extension='.tsv')
    jsons = layout.get(suffix='timeseries', extension='.json')
    assert len(preprocs)==len(masks)==len(dfs)==len(jsons), 'missings fmriprep files'
    return layout, atlas, preprocs, masks, dfs, jsons

def select_components(pipeline, json):
    json = json.filter(regex='a_comp_cor')
    if 'CompCorCombined' in pipeline: # [Behzadi2007] 
        comp_cor = json.columns[json.loc['Mask']=='combined'].to_list()
    elif 'CompCorSeparate' in pipeline: # [Muschelli2014]
        comp_cor = json.columns[json.loc['Mask']=='CSF'].to_list()[:5]
        comp_cor.extend(json.columns[json.loc['Mask']=='WM'].to_list()[:5])
    if 'CompCor' in pipeline:
        return '|'.join(comp_cor)

def build_path(layout, derivatives, pipeline, sub, task, space):
    pattern = 'sub-{subject}_task-{task}_space-{space}_pipeline-{pipeline}_{suffix}.{extension}'
    path = Path(f'{derivatives}/denoise/sub-{sub}')
    path.mkdir(parents=True, exist_ok=True)
    entities = {
            'subject': sub,
            'task': task,
            'space': space,
            'pipeline': pipeline,
            'desc': 'denoised',
            'suffix': 'confounds',
            'extension': 'tsv'}
    path_confounds = path / layout.build_path(entities, pattern, validate=False, absolute_paths=False)
    entities['suffix'] = 'plot'; entities['extension'] = 'png'
    path_plot = path / layout.build_path(entities, pattern, validate=False, absolute_paths=False)
    entities['suffix'] = 'connMat'; entities['extension'] = 'npy'
    path_matrix = path / layout.build_path(entities, pattern, validate=False, absolute_paths=False)
    return path_confounds, path_plot, path_matrix

def get_outliers(df, fd_threshold):
    df = df.drop(df.filter(regex='motion_outlier').columns, axis=1)
    df['motion_outliers_all'] = df.framewise_displacement.gt(fd_threshold)
    fd_outliers = df.motion_outliers_all.sum() / len(df.index)
    volumes = df.index[df['motion_outliers_all']].tolist()
    for idx, outlier in enumerate(volumes):
        df[f'motion_outlier{idx:02}'] = 0
        df.at[outlier, f'motion_outlier{idx:02}'] = 1
    return df, fd_outliers

def plot_dist(ax, matrix, matrix_clean):
    sns.distplot(matrix.flatten(), color='#B62E33', hist=False, ax=ax)
    sns.distplot(matrix_clean.flatten(), color='#3C83BC', hist=False, ax=ax)
    ax.axvline(x=0, c='k', alpha=.3, linestyle='dashed')

def plot_connectome(ax, matrix, matrix_clean, sub, task):
    ax.set_title(f'sub-{sub} task-{task}', loc='left', weight='bold', size='medium')
    plotting.plot_matrix(matrix, colorbar=False, tri='full', axes=ax)
    plotting.plot_matrix(matrix_clean, colorbar=False, tri='lower', axes=ax)

def plot_fd(ax, fd, fd_threshold, fd_outliers):
    ax.set_title(f'FD   mean: {fd.mean():.2}   outliers > {fd_threshold}mm: {fd_outliers:.2%}', loc='left', size='small')
    ax.axhline(y=fd_threshold, color='r', linestyle=':', lw=.6)
    ax.plot(fd, color='k', lw=0.5)
    outliers = [idx for idx, val in enumerate(fd) if val>fd_threshold]
    ax.plot(outliers, [0]*len(outliers), color='r', linestyle='none', marker='|')

def plot_carpet(ax, preproc, preproc_clean, mask):
    plotting.plot_carpet(preproc, mask_img=mask, axes=ax[0])
    plotting.plot_carpet(preproc_clean, mask_img=mask, axes=ax[1])

def main():  

    pipelines = {'6HMPCompCorSeparateSpikeReg': '[rot,trans]_[xyz]$|cosine|motion_outlier[0-9]+',
                 '6HMPCompCorSeparateSpikeRegGS': '[rot,trans]_[xyz]$|cosine|motion_outlier[0-9]+|global_signal$',
                 '24HMPCompCorSeparateSpikeReg': '[rot,trans]_[xyz]|cosine|motion_outlier[0-9]+',
                 '6HMPCompCorCombinedSpikeReg': '[rot,trans]_[xyz]$|cosine|motion_outlier[0-9]+', 
                 '6HMPCompCorCombinedSpikeRegGS': '[rot,trans]_[xyz]$|cosine|motion_outlier[0-9]+|global_signal$',
                 '6HMPWMSpikeReg': '[rot,trans]_[xyz]$|cosine|motion_outlier[0-9]+'}

    args = get_args()
    derivatives = str(args.derivatives)
    pipeline = args.pipeline
    fd_threshold = args.fd_threshold
    smooth_fwhm = args.smooth_fwhm

    if not args.report_only:
        layout, atlas, preprocs, masks, dfs, jsons = get_data(derivatives)
        for mask, df, json, preproc in zip(masks, dfs, jsons, preprocs):
            
            sub, task, space = preproc.entities['subject'], preproc.entities['task'], preproc.entities['space']
            path_confounds, path_plot, path_matrix = build_path(layout, derivatives, pipeline, sub, task, space)
            print(f'sub-{sub} task-{task}')

            mask = mask.get_image()
            json = pd.read_json(json)
            comp_cor = select_components(pipeline, json)
        
            df = df.get_df().fillna(0)
            df, fd_outliers = get_outliers(df, fd_threshold)
            conf = df.filter(regex=f'{pipelines[pipeline]}|{comp_cor}')
            conf.to_csv(path_confounds, sep='\t', header=True, index=False)

            preproc = preproc.get_image()
            preproc_clean = image.clean_img(preproc, detrend=False, standardize=True, confounds=conf, mask_img=mask)

            masker = NiftiLabelsMasker(atlas, mask_img=mask, smoothing_fwhm=smooth_fwhm, 
                                       memory=f'{derivatives}/denoise_cache', memory_level=3)
            ts = masker.fit_transform(preproc)
            ts_clean = masker.fit_transform(preproc_clean)

            correlation = ConnectivityMeasure(kind='correlation', discard_diagonal=True)
            matrix = correlation.fit_transform([ts])[0]
            np.fill_diagonal(matrix, np.nan)
            matrix_clean = correlation.fit_transform([ts_clean])[0]
            np.fill_diagonal(matrix_clean, np.nan)
            np.save(path_matrix, matrix_clean)

            fig, axs = plt.subplots(5, figsize=(4,8), gridspec_kw={'height_ratios': [2, 10, 1, 2, 2]})
            for ax in axs: ax.axis('off'); ax.margins(0,.02)
            plot_dist(axs[0], matrix, matrix_clean)
            plot_connectome(axs[1], matrix, matrix_clean, sub, task)
            plot_fd(axs[2], df.framewise_displacement, fd_threshold, fd_outliers)
            plot_carpet(axs[3:], preproc, preproc_clean, mask)

            fig.savefig(path_plot, dpi=300, bbox_inches='tight')
            plt.close(fig)
            del fig, axs

    print(f'group-level summary: pipeline-{pipeline}_report.html')
    denoise = f'{derivatives}/denoise'
    npys, plots = report.find_files(denoise, pipeline)
    fig, axs = plt.subplots(ncols=3, figsize=(16,2), gridspec_kw={'width_ratios': [2,2,2]})
    for ax in axs: ax.axis('off'); ax.margins(0,0)
    report.plot_atlas(axs[0])
    edges = report.plot_summary_dist(axs[1], npys, pipeline)
    np.save(f'{denoise}/group/pipeline-{pipeline}_connMat.npy', edges)
    report.compare(denoise, axs[2])
    fig.savefig(f'{denoise}/group/pipeline-{pipeline}_plot.png', dpi=300)
    report.html_report(denoise, pipeline, plots)
    plt.close(fig)
    del fig, axs

if __name__ == '__main__': main()