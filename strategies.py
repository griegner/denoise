
strategies = {

    'HMPCompCorScrub': {'strategy': ['high_pass', 'motion', 'compcor', 'scrub'], 
                         'motion': 'basic', 
                         'compcor': 'anat', 
                         'n_compcor': 5, 
                         'acompcor_combined': True, 
                         'fd_thresh': 0.9, 
                         'std_dvars_thresh': None},

    'HMPCompCorScrubGS': {'strategy': ['high_pass', 'motion', 'compcor', 'scrub', 'global'], 
                         'motion': 'basic', 
                         'compcor': 'anat', 
                         'n_compcor': 5, 
                         'acompcor_combined': True, 
                         'fd_thresh': 0.9, 
                         'std_dvars_thresh': None, 
                         'global_signal': 'basic'},               

    'HMPWMCSFScrub': {'strategy': ['high_pass', 'motion', 'wm_csf', 'scrub'], 
                         'motion': 'basic', 
                         'wm_csf': 'full',
                         'fd_thresh': 0.9, 
                         'std_dvars_thresh': None}
    }

