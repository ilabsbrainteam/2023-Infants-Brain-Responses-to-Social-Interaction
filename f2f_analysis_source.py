#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2F analysis functions, particularly for power spectra. Called from F2F_PSDxx
notebooks.

In order for the MMF analysis script to "borrow" these functions,
the conditions input (via the 'conditions' argument in create_srcpsds()),
is interpreted as a TALK and IGNORE condition (via the set tuple
'conditions_static'). This standardizes the naming of the resulting stc files.

New bands from 2020-01-23.

@author: neuromac
"""

import os
import time
import numpy as np
from scipy.stats import ttest_ind

import mnefun
import mne
from mne.minimum_norm import compute_source_psd_epochs, read_inverse_operator

try:  # doesn't work as bootstrapped import for mmf scripts
    from f2f_analysis_sensor import remove_projectors
except:
    print('Skipped importing from f2f_analysis_sensor')

# ---- Run-time parameters ----------------------- #
bands_type = 'reduced'   # 'infant', 'original', etc (defs below)
flimits = (0, 60)
conditions_static = ('talk', 'ignore')   # 0, 1 index into PSDs

method = 'sLORETA'
lambda2 = 1. / 9.
mt_bandwidth = 2.0
morph_subject = '14mo_surr'

PCT_LIMITS_BYTYPE = [(97, 99., 100), (92, 97, 99.5), (99, 99.5, 99.8)]
MAX_RANK = 0.998

# ------------------------------------------------ #

if bands_type=='infant':
    bands_def = {'low theta':(3.0,4.5), 'high theta':(4.5,6.0),
                 'alpha':(6.0,9.0), 'beta':(9.0,20.0), 'gamma':(20.0,60.0)}
elif bands_type=='reduced':
    bands_def = {'theta':(4.0,8.0), 'alpha':(8.0,12.0), 'beta':(12.0,28.0),
                 'gamma':(28.0,60.0)}
elif bands_type=='nogamma':
    bands_def = {'theta':(4.0,8.0), 'alpha':(8.0,12.0), 'beta':(12.0,28.0)}
else:
    bands_def = {'theta':(4.0,8.0), 'alpha':(8.0,12.0), 'beta':(12.0,28.0),
                 'gamma':(28.0,50.0), 'high gamma':(50.0,60.0)}
    
band_limits = (min([min(tupl) for key,tupl in bands_def.items()]),
               max([max(tupl) for key,tupl in bands_def.items()]))
n_bands = len(bands_def)


def create_srcpsds(subjects, p, epoch_range=None, activate_proj=True,
                   conditions=conditions_static,
                   morph=None, scaling='globalpower_logmean', extra_tag=''):
    '''Create power spectra for list of subjects and save as fif files
    with the ending 'volpsd-vl.fif' for volumetric or 'srfpsd-lh.fif'/'rh.fif' 
    for surface source spaces.
    NOTE: The inverse model used has free orientations and was created using
    the empty room covariance matrix.
    '''
    assert morph in (None, morph_subject)
    
    print(f'Creating power spectra using {bands_type} frequency bands.')
    print('The frequency bands will be:')
    print('**', bands_def, '**')
    
    n_subj = len(subjects)
    fmin, fmax = p.psd_params['fmin'], p.psd_params['fmax']
    n_jobs = p.n_jobs
    
    # Define source object required for morphing #
    subject = subjects[0]  # first determine what type of source model
    inv_file = subject + '-meg-erm-free-inv.fif'
    inv_file = os.path.join(p.work_dir, subject, p.inverse_dir, inv_file)
    inv = read_inverse_operator(inv_file, verbose=False)

    model_type = inv['src'].kind
    assert model_type in ('surface', 'volume')
    
    if model_type=='volume':
        src_morph_name = os.path.join(p.subjects_dir,  # invalid if no morph
                        morph, 'bem', morph + '-vol-5-src.fif')
    else:
        src_morph_name = os.path.join(p.subjects_dir,
                        morph, 'bem', morph + '-oct-6-src.fif')
    if morph:
        src_morph = mne.read_source_spaces(src_morph_name)
    else:
        src_morph = None
    
    # Collect projectors to NOT apply below #
    projrm_list = []
    if type(activate_proj) is str:
        assert activate_proj.lower() == 'no_ecg'
        projrm_list.append('meg-ECG')
        activate_proj = True
    
    # Create spectra for each subject #
    for subject in subjects:
        # Load epochs, removing projectors as directed #
        epoch_file = p.epochs_prefix + '_80-sss_' + subject + '-epo.fif'
        epoch_file = os.path.join(
            p.work_dir, subject, p.epochs_dir, epoch_file)
        epochs = mne.read_epochs(epoch_file, proj=False, preload=True)
        for proj_str in projrm_list:
            rmcnt = remove_projectors(epochs, proj_str)
            print(f'  {rmcnt} projectors removed.')
        if activate_proj:
            epochs.apply_proj()
        
        # Load inverse model - this is for the empty room!! #
        inv_file = subject + '-meg-erm-free-inv.fif'
        inv_file = os.path.join(p.work_dir, subject, p.inverse_dir, inv_file)
        inv = read_inverse_operator(inv_file, verbose=False)

        src_subject = inv['src']   # some subjects have "OTP" label
        assert src_subject.kind == model_type
        if src_subject[0]['subject_his_id'] != subject:
            assert src_subject[0]['subject_his_id'][:7] == subject[:7]
            print('    ** Adjusting subject label for source space **')
            for src in src_subject:
                src['subject_his_id'] = subject
        
        vertices = [s['vertno'].copy() for s in inv['src']]
        
        # Create source objects containing banded PSDs #
        sources = make_psd_sources(epochs, inv, vertices, model_type,
                    epoch_range=epoch_range, scaling=scaling,
                    conditions=conditions,
                    fmin=fmin, fmax=fmax, subject=subject, n_jobs=n_jobs)
        
        psd_dir = os.path.join(p.work_dir, subject, p.psd_dir)
        if not os.path.isdir(psd_dir):
            os.mkdir(psd_dir)
            print(f'  Created psd directory for {subject}.')
        
        # Optionally create morphed versions #
        sources_morphed = {}
        if morph:
            morph_fcn = mne.compute_source_morph(src=src_subject,
                                src_to=src_morph, smooth=10, spacing=None,
                                subject_from=subject, subject_to=None)
            for cond, src in sources.items():
                sources_morphed[cond] = morph_fcn.apply(src)
        
        # Save sources as separate files, including morphed versions #
        for cond, src in sources.items():
            output_stem = os.path.join(psd_dir, subject + extra_tag +
                            '_' + cond + '_psd')
            src.save(output_stem, ftype='stc', verbose=None)
        
        for cond, src in sources_morphed.items():
            output_stem = os.path.join(psd_dir, subject + extra_tag +
                            '_' + cond + '_psd_morphed')
            src.save(output_stem, ftype='stc', verbose=None)

    print(f'Processed source spectra for {n_subj} subjects.')


def make_psd_sources(epochs, inv, vertices, model_type, epoch_range=None, 
                    scaling='global_logmean', fmin=0, fmax=100,
                    conditions = conditions_static,
                    subject=None, n_jobs=1):
    '''Creates PSDs as STC numpy arrays, averages across epochs, then returns
    spectra as source or volume-source estimate objects.
    Conditions tuple expected to be in order of ('talk', 'ignore') but 
    can have different names (e.g. ('attend-1', 'ignore-1')).
    No longer saves "diff" type PSDs, as these may be trivially calculated.
    '''
    assert scaling in {'global_logmean', 'global', 'local',
                       'local_logmean', 'logmean', None}
    assert fmax >= band_limits[1]
    assert len(conditions) <= len(conditions_static)

    stcs_all = []
    for cond in conditions:
        # Determine the slice of epochs to include #
        epochs_sub = epochs[cond]
        if not epoch_range:
            idx = slice(0, len(epochs_sub))
        elif len(epoch_range)==1:   # interp as target to end
            idx = slice(epoch_range[0], len(epochs_sub))
        elif len(epoch_range)==2:  # interp as start to pos
            idx = slice(epoch_range[0], epoch_range[1]+1)
        else:
            ValueError('Epochs_range is wrong size')
        epochs_sub = epochs_sub[idx]
        
        # Create band spectra for condition, separately for each epoch #
        freqs = np.fft.rfftfreq(len(epochs_sub.times),
                            1. / epochs_sub.info['sfreq'])
        freqs = freqs[(freqs<=fmax)]
        
        stcs = []
        for stc in compute_source_psd_epochs(
                    epochs_sub, inv, lambda2, method, pick_ori=None,
                    pca=True, inv_split=1, n_jobs=n_jobs, adaptive=False,
                    low_bias=True, bandwidth=mt_bandwidth, fmax=fmax,
                    prepared=False, return_sensor=False,
                    return_generator=True, label=None, verbose=False):
            stcs.append(np.array([np.mean(
                stc.data[..., (bands_def[band][0] <= freqs) & \
                         (freqs < bands_def[band][1])],
                axis=-1) for band in bands_def])
                )  # band becomes first dim during this loop
        stcs = np.array(stcs)
        stcs = stcs.transpose(0, 2, 1)  # put band back to last dim
        
        stcs_all.append(stcs)    # each has #epochs x #vertices x #bands

    # Calculate normalizing factors for all channels #
    # First sum across frequency bands, then average across epochs.
    stcs_scalings = np.concatenate(stcs_all)  # consider talk AND ignore
    stcs_scalings = stcs_scalings.sum(axis=-1)
    stcs_scalings = stcs_scalings.mean(axis=0)  # vector of len = #vertices
    
    # Normalize, then store as source objects for each condition #
    if model_type == 'volume':
        srcClass = mne.VolSourceEstimate
    else:
        srcClass = mne.SourceEstimate

    srcs_all = {}
    for i, cond in enumerate(conditions):
        stcs = stcs_all[i].copy()
        n_epochs, n_verts, n_freqs = stcs.shape

        # Optionally normalize then log-transform #
        if scaling in {'local_logmean', 'local'}:
            for vv in range(n_verts):
                stcs[:, vv, :] = stcs[:, vv, :] / stcs_scalings[vv]
        elif scaling in {'global_logmean', 'global'}:
            stcs = stcs / stcs_scalings.mean()
        if scaling in {'local_logmean', 'global_logmean', 'logmean'}:
            stcs = 10 * np.log10(stcs)   # can be negative values here
        
        cond_str = conditions_static[i]  # OVERRIDE to talk, ignore !!
        
        srcs_all[cond_str] = srcClass(stcs.mean(axis=0), vertices,
            tmin=0, tstep=1, subject=subject)  # save MEAN across epochs !!
    
    # Also store source objects for an across-epochs t-test #   
    t_, p = ttest_ind(stcs_all[0], stcs_all[1], axis=0)
    p_flip = -np.log10(p)   # more signif. = higher positive value
    p_flip *= np.sign(t_)
    srcs_all['tdiff'] = srcClass(p_flip, vertices,
                                 tmin=0, tstep=1, subject=subject)

    return srcs_all


def create_srcpsd_summary(subjects, p, mode='individual',
              model_type='volume', view_type=None, plot_type='inflated',
              pattern_in='', pattern_out='_srcpsd', grpplot_mode = 'standard',
              path_out='./', title='%s Source PSDs', subject_paths=None,
              limits_bytype=[None, None, None]):
    '''2020-10-03: Now can render volumetric or surface spaces.
    For surface plots, the view defaults to ['lat', 'med']; set 'view_type'
    to 'front' to set view to ['lat', 'ros'].
    
    2020-02-09: Added subject_paths to allow pooling across work directories.
    Make sure the p.psd_dir and .inverse_dir work for all subjects.
    pattern_in can also now be a list.
    2021-04-21: Added limits_bytype to pass in color limits (surface only).
    Ordering is condition | diff | tdiff. Careful - it's mutable!
    
    Note that the "stcs" loaded here are actually AVERAGES across epochs.
    '''
    assert mode in ('individual', 'group')
    assert model_type in ('surface', 'volume')
    assert grpplot_mode in ('standard', 'condition')
    if not type(p) == mnefun._mnefun.Params:
        raise ValueError('"p" must be an MNEFun parameter object.')
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    
    if not subject_paths:
        subject_paths = [p.work_dir] * len(subjects)
    assert type(subject_paths) == list
    assert len(subject_paths) == len(subjects)
    
    if not type(pattern_in) == list:
        pattern_in_list = [pattern_in] * len(subjects)
    else:
        pattern_in_list = pattern_in

    if model_type == 'volume':
        end_pattern1 = '_psd_morphed-vl.stc'
        end_pattern2 = '_psd-v1.stc'
    else:
        end_pattern1 = '_psd_morphed-lh.stc'
        end_pattern2 = '_psd-lh.stc'

    assert plot_type in ('inflated', 'pial', 'white')
    assert view_type in (None, 'front')
    if view_type == 'front':
        views = ['lat', 'ros']
    else:
        views = ['lat', 'med']
    
    cnt = 0   # for accumulating group averages
    cnt_target = len(subjects)
    
    for subject, work_dir, pattern_in in \
      zip(subjects, subject_paths, pattern_in_list):
        # Get the morphed spectral (epoch-avgd) STCs of current subject #
        stc_path = os.path.join(work_dir, subject, p.psd_dir)
        
        file_pattern = subject + pattern_in + '_talk' + end_pattern1
        stc_file = os.path.join(stc_path, file_pattern)
        stc_talk = mne.read_source_estimate(stc_file)

        file_pattern = subject + pattern_in + '_ignore' + end_pattern1
        stc_file = os.path.join(stc_path, file_pattern)
        stc_ignore = mne.read_source_estimate(stc_file)
        
        file_pattern = subject + pattern_in + '_tdiff' + end_pattern1
        stc_file = os.path.join(stc_path, file_pattern)
        stc_tdiff = mne.read_source_estimate(stc_file)
                
        stc_diff = stc_talk.copy()
        # stc_diff.data = stc_talk.data - stc_ignore.data
        stc_diff.data = 10**(stc_talk.data) - 10**(stc_ignore.data)

        stc_talk.data = stc_talk.data + 20  # force positive (after diff)
        stc_ignore.data = stc_ignore.data + 20
        stc_talk.data[(stc_talk.data < 0)] = 0
        stc_ignore.data[(stc_ignore.data < 0)] = 0

        stc_mean = stc_talk.copy()
        stc_mean.data = (stc_talk.data + stc_ignore.data) / 2

        assert stc_tdiff.data.shape[1]==n_bands
        
        # Accumulate for group average #
        if mode == 'group' and cnt == 0:    # initialize accum arrays
            stc_talk_avg = stc_talk.copy()
            stc_talk_avg.data[:] = 0
            stc_ignore_avg = stc_ignore.copy()
            stc_ignore_avg.data[:] = 0
            stc_diff_avg = stc_diff.copy()
            stc_diff_avg.data[:] = 0
            stc_mean_avg = stc_mean.copy()
            stc_mean_avg.data[:] = 0
            stc_talk_accum = []
            stc_ignore_accum = []
        if mode == 'group':
            stc_talk_avg.data += stc_talk.data
            stc_ignore_avg.data += stc_ignore.data
            stc_diff_avg.data += stc_diff.data
            stc_mean_avg.data += stc_mean.data
            stc_talk_accum.append(stc_talk.data)
            stc_ignore_accum.append(stc_ignore.data)
            cnt += 1

        # Get the non-morphed spectral STCs of current subject #
        file_pattern = subject + pattern_in + '_talk' + end_pattern2
        stc_file = os.path.join(stc_path, file_pattern)
        stc_talk_nomorph = mne.read_source_estimate(stc_file)

        file_pattern = subject + pattern_in + '_ignore' + end_pattern2
        stc_file = os.path.join(stc_path, file_pattern)
        stc_ignore_nomorph = mne.read_source_estimate(stc_file)
        
        # Make the no-morph versions strictly positive (mean handled below) #
        stc_talk_nomorph.data = stc_talk_nomorph.data + 20
        stc_talk_nomorph.data[(stc_talk_nomorph.data < 0)] = 0
        stc_ignore_nomorph.data = stc_ignore_nomorph.data + 20
        stc_ignore_nomorph.data[(stc_ignore_nomorph.data < 0)] = 0
        
        # Set up plotting - what to plot #
        if mode == 'group' and cnt < cnt_target:
            continue
        elif mode == 'group':
            stc_talk_avg.data *= 1/cnt  # final averages across subjects
            stc_ignore_avg.data *= 1/cnt
            stc_diff_avg.data *= 1/cnt
            stc_mean_avg.data *= 1/cnt
            
            t_, p_ = ttest_ind(np.array(stc_talk_accum),
                              np.array(stc_ignore_accum), axis=0)
            p_flip = -np.log10(p_)   # becomes positive
            p_flip *= np.sign(t_)
            stc_tdiff_group = stc_talk.copy()
            stc_tdiff_group.data = p_flip
            
            # stc_mean_avg.data += 40    # for surf. maps!!
            # stc_mean_avg.data[(stc_mean_avg.data < 0)] = 0
            
            if grpplot_mode == 'standard':
                stc_plot = [stc_diff_avg, stc_tdiff_group, stc_mean_avg]
                stc_descrip = {'talk-ignore':1, 't-test':2, 'mean':0}
            else:
                # stc_talk_avg.data += 40  # for surf. maps!!
                # stc_talk_avg.data[(stc_talk_avg.data < 0)] = 0
                # stc_ignore_avg.data += 40  # for surf. maps!!
                # stc_ignore_avg.data[(stc_ignore_avg.data < 0)] = 0
                
                stc_plot = [stc_talk_avg, stc_ignore_avg, stc_diff_avg]
                stc_descrip = {'talk':0, 'ignore':0, 'talk-ignore':1}
            
            prefix = 'group'
        else:
            stc_plot = [stc_talk_nomorph, stc_ignore_nomorph,
                        stc_diff]  # must match 'src_plot' below
            stc_descrip = {'talk (sbj)':0, 'ignore (sbj)':0, 'talk-ignore':1}
            prefix = subject
            print(f'Compiling report for {subject}.')

        if model_type == 'volume':
            # Set up plotting 2 - what source spaces to use #
            print('Loading morph source space.')
            src_file = os.path.join(p.subjects_dir, morph_subject,
                             'bem', morph_subject + '-vol-5-src.fif')
            src_morph = mne.read_source_spaces(src_file)
            if mode == 'group':
                src_plot = [src_morph] * len(stc_plot)
            else:
                inv_file = subject + '-80-sss-meg-free-inv.fif'
                inv_file = os.path.join(work_dir, subject,
                                        p.inverse_dir, inv_file)
                inv = read_inverse_operator(inv_file, verbose=False)
                src = inv['src']
                src_plot = [src, src, src_morph]  # || stc_plot
            
            # Create brain maps of STC activity #
            figure_list, figure_info = make_volumeplots(stc_plot,
                                info=stc_descrip, mode=mode, srcs=src_plot)
            scale = 1.5
        else:
            # Set up plotting 2 - what subject name to use #
            if mode == 'group':
                src_plot = [morph_subject] * len(stc_plot)
            else:
                src_plot = [subject, subject, morph_subject]
            # Create brain maps of STC activity #
            figure_list, figure_info = make_surfaceplots(stc_plot,
                                info=stc_descrip, mode=mode,
                                src_subjects=src_plot,
                                limits_bytype=limits_bytype,
                                views_list=views, plot_type=plot_type)
            scale = 1
        
        # Compile all figures into a report #
        if not os.path.exists(path_out):
            os.mkdir(path_out)

        if '%s ' in title and not mode=='group':
            title_use = title.replace('%s ', '{sbj} ')  # don't overwrite!
            title_use = title_use.format(sbj=subject)
        elif '%s ' in title:
            title_use = title.replace('%s ', 'Group')
        else:
            title_use = title

        report = mne.Report(title=title_use, image_format='png')
        for fig, info in zip(figure_list, figure_info):
            report.add_figs_to_section(fig, captions=info[0], scale=scale,
                    section='Condition: ' + info[1])
        
        report_file = os.path.join(path_out, prefix + pattern_out + '.html')
        report.save(report_file, open_browser=False, overwrite=True)


def make_surfaceplots(stcs, info, mode='individual', src_subjects=None,
                      views_list=['lat', 'med'], plot_type='inflated',
                      limits_bytype=[None, None, None]):
    '''Generate Mayavi source plots (screenshots) for a list of STCs.
    The info output is a two-element list for figure caption and report section.
    '''
    def get_ranklvl(data_mat, lvl_fraction):
        pt = int(data_mat.shape[0] * lvl_fraction)
        rank_lvl = np.sort(data_mat, axis=0)
        rank_lvl = rank_lvl[pt, :]
        return rank_lvl
        
    if len(src_subjects) != len(stcs):
        raise ValueError('Need a source subject name for each figure.')
    assert type(info)==dict and len(info)==len(stcs)
    
    if plot_type == 'inflated':
        alpha = 1.00
    else:
        alpha = 0.75
    
    kind_bytype = ['value'] * 3
    ulimits_bytype = limits_bytype.copy()
    for i in (1,2):  # default for diff and tdiff types
        if ulimits_bytype[i] is None:
            ulimits_bytype[i] = [PCT_LIMITS_BYTYPE[i]] * n_bands
            kind_bytype[i] = 'percent'
    
    if ulimits_bytype[0] is None:  # default for talk and ignore
        max_lvls = np.zeros((n_bands))
        for stc, stc_type in zip(stcs, info):
            lid =info[stc_type]
            if lid != 0:
                continue
            lvls = get_ranklvl(stc.data, MAX_RANK)
            max_lvls = np.max([lvls, max_lvls], axis=0)
        assert all(max_lvls)
        ulimits_bytype[0] = []
        kind_bytype[0] = 'value'
        for ib in range(n_bands):
            clim = np.array(PCT_LIMITS_BYTYPE[0])/100 * max_lvls[ib]
            ulimits_bytype[0].append(clim)
    
    sfig_list, sfig_info = list(), list()

    for stc, stc_type, src_name in zip(stcs, info, src_subjects):
        lid = info[stc_type]
        clims = list()
        
        if lid == 0:
            for ib in range(n_bands):
                clims.append(dict(kind=kind_bytype[lid],
                                  lims=ulimits_bytype[lid][ib]))
        else:
            for ib in range(n_bands):
                clims.append(dict(kind=kind_bytype[lid],
                                  pos_lims=ulimits_bytype[lid][ib]))
        plot_subject = src_name
        
        for i, band in enumerate(bands_def):
            size = 800  #(500, 500)
            brain = stc.copy().crop(i, i).plot(surface=plot_type,
                subjects_dir=None, time_viewer=False,
                initial_time=1., subject=plot_subject, hemi='split',
                views=views_list, size=size, alpha=alpha,
                clim=clims[i], backend='mayavi', verbose=False)
            
            time.sleep(0.2)
            sfig = brain.screenshot()  # trim_bg(brain.screenshot(), 255)
            sfig_list.append(sfig)
            
            caption = stc_type + ' | ' + band.upper()
            sfig_info.append([caption, stc_type])
            
            time.sleep(0.1)
            brain.close()
            time.sleep(0.1)
            
    return sfig_list, sfig_info


def make_volumeplots(stcs, info=[], mode='subject', srcs=[]):
    # NOTE: Plot limits aren't updated; see make_surfaceplots() !!!
    # So, this code is currently BROKEN. Must fix the 'clim' lines below.
    if len(srcs) != len(stcs):
        raise ValueError('Need a source space for each figure.')
    if info:
        assert type(info)==list and len(info)==len(stcs)
    else:
        info = [''] * len(stcs)

    vfig_list, vfig_info = list(), list()

    for stc, stc_type, src in zip(stcs, info, srcs):
        if mode=='group' and (stc_type in conditions_static):
            clims = ['auto'] * n_bands
        elif mode=='group' and stc_type=='talk-ignore':
            clims = list()
        elif mode=='group' and stc_type=='t-test':
            clims = list()
        else:
            clims = ['auto'] * n_bands   # free scaling for individuals
        clims = ['auto'] * n_bands    # SMB: temp. for preliminary scaling!!
    
        plot_subject = src[0]['subject_his_id']
        
        for i, band in enumerate(bands_def):
            brain = stc.copy().crop(i, i).plot(src, mode='stat_map',
                initial_time=1., subject=plot_subject, clim=clims[i], 
                subjects_dir=None, verbose=False, show=False)
            brain.axes[1].remove()   # remove the unneeded line plot
            
            vfig_list.append(brain)
            caption = stc_type + ' | ' + band.upper()
            vfig_info.append([caption, stc_type])
            
    return vfig_list, vfig_info



#---- BELOW IS "PRE-PSD" CODE ----------------------------------------------#
GROUP_CONTRAST_LIMS = [(2, 3.5, 5), (1, 2, 3), (1, 2, 3),  # for pre-psd
                       (1, 2, 3), (.5, 1, 2)]

def make_sourceplots(stcs, info=None, mode='subject'):
    '''Generate Mayavi source plots (screenshots) for a list of STCs.
    The info output is a two-element list for figure caption and section.
    '''
    if info:
        assert type(info)==list and len(info)==len(stcs)
    else:
        info = [''] * len(stcs)
    
    if mode=='group':
        clims = list()
        for i in range(n_bands):
            clims.append(dict(kind='value', pos_lims=GROUP_CONTRAST_LIMS[i]))
    else:
        clims = ['auto'] * n_bands   # free scaling for subjects
    
    mfig_list, mfig_info = list(), list()

    for stc, stc_type in zip(stcs, info):  # Can't guarantee vertices align
        for i, band in enumerate(bands_def):  #   with the morph subject!!
            brain = stc.copy().crop(i, i).plot(figure=None,
                subjects_dir=None, time_viewer=False,
                initial_time=1., subject=morph_subject, hemi='split',
                views=['lat','med'], size=(500, 500),
                clim=clims[i], backend='mayavi', verbose=False)
            
            mfig = brain.screenshot()  # trim_bg(brain.screenshot(), 255)
            mfig_list.append(mfig)
            
            caption = band
            mfig_info.append([caption, stc_type])
    
            time.sleep(0.3)
            brain.close()
            time.sleep(0.2)
            
    return mfig_list, mfig_info


def create_prestc_summary(subjects, p='./', mode='individual', amp='abs',
              pattern_in='_a', pattern_out='_stc_report',
              path_out='./', title='Summary'):
    '''Create report(s) summarizing older spectral analysis for one or more
    subjects. Argument p can be a params object (for subject-level loading)
    or the full path to the subject PSDs (if all are in one directory).
    Mode is 'individual' or 'group', the latter creating an average.
    Spectrum amplitudes are normalized power (probably without log trans-
    formation), grouped into five bands.
    '''
    if not type(p) is str:
        ValueError('Subject-level path not yet implemented.')
    if type(subjects) is dict:
        s_dict = True
    else: s_dict = False
    assert mode in ('individual', 'group')

    cnt = 0   # for accumulating group averages
    cnt_target = len(subjects)
    
    for subject in subjects:
        # Get the spectral STCs of current subject #
        if s_dict:
            subject_str = subjects[subject]
        else: subject_str = subject
        stc_file = os.path.join(p, subject_str + pattern_in + '_talk')
        stc_talk = mne.read_source_estimate(stc_file)
        stc_file = os.path.join(p, subject_str + pattern_in + '_ignore')
        stc_ignore = mne.read_source_estimate(stc_file)
        
        assert stc_talk.data.shape[1]==n_bands
        assert stc_talk.data.shape == stc_ignore.data.shape
    
        # Create a talk-ignore "contrast" STC #
        stc_contrast = stc_talk.copy()
        stc_contrast.data = stc_talk.data - stc_ignore.data
        
        # Create a talk+ignore "average" STC #
        stc_sum = stc_talk.copy()
        stc_sum.data = (stc_talk.data + stc_ignore.data) / 2
    
        # Accumulate for group average #
        if mode == 'group' and cnt==0:
            stc_contrast_avg = stc_contrast.copy()    # initialize
            stc_contrast_avg.data[:] = 0
            stc_sum_avg = stc_sum.copy()    # initialize
            stc_sum_avg.data[:] = 0
            stc_talk_avg = stc_talk.copy()
            stc_talk_avg.data[:] = 0
            stc_ignore_avg = stc_ignore.copy()
            stc_ignore_avg.data[:] = 0
        if mode == 'group':
            stc_contrast_avg.data += stc_contrast.data
            stc_sum_avg.data += stc_sum.data
            stc_talk_avg.data += stc_talk.data
            stc_ignore_avg.data += stc_ignore.data
            cnt += 1
    
        # Plot results #
        if mode == 'group' and cnt<cnt_target:
            continue
        elif mode == 'group':
            stc_contrast_avg.data *= 1/cnt  # final averages
            stc_sum_avg.data *= 1/cnt
            stc_talk_avg.data *= 1/cnt
            stc_ignore_avg.data *= 1/cnt
            
            stc_plot = [stc_contrast_avg]
            stc_descrip = ['talk-ignore']
            prefix = 'group' 
        else:
            stc_plot = [stc_contrast]   # only contrast for indiv.s
            stc_descrip = ['talk-ignore']
            prefix = subject
            print(f'Compiling report for {subject}.')
            
        # Create brain maps of STC activity #
        figure_list, figure_info = make_sourceplots(stc_plot,
                                    info=stc_descrip, mode=mode)
        
        # Compile all figures into a report #
        if not os.path.exists(path_out):
            os.mkdir(path_out)

        report = mne.Report(title=title, image_format='png')
        for fig, info in zip(figure_list, figure_info):
            report.add_figs_to_section(fig, captions=info[0],
                    section='Condition: ' + info[1])
        report_file = os.path.join(path_out, prefix + pattern_out + '.html')
        report.save(report_file, open_browser=False, overwrite=True)
