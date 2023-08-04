# -*- coding: utf-8 -*-

"""
@author: mdclarke

Originally "score.py" from MC repository at github.com/mdclarke/f2f.
Revised by SMB for F2F MNEFun processing.

Scoring and primary report functions for F2F.

"""
import os
import numpy as np
import datetime

import sortednp as snp

import mne
from mnefun._paths import get_raw_fnames, get_event_fnames, get_report_fnames
from mnefun._report import _report_events

mri_reference = '14mo_surr'


def score_split(p, subjects, run_indices, old_dir = 'lists', id_dict = None,
                tlimits = (0, 1.5), dur = 1.0, step = 0.5):
    '''Create an event list from an existing list, giving multiple
    new "sub-events" for each original one. The original list's subdirectory
    name must be given; the new list will be written to the subdirectory named
    according to p.list_dir, so it should be different.
    
    For simplicity, 'run_indices' is ignored.
    '''
    if old_dir == p.list_dir:
        raise ValueError('Old event subdirectory must be different than ' +
                         'the current one.')
    
    for subj in subjects:
        print(f'Scoring subject {subj}...')
        
        # Load a raw file to get the sampling rate and other info #
        raw_file = get_raw_fnames(p, subj, which='raw',
                                  erm=False, add_splits=False)[0]
        raw = mne.io.read_raw_fif(raw_file, allow_maxshield=True,
                                  verbose=False)
        sfreq = raw.info['sfreq']
        first_samp, last_samp = raw.first_samp, raw.last_samp
        
        # Load events #
        # Note that event timings are absolute (like 'raw.first_samp', the
        # first possible event is typically several seconds past time 0).
        path = os.path.join(p.work_dir, subj, old_dir)
        evt_file = os.path.join(path, 'ALL_' + subj + '-eve.lst')
        events = mne.read_events(evt_file)
        
        # Make sure target ids are in the event list #
        id_list = np.unique(events[:,2])
        id_target = list(id_dict.keys())
        for id in id_target:
            assert id in id_list
        
        # Create new event list with event subintervals #
        idx1 = np.isin(events[:, 2], id_target)
        idx2 = (first_samp + tlimits[0]*sfreq <= events[:, 0])
        idx3 = (events[:, 0] < last_samp - tlimits[1]*sfreq)
        idx4 = (np.diff(events[:,0]) >= tlimits[1]*sfreq)
        idx4 = np.concatenate((idx4, [True]))
        idx = idx1 & idx2 & idx3 & idx4
        sub_events = split_events(events[idx,:], sfreq, tlimits, dur, step)
        for id in id_dict:     # replace old id with new
            id_idx = (sub_events[:, 2] == id)
            sub_events[id_idx, 2] = id_dict[id]
            
        # Combine original unchanged events with new split sub-events # 
        orig_events = events[np.invert(idx1), :]
        evt_times, index_tuple = snp.merge(orig_events[:,0], sub_events[:,0],
                                           indices=True)  # sort on evt times
        n_events = len(evt_times)
        new_events = np.empty((n_events, 3), dtype=int)
        new_events[index_tuple[0],:] = orig_events
        new_events[index_tuple[1],:] = sub_events
        
        assert all(np.diff(new_events[:,0]) >= 0)  # should be in order!
                   
        new_file = get_event_fnames(p, subj, run_indices=None)[0]
        mne.write_events(new_file, new_events)


def split_events(events, sfreq, limits, dur, step=0.0, max_samp=np.inf):
    '''
    'dur' and 'step', referring to the sub-epoch that will be generated from
    the sub-events, are sufficient for the spacing and number of sub-events.
    'limits' gives the time offsets to consider wrt each input event
    (<0 is prior).
    Sub-events won't be created from main events with timings greater than
    or equal to 'max_samp' (in samples; others in seconds).
    '''
    ta, tb = np.round(limits[0]*sfreq), np.round(limits[1]*sfreq)
    tdur, tstep = np.round(dur*sfreq), np.round(step*sfreq)
    
    # An error here means original events weren't spaced enough #
    assert tb > ta
    assert all(np.diff(events[:,0]) >= (tb-ta))
    
    # Loop through original events and create new "sub" events #
    sub_events = []
    for evt in events:
        etime, _, eid = evt
        if etime >= max_samp:  # skip original events too close to end
            continue
        tt = etime + ta
        while tt <= etime + tb - tdur:
            sub_events.append([tt, 0, eid])
            tt += tstep
    
    return np.array(sub_events, dtype=int)


def score(p, subjects, run_indices):
    '''Modified from mnefun._scoring::default_score().
    This was the original code for each event leading to
    a single 7-second epoch for each trial. Use score_split()
    to obtain shorter intevals for sub-epochs.
    '''
    for si, subj in enumerate(subjects):
        print('Scoring subject %s' % subj)

        # Figure out what our filenames should be
        raw_fnames = get_raw_fnames(p, subj, 'raw', False, False,
                                    run_indices[si])
        eve_fnames = get_event_fnames(p, subj, run_indices[si])

        for raw_fname, eve_fname in zip(raw_fnames, eve_fnames):
            print(f'  Raw = {raw_fname}... ')
                
            raw = mne.io.read_raw_fif(raw_fname, allow_maxshield='yes',
                                      verbose=True)
            events = mne.find_events(raw, stim_channel='STI101',
                                     shortest_event=1, mask=128,
                                     mask_type='not_and',
                                     verbose=True)
            events[:, 2] += 10
            if subj.startswith('f2f_108'):  # weird triggering
                mask = (events[:, 2] == 19)
                events[mask, 2] = 13
            n_bad = (~np.in1d(events[:, 2], [11, 13, 15])).sum()
            if n_bad > 0:
                print('Found %d unknown events!' % n_bad)
            else:
                print()

            mne.write_events(eve_fname, events)
            

def fix_mri(p, subjects, bem_only=False):
    '''Scale target MRI files for those subjects missing the "mri" folder
    in the subject anatomy directory. Also creates the -sol.fif file
    in the "bem" folder; optionally, can only do this part.
    '''
    print('Scaling MRIs for subjects:')
    target_pattern = 'T1.mgz'
    
    cnt = 0
    for subj in subjects:
        target_file = os.path.join(p.subjects_dir, subj, 'mri',
                                   target_pattern)
        
        if not bem_only:
            if os.path.exists(target_file):
                continue
        
            config = mne.coreg.read_mri_cfg(subj, p.subjects_dir)
            assert config.pop('n_params') in (1, 3)
            assert config['subject_from'] == mri_reference
        
            mne.coreg.scale_mri(mri_reference, subject_to=subj,
                            subjects_dir=p.subjects_dir,
                            scale=config['scale'], overwrite=True,
                            labels=False, annot=False)
        
        print(f'  {subj}')
                
        sol_fileIN = os.path.join(p.subjects_dir, subj, 'bem',
                                subj+'-5120-bem.fif')
        sol_fileOUT = os.path.join(p.subjects_dir, subj, 'bem',
                                subj+'-5120-bem-sol.fif')
        sol = mne.make_bem_solution(sol_fileIN)
        mne.write_bem_solution(sol_fileOUT, sol)
        
        cnt += 1
        
    print(f'Finished scaling for {cnt} subjects.\n')


def delete_sssfiles(p, head=True, annot=False):
    '''Remove files that can prevent SSS processing from executing or
    wouldn't otherwise be overwritten.
      Renamed from delete_chpifiles().
    '''
    def delete_files(subjects, patterns):
        n_removed = n_subj = 0
        for subj in subjects:
            n_current = n_removed
            for pattern in patterns:
                fname = os.path.join(p.work_dir, subj, p.raw_dir,
                                     subj + pattern)
                if os.path.exists(fname):
                    os.remove(fname)
                    n_removed += 1
            if n_removed > n_current:
                n_subj += 1
        return n_removed, n_subj
        
    file_patterns1 = {'_raw-counts.h5', '_raw-chpi_locs.h5',
                     '_raw.pos', '_twa_pos.fif'}
    file_patterns2 = {'_raw-annot.fif', '_raw_maxbad.txt',
                      '_erm_raw_maxbad.txt'}
    
    sidx = p.subject_indices
    if sidx is None:
        sidx = np.arange(len(p.subjects))
    subjects = np.array(p.subjects)[sidx].tolist()
    
    if head:
        n_f, n_s = delete_files(subjects, file_patterns1)
        print(f'{n_f} cHPI and head position files deleted for {n_s} of',
              f'{len(subjects)} subjects.')
    if annot:
        n_f, n_s = delete_files(subjects, file_patterns2)
        print(f'{n_f} annotation and max-bad files deleted for {n_s} of',
              f'{len(subjects)} subjects.')


def backup_file(fname):
    '''Rename file with a datetime suffix to prevent overwriting.
    Returns True if a name change was made, False otherwise.
    '''
    if not os.path.exists(fname):
        return False
    
    file_path, file_base = os.path.split(fname)
    file_base, file_ext = os.path.splitext(file_base)
    date_str = datetime.datetime.today()
    date_str = date_str.strftime('_%Y%m-%H%M')
    
    fname_new = os.path.join(file_path, file_base + date_str + file_ext)
    os.rename(fname, fname_new)
    print(f'File {fname} renamed to\n\t{os.path.split(fname_new)[1]}.')
    
    return True


def report_postfunction(report, p, subj, overwrite=False,
                        save_path=None, save_prefix='', save_postfix=''):
    '''Save the mnefun report with path and naming options.'''
    # Add events sections to the end of the report #    
    evt_dir = get_event_fnames(p, subj)[0]
    evt_dir, _ = os.path.split(evt_dir)
    jpg_file = os.path.join(evt_dir, 'event_timings.jpg')
    if os.path.exists(jpg_file):  # use mnefun events image if one exists
        import matplotlib.pyplot as plt
        fig = plt.imread(jpg_file)
        report.add_figs_to_section(fig, captions='Events: Orig | Processed',
                                   scale=None, section='Event Timings')
    else:
        ssp_fname = get_raw_fnames(p, subj, which='pca', erm=False)[0]
        raw = mne.io.read_raw_fif(ssp_fname, verbose=False)
        _report_events(report, [raw], p=p, subj=subj)

    # Prevent current report in subject directory from being overwritten #
    report_name = get_report_fnames(p, subj)[0]
    if not overwrite and os.path.exists(report_name):
        backup_file(report_name)
    
    # Save an additional copy of the report in a designated directory #
    # (The report will also be saved as usual in the subject directory.)
    file_path, file_base = os.path.split(report_name)
    file_base, file_ext = os.path.splitext(file_base)
    if save_path: # otherwise, save the file in the usual subject directory
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        file_path = save_path
    
    report_name_new = save_prefix + file_base + save_postfix + file_ext
    report_name_new = os.path.join(file_path, report_name_new)
    
    # If a report with the new name already exists, rename it #
    if os.path.exists(report_name_new):
        backup_file(report_name_new)
    
    # Finally, save the report #
    report.save(report_name_new, open_browser=False, overwrite=True)
    print('Report post-function: Saved report at\n',
          f'\t{report_name_new}.')
        
        