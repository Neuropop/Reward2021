import os
import socket
import argparse

import nipype.interfaces.fsl as fsl
import nipype.interfaces.spm as spm
import nipype.algorithms.misc as misc
import nipype.algorithms.modelgen as modelgen

import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio

import nipype.pipeline.engine as pe

"""
    level1 workflow

    This workflow includes:
    - smooth preprocessed data 
    - session information definition
    - model specification
    - first level estimation
    - contrast estimation
"""

"""
----------
Parameters
----------
"""
BASE_DIR_PATH = '/work'

DATA_DIR_PATH = os.path.join(BASE_DIR_PATH, 'bids')
DATA_SINK_PATH = os.path.join(BASE_DIR_PATH, 'bids/derivatives/analyzed')
WORKING_DIR_PATH = os.path.join(BASE_DIR_PATH, 'bids/derivatives/tmp')
CRASHDUMP_PATH = os.path.join(BASE_DIR_PATH, 'bids/derivatives/tmp/crashdump')

HIGHPASS_CUTOFF = 128
TR = 2.5

REGRESSORS = ['a_comp_cor_%02d' % i for i in range(6)] + \
             ['framewise_displacement'] + ['trans_%s' % i for i in ['x', 'y', 'z']] + \
             ['rot_%s' % i for i in ['x', 'y', 'z']] + \
             ['respiration']

CONDITIONS = ['ABUT', 'CIS', 'HMHA', 'ISO', 'LIM', 'MSH', 'TER', 'AIR']

SUBJECTS = ['sub-ANDCO', 'sub-BAIKE', 'sub-BARCH', 'sub-BARLO',
            'sub-BERCL', 'sub-BONBA', 'sub-BORNE', 'sub-DACPA',
            'sub-DEOME', 'sub-DIRAL', 'sub-DUCLA', 'sub-DUMME',
            'sub-DURMA', 'sub-FABSI', 'sub-GABLI', 'sub-HEIAN',
            'sub-HEMJU', 'sub-KHEHA', 'sub-LANTI', 'sub-MAKWI',
            'sub-MALSE', 'sub-MANMA', 'sub-MARLE', 'sub-MICAU',
            'sub-PACCL', 'sub-PICMI', 'sub-POUAR', 'sub-POUNI',
            'sub-RIGLI', 'sub-VINUG']

"""
--------
Workflow
--------
"""

workflow = pe.Workflow(name='level_1')
workflow.base_dir = WORKING_DIR_PATH

"""
---------
Functions
---------
"""


def _get_session_informations(events_files, confounds_regressors, stim_regressors, conditions, regressors):
    import pandas as pn
    import numpy as np
    from nipype.interfaces.base import Bunch

    output = list()
    # print(events_files, confounds_regressors, stim_regressors, conditions)
              
    for events_file, confounds_regressors_file, stim_regressors_file, idx in zip(events_files,
                                                                               confounds_regressors,
                                                                               stim_regressors,
                                                                               range(len(events_files))):
        events = pn.read_csv(events_file,
                             sep="\s+", header=0)


        # --

        confounds1 = pn.read_csv(confounds_regressors_file,
                                 sep="\s+", header=0, na_values="Na")

        confounds1.framewise_displacement.fillna(0, inplace=True)
        confounds2 = pn.read_csv(stim_regressors_file,
                                 sep="\s+", header=None, na_values="Na")
        confounds2.columns = ['respiration']

        confounds = pn.concat([confounds1, confounds2], axis=1)
        confounds = confounds.loc[:,regressors]
        
        infos = Bunch(conditions=conditions,
                      onsets=[list(events[events.trial_type == condition].onset) for condition in conditions],
                      durations=[list(events[events.trial_type == condition].duration) for condition in conditions],
                      regressors=[list(confounds[regressor]) for regressor in regressors],
                      regressor_names=regressors)
        output.insert(idx, infos)
    return output


def _plot_design_matrix(mat_file):
    import os
    # Read the design matrix
    import scipy.io as io
    import numpy as np
    spm_mat_struct = io.loadmat(mat_file, struct_as_record=False, squeeze_me=True)['SPM']
    design_matrix = spm_mat_struct.xX.X
    regressor_names = spm_mat_struct.xX.name

    # Plot the design matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    design_matrix_normalized = \
        design_matrix / np.maximum(1.e-12, np.sqrt(np.sum(design_matrix ** 2, 0)))
    plt.figure(figsize=(12, 8))

    plt.imshow(design_matrix_normalized, interpolation='nearest', aspect='auto', cmap='gray')
    plt.xlabel('Regressors')
    plt.ylabel('Scan number')
    plt.xticks(range(len(regressor_names)),
               regressor_names, rotation=60, ha='right')
    plt.tight_layout()
    plt.savefig('design_matrix.png')
    return os.path.abspath('design_matrix.png')


def _plot_regressors_correlation(mat_file):
    import os
    # Read the design matrix
    import scipy.io as io
    import numpy as np
    spm_mat_struct = io.loadmat(mat_file, struct_as_record=False, squeeze_me=True)['SPM']
    design_matrix = spm_mat_struct.xX.X
    regressor_names = spm_mat_struct.xX.name

    # Plot the regressors correlations
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    cc_matrix = np.corrcoef(design_matrix.T[:-1, :-1])
    plt.imshow(cc_matrix, interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Regressors')
    plt.ylabel('Regressors')
    plt.xticks(range(len(regressor_names[:-1])),
               regressor_names[:-1], rotation=60, ha='right')
    plt.yticks(range(len(regressor_names[:-1])),
               regressor_names[:-1], rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('cc_matrix.png')
    return os.path.abspath('cc_matrix.png')


def _specify_contrast(subject_id, conditions):

    # CONDITIONS = ['ABUT', 'CIS', 'HMHA', 'ISO', 'LIM', 'MSH', 'TER', 'AIR']
    # weights
    w = 1
    w7 = 1 / 7.

    contrast_01 = ['ABUT - AIR', 'T', conditions, [w, 0, 0, 0, 0, 0, 0, -w], [1, 1, 1, 1]]
    contrast_02 = ['CIS - AIR', 'T', conditions, [0, w, 0, 0, 0, 0, 0, -w], [1, 1, 1, 1]]
    contrast_03 = ['HMHA - AIR', 'T', conditions, [0, 0, w, 0, 0, 0, 0, -w], [1, 1, 1, 1]]
    contrast_04 = ['ISO - AIR', 'T', conditions, [0, 0, 0, w, 0, 0, 0, -w], [1, 1, 1, 1]]
    contrast_05 = ['LIM - AIR', 'T', conditions, [0, 0, 0, 0, w, 0, 0, -w], [1, 1, 1, 1]]
    contrast_06 = ['MSH - AIR', 'T', conditions, [0, 0, 0, 0, 0, w, 0, -w], [1, 1, 1, 1]]
    contrast_07 = ['TER - AIR', 'T', conditions, [0, 0, 0, 0, 0, 0, w, -w], [1, 1, 1, 1]]
    contrast_08 = ['ODORS - AIR', 'T', conditions, [w7, w7, w7, w7, w7, w7, w7, -w], [1, 1, 1, 1]]
                              
    contrasts = [contrast_01, contrast_02, contrast_03, contrast_04,
                 contrast_05, contrast_06, contrast_07, contrast_08]

    return contrasts

"""
-------------------
Input/Output Stream
-------------------
"""

# Input
subject_template = dict(func=[['subject_id', 'subject_id',
                               'task-olfactoryperception',
                               ['run-01', 'run-02', 'run-03', 'run-04'],
                               'space-MNI152NLin2009cAsym_desc-preproc_bold']],
                        mask=[['subject_id', 'subject_id',
                               'task-olfactoryperception',
                               ['run-01', 'run-02', 'run-03', 'run-04'],
                               'space-MNI152NLin2009cAsym_desc-brain_mask']],
                        events=[['subject_id', 'subject_id',
                                 'task-olfactoryperception',
                                 ['run-01', 'run-02', 'run-03', 'run-04'],
                                 'events']],
                        confounds_regressors=[['subject_id', 'subject_id',
                                               'task-olfactoryperception',
                                               ['run-01', 'run-02', 'run-03', 'run-04'],
                                               'desc-confounds_regressors']],
                        stim_regressors=[['subject_id', 'subject_id',
                                          'task-olfactoryperception',
                                          ['run-01', 'run-02', 'run-03', 'run-04'],
                                           'desc-stim_regressors']])

infosource = pe.Node(interface=niu.IdentityInterface(fields=['subject_id']),
                     name='infosource')
infosource.iterables = ('subject_id', SUBJECTS)

datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=['func',
                                                          'mask',
                                                          'events',
                                                          'confounds_regressors',
                                                          'stim_regressors']),
                     name='datasource')

datasource.inputs.base_directory = DATA_DIR_PATH
datasource.inputs.template = '*'
datasource.inputs.field_template = dict(func='derivatives/fmriprep/%s/func/%s_%s_%s_%s.nii.gz',
                                        mask='derivatives/fmriprep/%s/func/%s_%s_%s_%s.nii.gz',
                                        events='%s/func/%s_%s_%s_%s.tsv',
                                        confounds_regressors='derivatives/fmriprep/%s/func/%s_%s_%s_%s.tsv',
                                        stim_regressors='derivatives/fmriprep/%s/func/%s_%s_%s_%s.tsv')
datasource.inputs.template_args = subject_template
datasource.inputs.sort_filelist = True

workflow.connect(infosource, 'subject_id', datasource, 'subject_id')

# Output
datasink = pe.Node(interface=nio.DataSink(), name='datasink')
datasink.inputs.base_directory = DATA_SINK_PATH
workflow.connect(infosource, 'subject_id', datasink, 'container')

substitutions = list()
for subject_id in SUBJECTS:
    substitutions += [('_subject_id_%s/' % subject_id, '')]
datasink.inputs.substitutions = substitutions

"""
-----
Nodes
-----
"""

# session events and confounds
get_session_informations = pe.Node(niu.Function(input_names=['events_files', 'confounds_regressors',
                                                             'stim_regressors', 'conditions',
                                                             'regressors'],
                                                output_names=['informations'],
                                                function=_get_session_informations),
                                   name='get_session_informations')
get_session_informations.inputs.conditions = CONDITIONS
get_session_informations.inputs.regressors = REGRESSORS
workflow.connect(datasource, 'events', get_session_informations, 'events_files')
workflow.connect(datasource, 'confounds_regressors', get_session_informations, 'confounds_regressors')
workflow.connect(datasource, 'stim_regressors', get_session_informations, 'stim_regressors')

# contrasts of interest
contrasts_of_interest = pe.Node(niu.Function(inputnames=['subject_id',
                                                         'conditions'],
                                             output_names=['contrasts'],
                                             function=_specify_contrast),
                                name='contrasts_of_interest')
contrasts_of_interest.inputs.conditions = CONDITIONS
workflow.connect(infosource, 'subject_id', contrasts_of_interest, 'subject_id')

# fmri model specifications
unzip_source = pe.MapNode(misc.Gunzip(),
                          iterfield=['in_file'],
                          name='unzip_source')
workflow.connect(datasource, 'func', unzip_source, 'in_file')

smooth = pe.Node(interface=spm.Smooth(fwhm=[8, 8, 8]),
                    name='smooth')
workflow.connect(unzip_source, 'out_file', smooth, 'in_files')

modelspec = pe.Node(interface=modelgen.SpecifySPMModel(),
                    name='modelspec')
modelspec.inputs.input_units = 'secs'
modelspec.inputs.output_units = 'secs'
modelspec.inputs.time_repetition = TR
modelspec.inputs.high_pass_filter_cutoff = HIGHPASS_CUTOFF
workflow.connect(get_session_informations, 'informations', modelspec, 'subject_info')
workflow.connect(smooth, 'smoothed_files', modelspec, 'functional_runs')

# merge runs's masks
merge_masks = pe.Node(interface=fsl.Merge(dimension='t'),
                     name='merge_masks')
workflow.connect(datasource, 'mask', merge_masks, 'in_files')

# create mean runs mask
mean_mask = pe.Node(interface=fsl.MeanImage(args='-bin', output_type='NIFTI'),
                    name='mean_mask')
workflow.connect(merge_masks, 'merged_file', mean_mask, 'in_file')

# generate a first level SPM.mat
level1design = pe.Node(interface=spm.Level1Design(),
                       name='level1design')
level1design.inputs.timing_units = modelspec.inputs.output_units
level1design.inputs.interscan_interval = modelspec.inputs.time_repetition
level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
level1design.inputs.model_serial_correlations = 'AR(1)'
workflow.connect(modelspec, 'session_info', level1design, 'session_info')
workflow.connect(mean_mask, 'out_file', level1design, 'mask_image')

# plot the design matrix
plot_design_matrix = pe.Node(niu.Function(input_names=['mat_file'],
                                          output_names=['out_file'],
                                          function=_plot_design_matrix),
                             name='plot_design_matrix')
workflow.connect(level1design, 'spm_mat_file', plot_design_matrix, 'mat_file')
plot_design_matrix_rename = pe.Node(
    niu.Rename(format_string='%(subject_id)s_task-olfactoryperception_bold_dmatrix.png'),
    name='plot_design_matrix_rename')
workflow.connect(plot_design_matrix, 'out_file', plot_design_matrix_rename, 'in_file')
workflow.connect(infosource, 'subject_id', plot_design_matrix_rename, 'subject_id')
workflow.connect(plot_design_matrix_rename, 'out_file', datasink, 'report.@dmatrix')

# plot the correlation between regressors
plot_regressors_correlation = pe.Node(niu.Function(input_names=['mat_file'],
                                                   output_names=['out_file'],
                                                   function=_plot_regressors_correlation),
                                      name='plot_regressors_correlation')
workflow.connect(level1design, 'spm_mat_file', plot_regressors_correlation, 'mat_file')
plot_regressors_correlation_rename = pe.Node(
    niu.Rename(format_string='%(subject_id)s_task_olfactory_perception_bold_ccmatrix.png'),
    name='plot_regressors_correlation_rename')
workflow.connect(plot_regressors_correlation, 'out_file', plot_regressors_correlation_rename, 'in_file')
workflow.connect(infosource, 'subject_id', plot_regressors_correlation_rename, 'subject_id')
workflow.connect(plot_regressors_correlation_rename, 'out_file', datasink, 'report.@ccmatrix')

# determine the parameters of the model
level1estimate = pe.Node(interface=spm.EstimateModel(),
                         name='level1estimate')
level1estimate.inputs.estimation_method = {'Classical': 1}
workflow.connect(level1design, 'spm_mat_file', level1estimate, 'spm_mat_file')

# estimate the first level contrasts
contrastestimate = pe.Node(interface=spm.EstimateContrast(),
                           name='contrastestimate')
contrastestimate.overwrite = True
contrastestimate.config = {'execution': {'remove_unnecessary_outputs': False}}
contrastestimate.use_derivs = False
workflow.connect(contrasts_of_interest, 'contrasts', contrastestimate, 'contrasts')
workflow.connect(level1estimate, 'spm_mat_file', contrastestimate, 'spm_mat_file')
workflow.connect(level1estimate, 'beta_images', contrastestimate, 'beta_images')
workflow.connect(level1estimate, 'residual_image', contrastestimate, 'residual_image')
workflow.connect(level1estimate, 'spm_mat_file', datasink, 'level1.estimate.@spm_mat_file')
workflow.connect(level1estimate, 'beta_images', datasink, 'level1.estimate.@beta_images')
workflow.connect(level1estimate, 'residual_image', datasink, 'level1.estimate.@residual_image')
workflow.connect(contrastestimate, 'con_images', datasink, 'level1.contrasts.@con_images')

"""
--------------------
Execute the Workflow
--------------------
"""

if __name__ == '__main__':
    workflow.write_graph(graph2use='colored', format='png', simple_form=False)
    workflow.run(plugin='MultiProc', plugin_args={'n_procs': 8})
