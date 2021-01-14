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
    This workflow includes:
    - region of interest stats extraction:
        - min
        - max
        - mean
        - entropy
    - for each ROI:
        - left
        - right
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

TR = 2.5

CONDITIONS = ['ABUT', 'CIS', 'HMHA', 'ISO', 'LIM', 'MSH', 'TER', 'AIR']

SUBJECTS = ['sub-ANDCO', 'sub-BAIKE', 'sub-BARCH', 'sub-BARLO',
            'sub-BERCL', 'sub-BONBA', 'sub-BORNE', 'sub-DACPA',
            'sub-DEOME', 'sub-DIRAL', 'sub-DUCLA', 'sub-DUMME',
            'sub-DURMA', 'sub-FABSI', 'sub-GABLI', 'sub-HEIAN',
            'sub-HEMJU', 'sub-KHEHA', 'sub-LANTI', 'sub-MAKWI',
            'sub-MALSE', 'sub-MANMA', 'sub-MARLE', 'sub-MICAU',
            'sub-PACCL', 'sub-PICMI', 'sub-POUAR', 'sub-POUNI',
            'sub-RIGLI', 'sub-VINUG']

CONTRASTS = ['con_%04d' % i for i in range(1, 8)]

ROIS = ['anteriorpiriformleft', 'anteriorpiriformright',
        'posteriorpiriformleft', 'posteriorpiriformright',
        'leftolfactorytubercle', 'rightolfactorytubercle']

"""
--------
Workflow
--------
"""

workflow = pe.Workflow(name='rois_stats')
workflow.base_dir = WORKING_DIR_PATH

"""
---------
Functions
---------
"""

# CONDITIONS = ['ABUT', 'CIS', 'HMHA', 'ISO', 'LIM', 'MSH', 'TER', 'AIR']

def _collect_stats(subject_id,
                   contrast_id,
                   stats,
                   rois):
    CONTRASTS_NAMES = {'con_0001': 'ABUT - AIR', 
                       'con_0002': 'CIS - AIR',
                       'con_0003': 'HMHA - AIR', 
                       'con_0004': 'ISO - AIR',
                       'con_0005': 'LIM - AIR', 
                       'con_0006': 'MSH - AIR',
                       'con_0007': 'TER  - AIR'
    }
    import os
    import pandas as pn
    out = list()
    contrast_name = CONTRASTS_NAMES[contrast_id]
    for values, roi in zip(stats, rois):
        out.append([subject_id, contrast_id, contrast_name, roi] + values)

    df = pn.DataFrame(out, columns=['subject_id',
                                    'contrast_id', 'contrast_name', 'roi',
                                    'min', 'max', 'mean', 'entropy',
                                    'nb_voxels', 'volume'])
    new_filename = os.path.abspath('%s_stats.csv' % subject_id)
    df.to_csv(new_filename)
    return new_filename


def _collect_subjects(in_files):
    import os
    import pandas as pn
    out = list()
    for in_file in in_files:
        out.append(pn.read_csv(in_file, header=0, index_col=0))
    df = pn.concat(out, axis=0)
    new_filename = os.path.abspath('brain_stats.csv')
    df.to_csv(new_filename)
    return new_filename


"""
-------------------
Input/Output Stream
-------------------
"""

subject_template = dict(copes=[['subject_id', 'contrast_id']],
                        rois=[['subject_id', 'subject_id',
                               'space-MNI152NLin2009cAsym_desc',
                               ROIS]],
                        )

infosource = pe.Node(interface=niu.IdentityInterface(fields=['subject_id', 'contrast_id']),
                     name='infosource')
infosource.iterables = [('subject_id', SUBJECTS), ('contrast_id', CONTRASTS)]

datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id', 'contrast_id'],
                                               outfields=['copes', 'rois']),
                     name='datasource')

datasource.inputs.base_directory = DATA_DIR_PATH
datasource.inputs.template = '*'
datasource.inputs.field_template = dict(copes='derivatives/analyzed/%s/level1/contrasts/%s.nii',
                                        rois='derivatives/rois/%s/anat/%s_%s-%s_mask.nii.gz')
datasource.inputs.template_args = subject_template
datasource.inputs.sort_filelist = True

workflow.connect(infosource, 'subject_id', datasource, 'subject_id')
workflow.connect(infosource, 'contrast_id', datasource, 'contrast_id')

# Output
datasink = pe.Node(interface=nio.DataSink(), name='datasink')
datasink.inputs.base_directory = DATA_SINK_PATH
datasink.inputs.container = ''

substitutions = list()
for subject_id in SUBJECTS:
    substitutions += [('_subject_id_%s/' % subject_id, '')]
datasink.inputs.substitutions = substitutions

"""
-----
Nodes
-----
"""
# extract stats from each ROI
# ---------------------

ROI_stats = pe.MapNode(fsl.ImageStats(op_string='-k %s -R -m -e -v'),
                          iterfield=['mask_file'],
                          name='ROI_stats')
workflow.connect(datasource, 'copes', ROI_stats, 'in_file')
workflow.connect(datasource, 'rois', ROI_stats, 'mask_file')


# collect subject stats
# ---------------------
collect_stats = pe.Node(niu.Function(input_names=['subject_id',
                                                  'contrast_id',
                                                  'stats',
                                                  'rois'],
                                     output_names=['out_file'],
                                     function=_collect_stats),
                        name='collect_stats')
collect_stats.inputs.rois = ROIS
workflow.connect(infosource, 'subject_id', collect_stats, 'subject_id')
workflow.connect(infosource, 'contrast_id', collect_stats, 'contrast_id')
workflow.connect(ROI_stats, 'out_stat', collect_stats, 'stats')


collect = pe.JoinNode(interface=niu.Function(input_names=['in_files'],
                                             output_names=['out_file'],
                                             function=_collect_subjects),
                      joinsource="infosource",
                      joinfield=['in_files'],
                      unique=True,
                      name="collect")
workflow.connect(collect_stats, 'out_file', collect, 'in_files')
workflow.connect(collect, 'out_file', datasink, '@stats')

"""
--------------------
Execute the Workflow
--------------------
"""

if __name__ == '__main__':
    workflow.write_graph(graph2use='colored', format='png', simple_form=False)
    workflow.run(plugin='MultiProc', plugin_args={'n_procs': 8})
