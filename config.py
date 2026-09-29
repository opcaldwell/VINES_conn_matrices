#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fMRI Pipeline Configuration 
===========================
Contains all settings for the fMRI preprocessing pipeline using fMRIPrep.
                                        
Author: Mohammad Abbasi (mabbasi@stanford.edu)
Adapted by: Owen Caldwell (ocaldwe1@uvm.edu)
"""

import os

#
=============================================================================
# 1. GENERAL CONFIGURATION
#
=============================================================================

# Dataset Information
DATASET_NAME = "VINES"

#
=============================================================================
# 2. PATH CONFIGURATION
#
=============================================================================

# Input/Output Paths--be sure to update output_dir with sub ####
OUTPUT_DIR = "/gpfs1/pi/abrieant/ImagingData/sub-3380"       # fMRIPrep source
directory
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")            # Log directory

#
=============================================================================
# 3. LOGGING CONFIGURATION
#
=============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

#
=============================================================================
# 4. CONN. MATRIX CONFIGURATION
#
=============================================================================

# Just Shen 368 for the time being
ATLAS_NAME = "Shen-368"
# Move atlas file onto VACC in scripts subdir
ATLAS_FILE =
"/gpfs1/pi/abrieant/ImagingData/scripts/Shen_1mm_368_parcellation.nii.gz"
ATLAS_NODES = 368

DEFAULT_CONFOUNDS = ["csf", "white_matter", "global_signal",
                     "trans_x", "trans_y", "trans_z",
                     "rot_x", "rot_y", "rot_z"
                    ]

DEFAULT_CONNECTIVITY_TYPES = ['full-corr']
AVAILABLE_CONNECTIVITY_TYPES = ['full-corr', 'partial-corr', 'tangent',
                                'covariance']

CONNECTIVITY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "connectivity_matrices")

SESSION_PREFIX = "ses-"
REQUIRED_BOLD_SUFFIX = "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
REQUIRED_CONFOUNDS_SUFFIX = "_desc-confounds_timeseries.tsv"

MIN_TIME_POINTS = 50
MIN_VOLUMES_PASS = 180
MIN_VOLUMES_FAIL = 120
FD_PASS_THRESHOLD = 0.3
FD_WARN_THRESHOLD = 0.5
HIGH_MOTION_PASS_RATIO = 20
HIGH_MOTION_WARN_RATIO = 30
