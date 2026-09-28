#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fMRI Pipeline Configuration 
===========================
Contains all settings for the fMRI preprocessing pipeline using fMRIPrep.

Author: Mohammad Abbasi (mabbasi@stanford.edu)
"""

import os

# =============================================================================
# 1. GENERAL CONFIGURATION
# =============================================================================

# Dataset Information
DATASET_NAME = "ExampleDataset"

# =============================================================================
# 2. PATH CONFIGURATION
# =============================================================================

# Root directory for the project
ROOT_DIR = "/path/to/project/root"

# Input/Output Paths
INPUT_DIR = "/path/to/bids/input"   # BIDS input directory
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")       # fMRIPrep output directory
WORK_DIR = os.path.join(ROOT_DIR, "work")           # Working directory
LOG_DIR = os.path.join(ROOT_DIR, "logs")            # Log directory

# Subject Management
SUBJECTS_FILE = os.path.join(ROOT_DIR, "remaining_subjects_array.txt")

# =============================================================================
# 3. CONTAINER CONFIGURATION
# =============================================================================

# Singularity/Apptainer Image
SINGULARITY_IMAGE = "/path/to/fmriprep_latest.sif"

# FreeSurfer License
FREESURFER_LICENSE = "/path/to/license.txt"
FREESURFER_LICENSE_CONTAINER = "/opt/freesurfer/license.txt"

# BIDS Filter (optional)
BIDS_FILTER_FILE = "/path/to/bids_filter.json"

# =============================================================================
# 4. HARDWARE CONFIGURATION
# =============================================================================

# GPU Configuration
GPUS = [0, 1, 2, 3]  # Available GPU IDs
USE_GPU = True        # Set to False for CPU-only processing

# Parallel Processing
MAX_PARALLEL = 2  # Maximum parallel processes

# Resource Allocation per Job
MEMORY_MB = 50000      # Memory allocation per job (50GB per subject)
NPROCS = 12            # Number of processes per job
OMP_NTHREADS = 6       # OpenMP threads per job

# =============================================================================
# 5. FMRIPREP CONFIGURATION
# =============================================================================

# Output Spaces
OUTPUT_SPACES = ["MNI152NLin2009cAsym:res-2"]

# Processing Options
LONGITUDINAL = True          
WRITE_GRAPH = True          
SKIP_BIDS_VALIDATION = True 
RESOURCE_MONITOR = True     
STOP_ON_FIRST_CRASH = True  
FS_NO_RECONALL = False      

# =============================================================================
# 6. WORKFLOW CONFIGURATION
# =============================================================================

FORCE_REPROCESSING = False   
CLEANUP_WORK = True         

POLLING_INTERVAL = 10       
LAUNCH_DELAY = 2            

# =============================================================================
# 7. LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# 8. CONNECTIVITY MATRIX CONFIGURATION
# =============================================================================

GENERATE_CONNECTIVITY = True    

DEFAULT_ATLASES = ['aal', 'schaefer-100']  
AVAILABLE_ATLASES = [
    'aal', 'difumo-256', 'difumo-1024', 
    'harvard_oxford', 'schaefer-100', 'schaefer-400'
]

DEFAULT_CONFOUNDS = [
    "csf", "white_matter", "global_signal", 
    "trans_x", "trans_y", "trans_z", 
    "rot_x", "rot_y", "rot_z"
]

DEFAULT_CONNECTIVITY_TYPES = ['full-corr']
AVAILABLE_CONNECTIVITY_TYPES = ['full-corr', 'partial-corr', 'tangent', 'covariance']

CONNECTIVITY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "connectivity_matrices")

HANDLE_SESSIONS = True              
SESSION_PREFIX = "ses-"             
REQUIRED_BOLD_SUFFIX = "_desc-preproc_bold.nii.gz"      
REQUIRED_CONFOUNDS_SUFFIX = "_desc-confounds_timeseries.tsv"

USE_LARGEST_FILE = True             
MIN_TIME_POINTS = 50               

MIN_VOLUMES_PASS = 180             
MIN_VOLUMES_FAIL = 120             
FD_PASS_THRESHOLD = 0.3            
FD_WARN_THRESHOLD = 0.5            
HIGH_MOTION_PASS_RATIO = 20        
HIGH_MOTION_WARN_RATIO = 30        

# =============================================================================
# 9. LEGACY COMPATIBILITY
# =============================================================================

LOGDIR = LOG_DIR
WORKDIR = WORK_DIR
OUTDIR = OUTPUT_DIR
SIF = SINGULARITY_IMAGE
LICENSE_IN_CNT = FREESURFER_LICENSE_CONTAINER
ROOT = ROOT_DIR
LIST = SUBJECTS_FILE

# =============================================================================
# MAIN (For Debugging)
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("fMRI Pipeline Configuration")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max Parallel: {MAX_PARALLEL}")
    print(f"GPUs: {GPUS}")
    print("=" * 60)
