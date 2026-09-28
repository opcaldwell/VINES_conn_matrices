#!/usr/bin/env python3
"""
Standalone Connectivity Matrix Runner
Run connectivity matrix generation independently or after fMRIPrep completion

Author: Mohammad Hassan Abbasi
Date: September 15, 2025
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Import configuration and connectivity processor
try:
    from config import *
    from connectivity_matrix import ConnectivityProcessor
    from subject_manager import discover_subjects
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    sys.exit(1)

def setup_logging():
    """Setup logging for standalone connectivity processing"""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    log_file = os.path.join(LOG_DIR, f'run_connectivity_{timestamp}.log')
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger('run_connectivity')

def find_completed_subjects():
    """Find subjects that have completed fMRIPrep processing"""
    completed_subjects = []
    
    if not os.path.exists(OUTPUT_DIR):
        return completed_subjects
        
    # Look for subjects with fMRIPrep outputs
    for item in os.listdir(OUTPUT_DIR):
        if item.startswith('sub-'):
            subject_dir = os.path.join(OUTPUT_DIR, item)
            func_dir = os.path.join(subject_dir, 'func')
            
            # Check if preprocessed files exist
            if os.path.exists(func_dir):
                import glob
                preproc_files = glob.glob(os.path.join(func_dir, '*_desc-preproc_bold.nii.gz'))
                confounds_files = glob.glob(os.path.join(func_dir, '*_desc-confounds_timeseries.tsv'))
                
                if preproc_files and confounds_files:
                    subject_id = item.replace('sub-', '')
                    completed_subjects.append(subject_id)
                    
    return sorted(completed_subjects)

def check_connectivity_completed(subject):
    """Check if connectivity matrices are already generated for a subject"""
    subject_conn_dir = os.path.join(CONNECTIVITY_OUTPUT_DIR, f"sub-{subject}")
    
    if not os.path.exists(subject_conn_dir):
        return False
        
    # Check for connectivity matrix files
    import glob
    matrix_files = glob.glob(os.path.join(subject_conn_dir, "*_connectivity.csv"))
    return len(matrix_files) > 0

def main():
    parser = argparse.ArgumentParser(description='Generate connectivity matrices from fMRIPrep outputs')
    parser.add_argument('--subjects', nargs='+', 
                       help='Specific subject IDs to process (without sub- prefix)')
    parser.add_argument('--subjects-file', 
                       help='File containing subject IDs (one per line)')
    parser.add_argument('--all-completed', action='store_true',
                       help='Process all subjects with completed fMRIPrep outputs')
    parser.add_argument('--atlases', nargs='+', 
                       default=DEFAULT_ATLASES,
                       choices=AVAILABLE_ATLASES,
                       help='Atlas names for parcellation')
    parser.add_argument('--confounds', nargs='+', 
                       default=DEFAULT_CONFOUNDS,
                       help='Confound regressors to remove')
    parser.add_argument('--corr-kinds', nargs='+', 
                       default=DEFAULT_CONNECTIVITY_TYPES,
                       choices=AVAILABLE_CONNECTIVITY_TYPES,
                       help='Types of connectivity matrices')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if connectivity matrices exist')
    parser.add_argument('--check', action='store_true',
                       help='Check which subjects have completed fMRIPrep')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    if args.check:
        # Just show status
        completed = find_completed_subjects()
        logger.info(f"Subjects with completed fMRIPrep: {len(completed)}")
        for subject in completed[:10]:  # Show first 10
            conn_status = "✅" if check_connectivity_completed(subject) else "❌"
            logger.info(f"  {subject} {conn_status}")
        if len(completed) > 10:
            logger.info(f"  ... and {len(completed) - 10} more")
        return
    
    # Get subject list
    subjects = []
    if args.subjects:
        subjects = [s.replace('sub-', '') for s in args.subjects]
    elif args.subjects_file:
        with open(args.subjects_file, 'r') as f:
            subjects = [line.strip().replace('sub-', '') for line in f if line.strip()]
    elif args.all_completed:
        subjects = find_completed_subjects()
    else:
        logger.error("Must specify subjects using --subjects, --subjects-file, or --all-completed")
        sys.exit(1)
    
    if not subjects:
        logger.error("No subjects to process")
        sys.exit(1)
        
    logger.info(f"Processing connectivity for {len(subjects)} subjects")
    logger.info(f"Atlases: {args.atlases}")
    logger.info(f"Connectivity types: {args.corr_kinds}")
    
    # Initialize processor
    processor = ConnectivityProcessor()
    
    # Fetch atlases
    atlases = processor.fetch_atlases(args.atlases)
    if not atlases:
        logger.error("Could not fetch any atlases")
        sys.exit(1)
    
    # Process subjects
    successful = 0
    failed = 0
    skipped = 0
    
    for i, subject in enumerate(subjects, 1):
        logger.info(f"Processing subject {subject} ({i}/{len(subjects)})")
        
        # Check if already completed (unless force is specified)
        if not args.force and check_connectivity_completed(subject):
            logger.info(f"Connectivity matrices already exist for {subject}, skipping")
            skipped += 1
            continue
            
        try:
            success = processor.process_subject_connectivity(
                subject, atlases, args.confounds, args.corr_kinds
            )
            
            if success:
                successful += 1
                logger.info(f"✅ Completed connectivity for {subject}")
            else:
                failed += 1
                logger.error(f"❌ Failed connectivity for {subject}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {subject}: {e}")
            failed += 1
    
    # Summary
    logger.info("=" * 50)
    logger.info("Connectivity Processing Summary")
    logger.info("=" * 50)
    logger.info(f"Total subjects: {len(subjects)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Output directory: {CONNECTIVITY_OUTPUT_DIR}")
    logger.info("=" * 50)

if __name__ == '__main__':
    main()


