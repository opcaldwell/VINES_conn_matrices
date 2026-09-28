#!/usr/bin/env python3
"""
Subject Manager Helper
Utilities for managing subject lists for fMRI pipeline

Author: Mohammad Abbasi (mabbasi@stanford.edu)
"""

import os
import glob
import argparse
import sys
from pathlib import Path

# Import configuration
try:
    from config import INPUT_DIR, ROOT_DIR, SUBJECTS_FILE
except ImportError as e:
    print(f"ERROR: Could not import configuration from config.py: {e}")
    sys.exit(1)

def discover_subjects(input_dir=None):
    """
    Discover all subjects in BIDS dataset
    
    Args:
        input_dir: Input directory (default from config)
        
    Returns:
        List of subject IDs (without 'sub-' prefix)
    """
    if input_dir is None:
        input_dir = INPUT_DIR
        
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        return []
        
    # Find all subject directories
    subject_dirs = glob.glob(os.path.join(input_dir, "sub-*"))
    subjects = []
    
    for subj_dir in subject_dirs:
        if os.path.isdir(subj_dir):
            # Extract subject ID (remove 'sub-' prefix)
            subj_id = os.path.basename(subj_dir)
            if subj_id.startswith('sub-'):
                subjects.append(subj_id)
                
    subjects.sort()
    return subjects

def discover_subjects_with_fmri(input_dir=None):
    """
    Discover subjects that have fMRI data
    For simplicity, assumes all subjects have fMRI data
    
    Args:
        input_dir: Input directory (default from config)
        
    Returns:
        List of subject IDs (same as discover_subjects)
    """
    # For simplicity, assume all subjects have fMRI data
    return discover_subjects(input_dir)

def create_subjects_file(subjects, output_file=None, overwrite=False):
    """
    Create subjects text file
    
    Args:
        subjects: List of subject IDs
        output_file: Output file path (default from config)
        overwrite: Whether to overwrite existing file
        
    Returns:
        Path to created file
    """
    if output_file is None:
        output_file = SUBJECTS_FILE
        
    # Create directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check if file exists
    if os.path.exists(output_file) and not overwrite:
        response = input(f"File {output_file} exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return None
            
    # Write subjects to file
    with open(output_file, 'w') as f:
        for subject in subjects:
            f.write(f"{subject}\n")
            
    print(f"Created subjects file: {output_file}")
    print(f"Number of subjects: {len(subjects)}")
    
    return output_file

def show_subjects_info(input_dir=None):
    """
    Show information about subjects in dataset
    """
    if input_dir is None:
        input_dir = INPUT_DIR
        
    print(f"Dataset: {input_dir}")
    print("=" * 60)
    
    all_subjects = discover_subjects(input_dir)
    
    print(f"Total subjects found: {len(all_subjects)}")
        
    print(f"\nFirst 10 subjects:")
    for i, subject in enumerate(all_subjects[:10]):
        print(f"  {i+1:2d}. {subject}")
        
    if len(all_subjects) > 10:
        print(f"  ... and {len(all_subjects) - 10} more")
        
    print("=" * 60)

def add_subject_to_queue(subject, queue_file=None, position='end'):
    """
    Add a subject to the processing queue
    
    Args:
        subject: Subject ID
        queue_file: Queue file path (default from config)
        position: 'start' or 'end'
    """
    if queue_file is None:
        queue_file = SUBJECTS_FILE
        
    # Ensure subject has 'sub-' prefix
    if not subject.startswith('sub-'):
        subject = f"sub-{subject}"
        
    # Read existing subjects
    existing_subjects = []
    if os.path.exists(queue_file):
        with open(queue_file, 'r') as f:
            existing_subjects = [line.strip() for line in f if line.strip()]
            
    # Check if subject already in queue
    if subject in existing_subjects:
        print(f"Subject {subject} already in queue")
        return
        
    # Add subject
    if position == 'start':
        existing_subjects.insert(0, subject)
    else:
        existing_subjects.append(subject)
        
    # Write updated list
    with open(queue_file, 'w') as f:
        for subj in existing_subjects:
            f.write(f"{subj}\n")
            
    print(f"Added {subject} to queue at {position}")

def remove_subject_from_queue(subject, queue_file=None):
    """
    Remove a subject from the processing queue
    """
    if queue_file is None:
        queue_file = SUBJECTS_FILE
        
    # Ensure subject has 'sub-' prefix
    if not subject.startswith('sub-'):
        subject = f"sub-{subject}"
        
    if not os.path.exists(queue_file):
        print(f"Queue file not found: {queue_file}")
        return
        
    # Read existing subjects
    with open(queue_file, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]
        
    # Remove subject
    if subject in subjects:
        subjects.remove(subject)
        
        # Write updated list
        with open(queue_file, 'w') as f:
            for subj in subjects:
                f.write(f"{subj}\n")
                
        print(f"Removed {subject} from queue")
    else:
        print(f"Subject {subject} not found in queue")

def show_queue_status(queue_file=None):
    """
    Show current queue status
    """
    if queue_file is None:
        queue_file = SUBJECTS_FILE
        
    if not os.path.exists(queue_file):
        print(f"Queue file not found: {queue_file}")
        return
        
    with open(queue_file, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]
        
    print(f"Queue file: {queue_file}")
    print(f"Subjects in queue: {len(subjects)}")
    
    if subjects:
        print("\nNext 10 subjects:")
        for i, subject in enumerate(subjects[:10]):
            print(f"  {i+1:2d}. {subject}")
            
        if len(subjects) > 10:
            print(f"  ... and {len(subjects) - 10} more")

def main():
    parser = argparse.ArgumentParser(description='Subject Manager for fMRI Pipeline')
    parser.add_argument('--input-dir', default=INPUT_DIR,
                       help='Input BIDS directory')
    parser.add_argument('--output-file', default=SUBJECTS_FILE,
                       help='Output subjects file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List subjects
    list_parser = subparsers.add_parser('list', help='List all subjects')
    
    # Create subjects file
    create_parser = subparsers.add_parser('create', help='Create subjects file')
    create_parser.add_argument('--overwrite', action='store_true',
                             help='Overwrite existing file')
    
    # Add subject
    add_parser = subparsers.add_parser('add', help='Add subject to queue')
    add_parser.add_argument('subject', help='Subject ID')
    add_parser.add_argument('--position', choices=['start', 'end'], default='end',
                          help='Position to add subject')
    
    # Remove subject
    remove_parser = subparsers.add_parser('remove', help='Remove subject from queue')
    remove_parser.add_argument('subject', help='Subject ID')
    
    # Show info
    info_parser = subparsers.add_parser('info', help='Show dataset information')
    
    # Show queue status
    status_parser = subparsers.add_parser('status', help='Show queue status')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        subjects = discover_subjects(args.input_dir)
        print("All subjects:")
            
        for subject in subjects:
            print(subject)
            
    elif args.command == 'create':
        subjects = discover_subjects(args.input_dir)
        create_subjects_file(subjects, args.output_file, args.overwrite)
        
    elif args.command == 'add':
        add_subject_to_queue(args.subject, args.output_file, args.position)
        
    elif args.command == 'remove':
        remove_subject_from_queue(args.subject, args.output_file)
        
    elif args.command == 'info':
        show_subjects_info(args.input_dir)
        
    elif args.command == 'status':
        show_queue_status(args.output_file)
        
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
