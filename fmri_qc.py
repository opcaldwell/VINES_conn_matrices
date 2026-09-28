#!/usr/bin/env python3
"""
fMRI Quality Control Script
Author: Mohammad Abbasi (mabbasi@stanford.edu)
Created: September 2025

This script generates QC reports for fMRI preprocessing including:
- Motion parameters (Mean FD, Max FD, High-motion ratio)
- Volume counts and data quality metrics
- HTML reports with detailed visualizations and connectivity matrices
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime
from pathlib import Path
import re
from urllib.parse import quote
import glob

# Import configuration
try:
    from config import *
except ImportError as e:
    print(f"ERROR: Could not import configuration from config.py: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class FMRIQualityControl:
    def __init__(self):
        self.dataset_name = DATASET_NAME
        self.output_dir = OUTPUT_DIR
        self.connectivity_dir = CONNECTIVITY_OUTPUT_DIR
        self.qc_data = []
        
        # Ensure QC output directories exist
        self.qc_output_dir = os.path.join(OUTPUT_DIR, "qc_reports")
        self.subjects_dir = os.path.join(self.qc_output_dir, "subjects")
        os.makedirs(self.qc_output_dir, exist_ok=True)
        os.makedirs(self.subjects_dir, exist_ok=True)
        
        logger.info(f"QC output directory: {self.qc_output_dir}")

    def find_subjects(self):
        """Find all subjects with fMRIPrep output"""
        subjects = []
        if os.path.exists(self.output_dir):
            for item in os.listdir(self.output_dir):
                if item.startswith('sub-') and os.path.isdir(os.path.join(self.output_dir, item)):
                    subject_id = item.replace('sub-', '')
                    subjects.append(subject_id)
        subjects.sort()
        logger.info(f"Found {len(subjects)} subjects: {subjects}")
        return subjects

    def find_sessions_for_subject(self, subject):
        """Find all sessions for a subject"""
        subject_dir = os.path.join(self.output_dir, f"sub-{subject}")
        sessions = []
        
        if os.path.exists(subject_dir):
            for item in os.listdir(subject_dir):
                if item.startswith('ses-') and os.path.isdir(os.path.join(subject_dir, item)):
                    sessions.append(item)
        
        if not sessions:
            # Check for non-session based structure
            func_dir = os.path.join(subject_dir, "func")
            if os.path.exists(func_dir):
                sessions = ["nosession"]  # Use "nosession" as session name for non-session data
        
        sessions.sort()
        return sessions

    def find_best_confounds_file(self, subject, session):
        """Find confounds file with most volumes (>50) for a subject/session"""
        if session == "nosession":
            # Non-session based
            func_dir = os.path.join(self.output_dir, f"sub-{subject}", "func")
        else:
            # Session based
            func_dir = os.path.join(self.output_dir, f"sub-{subject}", session, "func")
        
        if not os.path.exists(func_dir):
            return None, None, None
        
        # Find confounds files
        confounds_pattern = os.path.join(func_dir, "*_desc-confounds_timeseries.tsv")
        confounds_files = glob.glob(confounds_pattern)
        
        best_file = None
        best_volumes = 0
        best_info = {}
        
        for confounds_file in confounds_files:
            try:
                df = pd.read_csv(confounds_file, delimiter='\t')
                volumes = len(df)
                
                if volumes > 50 and volumes > best_volumes:
                    # Extract run info from filename
                    filename = os.path.basename(confounds_file)
                    
                    # Parse task and direction
                    task_match = re.search(r'task-(\w+)', filename)
                    dir_match = re.search(r'dir-(\w+)', filename)
                    run_match = re.search(r'run-(\w+)', filename)
                    
                    task = task_match.group(1) if task_match else 'unknown'
                    direction = dir_match.group(1) if dir_match else 'unknown'
                    run = run_match.group(1) if run_match else 'unknown'
                    
                    best_file = confounds_file
                    best_volumes = volumes
                    best_info = {
                        'task': task,
                        'direction': direction,
                        'run': run,
                        'volumes': volumes,
                        'confounds_df': df
                    }
                    
            except Exception as e:
                logger.warning(f"Could not read {confounds_file}: {e}")
                continue
        
        if best_file:
            logger.info(f"Best confounds for {subject}/{session}: {os.path.basename(best_file)} ({best_volumes} volumes)")
            return best_file, best_info, best_volumes
        else:
            logger.warning(f"No suitable confounds file found for {subject}/{session}")
            return None, None, 0

    def calculate_motion_metrics(self, confounds_df):
        """Calculate motion metrics from confounds dataframe with proper volume count"""
        try:
            # Check if framewise_displacement column exists
            if 'framewise_displacement' not in confounds_df.columns:
                logger.warning("framewise_displacement column not found")
                return None, None, None, None
            
            # Convert to numeric and drop NaN values (first row is usually NaN)
            fd_values = pd.to_numeric(confounds_df['framewise_displacement'], errors='coerce').dropna()
            
            if len(fd_values) == 0:
                logger.warning("No valid FD values found")
                return None, None, None, None
            
            # Calculate metrics
            volumes = len(fd_values)  # Use FD length for accurate volume count
            mean_fd = np.mean(fd_values)
            max_fd = np.max(fd_values)
            high_motion_ratio = np.sum(fd_values > 0.5) / len(fd_values) * 100
            
            return mean_fd, max_fd, high_motion_ratio, volumes
            
        except Exception as e:
            logger.error(f"Error calculating motion metrics: {e}")
            return None, None, None, None

    def process_subject(self, subject):
        """Process QC for a single subject"""
        logger.info(f"Processing QC for subject {subject}")
        
        sessions = self.find_sessions_for_subject(subject)
        
        if not sessions:
            logger.warning(f"No sessions found for subject {subject}")
            return
        
        for session in sessions:
            session_display = session.replace('ses-', '') if session.startswith('ses-') else session
            
            # Find best confounds file
            confounds_file, confounds_info, volumes = self.find_best_confounds_file(subject, session)
            
            if not confounds_file:
                # Add entry for missing data
                self.qc_data.append({
                    'Subject': subject,
                    'Session': session_display,
                    'Task': 'N/A',
                    'Direction': 'N/A',
                    'Volumes': 0,
                    'Mean_FD': 'N/A',
                    'Max_FD': 'N/A',
                    'High_Motion_Ratio': 'N/A',
                    'Status': 'FAIL',
                    'Notes': 'No suitable confounds file found'
                })
                continue
            
            # Calculate motion metrics with accurate volume count
            mean_fd, max_fd, high_motion_ratio, accurate_volumes = self.calculate_motion_metrics(confounds_info['confounds_df'])
            
            # Use accurate volume count if available, otherwise fall back to original
            final_volumes = accurate_volumes if accurate_volumes is not None else volumes
            
            # Determine status using improved thresholds with clear notes
            status = 'PASS'
            notes = []
            
            if final_volumes < MIN_VOLUMES_FAIL:
                status = 'FAIL'
                notes.append('Too short scan, not analyzable')
            elif mean_fd is None:
                status = 'FAIL'
                notes.append('Motion metrics unavailable, not analyzable')
            elif mean_fd > FD_WARN_THRESHOLD:
                status = 'FAIL'
                notes.append(f'Excessive motion (meanFD={mean_fd:.2f}mm), unusable')
            elif final_volumes < MIN_VOLUMES_PASS:
                status = 'REVIEW'
                notes.append(f'Short scan ({final_volumes} volumes), usable but flagged for QC')
            elif mean_fd > FD_PASS_THRESHOLD:
                status = 'REVIEW'
                notes.append(f'High motion (meanFD={mean_fd:.2f}mm), keep for biomarker analysis, flagged for QC')
            elif high_motion_ratio > HIGH_MOTION_WARN_RATIO:
                status = 'REVIEW'
                notes.append(f'High motion ratio ({high_motion_ratio:.1f}% > {HIGH_MOTION_WARN_RATIO}%), usable but flagged')
            elif high_motion_ratio > HIGH_MOTION_PASS_RATIO:
                status = 'REVIEW'
                notes.append(f'Moderate motion ratio ({high_motion_ratio:.1f}% > {HIGH_MOTION_PASS_RATIO}%), usable but flagged')
            
            if not notes:
                notes.append('Good quality data')
            
            # Add to QC data
            self.qc_data.append({
                'Subject': subject,
                'Session': session_display,
                'Task': confounds_info['task'],
                'Direction': confounds_info['direction'],
                'Volumes': final_volumes,
                'Mean_FD': mean_fd if mean_fd is not None else 'N/A',
                'Max_FD': max_fd if max_fd is not None else 'N/A',
                'High_Motion_Ratio': high_motion_ratio if high_motion_ratio is not None else 'N/A',
                'Status': status,
                'Notes': '; '.join(notes)
            })
            
            # Generate individual subject report
            self.generate_subject_report(subject, session_display, confounds_info, mean_fd, max_fd, high_motion_ratio, status)

    def generate_subject_report(self, subject, session, confounds_info, mean_fd, max_fd, high_motion_ratio, status):
        """Generate detailed HTML report for individual subject"""
        report_filename = f"sub-{subject}_ses-{session}_qc.html"
        report_path = os.path.join(self.subjects_dir, report_filename)
        
        # Check for connectivity matrices (filter by session)
        connectivity_subj_dir = os.path.join(self.connectivity_dir, f"sub-{subject}")
        connectivity_files = []
        visualization_files = []
        
        if os.path.exists(connectivity_subj_dir):
            all_connectivity_files = glob.glob(os.path.join(connectivity_subj_dir, "*_connectivity.csv"))
            all_visualization_files = glob.glob(os.path.join(connectivity_subj_dir, "*_visualization.png"))
            
            # Filter files to only include those from this specific session
            if session != "nosession":
                session_pattern = f"_ses-{session}_"
            else:
                session_pattern = "_nosession_"
            
            connectivity_files = [f for f in all_connectivity_files if session_pattern in os.path.basename(f)]
            visualization_files = [f for f in all_visualization_files if session_pattern in os.path.basename(f)]
        
        # Check for fMRIPrep HTML report
        fmriprep_html = os.path.join(self.output_dir, f"sub-{subject}.html")
        fmriprep_html_exists = os.path.exists(fmriprep_html)
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>fMRI QC Report - {subject} Session {session}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f7fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .summary-card h3 {{ margin: 0 0 10px 0; color: #2c3e50; }}
        .summary-card .value {{ font-size: 1.5em; font-weight: bold; color: #3498db; }}
        .status-pass {{ color: #27ae60; font-weight: bold; }}
        .status-warning {{ color: #f39c12; font-weight: bold; }}
        .status-fail {{ color: #e74c3c; font-weight: bold; }}
        .files-section {{ margin: 20px 0; }}
        .file-list {{ list-style: none; padding: 0; }}
        .file-list li {{ background: #ecf0f1; margin: 5px 0; padding: 10px; border-radius: 5px; }}
        .file-list a {{ color: #2980b9; text-decoration: none; }}
        .file-list a:hover {{ text-decoration: underline; }}
        .connectivity-viz {{ text-align: center; margin: 20px 0; }}
        .connectivity-viz img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; cursor: pointer; }}
        .connectivity-viz img:hover {{ opacity: 0.8; }}
        
        /* Modal styles for image popup */
        .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.9); }}
        .modal-content {{ margin: auto; display: block; width: 80%; max-width: 1200px; }}
        .close {{ position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }}
        .close:hover, .close:focus {{ color: #bbb; text-decoration: none; cursor: pointer; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>fMRI Quality Control Report</h1>
        <p style="text-align: center; color: #7f8c8d;">Subject: {subject} | Session: {session}</p>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Data Quality</h3>
                <div class="value">
                    <span class="status-{status.lower()}">{status}</span>
                </div>
            </div>
            <div class="summary-card">
                <h3>Task Information</h3>
                <div class="value">{confounds_info.get('task', 'N/A') if confounds_info else 'N/A'}</div>
                <div>Direction: {confounds_info.get('direction', 'N/A') if confounds_info else 'N/A'}</div>
            </div>
            <div class="summary-card">
                <h3>Volume Count</h3>
                <div class="value">{confounds_info.get('volumes', 0) if confounds_info else 0}</div>
            </div>
            <div class="summary-card">
                <h3>Motion Metrics</h3>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <div>
                        <div style="font-size: 1.0em; font-weight: bold; color: #3498db;">Mean FD</div>
                        <div style="font-size: 1.1em; color: #2c3e50;">{mean_fd:.3f} mm</div>
                    </div>
                    <div>
                        <div style="font-size: 1.0em; font-weight: bold; color: #e74c3c;">Max FD</div>
                        <div style="font-size: 1.1em; color: #2c3e50;">{max_fd:.3f} mm</div>
                    </div>
                </div>
            </div>
            <div class="summary-card">
                <h3>High Motion Ratio</h3>
                <div class="value">{high_motion_ratio:.1f}%</div>
                <div style="font-size: 0.8em; color: #7f8c8d;">FD > 0.5 mm</div>
            </div>
        </div>
        
        <h2>fMRIPrep Report</h2>
        <div class="files-section">
"""

        if fmriprep_html_exists:
            fmriprep_rel_path = os.path.relpath(fmriprep_html, self.subjects_dir)
            html_content += f"""
            <p><a href="{fmriprep_rel_path}" target="_blank">View fMRIPrep HTML Report</a></p>
"""
        else:
            html_content += f"""
            <p style="color: #e74c3c;">fMRIPrep HTML report not found</p>
"""

        # Add connectivity section
        html_content += f"""
        <h2>Connectivity Analysis</h2>
        <div class="files-section">
"""

        if connectivity_files:
            html_content += f"""
            <p>Found {len(connectivity_files)} connectivity matrices:</p>
            <ul class="file-list">
"""
            for conn_file in connectivity_files:
                conn_rel_path = os.path.relpath(conn_file, self.subjects_dir)
                filename = os.path.basename(conn_file)
                html_content += f"""
                <li><a href="{conn_rel_path}" target="_blank">{filename}</a></li>
"""
            html_content += "</ul>"
        else:
            html_content += f"""
            <p style="color: #e74c3c;">No connectivity matrices found</p>
"""

        # Add visualizations for this specific session only
        if visualization_files:
            html_content += f"""
            <h3>Connectivity Visualizations (Session {session})</h3>
"""
            for viz_file in visualization_files:
                viz_rel_path = os.path.relpath(viz_file, self.subjects_dir)
                viz_name = os.path.basename(viz_file).replace('_visualization.png', '')
                # Clean up the display name - remove subject prefix and session info
                display_name = viz_name.replace(f'sub-{subject}_{session}_', '').replace('_', ' ').title()
                html_content += f"""
            <div class="connectivity-viz">
                <h4 style="margin: 15px 0; color: #34495e;">{display_name}</h4>
                <img src="{viz_rel_path}" alt="Connectivity Matrix Visualization" onclick="openModal(this.src, this.alt)">
            </div>
"""

        html_content += f"""
        </div>
        
        <h2>Data Files</h2>
        <div class="files-section">
            <h3>Time Series Data</h3>
"""

        # Add time series files for this specific session only
        if os.path.exists(connectivity_subj_dir):
            all_timeseries_files = glob.glob(os.path.join(connectivity_subj_dir, "*_timeseries.csv"))
            
            # Filter timeseries files for this session
            if session != "nosession":
                session_pattern = f"_ses-{session}_"
            else:
                session_pattern = "_nosession_"
            timeseries_files = [f for f in all_timeseries_files if session_pattern in os.path.basename(f)]
            
            if timeseries_files:
                html_content += f"<h4 style='margin-top: 15px; color: #2c3e50;'>Time Series Files (Session {session})</h4>"
                html_content += "<ul class='file-list'>"
                for ts_file in timeseries_files:
                    ts_rel_path = os.path.relpath(ts_file, self.subjects_dir)
                    filename = os.path.basename(ts_file)
                    # Clean up display name - remove subject and session prefix
                    display_name = filename.replace(f'sub-{subject}_{session}_', '').replace('_', ' ')
                    html_content += f"""
                    <li><a href="{ts_rel_path}" target="_blank">{display_name}</a></li>
"""
                html_content += "</ul>"
            else:
                html_content += "<p style='color: #e74c3c;'>No time series files found for this session</p>"

        html_content += f"""
        </div>
        
        <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
            <div style="margin-bottom: 15px;">
                <a href="../fmri_qc_summary.html" style="display: inline-block; padding: 10px 20px; background-color: #3498db; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">← Back to Summary</a>
            </div>
            <p style="color: #7f8c8d;">Designed & Developed by <a href="https://stai.stanford.edu/" style="color: #3498db;">Stanford Translational AI (STAI)</a></p>
        </div>
    </div>

    <!-- Modal for image popup -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        function openModal(src, alt) {{
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.style.display = 'block';
            modalImg.src = src;
            modalImg.alt = alt;
        }}
        
        function closeModal() {{
            document.getElementById('imageModal').style.display = 'none';
        }}
        
        // Close modal when clicking outside the image
        window.onclick = function(event) {{
            const modal = document.getElementById('imageModal');
            if (event.target == modal) {{
                modal.style.display = 'none';
            }}
        }}
    </script>
</body>
</html>
"""

        # Write HTML file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated subject report: {report_path}")

    def generate_summary_report(self):
        """Generate main HTML summary report"""
        if not self.qc_data:
            logger.warning("No QC data to generate summary report")
            return
        
        df = pd.DataFrame(self.qc_data)
        output_path = os.path.join(self.qc_output_dir, "fmri_qc_summary.html")
        
        # Calculate statistics
        total_subjects = len(df['Subject'].unique())
        total_sessions = len(df)
        pass_count = len(df[df['Status'] == 'PASS'])
        review_count = len(df[df['Status'] == 'REVIEW'])
        fail_count = len(df[df['Status'] == 'FAIL'])
        
        # Motion statistics for passed/warning subjects
        motion_df = df[df['Mean_FD'] != 'N/A'].copy()
        if not motion_df.empty:
            motion_df['Mean_FD'] = pd.to_numeric(motion_df['Mean_FD'])
            avg_mean_fd = motion_df['Mean_FD'].mean()
            avg_volumes = df[df['Volumes'] > 0]['Volumes'].mean()
        else:
            avg_mean_fd = 0
            avg_volumes = 0
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.dataset_name} fMRI Quality Control Summary</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f7fa; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border-left: 4px solid #3498db; }}
        .stat-card h3 {{ margin: 0 0 10px 0; color: #2c3e50; }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .status-pass {{ color: #27ae60; font-weight: bold; }}
        .status-review {{ color: #f39c12; font-weight: bold; }}
        .status-fail {{ color: #e74c3c; font-weight: bold; }}
        .status-unknown {{ color: #95a5a6; }}
        .filters {{ margin: 20px 0; padding: 15px; background: #ecf0f1; border-radius: 8px; }}
        .filters input, .filters select {{ margin: 5px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; position: sticky; top: 0; }}
        tr:hover {{ background-color: #f8f9fa; }}
        .subject-merged {{ vertical-align: middle; background-color: #f1f2f6; font-weight: bold; }}
        .subject-separator {{ border-top: 2px solid #3498db; }}
        a {{ color: #2980b9; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
    <script>
        function filterTable() {{
            const subjectFilter = document.getElementById('subjectFilter').value.toLowerCase();
            const statusFilter = document.getElementById('statusFilter').value;
            const table = document.getElementById('qcTable');
            const rows = table.getElementsByTagName('tr');
            
            for (let i = 1; i < rows.length; i++) {{
                const row = rows[i];
                const cells = row.getElementsByTagName('td');
                if (cells.length === 0) continue;
                
                const subject = cells[0] ? cells[0].textContent.toLowerCase() : '';
                const statusCell = cells[9] ? cells[9].querySelector('span') : null;
                const status = statusCell ? statusCell.textContent : '';
                
                let showRow = true;
                
                if (subjectFilter && !subject.includes(subjectFilter)) {{
                    showRow = false;
                }}
                
                if (statusFilter && statusFilter !== 'ALL' && status !== statusFilter) {{
                    showRow = false;
                }}
                
                row.style.display = showRow ? '' : 'none';
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>{self.dataset_name} fMRI Quality Control Summary</h1>
        <p style="text-align: center; color: #7f8c8d;">fMRI Preprocessing QC Report</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Subjects & Sessions</h3>
                <div class="value">{total_subjects}</div>
                <div>Subjects</div>
                <div style="font-size: 1.2em; color: #3498db; margin-top: 5px;">{total_sessions} Sessions</div>
            </div>
            <div class="stat-card">
                <h3>PASS</h3>
                <div class="value" style="color: #27ae60;">{pass_count}</div>
            </div>
            <div class="stat-card">
                <h3>REVIEW</h3>
                <div class="value" style="color: #f39c12;">{review_count}</div>
            </div>
            <div class="stat-card">
                <h3>FAIL</h3>
                <div class="value" style="color: #e74c3c;">{fail_count}</div>
            </div>
            <div class="stat-card">
                <h3>Average Motion</h3>
                <div class="value">{avg_mean_fd:.3f}</div>
                <div>Mean FD (mm)</div>
            </div>
        </div>
        
        <!-- Status Explanation Section -->
        <div style="margin: 20px 0; padding: 20px; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">
            <h3 style="color: #2c3e50; margin-bottom: 15px;">📋 Quality Control Status Guide</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                <div style="padding: 15px; background: white; border-radius: 8px; border-left: 4px solid #27ae60;">
                    <h4 style="color: #27ae60; margin: 0 0 10px 0;">PASS</h4>
                    <p style="margin: 0; color: #2c3e50;">High-quality, usable data meeting all criteria:<br>
                    • ≥ {MIN_VOLUMES_PASS} volumes<br>
                    • Mean FD ≤ {FD_PASS_THRESHOLD} mm<br>
                    • Motion ratio ≤ {HIGH_MOTION_PASS_RATIO}%</p>
                </div>
                <div style="padding: 15px; background: white; border-radius: 8px; border-left: 4px solid #f39c12;">
                    <h4 style="color: #f39c12; margin: 0 0 10px 0;">REVIEW</h4>
                    <p style="margin: 0; color: #2c3e50;">Usable data but flagged for attention:<br>
                    • Moderate motion or shorter scans<br>
                    • Keep for biomarker analysis<br>
                    • Consider motion correction</p>
                </div>
                <div style="padding: 15px; background: white; border-radius: 8px; border-left: 4px solid #e74c3c;">
                    <h4 style="color: #e74c3c; margin: 0 0 10px 0;">FAIL</h4>
                    <p style="margin: 0; color: #2c3e50;">Unusable data requiring exclusion:<br>
                    • < {MIN_VOLUMES_FAIL} volumes<br>
                    • Mean FD > {FD_WARN_THRESHOLD} mm<br>
                    • Missing motion metrics</p>
                </div>
            </div>
        </div>
        
        <div class="filters">
            <strong>Filters:</strong>
            <input type="text" id="subjectFilter" placeholder="Filter by subject..." onkeyup="filterTable()">
            <select id="statusFilter" onchange="filterTable()">
                <option value="ALL">All Status</option>
                <option value="PASS">PASS</option>
                <option value="REVIEW">REVIEW</option>
                <option value="FAIL">FAIL</option>
            </select>
        </div>
        
        <div style="overflow-x: auto;">
            <table id="qcTable">
                <thead>
                    <tr>
                        <th>Subject</th>
                        <th>Session</th>
                        <th>Task</th>
                        <th>Direction</th>
                        <th>Volumes</th>
                        <th>Mean FD (mm)</th>
                        <th>Max FD (mm)</th>
                        <th>High Motion (%)</th>
                        <th>Notes</th>
                        <th>Status</th>
                        <th>Report</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add table rows with subject merging
        current_subject = None
        subject_sessions = df.groupby('Subject').size().to_dict()
        
        def format_status(status):
            if status == 'PASS':
                return f'<span class="status-pass">{status}</span>'
            elif status == 'REVIEW':
                return f'<span class="status-review">{status}</span>'
            elif status == 'FAIL':
                return f'<span class="status-fail">{status}</span>'
            else:
                return f'<span class="status-unknown">{status}</span>'
        
        def format_number(value):
            if pd.isna(value) or value == 'N/A':
                return '—'
            try:
                num = float(value)
                if np.isnan(num):
                    return '—'
                return f"{num:.3f}" if isinstance(num, float) else str(int(num))
            except:
                return '—' if value in (None, 'N/A', '') else str(value)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            is_new_subject = current_subject != row['Subject']
            subject_cell = ""
            row_class = ""
            
            if is_new_subject:
                if current_subject is not None:
                    row_class = ' class="subject-separator"'
                current_subject = row['Subject']
                session_count = subject_sessions[current_subject]
                if session_count > 1:
                    subject_cell = f'<td rowspan="{session_count}" class="subject-merged">{row["Subject"]}</td>'
                else:
                    subject_cell = f'<td>{row["Subject"]}</td>'
            else:
                subject_cell = ""
            
            # Generate report link
            report_filename = f"sub-{row['Subject']}_ses-{row['Session']}_qc.html"
            report_link = f'<a href="subjects/{report_filename}">📊 View</a>'
            
            html_content += f"""
                    <tr{row_class}>
                        {subject_cell}
                        <td>{row['Session']}</td>
                        <td>{row['Task']}</td>
                        <td>{row['Direction']}</td>
                        <td>{row['Volumes']}</td>
                        <td>{format_number(row['Mean_FD'])}</td>
                        <td>{format_number(row['Max_FD'])}</td>
                        <td>{format_number(row['High_Motion_Ratio'])}</td>
                        <td>{row['Notes']}</td>
                        <td>{format_status(row['Status'])}</td>
                        <td>{report_link}</td>
                    </tr>
"""
        
        html_content += f"""
                </tbody>
            </table>
        </div>
        
        <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
            <p>Designed & Developed by <a href="https://stai.stanford.edu/" style="color: #3498db;">Stanford Translational AI (STAI)</a></p>
        </div>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        # Save CSV data
        csv_path = os.path.join(self.qc_output_dir, "fmri_qc_data.csv")
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Generated summary report: {output_path}")
        logger.info(f"Saved QC data: {csv_path}")
        
        return output_path

    def run_qc(self, subjects=None):
        """Run QC for all or specified subjects"""
        if subjects is None:
            subjects = self.find_subjects()
        
        logger.info(f"Starting QC analysis for {len(subjects)} subjects")
        
        for subject in subjects:
            try:
                self.process_subject(subject)
            except Exception as e:
                logger.error(f"Error processing subject {subject}: {e}")
                continue
        
        # Generate summary report
        if self.qc_data:
            summary_path = self.generate_summary_report()
            
            # Print summary line for each subject/session
            print("\n" + "="*80)
            print("fMRI Quality Control Summary")
            print("="*80)
            
            for entry in self.qc_data:
                subj = entry['Subject']
                sess = entry['Session']
                task = entry['Task']
                direction = entry['Direction']
                mean_fd = entry['Mean_FD']
                max_fd = entry['Max_FD']
                high_motion = entry['High_Motion_Ratio']
                volumes = entry['Volumes']
                
                # Format values
                mean_fd_str = f"{mean_fd:.2f}mm" if mean_fd != 'N/A' else 'N/A'
                max_fd_str = f"{max_fd:.2f}mm" if max_fd != 'N/A' else 'N/A'
                high_motion_str = f"{high_motion:.0f}%" if high_motion != 'N/A' else 'N/A'
                
                print(f"sub-{subj} | {sess} | {task} | dir={direction} | meanFD={mean_fd_str} | maxFD={max_fd_str} | >0.5mm={high_motion_str} | vols={volumes}")
            
            print("="*80)
            print(f"📊 Summary report: {summary_path}")
            print(f"📁 Individual reports: {self.subjects_dir}")
            
        else:
            logger.warning("No QC data generated")

def main():
    parser = argparse.ArgumentParser(description='Generate fMRI quality control reports')
    parser.add_argument('--subjects', nargs='+', help='Subject IDs to process (without sub- prefix)')
    parser.add_argument('--all', action='store_true', help='Process all subjects')
    
    args = parser.parse_args()
    
    qc = FMRIQualityControl()
    
    if args.subjects:
        subjects = args.subjects
    elif args.all:
        subjects = None  # Process all subjects
    else:
        print("Please specify --subjects or --all")
        parser.print_help()
        return
    
    qc.run_qc(subjects)

if __name__ == '__main__':
    main()
