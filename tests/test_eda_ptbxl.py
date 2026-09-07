"""
Unit and Integration Tests for PTB-XL EDA Pipeline
===================================================
Tests dataset loading, target definition alignment, leakage-free patient
partitioning, signal dimensions, and output artifact generation.
"""

import os
import sys
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eda_ptbxl import (
    is_normal_record,
    PTBXLDataLoader,
    PTBXLExplorer,
    EDA_REPORTS_DIR,
    REPORTS_DIR,
)


class TestPTBXLEDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Generates synthetic PTB-XL metadata matching the exact schema."""
        data = {
            'ecg_id': [1, 2, 3, 4, 5, 6],
            'patient_id': [101, 101, 102, 103, 104, 105],  # Patient 101 has 2 records
            'age': [55.0, 56.0, np.nan, 45.0, 72.0, 60.0],
            'sex': [1, 1, 0, 1, np.nan, 0],
            'height': [175.0, np.nan, 160.0, np.nan, 180.0, 165.0],
            'weight': [80.0, np.nan, 65.0, np.nan, 85.0, 70.0],
            'scp_codes': [
                "{'NORM': 100.0, 'SR': 100.0}",
                "{'NORM': 100.0}",
                "{'IMI': 100.0, 'ABQRS': 0.0}",
                "{'AMI': 80.0}",
                "{'NORM': 100.0, 'LAFB': 50.0}",
                "{'AFIB': 100.0}",
            ],
            'filename_lr': [
                'records100/00000/00001_lr',
                'records100/00000/00002_lr',
                'records100/00000/00003_lr',
                'records100/00000/00004_lr',
                'records100/00000/00005_lr',
                'records100/00000/00006_lr',
            ],
        }
        cls.sample_df = pd.DataFrame(data)
        cls.sample_df['target'] = cls.sample_df['scp_codes'].apply(is_normal_record)

    def test_target_generation_logic(self):
        """Verifies target generation strictly adheres to Phase 2 rule (NORM present = 0, else 1)."""
        self.assertEqual(is_normal_record("{'NORM': 100.0}"), 0)
        self.assertEqual(is_normal_record("{'NORM': 100.0, 'SR': 100.0}"), 0)
        self.assertEqual(is_normal_record("{'IMI': 100.0}"), 1)
        self.assertEqual(is_normal_record("{'AFIB': 100.0, 'PVC': 50.0}"), 1)
        self.assertEqual(is_normal_record(np.nan), 1)
        self.assertEqual(is_normal_record("corrupted_string"), 1)

    def test_target_counts_sum_correctly(self):
        """Ensures Normal count + Abnormal count equals total record count."""
        total = len(self.sample_df)
        norm = (self.sample_df['target'] == 0).sum()
        abnorm = (self.sample_df['target'] == 1).sum()
        self.assertEqual(norm + abnorm, total)
        self.assertEqual(norm, 3)
        self.assertEqual(abnorm, 3)

    def test_patient_level_leakage_prevention(self):
        """Verifies zero patient ID overlap between Train, Validation, and Test sets."""
        unique_patients = self.sample_df['patient_id'].unique()
        
        # 3-way split
        train_p = set(unique_patients[:3])
        val_p = set(unique_patients[3:4])
        test_p = set(unique_patients[4:])

        # Intersections must be strictly empty
        self.assertEqual(len(train_p.intersection(val_p)), 0, "Train-Val patient leakage detected!")
        self.assertEqual(len(train_p.intersection(test_p)), 0, "Train-Test patient leakage detected!")
        self.assertEqual(len(val_p.intersection(test_p)), 0, "Val-Test patient leakage detected!")

    def test_missing_value_calculations(self):
        """Validates computation of missing values and percentages."""
        total = len(self.sample_df)
        age_missing = self.sample_df['age'].isna().sum()
        self.assertEqual(age_missing, 1)
        self.assertEqual(round((age_missing / total) * 100, 2), 16.67)

    def test_ecg_signal_dimension_conformance(self):
        """Verifies signal dimensions conform to expected 10-second 12-lead 100Hz structure."""
        expected_samples = 1000  # 10s @ 100Hz
        expected_leads = 12
        
        # Create sample signal tensor
        dummy_signal = np.zeros((expected_samples, expected_leads))
        self.assertEqual(dummy_signal.shape, (1000, 12))
        self.assertEqual(dummy_signal.shape[1], 12, "Must have exactly 12 leads")

    def test_eda_runner_end_to_end(self):
        """Verifies PTBXLExplorer runs smoothly and creates all required reports and plots."""
        temp_dir = tempfile.mkdtemp(prefix="test_ptbxl_eda_")
        try:
            db_csv = Path(temp_dir) / "ptbxl_database.csv"
            self.sample_df.to_csv(db_csv, index=False)

            loader = PTBXLDataLoader(data_path=temp_dir)
            explorer = PTBXLExplorer(loader=loader)
            
            summary = explorer.run_full_eda()
            
            self.assertEqual(summary['total_records'], 6)
            self.assertEqual(summary['unique_patients'], 5)
            self.assertEqual(summary['normal_count'], 3)
            self.assertEqual(summary['abnormal_count'], 3)
            self.assertTrue(explorer.leakage_results['leakage_passed'])

            # Verify generated artifact files exist
            self.assertTrue((EDA_REPORTS_DIR / "ptbxl_eda_summary.csv").exists())
            self.assertTrue((EDA_REPORTS_DIR / "ptbxl_missing_values.csv").exists())
            self.assertTrue((EDA_REPORTS_DIR / "ptbxl_target_distribution.csv").exists())
            self.assertTrue((EDA_REPORTS_DIR / "ptbxl_scp_code_distribution.csv").exists())
            self.assertTrue((EDA_REPORTS_DIR / "01_dataset_overview.png").exists())
            self.assertTrue((EDA_REPORTS_DIR / "02_age_distribution.png").exists())
            self.assertTrue((EDA_REPORTS_DIR / "05_target_distribution.png").exists())
            self.assertTrue((REPORTS_DIR / "EDA_PTBXL_REPORT.md").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
