"""
validate_sync_output.py
Validates the output of the SyncPianoMotionDataset.py script.
"""

import pandas as pd
from pathlib import Path
import numpy as np

def validate_output():
    """Validates the features.csv file."""

    # Path relative to repo root
    features_csv = Path("Machine_Learning_Course/Data/PianoMotion10M/features.csv")

    if not features_csv.exists():
        # Fallback check if running from Code/Data_Pipeline
        features_csv_alt = Path("../../Data/PianoMotion10M/features.csv")
        if features_csv_alt.exists():
            features_csv = features_csv_alt
        else:
            raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    print(f"Validating: {features_csv}")
    df = pd.read_csv(features_csv)

    if df.empty:
        raise ValueError("The features.csv file is empty.")

    expected_cols = [
        'finger_velocity_x', 'finger_velocity_y', 'finger_velocity_z',
        'finger_acceleration_x', 'finger_acceleration_y', 'finger_acceleration_z',
        'finger_position_x', 'finger_position_y', 'finger_position_z',
        'depth_feature', 'posture_feature', 'euclidean_distance',
        'distance_from_wrist', 'fingertip_to_palm_center_distance',
        'wrist_velocity_x', 'wrist_velocity_y', 'wrist_velocity_z',
        'relative_velocity_x', 'relative_velocity_y', 'relative_velocity_z',
        'avg_velocity_x', 'avg_velocity_y', 'avg_velocity_z',
        'avg_acceleration_x', 'avg_acceleration_y', 'avg_acceleration_z',
        'ground_truth_label'
    ]

    missing_cols = [col for col in expected_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing expected feature columns: {missing_cols}")

    # Check label distribution
    label_counts = df['ground_truth_label'].value_counts()
    print(f"Label distribution:\n{label_counts}")

    if 1.0 not in label_counts:
        print("⚠️  WARNING: No positive labels (presses) found! Check chord grouping logic.")
        # We might treat this as an error if we expect the sample to have presses
        # The sample file BV1Jf421Z732.mid definitely has notes.
        raise ValueError("Dataset contains no positive labels.")

    # Check for NaN
    if df.isnull().values.any():
        print("⚠️  WARNING: Dataset contains NaNs.")
        print(df.isnull().sum())
        # Fill NaNs check
        # raise ValueError("Dataset contains NaNs")

    print("✅ Validation successful: features.csv has the correct format and 26+1 columns.")

if __name__ == "__main__":
    try:
        validate_output()
        print("✅ All checks passed!")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Validation failed: {e}")
        import sys
        sys.exit(1)
