"""
experiment_controller.py
Controller script for running Incremental Learning experiments on Cloud Kernels (Colab/Kaggle).
Implements "Generate Once, Slice Many" strategy.
"""

# --- 1. Universal Header ---
import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add repo root to path so we can import modules
# Assuming this script is running in a Colab cell or from repo root
# Adjust repo_root based on where the file is. If it's in root:
repo_root = Path(__file__).parent.resolve()
sys.path.append(str(repo_root))

from Machine_Learning_Course.Code.Data_Pipeline.ML_Pipeline_Prep import PianoMotionMLPipeline
from Machine_Learning_Course.Code.Data_Pipeline.SyncPianoMotionDataset import SyncPianoMotionDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = repo_root / "Machine_Learning_Course" / "Data" / "PianoMotion10M"
FEATURES_CSV = DATA_DIR / "features.csv"
RESULTS_DIR = DATA_DIR / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Verification Mode: Uses small dummy dataset if True
VERIFICATION_MODE = False

def create_dummy_verification_data():
    """Duplicates sample data to simulate multiple sequences for verification."""
    logger.info("🛠️ Creating dummy data for VERIFICATION_MODE...")

    # Locate sample JSON in Data Pipeline folder
    pipeline_dir = repo_root / "Machine_Learning_Course" / "Code" / "Data_Pipeline"
    sample_json = pipeline_dir / "BV1Jf421Z732_seq_0000.json"

    if not sample_json.exists():
        logger.error(f"Sample file not found at {sample_json}")
        return

    # Create 15 dummy copies
    for i in range(1, 16):
        dummy_path = pipeline_dir / f"dummy_seq_{i:02d}.json"
        with open(sample_json, 'rb') as src, open(dummy_path, 'wb') as dst:
            dst.write(src.read())

    # Run Sync to generate features.csv with these new files
    # We point Sync to the pipeline dir where dummies are
    logger.info("Running SyncPianoMotionDataset on dummy files...")
    processor = SyncPianoMotionDataset(dataset_dir=pipeline_dir)
    processor.run(max_files=None, output_csv="features.csv")

    # Move/Copy features.csv to expected DATA_DIR location if needed
    # Sync script now saves to Machine_Learning_Course/Data/PianoMotion10M/features.csv by default relative to repo
    # So it should be in the right place.

    logger.info("✅ Dummy data generation complete.")

def main():
    logger.info("\n" + "="*60)
    logger.info("🧪 PIANOMOTION EXPERIMENT CONTROLLER")
    logger.info("="*60)

    if VERIFICATION_MODE:
        create_dummy_verification_data()

    # --- 2. Data Strategy: Load Once ---
    if not FEATURES_CSV.exists():
        logger.error(f"Features file not found: {FEATURES_CSV}")
        logger.info("Please run SyncPianoMotionDataset.py first.")
        return

    logger.info(f"Loading full dataset from {FEATURES_CSV}...")
    full_df = pd.read_csv(FEATURES_CSV)

    if 'sequence_id' not in full_df.columns:
        logger.error("❌ 'sequence_id' column missing. Please update SyncPianoMotionDataset.py.")
        return

    unique_sequences = full_df['sequence_id'].unique()
    n_sequences = len(unique_sequences)
    logger.info(f"Loaded {len(full_df)} samples from {n_sequences} sequences.")

    # --- 3. Feature Selection (One-Time) ---
    logger.info("\n" + "-"*60)
    logger.info("🔍 PHASE 0: Global Feature Selection")
    logger.info("-" * 60)

    # We run RFE once on the full dataset (or a large representative subset)
    # For speed in verification, we use full_df. In production, this is correct.

    # Initialize pipeline just for RFE
    rfe_pipeline = PianoMotionMLPipeline(dataframe=full_df)
    X_train, _, y_train, _ = rfe_pipeline.load_and_prepare_data()

    # Perform RFE
    selected_features = rfe_pipeline.perform_rfe(X_train, y_train)
    selected_features_list = list(selected_features)

    logger.info(f"✅ Locked {len(selected_features_list)} features for all experiments.")

    # --- 4. Experiment Workflow ---

    # Step A: Hyperparameter Exploration (1-10 Files)
    logger.info("\n" + "="*60)
    logger.info("🔬 STEP A: Hyperparameter Exploration (1-10 Files)")
    logger.info("="*60)

    best_params_history = []
    step_a_results = []

    # Determine loop range (up to 10, or less if we have fewer sequences)
    max_a = min(10, n_sequences)

    for n in range(1, max_a + 1):
        logger.info(f"\n--- Experiment A.{n}: Training on {n} Sequence(s) ---")

        # Slice DataFrame
        target_seqs = unique_sequences[:n]
        df_slice = full_df[full_df['sequence_id'].isin(target_seqs)]

        # Initialize Pipeline
        pipeline = PianoMotionMLPipeline(dataframe=df_slice, random_state=42)

        # Run Pipeline (Skip SVM for speed, Use Pre-selected Features)
        # We allow tuning here to find best params
        output_path = pipeline.run_pipeline(
            output_dir=RESULTS_DIR / f"step_a_{n}",
            selected_features=selected_features_list,
            skip_svm=True
        )

        # Log Best Params
        rf_model = pipeline.models['Random Forest']
        best_params = rf_model.get_params()
        # Filter for relevant keys
        relevant_keys = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
        filtered_params = {k: best_params[k] for k in relevant_keys}

        best_params_history.append(filtered_params)

        # Store score
        score = pipeline.results['Random Forest']['f1_score']
        step_a_results.append({'n_sequences': n, 'f1_score': score, 'params': filtered_params})
        logger.info(f"Result A.{n}: F1={score:.4f}")

    # Select Optimal Hyperparameters (Simple approach: Majority vote or parameters from best run)
    # Let's pick the params from the run with the best F1 score on the largest subset (A.10)
    # OR find the most common config. For simplicity/stability, using the config from the largest sample in Step A is robust.
    optimal_params = best_params_history[-1]
    logger.info(f"\n🏆 Optimal Hyperparameters Selected (from n={max_a}): {optimal_params}")

    # Step B: Incremental Learning Curve (1..All Files)
    logger.info("\n" + "="*60)
    logger.info("📈 STEP B: Incremental Learning Curve")
    logger.info("="*60)

    step_b_results = []

    # Step size for loop
    step_size = 5 if n_sequences > 20 else 1

    # Range: 1 to Total, stepping by 5
    # Ensure we hit the max
    x_values = list(range(1, n_sequences + 1, step_size))
    if x_values[-1] != n_sequences:
        x_values.append(n_sequences)

    for n in x_values:
        logger.info(f"\n--- Experiment B.{n}: Training on {n} Sequence(s) (Fixed Params) ---")

        # Slice
        target_seqs = unique_sequences[:n]
        df_slice = full_df[full_df['sequence_id'].isin(target_seqs)]

        # Pipeline
        pipeline = PianoMotionMLPipeline(dataframe=df_slice, random_state=42)

        # Run with FIXED params
        output_path = pipeline.run_pipeline(
            output_dir=RESULTS_DIR / f"step_b_{n}",
            selected_features=selected_features_list,
            fixed_rf_params=optimal_params, # Force fixed params
            skip_svm=True
        )

        # Log Metrics
        metrics = pipeline.results['Random Forest']
        step_b_results.append({
            'n_sequences': n,
            'n_samples': len(df_slice),
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })

    # --- 5. Final Reporting & Visualization ---
    logger.info("\n" + "="*60)
    logger.info("📊 FINAL REPORT")
    logger.info("="*60)

    results_df = pd.DataFrame(step_b_results)
    print(results_df)

    results_csv = RESULTS_DIR / "incremental_learning_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Results saved to {results_csv}")

    # Plot Learning Curve
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['n_sequences'], results_df['f1_score'], marker='o', label='F1-Score')
    plt.plot(results_df['n_sequences'], results_df['precision'], marker='s', linestyle='--', label='Precision')
    plt.plot(results_df['n_sequences'], results_df['recall'], marker='^', linestyle='--', label='Recall')

    plt.title('Incremental Learning Curve - Piano Motion Classification')
    plt.xlabel('Number of Sequences')
    plt.ylabel('Score (Weighted)')
    plt.legend()
    plt.grid(True)

    plot_path = RESULTS_DIR / "learning_curve.png"
    plt.savefig(plot_path)
    logger.info(f"Learning curve saved to {plot_path}")

    logger.info("\n✅ Experiment Controller Finished Successfully.")

if __name__ == "__main__":
    # Check for CLI args to enable verification mode
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='Run in verification mode with dummy data')
    args = parser.parse_args()

    if args.verify:
        VERIFICATION_MODE = True

    main()
