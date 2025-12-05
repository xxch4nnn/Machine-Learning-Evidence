"""
experiment_controller.py
Controller script for running Incremental Learning experiments on Cloud Kernels (Colab/Kaggle).
Implements "Generate Once, Slice Many" strategy.

Phase A: Hyperparameter Tuning (explores 1-10 sequences)
Phase B: Incremental Learning Curve (explores full dataset with logarithmic increments)
Output: Saves trained models (rf_model.pkl, scaler.pkl) to models/ directory
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
import joblib
from datetime import datetime

# Add repo root to path so we can import modules
repo_root = Path(__file__).parent.resolve()
sys.path.append(str(repo_root))

from Machine_Learning_Course.Code.Data_Pipeline.ML_Pipeline_Prep import PianoMotionMLPipeline
from Machine_Learning_Course.Code.Data_Pipeline.SyncPianoMotionDataset import SyncPianoMotionDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = repo_root / "Machine_Learning_Course" / "Data" / "PianoMotion10M"
FEATURES_CSV = DATA_DIR / "features_real_pianomotion10m.csv"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

    logger.info("Running SyncPianoMotionDataset on dummy files...")
    processor = SyncPianoMotionDataset(dataset_dir=pipeline_dir)
    processor.run(max_files=None, output_csv="features.csv")

    logger.info("✅ Dummy data generation complete.")


def generate_logarithmic_increments(total_size: int, num_points: int = 10) -> list:
    """
    Generate logarithmic increments for dataset sizes.
    
    Args:
        total_size: Total dataset size
        num_points: Desired number of increments
        
    Returns:
        List of incrementally spaced values using logarithmic scale
    """
    if total_size <= 1:
        return [1]
    
    # Create logarithmic spacing from log(1) to log(total_size)
    log_min = np.log(1)
    log_max = np.log(total_size)
    log_points = np.linspace(log_min, log_max, num_points)
    
    # Convert back from log space
    linear_points = np.exp(log_points)
    
    # Convert to integers and remove duplicates while preserving order
    increments = sorted(list(set(int(round(p)) for p in linear_points)))
    
    # Ensure we include 1 and total_size
    if 1 not in increments:
        increments.insert(0, 1)
    if total_size not in increments:
        increments.append(total_size)
    
    return sorted(list(set(increments)))


def main():
    logger.info("\n" + "="*80)
    logger.info("🎹 PIANOMOTION EXPERIMENT CONTROLLER - INCREMENTAL LEARNING (NO RFE)")
    logger.info("="*80)
    logger.info(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if VERIFICATION_MODE:
        create_dummy_verification_data()

    # --- STEP 0: Load Features ---
    # Check if using real features or dummy
    features_file = FEATURES_CSV if FEATURES_CSV.exists() else (DATA_DIR / "features.csv")
    
    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        logger.info("Please run DownloadRealPianoMotion10M.py first to generate features.")
        return

    logger.info(f"\n{'='*80}")
    logger.info("📥 STEP 0: Loading Features Dataset (ALL FEATURES - NO RFE)")
    logger.info(f"{'='*80}")
    logger.info(f"Loading dataset from {features_file}...")
    
    full_df = pd.read_csv(features_file)
    
    if 'sequence_id' not in full_df.columns and 'ground_truth_label' not in full_df.columns:
        logger.error("❌ Required columns ('sequence_id', 'ground_truth_label') missing.")
        return

    # If sequence_id is missing, create it from index or filename
    if 'sequence_id' not in full_df.columns:
        logger.warning("⚠️ 'sequence_id' not found. Creating synthetic sequence IDs...")
        full_df['sequence_id'] = ['seq_' + str(i // 100) for i in range(len(full_df))]

    unique_sequences = sorted(full_df['sequence_id'].unique())
    n_sequences = len(unique_sequences)
    n_samples = len(full_df)
    
    logger.info(f"✅ Loaded {n_samples} samples from {n_sequences} sequences")
    logger.info(f"   Label distribution: {full_df['ground_truth_label'].value_counts().to_dict()}")

    # --- STEP 0.5: Prepare All Features (No RFE) ---
    logger.info(f"\n{'='*80}")
    logger.info("✨ STEP 0.5: Using ALL Features (RFE Skipped)")
    logger.info(f"{'='*80}")

    # Get all feature columns (excluding label and sequence_id)
    exclude_cols = {'ground_truth_label', 'sequence_id', 'frame_index', 'subject', 'experiment', 'phase'}
    all_features = [col for col in full_df.columns if col not in exclude_cols]
    selected_features_list = all_features
    
    logger.info(f"✅ Using all {len(selected_features_list)} features:")
    for i, feat in enumerate(selected_features_list[:15], 1):
        logger.info(f"   {i}. {feat}")
    if len(selected_features_list) > 15:
        logger.info(f"   ... and {len(selected_features_list) - 15} more features")

    # --- PHASE A: Hyperparameter Exploration ---
    logger.info(f"\n{'='*80}")
    logger.info("🔬 PHASE A: HYPERPARAMETER TUNING (1-10 Sequences)")
    logger.info(f"{'='*80}")

    best_params_history = []
    step_a_results = []
    
    # Determine loop range (up to 10, or less if we have fewer sequences)
    max_a = min(10, n_sequences)
    logger.info(f"Running {max_a} hyperparameter tuning experiments...")

    for n in range(1, max_a + 1):
        logger.info(f"\n{'─'*80}")
        logger.info(f"📊 Experiment A.{n}: Training on {n} Sequence(s)")
        logger.info(f"{'─'*80}")

        # Slice DataFrame
        target_seqs = unique_sequences[:n]
        df_slice = full_df[full_df['sequence_id'].isin(target_seqs)]
        
        logger.info(f"   • Dataset size: {len(df_slice)} samples")
        logger.info(f"   • Sequences: {list(target_seqs[:3])}{'...' if len(target_seqs) > 3 else ''}")
        logger.info(f"   • Label distribution: {df_slice['ground_truth_label'].value_counts().to_dict()}")

        # Initialize Pipeline
        pipeline = PianoMotionMLPipeline(
            features_csv=str(features_file), 
            dataframe=df_slice, 
            random_state=42
        )

        # Run Pipeline
        output_path = pipeline.run_pipeline(
            output_dir=RESULTS_DIR / f"step_a_{n}",
            selected_features=selected_features_list,
            skip_svm=True  # Skip SVM for speed
        )

        # Extract Best Params from Random Forest
        rf_model = pipeline.models.get('Random Forest')
        if rf_model:
            best_params = rf_model.get_params()
            relevant_keys = [
                'n_estimators', 'max_depth', 'min_samples_split', 
                'min_samples_leaf', 'max_features', 'bootstrap', 'criterion'
            ]
            filtered_params = {k: best_params[k] for k in relevant_keys if k in best_params}
            best_params_history.append(filtered_params)

            # Store results
            score = pipeline.results.get('Random Forest', {}).get('f1_score', 0)
            step_a_results.append({
                'phase': 'A',
                'experiment': f'A.{n}',
                'n_sequences': n,
                'n_samples': len(df_slice),
                'f1_score': score,
                'accuracy': pipeline.results.get('Random Forest', {}).get('accuracy', 0),
                'precision': pipeline.results.get('Random Forest', {}).get('precision', 0),
                'recall': pipeline.results.get('Random Forest', {}).get('recall', 0),
            })
            
            logger.info(f"✅ Result A.{n}:")
            logger.info(f"   • F1-Score: {score:.4f}")
            logger.info(f"   • Accuracy: {pipeline.results['Random Forest']['accuracy']:.4f}")
            logger.info(f"   • Precision: {pipeline.results['Random Forest']['precision']:.4f}")
            logger.info(f"   • Recall: {pipeline.results['Random Forest']['recall']:.4f}")

    # --- Select Optimal Hyperparameters ---
    logger.info(f"\n{'='*80}")
    logger.info("🏆 SELECTING OPTIMAL HYPERPARAMETERS")
    logger.info(f"{'='*80}")
    
    # Use parameters from the best performing experiment (typically the largest dataset)
    optimal_params = best_params_history[-1]
    
    logger.info(f"Optimal hyperparameters selected from A.{max_a}:")
    for key, value in optimal_params.items():
        logger.info(f"   • {key}: {value}")

    # --- PHASE B: Incremental Learning Curve with Logarithmic Increments ---
    logger.info(f"\n{'='*80}")
    logger.info("📈 PHASE B: INCREMENTAL LEARNING CURVE (Logarithmic Dataset Growth)")
    logger.info(f"{'='*80}")

    step_b_results = []
    
    # Generate logarithmic increments based on number of sequences
    x_values = generate_logarithmic_increments(n_sequences, num_points=12)
    
    logger.info(f"Running incremental learning with {len(x_values)} experiments...")
    logger.info(f"Dataset increments (sequences): {x_values}")
    
    # Convert sequence counts to sample counts for reference
    sample_counts = []
    for seq_count in x_values:
        target_seqs = unique_sequences[:seq_count]
        sample_count = len(full_df[full_df['sequence_id'].isin(target_seqs)])
        sample_counts.append(sample_count)
    logger.info(f"Corresponding sample counts: {sample_counts}")

    best_b_model = None
    best_b_scaler = None
    best_b_features = None
    best_b_score = 0
    best_b_n = 0

    for idx, n in enumerate(x_values, 1):
        logger.info(f"\n{'─'*80}")
        logger.info(f"📊 Experiment B.{idx}: Training on {n} Sequence(s) (Fixed Optimal Params)")
        logger.info(f"{'─'*80}")

        # Slice DataFrame
        target_seqs = unique_sequences[:n]
        df_slice = full_df[full_df['sequence_id'].isin(target_seqs)]
        
        logger.info(f"   • Dataset size: {len(df_slice)} samples")
        logger.info(f"   • Sequences: {list(target_seqs[:3])}{'...' if len(target_seqs) > 3 else ''}")
        logger.info(f"   • Label distribution: {df_slice['ground_truth_label'].value_counts().to_dict()}")

        # Initialize Pipeline
        pipeline = PianoMotionMLPipeline(
            features_csv=str(features_file), 
            dataframe=df_slice, 
            random_state=42
        )

        # Run with FIXED optimal parameters
        output_path = pipeline.run_pipeline(
            output_dir=RESULTS_DIR / f"step_b_{n}",
            selected_features=selected_features_list,
            fixed_rf_params=optimal_params,  # Use fixed optimal params
            skip_svm=True
        )

        # Extract and log metrics
        rf_results = pipeline.results.get('Random Forest', {})
        metrics_dict = {
            'phase': 'B',
            'experiment': f'B.{idx}',
            'n_sequences': n,
            'n_samples': len(df_slice),
            'f1_score': rf_results.get('f1_score', 0),
            'accuracy': rf_results.get('accuracy', 0),
            'precision': rf_results.get('precision', 0),
            'recall': rf_results.get('recall', 0),
        }
        step_b_results.append(metrics_dict)

        current_score = rf_results.get('f1_score', 0)
        logger.info(f"✅ Result B.{idx}:")
        logger.info(f"   • F1-Score: {current_score:.4f}")
        logger.info(f"   • Accuracy: {rf_results.get('accuracy', 0):.4f}")
        logger.info(f"   • Precision: {rf_results.get('precision', 0):.4f}")
        logger.info(f"   • Recall: {rf_results.get('recall', 0):.4f}")

        # Track best model
        if current_score > best_b_score:
            best_b_score = current_score
            best_b_n = n
            best_b_model = pipeline.models.get('Random Forest')
            best_b_scaler = pipeline.scaler
            best_b_features = pipeline.selected_feature_names

    # --- Save Final Models ---
    logger.info(f"\n{'='*80}")
    logger.info("💾 SAVING FINAL MODELS")
    logger.info(f"{'='*80}")
    
    if best_b_model:
        logger.info(f"Best model achieved F1-Score of {best_b_score:.4f} on {best_b_n} sequences")
        logger.info(f"Saving to: {MODELS_DIR}")
        
        # Save best model
        joblib.dump(best_b_model, MODELS_DIR / "rf_model.pkl")
        logger.info(f"✅ Saved: {MODELS_DIR / 'rf_model.pkl'}")
        
        # Save scaler
        joblib.dump(best_b_scaler, MODELS_DIR / "scaler.pkl")
        logger.info(f"✅ Saved: {MODELS_DIR / 'scaler.pkl'}")
        
        # Save feature names
        joblib.dump(best_b_features, MODELS_DIR / "feature_names.pkl")
        logger.info(f"✅ Saved: {MODELS_DIR / 'feature_names.pkl'}")
        
        # Save model metadata
        metadata = {
            'best_score': float(best_b_score),
            'best_n_sequences': int(best_b_n),
            'optimal_params': optimal_params,
            'selected_features': selected_features_list,
            'num_features': len(selected_features_list),
            'timestamp': datetime.now().isoformat(),
            'total_sequences': n_sequences,
            'total_samples': n_samples,
            'rfe_used': False,
        }
        joblib.dump(metadata, MODELS_DIR / "model_metadata.pkl")
        logger.info(f"✅ Saved: {MODELS_DIR / 'model_metadata.pkl'}")

    # --- Generate Results Summary ---
    logger.info(f"\n{'='*80}")
    logger.info("📊 FINAL REPORT & VISUALIZATION")
    logger.info(f"{'='*80}")

    # Combine results
    all_results = step_a_results + step_b_results
    results_df = pd.DataFrame(all_results)
    
    # Display summary
    logger.info("\n" + "="*80)
    logger.info("PHASE A RESULTS (Hyperparameter Tuning)")
    logger.info("="*80)
    print(results_df[results_df['phase'] == 'A'].to_string(index=False))
    
    logger.info("\n" + "="*80)
    logger.info("PHASE B RESULTS (Incremental Learning - Logarithmic Increments)")
    logger.info("="*80)
    print(results_df[results_df['phase'] == 'B'].to_string(index=False))

    # Save results to CSV
    results_csv = RESULTS_DIR / "incremental_learning_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"\n✅ Results saved to: {results_csv}")

    # Plot Learning Curve
    logger.info("\nGenerating learning curve visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Phase A: Hyperparameter tuning
    phase_a = results_df[results_df['phase'] == 'A']
    axes[0, 0].plot(phase_a['n_sequences'], phase_a['f1_score'], marker='o', linewidth=2, markersize=8, label='F1-Score', color='#2E86AB')
    axes[0, 0].plot(phase_a['n_sequences'], phase_a['precision'], marker='s', linestyle='--', label='Precision', color='#A23B72')
    axes[0, 0].plot(phase_a['n_sequences'], phase_a['recall'], marker='^', linestyle='--', label='Recall', color='#F18F01')
    axes[0, 0].set_title('Phase A: Hyperparameter Tuning', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Sequences')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Phase B: Incremental learning
    phase_b = results_df[results_df['phase'] == 'B']
    axes[0, 1].plot(phase_b['n_sequences'], phase_b['f1_score'], marker='o', linewidth=2, markersize=8, label='F1-Score', color='#2E86AB')
    axes[0, 1].plot(phase_b['n_sequences'], phase_b['precision'], marker='s', linestyle='--', label='Precision', color='#A23B72')
    axes[0, 1].plot(phase_b['n_sequences'], phase_b['recall'], marker='^', linestyle='--', label='Recall', color='#F18F01')
    axes[0, 1].set_title('Phase B: Incremental Learning (Logarithmic Increments)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Sequences')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Phase B with sample count (log scale)
    axes[1, 0].semilogx(phase_b['n_samples'], phase_b['f1_score'], marker='o', linewidth=2, markersize=8, label='F1-Score', color='#2E86AB')
    axes[1, 0].semilogx(phase_b['n_samples'], phase_b['precision'], marker='s', linestyle='--', label='Precision', color='#A23B72')
    axes[1, 0].semilogx(phase_b['n_samples'], phase_b['recall'], marker='^', linestyle='--', label='Recall', color='#F18F01')
    axes[1, 0].set_title('Phase B: Learning Curve (Samples on Log Scale)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Samples (log scale)')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')
    
    # Accuracy comparison
    axes[1, 1].plot(phase_a['n_sequences'], phase_a['accuracy'], marker='o', linewidth=2, markersize=8, label='Phase A - Accuracy', color='#06A77D')
    axes[1, 1].plot(phase_b['n_sequences'], phase_b['accuracy'], marker='s', linewidth=2, markersize=8, label='Phase B - Accuracy', color='#D62828')
    axes[1, 1].set_title('Accuracy Comparison: Phase A vs Phase B', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Sequences')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Piano Motion Classification - Incremental Learning Analysis (No RFE, All Features)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / "learning_curve_logarithmic.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Learning curve saved to: {plot_path}")
    plt.close()

    # Summary statistics
    logger.info(f"\n{'='*80}")
    logger.info("📈 SUMMARY STATISTICS")
    logger.info(f"{'='*80}")
    logger.info(f"Total experiments run: {len(results_df)}")
    logger.info(f"  • Phase A (tuning): {len(phase_a)}")
    logger.info(f"  • Phase B (learning): {len(phase_b)}")
    logger.info(f"\nDataset Information:")
    logger.info(f"  • Total sequences: {n_sequences}")
    logger.info(f"  • Total samples: {n_samples}")
    logger.info(f"  • Total features used: {len(selected_features_list)} (RFE skipped)")
    logger.info(f"\nPhase A Statistics:")
    logger.info(f"  • Best F1-Score: {phase_a['f1_score'].max():.4f}")
    logger.info(f"  • Avg F1-Score: {phase_a['f1_score'].mean():.4f}")
    logger.info(f"\nPhase B Statistics:")
    logger.info(f"  • Best F1-Score: {phase_b['f1_score'].max():.4f}")
    logger.info(f"  • Avg F1-Score: {phase_b['f1_score'].mean():.4f}")
    logger.info(f"  • Improvement from min to max: {phase_b['f1_score'].max() - phase_b['f1_score'].min():.4f}")
    logger.info(f"\nBest Model:")
    logger.info(f"  • F1-Score: {best_b_score:.4f}")
    logger.info(f"  • Sequences used: {best_b_n}/{n_sequences}")
    logger.info(f"  • Experiment: {results_df.loc[results_df['f1_score'].idxmax(), 'experiment']}")

    logger.info(f"\n{'='*80}")
    logger.info("✅ EXPERIMENT CONTROLLER FINISHED SUCCESSFULLY")
    logger.info(f"🕐 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*80}")
    logger.info(f"\n📁 Output locations:")
    logger.info(f"   Models: {MODELS_DIR}/")
    logger.info(f"   Results: {RESULTS_DIR}/")
    logger.info(f"   Featured files:")
    logger.info(f"      • {MODELS_DIR}/rf_model.pkl")
    logger.info(f"      • {MODELS_DIR}/scaler.pkl")
    logger.info(f"      • {MODELS_DIR}/feature_names.pkl")
    logger.info(f"      • {MODELS_DIR}/model_metadata.pkl")
    logger.info(f"      • {RESULTS_DIR}/incremental_learning_results.csv")
    logger.info(f"      • {RESULTS_DIR}/learning_curve_logarithmic.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Piano Motion Incremental Learning Experiment Controller (No RFE, Logarithmic Increments)"
    )
    parser.add_argument(
        '--verify', 
        action='store_true', 
        help='Run in verification mode with dummy data (15 synthetic sequences)'
    )
    parser.add_argument(
        '--features-csv',
        type=str,
        default=None,
        help='Path to custom features CSV file'
    )
    
    args = parser.parse_args()

    if args.verify:
        VERIFICATION_MODE = True
    
    if args.features_csv:
        FEATURES_CSV = Path(args.features_csv)

    main()