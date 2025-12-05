"""
experiment_controller.py
Resumable Incremental Learning Controller for PianoMotion10M.
Implements "Generate Once, Slice Many" strategy with Robust State Management.
"""

import os
import sys
import json
import logging
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Add repo root to path to allow imports
repo_root = Path(__file__).parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

try:
    from Machine_Learning_Course.Code.Data_Pipeline.ML_Pipeline_Prep import PianoMotionMLPipeline
    from Machine_Learning_Course.Code.Data_Pipeline.SyncPianoMotionDataset import SyncPianoMotionDataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current sys.path: {sys.path}")
    # Fallback for when script is run directly from root without module structure
    try:
        sys.path.append(str(repo_root / "Machine_Learning_Course" / "Code" / "Data_Pipeline"))
        from ML_Pipeline_Prep import PianoMotionMLPipeline
        from SyncPianoMotionDataset import SyncPianoMotionDataset
    except ImportError:
        raise

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Explicitly pointing to the Data_Pipeline folder where features.csv is generated
DATA_PIPELINE_DIR = repo_root / "Machine_Learning_Course" / "Code" / "Data_Pipeline"
FEATURES_CSV = DATA_PIPELINE_DIR / "features.csv"
EXPERIMENT_DIR = DATA_PIPELINE_DIR / "experiments_incremental"
STATE_FILE = EXPERIMENT_DIR / "experiment_state.json"
RESULTS_CSV = EXPERIMENT_DIR / "incremental_results.csv"

MAX_SAMPLES = 100000

class ExperimentState:
    """
    Manages the state of the experiment to allow resumption after crashes.
    """
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.experiment_dir = state_file.parent
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"🔄 Resuming from Phase {state.get('phase', 'INIT')}")
                return state
            except Exception as e:
                logger.error(f"❌ Corrupt state file: {e}")

        # Default State
        return {
            "phase": "A", # A = Tuning, B = Scaling, DONE = Finished
            "best_params": None,
            "completed_steps": [], # List of n_files steps completed in Phase B
        }

    def save(self, state: Dict):
        # Atomic save
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=4)
        shutil.move(temp_file, self.state_file)
        logger.info("💾 State saved.")

class ExperimentController:
    """
    Orchestrates the Incremental Learning Experiment.
    """
    def __init__(self):
        self.state_manager = ExperimentState(STATE_FILE)
        self.state = self.state_manager.load()

        # Ensure data exists
        if not FEATURES_CSV.exists():
            logger.info("⚠️ features.csv not found. Launching Data Engine...")
            # Pass specific path if needed, or rely on default
            engine = SyncPianoMotionDataset(dataset_root=None)
            # If we want to force the output dir to be where features_csv is expected
            engine.output_dir = DATA_PIPELINE_DIR
            engine.run(limit=MAX_SAMPLES)

        if not FEATURES_CSV.exists():
            logger.error("❌ Failed to generate features.csv. Aborting.")
            sys.exit(1)

        logger.info(f"📚 Loading Data from {FEATURES_CSV}...")
        self.df = pd.read_csv(FEATURES_CSV)
        self.unique_sequences = self.df['sequence_id'].unique().tolist()
        logger.info(f"Loaded {len(self.df)} samples, {len(self.unique_sequences)} sequences.")

        if len(self.df) > MAX_SAMPLES:
            logger.info(f"Trimming dataset to {MAX_SAMPLES} samples.")
            self.df = self.df.iloc[:MAX_SAMPLES]

    def run_phase_a_tuning(self):
        """
        Phase A: Tuning.
        Load the first 10 unique sequences.
        Run GridSearchCV (via Pipeline) to find best_params.
        """
        logger.info("\n" + "="*60)
        logger.info("🔬 PHASE A: Tuning (First 10 Sequences)")
        logger.info("="*60)

        # Slice first 10 sequences
        target_seqs = self.unique_sequences[:10]
        subset_df = self.df[self.df['sequence_id'].isin(target_seqs)]

        if subset_df.empty:
            logger.error("No data found for Phase A.")
            return

        logger.info(f"Tuning on {len(subset_df)} samples from {len(target_seqs)} sequences.")

        # Run Pipeline with Tuning
        output_dir = EXPERIMENT_DIR / "phase_a_tuning"
        pipeline = PianoMotionMLPipeline(dataframe=subset_df)
        pipeline.load_and_prepare_data()

        # This runs RandomizedSearchCV inside
        pipeline.run_pipeline(output_dir=output_dir, skip_svm=True)

        # Capture Best Params from the trained model
        rf_model = pipeline.models.get('Random Forest')
        if rf_model:
            all_params = rf_model.get_params()
            # Filter to relevant RF params
            keys = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
            best_params = {k: all_params[k] for k in keys if k in all_params}

            logger.info(f"🏆 Best Params Found: {best_params}")

            self.state['best_params'] = best_params
            self.state['phase'] = 'B'
            self.state_manager.save(self.state)
        else:
            logger.error("Failed to retrieve RF model from pipeline.")

    def run_phase_b_scaling(self):
        """
        Phase B: Scaling.
        Loop n_files from 10 to TOTAL (Step=5).
        Train RF using fixed best_params.
        Accumulate results.
        """
        logger.info("\n" + "="*60)
        logger.info("📈 PHASE B: Scaling (Step 5 Files)")
        logger.info("="*60)

        best_params = self.state.get('best_params')
        if not best_params:
            logger.warning("⚠️ No best_params found. Using defaults.")
            best_params = {}

        total_files = len(self.unique_sequences)

        # Loop from 10 to Total, step 5
        for n_files in range(10, total_files + 1, 5):
            if n_files in self.state['completed_steps']:
                logger.info(f"⏩ Skipping completed step: {n_files} files")
                continue

            target_seqs = self.unique_sequences[:n_files]
            subset_df = self.df[self.df['sequence_id'].isin(target_seqs)]
            n_samples = len(subset_df)

            logger.info(f"\n--- Training Step: {n_files} files ({n_samples} samples) ---")

            # Run Pipeline with Fixed Params
            step_dir = EXPERIMENT_DIR / f"step_b_{n_files}"
            pipeline = PianoMotionMLPipeline(dataframe=subset_df)
            pipeline.load_and_prepare_data()

            # Train
            pipeline.run_pipeline(
                output_dir=step_dir,
                fixed_rf_params=best_params,
                skip_svm=True
            )

            # Log Metrics
            metrics = pipeline.results.get('Random Forest', {})
            result_row = {
                'n_files': n_files,
                'n_samples': n_samples,
                'f1_score': metrics.get('f1_score', 0),
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'fps': metrics.get('fps', 0)
            }

            # Append to CSV
            results_df = pd.DataFrame([result_row])
            write_header = not RESULTS_CSV.exists()
            results_df.to_csv(RESULTS_CSV, mode='a', header=write_header, index=False)
            logger.info(f"📝 Results appended to {RESULTS_CSV}")

            # Update State
            self.state['completed_steps'].append(n_files)
            self.state_manager.save(self.state)

        logger.info("✅ Phase B Complete.")
        self.state['phase'] = 'DONE'
        self.state_manager.save(self.state)

    def run(self):
        logger.info("🚀 Starting Experiment Controller")

        if self.state['phase'] == 'DONE':
            logger.info("✅ Experiment previously finished. Delete state file to restart.")
            return

        if self.state['phase'] == 'A':
            self.run_phase_a_tuning()

        if self.state['phase'] == 'B':
            self.run_phase_b_scaling()

        logger.info("🏁 Controller Finished.")

if __name__ == "__main__":
    controller = ExperimentController()
    controller.run()
