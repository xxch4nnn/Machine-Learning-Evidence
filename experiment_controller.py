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
import joblib
from pathlib import Path
from typing import Dict, List, Optional

# Add repo root to path
repo_root = Path(__file__).parent.resolve()
sys.path.append(str(repo_root))

from Machine_Learning_Course.Code.Data_Pipeline.ML_Pipeline_Prep import PianoMotionMLPipeline
from Machine_Learning_Course.Code.Data_Pipeline.SyncPianoMotionDataset import SyncPianoMotionDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = repo_root / "Machine_Learning_Course" / "Data" / "PianoMotion10M"
EXPERIMENT_DIR = DATA_DIR / "experiments_incremental"
STATE_FILE = EXPERIMENT_DIR / "state.json"
MASTER_FEATURES_CSV = EXPERIMENT_DIR / "features_accumulated.csv"
RESULTS_CSV = EXPERIMENT_DIR / "incremental_results.csv"

MAX_SAMPLES = 100000

class CheckpointManager:
    """
    Manages the state of the experiment to allow resumption after crashes.
    """
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.experiment_dir = state_file.parent
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> Dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"🔄 Loaded state: Phase {state.get('phase', 'INIT')}, {len(state.get('processed_sequences', []))} sequences processed.")
                return state
            except Exception as e:
                logger.error(f"❌ Corrupt state file: {e}")

        # Default State
        return {
            "phase": "A", # A = Tuning, B = Scaling, DONE = Finished
            "processed_sequences": [], # List of sequence IDs completed
            "best_params": None,
            "results_history": [],
            "sample_count": 0
        }

    def save_state(self, state: Dict):
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
        self.checkpoint_manager = CheckpointManager(STATE_FILE)
        self.state = self.checkpoint_manager.load_state()
        self.dataset_engine = SyncPianoMotionDataset()

        # Load accumulated data if exists
        if MASTER_FEATURES_CSV.exists():
            self.cumulative_df = pd.read_csv(MASTER_FEATURES_CSV)
            logger.info(f"📚 Loaded accumulated dataset: {len(self.cumulative_df)} samples.")
        else:
            self.cumulative_df = pd.DataFrame()

    def get_remaining_sequences(self) -> List[Dict]:
        """
        Returns list of sequence metadata that have NOT been processed yet.
        """
        all_sequences = self.dataset_engine.parser.list_sequences()
        processed_set = set(self.state['processed_sequences'])

        remaining = [s for s in all_sequences if s['sequence'] not in processed_set]
        return remaining

    def run_phase_a_tuning(self):
        """
        Phase A: Process first 10 files (1 by 1).
        Run GridSearch on accumulated data to find best params.
        """
        logger.info("\n" + "="*60)
        logger.info("🔬 PHASE A: Tuning (Files 1-10)")
        logger.info("="*60)

        target_count = 10
        current_processed_count = len(self.state['processed_sequences'])

        if current_processed_count >= target_count:
            logger.info("✅ Phase A already complete.")
            if self.state['phase'] == 'A':
                self.state['phase'] = 'B'
                self.checkpoint_manager.save_state(self.state)
            return

        # We need to process up to 10 files
        # We fetch them one by one
        remaining = self.get_remaining_sequences()

        # Limit to needed amount for Phase A
        needed = target_count - current_processed_count
        to_process = remaining[:needed]

        # We manually iterate because yield_batch fetches dynamically
        # But here we want specific control.
        # Actually, let's use a helper that re-uses yield_batch logic but for specific list

        batch_gen = self._manual_batch_generator(to_process, batch_size=1)

        best_params_history = []

        for batch_df in batch_gen:
            if batch_df.empty: continue

            # Accumulate
            self.cumulative_df = pd.concat([self.cumulative_df, batch_df], ignore_index=True)
            self.cumulative_df.to_csv(MASTER_FEATURES_CSV, index=False)

            # Update state variables (locally)
            new_seqs = batch_df['sequence_id'].unique().tolist()
            self.state['processed_sequences'].extend(new_seqs)
            self.state['sample_count'] = len(self.cumulative_df)

            logger.info(f"--- Training Step A (Seqs: {len(self.state['processed_sequences'])}) ---")

            # Run Pipeline with Tuning
            pipeline = PianoMotionMLPipeline(dataframe=self.cumulative_df)
            pipeline.load_and_prepare_data()

            # Select features (RFE once or every time? Plan implies every time for tuning, or fixed?
            # Controller says: "Run GridSearchCV... Save best params")
            # We'll run full pipeline.
            output_dir = EXPERIMENT_DIR / f"step_a_{len(self.state['processed_sequences'])}"
            pipeline.run_pipeline(output_dir=output_dir, skip_svm=True)

            # Capture Params
            rf_model = pipeline.models['Random Forest']
            best_params = rf_model.get_params()
            relevant_keys = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
            filtered_params = {k: best_params[k] for k in relevant_keys}

            self.state['best_params'] = filtered_params

            # Save State
            self.checkpoint_manager.save_state(self.state)

        # Transition to Phase B
        self.state['phase'] = 'B'
        self.checkpoint_manager.save_state(self.state)
        logger.info("✅ Phase A Complete. Transitioning to Phase B.")

    def run_phase_b_scaling(self):
        """
        Phase B: Process remaining files in batches of 5.
        Stop when total samples > 100k.
        Use Fixed Best Params from Phase A.
        """
        logger.info("\n" + "="*60)
        logger.info("📈 PHASE B: Scaling (Batch Size 5)")
        logger.info("="*60)

        if self.state['sample_count'] >= MAX_SAMPLES:
            logger.info("🛑 Sample limit reached. Experiment complete.")
            self.state['phase'] = 'DONE'
            self.checkpoint_manager.save_state(self.state)
            return

        remaining = self.get_remaining_sequences()
        if not remaining:
            logger.info("✅ All available files processed.")
            self.state['phase'] = 'DONE'
            self.checkpoint_manager.save_state(self.state)
            return

        batch_gen = self._manual_batch_generator(remaining, batch_size=5)

        fixed_params = self.state.get('best_params')
        if not fixed_params:
            logger.warning("⚠️ No best params found from Phase A. Using defaults.")
            fixed_params = {}

        for batch_df in batch_gen:
            # Accumulate
            self.cumulative_df = pd.concat([self.cumulative_df, batch_df], ignore_index=True)
            self.cumulative_df.to_csv(MASTER_FEATURES_CSV, index=False)

            # Update State
            new_seqs = batch_df['sequence_id'].unique().tolist()
            self.state['processed_sequences'].extend(new_seqs)
            self.state['sample_count'] = len(self.cumulative_df)

            n_seqs = len(self.state['processed_sequences'])
            logger.info(f"\n--- Training Step B (Seqs: {n_seqs}, Samples: {self.state['sample_count']}) ---")

            # Run Pipeline (Fixed Params)
            pipeline = PianoMotionMLPipeline(dataframe=self.cumulative_df)
            # Use cached selected features if available to speed up?
            # ML Pipeline loads from file if passed.
            # We'll just let it run. RFE might be slow on large data.
            # Optimization: Load selected features from Phase A if possible.
            # We will rely on ML Pipeline's RFE (it's fast enough for <100k samples usually, or we could pass list)

            output_dir = EXPERIMENT_DIR / f"step_b_{n_seqs}"
            pipeline.run_pipeline(output_dir=output_dir, fixed_rf_params=fixed_params, skip_svm=True)

            # Log Metrics
            metrics = pipeline.results['Random Forest']
            result_row = {
                'phase': 'B',
                'n_sequences': n_seqs,
                'n_samples': self.state['sample_count'],
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'fps': metrics['fps']
            }
            self.state['results_history'].append(result_row)

            # Append to CSV
            results_df = pd.DataFrame([result_row])
            write_header = not RESULTS_CSV.exists()
            results_df.to_csv(RESULTS_CSV, mode='a', header=write_header, index=False)

            # Save State
            self.checkpoint_manager.save_state(self.state)

            # Check Stop Condition
            if self.state['sample_count'] >= MAX_SAMPLES:
                logger.info(f"🛑 limit reached ({self.state['sample_count']} >= {MAX_SAMPLES}). Stopping.")
                self.state['phase'] = 'DONE'
                self.checkpoint_manager.save_state(self.state)
                break

    def _manual_batch_generator(self, sequence_list: List[Dict], batch_size: int):
        """
        Helper to yield batches from a specific list of sequence metadata.
        Uses the Dataset Engine's internal logic logic but bypasses its own list_sequences().
        """
        current_batch = []

        for seq_meta in sequence_list:
            try:
                # Reuse logic by temporarily pointing dataset to file?
                # Or just duplicate the loading logic here?
                # Better: call a method on engine.
                # I'll expose a `process_single_sequence` method in engine or just replicate code?
                # The engine code is in SyncPianoMotionDataset.
                # I can modify SyncPianoMotionDataset to accept a list of sequences to process,
                # OR just manually load here using the engine's methods.
                # But `load_midi_labels` and `extract_features` are methods.

                # Let's read the file manually and call `extract_features`
                with open(seq_meta['pose_path'], 'r') as f:
                    raw = json.load(f)
                    if 'right' in raw: data = raw['right']
                    elif 'left' in raw: data = raw['left']
                    elif isinstance(raw, list): data = raw
                    else: continue

                frames = []
                for fr in data:
                    if len(fr) == 63: frames.append(np.array(fr).reshape(21, 3))
                    elif len(fr) == 62: frames.append(np.array(fr + [0]).reshape(21, 3))

                points_3d = np.array(frames)
                note_events = self.dataset_engine.load_midi_labels(seq_meta['midi_path'])

                df = self.dataset_engine.extract_features(points_3d, note_events, seq_meta['sequence'])

                if not df.empty:
                    current_batch.append(df)

            except Exception as e:
                logger.warning(f"Failed to process {seq_meta['sequence']}: {e}")

            if len(current_batch) >= batch_size:
                yield pd.concat(current_batch, ignore_index=True)
                current_batch = []

        if current_batch:
            yield pd.concat(current_batch, ignore_index=True)

    def run(self):
        logger.info("🚀 Starting Experiment Controller")

        if self.state['phase'] == 'DONE':
            logger.info("✅ Experiment marked as DONE. Delete state.json to restart.")
            return

        if self.state['phase'] == 'A':
            self.run_phase_a_tuning()

        if self.state['phase'] == 'B':
            self.run_phase_b_scaling()

        logger.info("🏁 Controller Finished.")

if __name__ == "__main__":
    controller = ExperimentController()
    controller.run()
