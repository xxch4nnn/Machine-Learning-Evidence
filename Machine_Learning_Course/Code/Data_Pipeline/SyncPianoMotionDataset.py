"""
SyncPianoMotionDataset.py
The Unified Data Engine for PianoMotion10M.
Handles Downloading, Parsing, 3D-to-2D Projection, Feature Extraction, and Batch Generation.
"""

import os
import sys
import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from tqdm import tqdm
import mido
from scipy.signal import savgol_filter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Generator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Downloader & Parser Logic (Merged from DownloadRealPianoMotion10M.py) ---

class PianoMotion10MDownloader:
    """
    Downloads and manages the PianoMotion10M dataset.
    """
    GITHUB_ZIP = "https://zenodo.org/records/13297386/files/annotation.zip?download=1"
    MIDI_ZIP_URL = "https://zenodo.org/records/13297386/files/midi.zip?download=1"

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.data_dir = self.output_dir / "data"

    def download(self) -> bool:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if data exists
        if self.data_dir.exists() and any(self.data_dir.iterdir()):
            logger.info("✅ Dataset directory exists. Skipping download.")
            return True

        zip_files = {
            "annotation.zip": self.GITHUB_ZIP,
            "midi.zip": self.MIDI_ZIP_URL,
        }

        try:
            for zip_name, url in zip_files.items():
                zip_path = self.output_dir / zip_name
                if not zip_path.exists():
                    logger.info(f"Downloading {zip_name} from {url}...")
                    urllib.request.urlretrieve(url, zip_path)

                logger.info(f"Extracting {zip_name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.output_dir)

                zip_path.unlink() # Cleanup
            return True
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False

class PianoMotion10MParser:
    """
    Parses the dataset directory structure.
    """
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.data_dir = dataset_dir / "data"

    def list_sequences(self) -> List[Dict]:
        """
        Scans the data directory and returns a list of all available sequence metadata.
        Returns: List of dicts {'subject': str, 'sequence': str, 'paths': {...}}
        """
        sequences = []
        if not self.data_dir.exists():
            return sequences

        # Walk through subjects
        for subject_dir in self.data_dir.iterdir():
            if not subject_dir.is_dir() or subject_dir.name == 'midi':
                continue

            # Handle nested structure (e.g., audio-002/audio/seq_id)
            search_roots = [subject_dir]
            if (subject_dir / 'audio').exists():
                search_roots.append(subject_dir / 'audio')

            for root in search_roots:
                for seq_dir in root.iterdir():
                    if seq_dir.is_dir():
                        # Validate it has pose data
                        pose_files = list(seq_dir.glob("*.json")) + list(seq_dir.glob("*.npz"))
                        if not pose_files:
                            continue

                        # Find matching MIDI
                        # MIDI is often in data/midi/midi/{seq_id}/*.mid
                        midi_dir = self.data_dir / 'midi' / 'midi' / seq_dir.name
                        midi_files = list(midi_dir.glob("*.mid"))

                        # Fallback to local
                        if not midi_files:
                            midi_files = list(seq_dir.glob("*.mid"))

                        if pose_files and midi_files:
                            sequences.append({
                                'subject': subject_dir.name,
                                'sequence': seq_dir.name,
                                'pose_path': pose_files[0],
                                'midi_path': midi_files[0],
                                'annotation_path': list(seq_dir.glob("*annotation*.json"))[0] if list(seq_dir.glob("*annotation*.json")) else None
                            })

        return sorted(sequences, key=lambda x: x['sequence'])


# --- 2. The Main Data Engine Class ---

class SyncPianoMotionDataset:
    """
    Unified Data Engine.
    1. Downloads/Parses Data.
    2. Projects 3D -> 2D using Camera Intrinsics.
    3. Extracts 2D Physics Features.
    4. Yields Batches for Incremental Learning.
    """

    # Camera Intrinsics (Hardcoded Defaults)
    CAM_W = 1920
    CAM_H = 1080
    FX = 1000
    FY = 1000
    CX = 960
    CY = 540

    def __init__(self, dataset_dir: str = None, fps: float = 30.0):
        if dataset_dir is None:
             # Default to repo structure
             self.dataset_dir = Path(__file__).parent.parent.parent / "Data" / "PianoMotion10M"
        else:
            self.dataset_dir = Path(dataset_dir)

        self.fps = fps
        self.frame_duration = 1.0 / fps

        # Initialize sub-components
        self.downloader = PianoMotion10MDownloader(self.dataset_dir)
        self.parser = PianoMotion10MParser(self.dataset_dir)

    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Projects 3D points (x, y, z) to Normalized 2D screen coordinates (u_norm, v_norm).
        Uses pinhole camera model.

        Args:
            points_3d: (N, 21, 3) array of 3D coordinates.

        Returns:
            (N, 21, 2) array of normalized 2D coordinates [0, 1].
        """
        # Unpack
        x = points_3d[..., 0]
        y = points_3d[..., 1]
        z = points_3d[..., 2]

        # Avoid division by zero
        z = np.where(z == 0, 1e-6, z)

        # Pinhole Projection
        u = (x / z) * self.FX + self.CX
        v = (y / z) * self.FY + self.CY

        # Normalize
        u_norm = u / self.CAM_W
        v_norm = v / self.CAM_H

        return np.stack([u_norm, v_norm], axis=-1)

    def load_midi_labels(self, midi_file: Path) -> Dict[int, int]:
        """
        Parses MIDI to get per-frame binary labels (Pressed=1, Hover/Release=0).
        Logic adapted to match 'Press' state priority.
        """
        frame_labels = {}
        try:
            mid = mido.MidiFile(str(midi_file))
            tempo = 500000

            # Get tempo
            for msg in mido.merge_tracks(mid.tracks):
                if msg.is_meta and msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break

            # Re-implementation of robust label loading (Time-based)
            # Returns dict: {frame_idx: label}
            # Simplified for robustness:
            # 0=Hover, 1=Press (First 3 frames), 2=Hold, 3=Release (Last 3 frames)

            # Reset
            time_sec = 0.0
            active_notes = {} # note -> start_time
            notes_log = [] # (note, start, end)

            for msg in mido.merge_tracks(mid.tracks):
                time_sec += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = time_sec
                elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start = active_notes.pop(msg.note)
                        end = time_sec
                        notes_log.append((msg.note, start, end))

            return notes_log

        except Exception as e:
            logger.error(f"MIDI Error {midi_file}: {e}")
            return []

    def extract_features(self, kinematics_3d: np.ndarray, note_events: List, seq_id: str) -> pd.DataFrame:
        """
        Extracts 2D Features from 3D Kinematics.
        Labels using Group & Rank heuristic.
        """
        if kinematics_3d is None or len(kinematics_3d) == 0:
            return pd.DataFrame()

        num_frames = len(kinematics_3d)

        # 1. Project 3D -> 2D
        points_2d = self.project_3d_to_2d(kinematics_3d) # (Frames, 21, 2)

        # 2. Smooth 2D points
        points_2d = savgol_filter(points_2d, window_length=5, polyorder=2, axis=0)

        # 3. Calculate Derivatives (2D)
        # Gradient returns list [grad_axis0, grad_axis1...], we want axis 0 (time)
        vel_2d = np.gradient(points_2d, axis=0) / self.frame_duration
        acc_2d = np.gradient(vel_2d, axis=0) / self.frame_duration

        # 4. Prepare Landmarks
        # Indices: Wrist=0, Index=8, Middle=12 (Palm Proxy for simplified logic? Original used 9)
        # Using 9 (Middle MCP) as Palm Center Proxy
        wrist_pos = points_2d[:, 0, :]
        palm_pos = points_2d[:, 9, :]
        wrist_vel = vel_2d[:, 0, :]

        fingertip_indices = [4, 8, 12, 16, 20] # Thumb to Pinky
        dip_indices = [3, 7, 11, 15, 19]

        # Calculate Relative Depth (Normalized Z) - before projection logic loss
        # Z is index 2. We use the raw kinematics_3d which are normalized/centered in dataset.
        # rel_depth = tip.z - wrist.z
        wrist_z_3d = kinematics_3d[:, 0, 2]

        all_rows = []

        # Labeling Preparation
        labels = np.zeros((num_frames, 5), dtype=int)

        # Group notes by start time (Chord grouping)
        # Sort by start time
        note_events.sort(key=lambda x: x[1])

        # Simple grouping: events within 66ms (2 frames)
        groups = []
        if note_events:
            curr_group = [note_events[0]]
            for i in range(1, len(note_events)):
                if (note_events[i][1] - curr_group[0][1]) < 0.07: # ~2 frames at 30fps
                    curr_group.append(note_events[i])
                else:
                    groups.append(curr_group)
                    curr_group = [note_events[i]]
            groups.append(curr_group)

        # Assign labels
        for group in groups:
            # Time to frame
            start_t = np.mean([n[1] for n in group])
            start_f = int(start_t * self.fps)

            if start_f >= num_frames: continue

            # Determine which fingers pressed based on Z-velocity (3D)
            # We use 3D Z-velocity just for the labeling heuristic (Internal Logic)

            # Calculate 3D Z-vel for labeling only
            z_vals = kinematics_3d[:, :, 2]
            z_vel = np.gradient(z_vals, axis=0)

            finger_z_vels = []
            for f_idx in fingertip_indices:
                finger_z_vels.append(z_vel[start_f, f_idx])

            # Rank: Most negative (downward) first
            ranked_indices = np.argsort(finger_z_vels) # Ascending (neg -> pos)

            # Assign
            for i, note_tuple in enumerate(group):
                if i < 5:
                    f_real_idx = ranked_indices[i] # 0..4

                    # Label frames
                    n_start = int(note_tuple[1] * self.fps)
                    n_end = int(note_tuple[2] * self.fps)
                    duration = n_end - n_start

                    # 4-State Logic
                    # 1=Press (3 frames), 2=Hold, 3=Release (3 frames)
                    if duration > 6:
                        labels[n_start:n_start+3, f_real_idx] = 1
                        labels[n_start+3:n_end-3, f_real_idx] = 2
                        labels[n_end-3:n_end, f_real_idx] = 3
                    else:
                        mid = n_start + (duration // 2)
                        labels[n_start:mid, f_real_idx] = 1
                        labels[mid:n_end, f_real_idx] = 3

        # Feature Construction
        for f in range(num_frames):
            for i, (tip_idx, dip_idx) in enumerate(zip(fingertip_indices, dip_indices)):
                row = {}

                # Base vectors
                tip_p = points_2d[f, tip_idx]
                tip_v = vel_2d[f, tip_idx]
                tip_a = acc_2d[f, tip_idx]
                wrist_p = wrist_pos[f]
                wrist_v = wrist_vel[f]
                palm_p = palm_pos[f]
                dip_p = points_2d[f, dip_idx]

                # --- NEW 2D FEATURE SET ---

                # 1. Position (Normalized)
                row['finger_pos_x'] = tip_p[0]
                row['finger_pos_y'] = tip_p[1]
                row['wrist_pos_x'] = wrist_p[0]
                row['wrist_pos_y'] = wrist_p[1]

                # 2. Velocity (Normalized/sec)
                row['finger_vel_x'] = tip_v[0]
                row['finger_vel_y'] = tip_v[1]
                row['finger_speed'] = np.linalg.norm(tip_v)

                row['wrist_vel_x'] = wrist_v[0]
                row['wrist_vel_y'] = wrist_v[1]
                row['wrist_speed'] = np.linalg.norm(wrist_v)

                # 3. Acceleration
                row['finger_acc_x'] = tip_a[0]
                row['finger_acc_y'] = tip_a[1]
                row['finger_acc_mag'] = np.linalg.norm(tip_a)

                # 4. Relative (Tip - Wrist)
                rel_pos = tip_p - wrist_p
                rel_vel = tip_v - wrist_v
                row['rel_finger_pos_x'] = rel_pos[0]
                row['rel_finger_pos_y'] = rel_pos[1]
                row['rel_finger_vel_x'] = rel_vel[0]
                row['rel_finger_vel_y'] = rel_vel[1]

                # 5. Distances
                row['dist_wrist'] = np.linalg.norm(rel_pos)
                row['dist_palm'] = np.linalg.norm(tip_p - palm_p)
                row['posture_dist'] = np.linalg.norm(tip_p - dip_p) # Tip to DIP

                # 6. Relative Depth (New for Phase 2.5)
                # Use raw 3D Z difference: tip.z - wrist.z
                # This matches MediaPipe's wrist-relative landmark.z behavior
                tip_z_3d = kinematics_3d[f, tip_idx, 2]
                row['rel_depth'] = tip_z_3d - wrist_z_3d[f]

                # 7. Rolling Averages (Last 5 frames)
                s_idx = max(0, f-4)
                row['avg_speed'] = np.mean(np.linalg.norm(vel_2d[s_idx:f+1, tip_idx], axis=1))
                row['avg_acc_mag'] = np.mean(np.linalg.norm(acc_2d[s_idx:f+1, tip_idx], axis=1))

                # 8. Lags (Speed)
                for lag in [2, 4, 6]:
                    if f >= lag:
                        l_idx = f - lag
                        row[f'lag_speed_{lag}'] = np.linalg.norm(vel_2d[l_idx, tip_idx])
                    else:
                        row[f'lag_speed_{lag}'] = 0.0

                # 9. Rolling Variance (Stability)
                if f > 4:
                    row['rolling_var_speed'] = np.var(np.linalg.norm(vel_2d[s_idx:f+1, tip_idx], axis=1))
                else:
                    row['rolling_var_speed'] = 0.0

                # Meta
                row['sequence_id'] = seq_id
                row['ground_truth_label'] = labels[f, i]

                all_rows.append(row)

        return pd.DataFrame(all_rows)


    def yield_batch(self, batch_size: int = 5) -> Generator[pd.DataFrame, None, None]:
        """
        Generator that yields DataFrames of features in batches of files.
        """
        # Ensure data is ready
        self.downloader.download()

        sequences = self.parser.list_sequences()
        logger.info(f"Found {len(sequences)} sequences.")

        current_batch = []

        for seq_meta in sequences:
            # Load Data
            try:
                # Load JSON
                with open(seq_meta['pose_path'], 'r') as f:
                    raw = json.load(f)
                    # Handle structure variants
                    if 'right' in raw:
                        data = raw['right']
                    elif 'left' in raw:
                        data = raw['left']
                    elif isinstance(raw, list):
                         data = raw
                    else:
                        continue # Skip unknown format

                # Fix shape
                frames = []
                for fr in data:
                    if len(fr) == 63: frames.append(np.array(fr).reshape(21, 3))
                    elif len(fr) == 62: frames.append(np.array(fr + [0]).reshape(21, 3))

                points_3d = np.array(frames)

                # Load MIDI
                note_events = self.load_midi_labels(seq_meta['midi_path'])

                # Extract
                df = self.extract_features(points_3d, note_events, seq_meta['sequence'])

                if not df.empty:
                    current_batch.append(df)

            except Exception as e:
                logger.warning(f"Failed to process {seq_meta['sequence']}: {e}")
                continue

            # Yield if batch full
            if len(current_batch) >= batch_size:
                yield pd.concat(current_batch, ignore_index=True)
                current_batch = []

        # Yield remaining
        if current_batch:
            yield pd.concat(current_batch, ignore_index=True)

    def run(self, max_files=None, output_csv="features.csv"):
        """
        Legacy entry point: Processes all (or max) files and saves CSV.
        """
        all_dfs = []
        gen = self.yield_batch(batch_size=5)

        count = 0
        try:
            for df_batch in gen:
                all_dfs.append(df_batch)
                count += len(df_batch['sequence_id'].unique())
                if max_files and count >= max_files:
                    break
        except StopIteration:
            pass

        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            out_path = self.dataset_dir / output_csv
            final_df.to_csv(out_path, index=False)
            logger.info(f"Saved {len(final_df)} samples to {out_path}")
        else:
            logger.warning("No data generated.")

if __name__ == "__main__":
    # Test Run
    dataset = SyncPianoMotionDataset()
    dataset.run(max_files=2)
