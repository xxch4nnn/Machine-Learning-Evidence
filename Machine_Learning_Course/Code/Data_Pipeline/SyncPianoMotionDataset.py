"""
SyncPianoMotionDataset.py
Synchronizes 3D hand kinematics from JSON files with MIDI keypress data.
Extracts features for all five fingers of the right hand and labels key presses
based on a heuristic that identifies the finger with the highest downward velocity
at the time of a MIDI note_on event.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from tqdm import tqdm
import mido
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SyncPianoMotionDataset:
    """
    Processes the PianoMotion10M dataset to create a synchronized feature set for
    machine learning.
    """
    def __init__(self, dataset_dir: str, fps: float = 30.0):
        """
        Initializes the data processor.

        Args:
            dataset_dir: Path to the PianoMotion10M dataset directory.
            fps: Frames per second of the motion capture data.
        """
        self.dataset_dir = Path(dataset_dir)
        self.fps = fps
        self.frame_duration = 1.0 / fps

    def load_midi_labels(self, midi_file: Path) -> dict:
        """
        Loads a MIDI file and extracts note press events with their start and end frames.

        Args:
            midi_file: Path to the MIDI file.

        Returns:
            A dictionary where keys are note numbers and values are lists of
            (start_frame, end_frame) tuples.
        """
        note_events = {}
        try:
            midi = mido.MidiFile(str(midi_file))
            ticks_per_beat = midi.ticks_per_beat or 480
            tempo = 500000  # Default tempo (120 BPM)

            # Find the first tempo change event
            for msg in mido.merge_tracks(midi.tracks):
                if msg.is_meta and msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break

            time_in_seconds = 0.0
            open_notes = {}

            for msg in mido.merge_tracks(midi.tracks):
                delta_ticks = msg.time
                time_in_seconds += mido.tick2second(delta_ticks, ticks_per_beat, tempo)

                if msg.type == 'note_on' and msg.velocity > 0:
                    open_notes[msg.note] = time_in_seconds
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in open_notes:
                        start_time = open_notes.pop(msg.note)
                        end_time = time_in_seconds
                        start_frame = int(start_time * self.fps)
                        end_frame = int(end_time * self.fps)

                        if msg.note not in note_events:
                            note_events[msg.note] = []
                        note_events[msg.note].append((start_frame, end_frame))

        except Exception as e:
            logger.error(f"Could not process MIDI file {midi_file}: {e}")
        return note_events

    def load_kinematics(self, kinematics_file: Path) -> np.ndarray:
        """
        Loads 3D hand kinematics from a JSON annotation file.

        Args:
            kinematics_file: Path to the kinematics JSON file.

        Returns:
            A numpy array of shape (frames, 21, 3) representing hand joint coordinates.
        """
        try:
            with kinematics_file.open('r') as f:
                data = json.load(f)

            # Accommodate both 'right' and 'left' hand data
            hand_data = data.get('right') or data.get('left')
            if not hand_data:
                return None

            # Process frames, padding if necessary
            processed_frames = []
            for frame_data in hand_data:
                if len(frame_data) == 62:
                    frame_data.append(0)  # Pad to 63 for 21 joints

                if len(frame_data) == 63:
                    processed_frames.append(np.array(frame_data).reshape(21, 3))
                else:
                    # Handle empty or malformed frames
                    processed_frames.append(np.zeros((21, 3)))

            return np.array(processed_frames)

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading or parsing kinematics file {kinematics_file}: {e}")
            return None

    def _group_chords(self, note_events: dict, window_frames: int = 2) -> list:
        """
        Groups MIDI note events that occur within a small time window (e.g., +/- 2 frames)
        to identify chords.

        Args:
            note_events: Dict of {note: [(start, end), ...]}
            window_frames: Time window in frames to group simultaneous notes.

        Returns:
            List of dicts: [{'start': frame, 'end': frame, 'notes': [note1, note2...]}, ...]
        """
        # Flatten all events into a list of (start_frame, end_frame, note)
        all_events = []
        for note, events in note_events.items():
            for start, end in events:
                all_events.append({'start': start, 'end': end, 'note': note})

        # Sort by start frame
        all_events.sort(key=lambda x: x['start'])

        if not all_events:
            return []

        chord_groups = []
        current_group = [all_events[0]]

        for i in range(1, len(all_events)):
            event = all_events[i]
            prev_event = current_group[-1]

            # Check if event starts within window of the first event in the current group
            # Using the first event helps anchor the chord window
            if abs(event['start'] - current_group[0]['start']) <= window_frames:
                current_group.append(event)
            else:
                # Process completed group
                group_data = {
                    'start': int(np.mean([e['start'] for e in current_group])), # Average start frame
                    'notes': [e['note'] for e in current_group],
                    # Store individual note durations if needed, but for labeling we might just need to know which fingers
                    # For simplicity, we'll store the list of events to access individual end times later if needed
                    'events': current_group
                }
                chord_groups.append(group_data)
                current_group = [event]

        # Append last group
        if current_group:
            group_data = {
                'start': int(np.mean([e['start'] for e in current_group])),
                'notes': [e['note'] for e in current_group],
                'events': current_group
            }
            chord_groups.append(group_data)

        return chord_groups

    def extract_and_label_features(self, kinematics: np.ndarray, note_events: dict) -> list:
        """
        Extracts 26 features for all fingers and labels them based on MIDI events.
        Includes chord grouping logic and relative position features.
        """
        from scipy.signal import savgol_filter

        # Apply Savitzky-Golay filter for smoothing
        kinematics = savgol_filter(kinematics, window_length=5, polyorder=2, axis=0)

        all_features = []
        num_frames = kinematics.shape[0]
        labels = np.zeros((num_frames, 5))  # Labels for 5 fingers: Thumb(0) to Pinky(4)
        fingertip_indices = [4, 8, 12, 16, 20]  # Landmarks for fingertips

        # --- 1. Label Assignment (Chord Aware) ---
        chord_groups = self._group_chords(note_events, window_frames=2)

        for chord in chord_groups:
            start_frame = chord['start']
            notes = chord['notes']
            events = chord['events']

            if 0 < start_frame < num_frames:
                # Get downward velocities (Z-axis) for all 5 fingers at the onset
                velocities = []
                for finger_idx in fingertip_indices:
                    # Use immediate velocity at onset
                    pos_current = kinematics[start_frame, finger_idx]
                    pos_prev = kinematics[start_frame - 1, finger_idx]
                    velocity = (pos_current - pos_prev) / self.frame_duration
                    velocities.append(velocity[2]) # Z-velocity

                # Rank fingers by velocity (most negative/downward is highest priority)
                # np.argsort returns indices that would sort the array.
                # We want smallest (most negative) first.
                ranked_fingers = np.argsort(velocities)

                # Assign each note to the next-ranked finger
                # If more notes than fingers, excess notes are ignored (limitation of hand size)
                for i, event in enumerate(events):
                    if i < len(ranked_fingers):
                        finger_idx = ranked_fingers[i] # 0-4 index for labels
                        note_start = event['start']
                        note_end = event['end']

                        # Clamp frames to be within the valid range of the labels array
                        note_start = max(0, min(note_start, num_frames - 1))
                        note_end = max(0, min(note_end, num_frames))

                        # Skip if the note is invalid or has zero duration
                        if note_start >= note_end:
                            continue

                        duration = note_end - note_start
                        press_window = 3
                        release_window = 3

                        # --- 4-State Labeling Logic ---
                        if duration >= (press_window + release_window):
                            # Case 1: Long note with Press, Hold, and Release states
                            # State 1: Press
                            labels[note_start : note_start + press_window, finger_idx] = 1
                            # State 2: Hold
                            labels[note_start + press_window : note_end - release_window, finger_idx] = 2
                            # State 3: Release
                            labels[note_end - release_window : note_end, finger_idx] = 3
                        else:
                            # Case 2: Short note, prioritize Press then Release
                            # Calculate midpoint, giving extra frame to Press for odd durations
                            midpoint = int(np.ceil(duration / 2.0))

                            # State 1: Press
                            labels[note_start : note_start + midpoint, finger_idx] = 1
                            # State 3: Release
                            labels[note_start + midpoint : note_end, finger_idx] = 3

        # --- 2. Feature Extraction (Vectorized) ---
        # Pre-calculate all raw values to support rolling averages and relative features

        # Landmarks
        wrist = kinematics[:, 0, :] # (Frames, 3)
        palm_center = kinematics[:, 9, :] # (Frames, 3) - Using Middle Finger MCP as proxy

        # Calculate basic derivatives (Absolute)
        # Pad with first frame to keep shape
        velocities = np.gradient(kinematics, axis=0) / self.frame_duration # (Frames, 21, 3)
        accelerations = np.gradient(velocities, axis=0) / self.frame_duration # (Frames, 21, 3)

        # Wrist kinematics (Absolute)
        wrist_vel = velocities[:, 0, :] # (Frames, 3)

        # Process each finger
        # finger_idx corresponds to 0..4 (Thumb..Pinky)
        # tip_idx is the MediaPipe landmark index
        dip_indices = [3, 7, 11, 15, 19] # DIP joints (IP for thumb)

        for frame_idx in range(num_frames):
            # Skip first few frames if needed for rolling window validness,
            # but typically we just handle edge cases or accept noisy starts.
            # The ML pipeline often expects data from frame 0 or 1.

            for finger_i, (tip_idx, dip_idx) in enumerate(zip(fingertip_indices, dip_indices)):
                features = {}

                # -- Raw Vectors --
                tip_pos_abs = kinematics[frame_idx, tip_idx]
                tip_vel_abs = velocities[frame_idx, tip_idx]
                tip_acc_abs = accelerations[frame_idx, tip_idx]

                wrist_pos_abs = wrist[frame_idx]
                wrist_vel_abs = wrist_vel[frame_idx]

                # -- Relative Calculation (Task A2) --
                # Position relative to wrist
                rel_pos = tip_pos_abs - wrist_pos_abs

                # Relative Velocity (Finger Vel - Wrist Vel)
                rel_vel = tip_vel_abs - wrist_vel_abs

                # -- Feature Mapping to 26 Columns --

                # 1. Finger Velocity (Absolute)
                features['finger_velocity_x'] = tip_vel_abs[0]
                features['finger_velocity_y'] = tip_vel_abs[1]
                features['finger_velocity_z'] = tip_vel_abs[2]

                # 2. Finger Acceleration (Absolute)
                features['finger_acceleration_x'] = tip_acc_abs[0]
                features['finger_acceleration_y'] = tip_acc_abs[1]
                features['finger_acceleration_z'] = tip_acc_abs[2]

                # 3. Finger Position (Relative per Task A2)
                features['finger_position_x'] = rel_pos[0]
                features['finger_position_y'] = rel_pos[1]
                features['finger_position_z'] = rel_pos[2]

                # 4. Depth Feature (Relative Z)
                features['depth_feature'] = rel_pos[2]

                # 5. Posture / Euclidean Distance (Tip to DIP/IP)
                dip_pos_abs = kinematics[frame_idx, dip_idx]
                posture = np.linalg.norm(tip_pos_abs - dip_pos_abs)
                features['posture_feature'] = posture
                features['euclidean_distance'] = posture # Duplicate as per ML pipeline expectation

                # 6. Distance from Wrist
                features['distance_from_wrist'] = np.linalg.norm(rel_pos)

                # 7. Fingertip to Palm Center
                palm_pos_abs = palm_center[frame_idx]
                features['fingertip_to_palm_center_distance'] = np.linalg.norm(tip_pos_abs - palm_pos_abs)

                # 8. Wrist Velocity (Absolute)
                features['wrist_velocity_x'] = wrist_vel_abs[0]
                features['wrist_velocity_y'] = wrist_vel_abs[1]
                features['wrist_velocity_z'] = wrist_vel_abs[2]

                # 9. Relative Velocity
                features['relative_velocity_x'] = rel_vel[0]
                features['relative_velocity_y'] = rel_vel[1]
                features['relative_velocity_z'] = rel_vel[2]

                # 10. Avg Velocity/Acceleration (Rolling Window)
                # We need to look back. Simple approach: average last 5 frames (inclusive)
                start_w = max(0, frame_idx - 4)
                window_vel = velocities[start_w : frame_idx+1, tip_idx]
                window_acc = accelerations[start_w : frame_idx+1, tip_idx]

                avg_vel = np.mean(window_vel, axis=0)
                avg_acc = np.mean(window_acc, axis=0)

                features['avg_velocity_x'] = avg_vel[0]
                features['avg_velocity_y'] = avg_vel[1]
                features['avg_velocity_z'] = avg_vel[2]

                features['avg_acceleration_x'] = avg_acc[0]
                features['avg_acceleration_y'] = avg_acc[1]
                features['avg_acceleration_z'] = avg_acc[2]

                # 11. Finger Speed (Magnitude of Velocity)
                features['finger_speed'] = np.linalg.norm(tip_vel_abs)

                # 12. Lag Features for Z-axis Velocity and Acceleration
                for lag in [2, 4, 6]:
                    if frame_idx >= lag:
                        features[f'velocity_z_lag_{lag}'] = velocities[frame_idx - lag, tip_idx, 2]
                    else:
                        features[f'velocity_z_lag_{lag}'] = 0.0

                for lag in [2, 4]:
                    if frame_idx >= lag:
                        features[f'acceleration_z_lag_{lag}'] = accelerations[frame_idx - lag, tip_idx, 2]
                    else:
                        features[f'acceleration_z_lag_{lag}'] = 0.0

                # 13. Rolling Variance for Z-axis Velocity and Acceleration
                # Window of 5 frames
                start_w_var = max(0, frame_idx - 4)
                window_vel_z = velocities[start_w_var : frame_idx+1, tip_idx, 2]
                window_acc_z = accelerations[start_w_var : frame_idx+1, tip_idx, 2]

                features['rolling_variance_velocity_z'] = np.var(window_vel_z)
                features['rolling_variance_acceleration_z'] = np.var(window_acc_z)

                # Label
                features['ground_truth_label'] = labels[frame_idx, finger_i]

                all_features.append(features)

        return all_features

    def run(self, max_files=None, output_csv="features.csv"):
        """
        Main entry point to run the dataset generation.

        Args:
            max_files: The maximum number of file pairs to process.
            output_csv: The name of the output CSV file.
        """
        logger.info("Starting dataset generation...")

        # If we are running from the script directory, we can try to find files relative to it
        # But the plan suggests we might just run it from repo root.
        # The original code used self.dataset_dir passed in init.

        annotation_dir = self.dataset_dir # / "annotation" / "annotation" # Adjusted based on file structure
        # Wait, the original code had nested paths. Let's check list_files.
        # The file list shows BV...json in Code/Data_Pipeline.
        # If the user wants me to use the LOCAL sample files, I should point to them.
        # But the class is designed for the full dataset structure.
        # I will adapt it to look in dataset_dir directly if the nested structure is missing.

        # Check for sample files in the current directory (Code/Data_Pipeline)
        # If dataset_dir is passed as '.', we might find them.

        all_features = []

        # Robust file finding
        # Try finding JSONs recursively
        kinematics_files = list(self.dataset_dir.rglob("*.json"))

        if max_files:
            kinematics_files = kinematics_files[:max_files]

        logger.info(f"Found {len(kinematics_files)} kinematics files.")

        for kinematics_file in tqdm(kinematics_files, desc="Processing files"):
            # Attempt to find matching MIDI
            # Assuming naming convention: BV1Jf421Z732_seq_0000.json -> BV1Jf421Z732.mid
            # The sequence suffix might not be in the MIDI filename if MIDI is per-video/session.

            # Strategy: Try to find a .mid file with the prefix of the json file
            file_stem = kinematics_file.stem
            # Remove _seq_XXXX
            base_name = file_stem.split('_seq_')[0]

            # Look in same dir or recursively
            midi_candidates = list(self.dataset_dir.rglob(f"{base_name}.mid"))

            if midi_candidates:
                midi_file = midi_candidates[0]

                kinematics = self.load_kinematics(kinematics_file)
                note_events = self.load_midi_labels(midi_file)

                if kinematics is not None and note_events:
                    features = self.extract_and_label_features(kinematics, note_events)

                    # Add sequence_id to each feature row
                    sequence_id = kinematics_file.stem
                    for feature_row in features:
                        feature_row['sequence_id'] = sequence_id

                    all_features.extend(features)
            else:
                logger.warning(f"MIDI file not found for {kinematics_file} (Base: {base_name})")

        if not all_features:
            logger.warning("No features were extracted. The dataset might be empty or paths are incorrect.")
            return

        df = pd.DataFrame(all_features)

        # Determine project root and save to the correct Data directory
        # Assuming this script is in Machine_Learning_Course/Code/Data_Pipeline
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        # Wait, __file__ is .../Code/Data_Pipeline/Sync.py
        # parent -> Pipeline
        # parent -> Code
        # parent -> Course
        # parent -> Root
        # Actually, let's just use relative path "Data/PianoMotion10M" from where we run, or fixed path.

        # If running from repo root: Machine_Learning_Course/Data/PianoMotion10M
        output_dir = Path("Machine_Learning_Course/Data/PianoMotion10M")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_csv

        df.to_csv(output_path, index=False)

        logger.info(f"Dataset generation complete. Saved to {output_path}")
        logger.info(f"Generated {len(df)} samples.")
        logger.info(f"Label distribution:\n{df['ground_truth_label'].value_counts()}")


if __name__ == "__main__":
    # Use the directory where this script is located as the default dataset dir for the sample files
    script_dir = Path(__file__).parent

    # For the sample run, we point to the script dir where the sample files are
    processor = SyncPianoMotionDataset(dataset_dir=script_dir)
    processor.run(max_files=None)
