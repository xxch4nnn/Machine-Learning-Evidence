# PianoMotion Pipeline Agents

## Data Format
- **Kinematics**: JSON files containing 3D coordinates (21 joints x 3 dimensions).
- **Labels**: MIDI files (.mid) synchronized to 30 FPS video.
- **Output**: `features.csv` with 26 feature columns + 1 `ground_truth_label`.

## Labeling Convention
The pipeline is transitioning from Binary to Multi-Class:
- 0: Hover (No contact)
- 1: Press (Downward attack)
- 2: Hold (Sustain)
- 3: Release (Lift off)

## Critical Files
- `Code/Data_Pipeline/SyncPianoMotionDataset.py`: Main feature extraction logic.
- `Code/Data_Pipeline/validate_sync_output.py`: Quality assurance script.
