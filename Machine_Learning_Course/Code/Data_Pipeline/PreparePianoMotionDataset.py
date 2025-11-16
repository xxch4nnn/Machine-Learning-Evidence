import os
import sys
sys.path.append(os.getcwd())
import json
import torch
import numpy as np
import pandas as pd
from manotorch.manolayer import ManoLayer
from scipy.spatial.transform import Rotation
from Machine_Learning_Course.Data.PianoMotion10M.datasets.utils import read_midi, TargetProcessor

def get_hand_pose(mano_params, is_right_hand):
    """
    Get hand pose from MANO parameters.
    """
    if is_right_hand:
        mano_side = "right"
    else:
        mano_side = "left"

    transl = np.array(mano_params[mano_side]["transl"])
    global_orient = np.array(mano_params[mano_side]["global_orient"])
    hand_pose = np.array(mano_params[mano_side]["hand_pose"])
    betas = np.array(mano_params[mano_side]["betas"])

    return transl, global_orient, hand_pose, betas

def get_3d_joints(mano_layer, transl, global_orient, hand_pose, betas):
    """
    Get 3D joint locations from MANO layer.
    """
    hand_pose = torch.from_numpy(hand_pose).unsqueeze(0)
    global_orient = torch.from_numpy(global_orient).unsqueeze(0)
    transl = torch.from_numpy(transl).unsqueeze(0)
    betas = torch.from_numpy(betas).unsqueeze(0)

    output = mano_layer(global_orient=global_orient, hand_pose=hand_pose, transl=transl, betas=betas)
    joints = output.joints.squeeze(0).detach().numpy()
    return joints

def extract_features(data_dir, output_csv):
    """
    Extracts features from the PianoMotion10M dataset and saves them to a CSV file.

    Args:
        data_dir (str): The path to the PianoMotion10M dataset directory.
        output_csv (str): The path to the output CSV file.
    """
    annotation_dir = os.path.join(data_dir, "annotation")
    midi_dir = os.path.join(data_dir, "midi")
    train_file = os.path.join(data_dir, "train.txt")

    mano_layer_rh = ManoLayer(mano_assets_root='Machine_Learning_Course/Data/PianoMotion10M/mano', use_pca=False, n_comps=45, side='right')

    features = []

    with open(train_file, 'r') as f:
        for line in f:
            up, video_name = line.strip().split(' ')
            json_dir = os.path.join(annotation_dir, up, video_name)
            midi_path = os.path.join(midi_dir, up, f"{video_name}.mid")

            if not os.path.exists(midi_path):
                continue

            midi_dict = read_midi(midi_path)
            target_processor = TargetProcessor(1.0, 30, 21, 88)

            # Sort files to ensure correct frame order for velocity/acceleration
            sorted_json_files = sorted(os.listdir(json_dir))

            last_tip_coords = None
            last_velocity = None

            for json_file in sorted_json_files:
                json_path = os.path.join(json_dir, json_file)

                with open(json_path, 'r') as f_json:
                    mano_params = json.load(f_json)

                if not mano_params["right"]:
                    continue

                # Get right hand pose
                transl_rh, global_orient_rh, hand_pose_rh, betas_rh = get_hand_pose(mano_params, True)

                # Get joint positions
                joints_rh = get_3d_joints(mano_layer_rh, transl_rh, global_orient_rh, hand_pose_rh, betas_rh)
                tip_rh_coords = joints_rh[7] # Index fingertip
                dip_rh_coords = joints_rh[6] # Index DIP joint

                # Feature extraction
                depth_feature = tip_rh_coords[2]
                tip_to_dip_distance = np.linalg.norm(tip_rh_coords - dip_rh_coords)

                # Velocity and Acceleration
                if last_tip_coords is not None:
                    velocity = tip_rh_coords - last_tip_coords
                    if last_velocity is not None:
                        acceleration = velocity - last_velocity
                    else:
                        acceleration = np.zeros(3)
                else:
                    velocity = np.zeros(3)
                    acceleration = np.zeros(3)

                # Update for next frame
                last_tip_coords = tip_rh_coords
                last_velocity = velocity

                # Align with MIDI data to get ground truth label
                fps = mano_params.get("fps", 30)
                frame_index = int(os.path.splitext(json_file)[0].split('_')[-1])
                frame_time = frame_index / fps

                target_dict, _, _ = target_processor.process(frame_time, midi_dict['midi_event_time'], midi_dict['midi_event'])
                frame_roll = target_dict['frame_roll']

                is_press = 1 if np.sum(frame_roll[0]) > 0 else 0

                features.append([
                    json_path, depth_feature, tip_to_dip_distance,
                    tip_rh_coords[0], tip_rh_coords[1], tip_rh_coords[2],
                    velocity[0], velocity[1], velocity[2],
                    acceleration[0], acceleration[1], acceleration[2],
                    is_press
                ])

    columns = [
        "json_path", "depth_feature", "tip_to_dip_distance",
        "tip_x", "tip_y", "tip_z",
        "velocity_x", "velocity_y", "velocity_z",
        "acceleration_x", "acceleration_y", "acceleration_z",
        "is_press"
    ]
    df = pd.DataFrame(features, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")


if __name__ == "__main__":
    DATA_DIR = "Machine_Learning_Course/Data/PianoMotion10M"
    OUTPUT_CSV = "Machine_Learning_Course/Data/PianoMotion10M/features.csv"
    extract_features(DATA_DIR, OUTPUT_CSV)
