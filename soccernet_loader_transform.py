import os
import json
import cv2
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import (
    R3D_18_Weights, 
    MC3_18_Weights, 
    R2Plus1D_18_Weights, 
    S3D_Weights, 
    MViT_V2_S_Weights,
    MViT_V1_B_Weights
)

# ==========================================
# 1. GLOBAL CONFIGURATION & FACTORY
# ==========================================
def get_video_transform(net_name):
    """
    Factory function to get the official preprocessing transforms.
    """
    if net_name == "r3d_18":
        weights = R3D_18_Weights.DEFAULT
    elif net_name == "mc3_18":
        weights = MC3_18_Weights.DEFAULT
    elif net_name == "r2plus1d_18":
        weights = R2Plus1D_18_Weights.DEFAULT
    elif net_name == "s3d":
        weights = S3D_Weights.DEFAULT
    elif net_name == "mvit_v2_s":
        weights = MViT_V2_S_Weights.DEFAULT
    elif net_name == "mvit_v1_b":
        weights = MViT_V1_B_Weights.DEFAULT
    else:
        print(f"Warning: Unknown model {net_name}. Defaulting to R2Plus1D.")
        weights = R2Plus1D_18_Weights.DEFAULT
    
    print(f"Loaded transforms for: {net_name}")
    return weights.transforms()

# Default global transform (will be set when imported, or can be reset)
# You can change this variable from your main notebook if needed:
# soccernet_loader.global_transform = soccernet_loader.get_video_transform('r2plus1d_18')
global_transform = get_video_transform('mvit_v2_s') 


# ==========================================
# 2. PREPROCESSING (The "Auto-Pilot")
# ==========================================
def frames_extract(video_path, frames_count=16):
    """
    Reads video, extracts 16 frames, and applies the GLOBAL transform.
    """
    # Fallback tensor (C, T, H, W) - MViT standard
    empty_tensor = torch.zeros(3, frames_count, 224, 224)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return empty_tensor 

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = 60
    end_frame = 90 
    indices = np.round(np.linspace(start_frame, end_frame, frames_count), decimals=0)
    
    frames_list = []

    for i, idx in enumerate(indices):
        if idx >= total_frames: break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        
        if not success: 
            if len(frames_list) > 0:
                frames_list.append(frames_list[-1])
            else:
                break
            continue

        # OpenCV is BGR, PyTorch needs RGB
        # Keep as uint8 (0-255) for the transform function
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(rgb_frame)

    cap.release()

    if len(frames_list) == 0:
        return empty_tensor

    while len(frames_list) < frames_count:
        frames_list.append(frames_list[-1])

    # 1. Stack into Numpy: (Time, Height, Width, Channels)
    video_data = np.stack(frames_list) 
    
    # 2. To Torch Tensor
    video_tensor = torch.from_numpy(video_data)

    # 3. Permute for Transform: (Time, H, W, C) -> (T, C, H, W)
    # Standard Torchvision video transforms input is (T, C, H, W)
    video_tensor = video_tensor.permute(0, 3, 1, 2) 

    # 4. Apply Auto-Pilot
    # This handles Resize, Crop, Normalize, and Permute to (C, T, H, W)
    final_tensor = global_transform(video_tensor)

    return final_tensor


# ==========================================
# 3. DATA LOADING LOGIC
# ==========================================
def load_soccer_data(root_dir, mode='train'):
    """
    Parses the annotations.json file and builds a DataFrame.
    """
    json_path = os.path.join(root_dir, "annotations.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find annotations.json at {json_path}")

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    rows_list = []
    
    for action_id in json_data['Actions']:
        # --- FILTERING ---
        if json_data['Actions'][action_id]['Action class'] in ['', 'Dont know']: continue
        if (json_data['Actions'][action_id]['Offence'] in ['', 'Between']) and json_data['Actions'][action_id]['Action class'] != 'Dive': continue
        if (json_data['Actions'][action_id]['Severity'] in ['', '2.0', '4.0']) and json_data['Actions'][action_id]['Action class'] != 'Dive' and json_data['Actions'][action_id]['Offence'] != 'No Offence': continue

        # --- ATTRIBUTES ---
        clips = json_data['Actions'][action_id]['Clips']
        
        if json_data['Actions'][action_id]['Offence'] in ['', 'Between']: offence = 'Offence'
        else: offence = json_data['Actions'][action_id]['Offence']

        if json_data['Actions'][action_id]['Severity'] in ['', '2.0', '4.0']: severity = '1.0'
        else: severity = json_data['Actions'][action_id]['Severity']

        action_class_str = json_data['Actions'][action_id]['Action class']

        # --- LABEL MAPPING ---
        if offence.lower() == 'no offence': card_label = 0
        elif severity == '1.0': card_label = 1
        elif severity == '3.0': card_label = 2
        elif severity == '5.0': card_label = 3
        else: card_label = 1

        # --- VIEW SELECTION ---
        all_indices = list(range(len(clips)))
        if len(all_indices) == 0: continue

        selected_paths = []
        
        if mode == 'train':
            if len(all_indices) < 2: continue
            # We collect ALL valid paths, sampling happens in Dataset class
            for idx in all_indices:
                path = os.path.join(root_dir, f'action_{action_id}/clip_{idx}.mp4')
                if os.path.exists(path): selected_paths.append(path)
            if len(selected_paths) < 2: continue
        else:
            for idx in all_indices:
                path = os.path.join(root_dir, f'action_{action_id}/clip_{idx}.mp4')
                if os.path.exists(path): selected_paths.append(path)
            if len(selected_paths) == 0: continue

        rows_list.append({
            'action_id': action_id,
            'view_paths': selected_paths,
            'Action_class': action_class_str,
            'card_label': card_label,
            'mode': mode
        })

    df = pd.DataFrame(rows_list)
    print(f"[{mode.upper()}] Processed {len(df)} actions.")
    return df


# ==========================================
# 4. DATASET CLASS
# ==========================================
class SoccerNetDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.action_mapping = {
            'Standing tackling': 0, 'Tackling': 1, 'Challenge': 2,
            'Holding': 3, 'Elbowing': 4, 'High leg': 5, 
            'Pushing': 6, 'Dive' : 7
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        
        # Dynamic Sampling based on Mode
        if row['mode'] == 'train':
            # Pick 2 random views for training
            selected_paths = random.sample(row.view_paths, 2)
        else:
            # Use all views for test/valid
            selected_paths = row.view_paths

        views = []
        for path in selected_paths:
            # Calling the optimized extractor
            tensor = frames_extract(path)
            views.append(tensor)
        
        features = torch.stack(views, dim=0)
        
        action_label = self.action_mapping.get(row.Action_class, 0)
        card_label = int(row.card_label)
        
        return features, torch.tensor(action_label).long(), torch.tensor(card_label).long()
