import os
import json
import cv2
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def center_crop(frame, size=224):
    h, w = frame.shape[:2]
    startx = w//2 - size//2
    starty = h//2 - size//2
    return frame[starty:starty+size, startx:startx+size]

def frames_extract(video_path, frames_count=16):
    """
    Reads video, extracts 16 frames, resizes (224x224), normalizes (ImageNet stats),
    and returns tensor (3, 16, 224, 224).
    """
    # Initialize with correct shape (Channels, Depth, Height, Width)
    frames_arr = np.zeros((3, frames_count, 224, 224), dtype=np.float32)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Return zeros if video fails to open
        return frames_arr 

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sampling logic (e.g., frames 60 to 90)
    start_frame = 60
    end_frame = 90 
    indices = np.round(np.linspace(start_frame, end_frame, frames_count), decimals=0)
    
    # ImageNet Statistics
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, idx in enumerate(indices):
        if idx >= total_frames: break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success: break

        # Resize & Crop
        h, w = frame.shape[:2]
        scale = 256 / min(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frame = center_crop(frame, 224)

        # Normalize
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_frame = (rgb_frame - mean) / std
        
        # Transpose to (Channels, Height, Width)
        rgb_frame = np.transpose(rgb_frame, (2, 0, 1))
        
        frames_arr[:, i, :, :] = rgb_frame

    cap.release()
    return frames_arr

# ==========================================
# 2. DATA LOADING LOGIC (JSON Parsing)
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
    
    # Iterate over all actions
    for action_id in json_data['Actions']:
        
        # --- YOUR EXACT FILTERING LOGIC (Indentation Fixed) ---
        if json_data['Actions'][action_id]['Action class'] == '' or json_data['Actions'][action_id]['Action class'] == "Dont know":
            continue

        if (json_data['Actions'][action_id]['Offence'] == '' or json_data['Actions'][action_id]['Offence'] == 'Between') and json_data['Actions'][action_id]['Action class'] != 'Dive':
            continue

        if (json_data['Actions'][action_id]['Severity'] == '' or json_data['Actions'][action_id]['Severity'] == '2.0' or json_data['Actions'][action_id]['Severity'] == '4.0') and json_data['Actions'][action_id]['Action class'] != 'Dive' and json_data['Actions'][action_id]['Offence'] != 'No Offence':
            continue

        # --- EXTRACT ATTRIBUTES (Your Logic) ---
        
        
        if json_data['Actions'][action_id]['Offence'] == '' or json_data['Actions'][action_id]['Offence'] == 'Between':
            offence = 'Offence'
        else:
            offence = json_data['Actions'][action_id]['Offence']

        
        if json_data['Actions'][action_id]['Severity'] == '' or json_data['Actions'][action_id]['Severity'] == '2.0' or json_data['Actions'][action_id]['Severity'] == '4.0':
            severity = '1.0'
        else:
            severity = json_data['Actions'][action_id]['Severity']

        action_class_str = json_data['Actions'][action_id]['Action class']

        # --- CALCULATE CARD LABEL (Required for Model) ---
        if offence == 'No Offence' or offence == 'No offence': 
            card_label = 0
        elif severity == '1.0': 
            card_label = 1
        elif severity == '3.0': 
            card_label = 2
        elif severity == '5.0': 
            card_label = 3
        else: 
            card_label = 1
            
        clips = json_data['Actions'][action_id]['Clips']
        # --- VIEW SELECTION LOGIC ---
        all_indices = list(range(len(clips)))
        
        # We need at least 1 view
        if len(all_indices) == 0: 
            continue
        
        if mode == 'train' and len(all_indices) < 2:
            continue

        selected_paths = []
        
        for idx in all_indices:
            path = os.path.join(root_dir, f'action_{action_id}/clip_{idx}.mp4')
            # Only add if file exists (Safety check)
            if os.path.exists(path):
                selected_paths.append(path)

        rows_list.append({
            'action_id': action_id,
            'view_paths': selected_paths, 
            'Action_class': action_class_str,
            'card_label': card_label,
            'mode' : mode
        })

    df = pd.DataFrame(rows_list)
    print(f"[{mode.upper()}] Processed {len(df)} actions.")
    return df

# ==========================================
# 3. DATASET CLASS
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
        
        
        if row['mode'] == 'train':
            # Pick 2 random views EVERY TIME this is called
            selected_paths = random.sample(row.view_paths, 2)
        else:
            # Test/Valid: Use all views
            selected_paths = row.view_paths

        # Load Frames
        views = []
        for path in selected_paths:
            tensor = torch.from_numpy(frames_extract(path))
            views.append(tensor)
        # Stack into (Views, 3, 16, 224, 224)
        features = torch.stack(views, dim=0)
        
        action_label = self.action_mapping.get(row.Action_class, 0)
        card_label = int(row.card_label)
        
        return features, torch.tensor(action_label).long(), torch.tensor(card_label).long()
