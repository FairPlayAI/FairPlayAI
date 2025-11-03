from DataLoader import dataframe
from config import *
import numpy as np
import cv2
import torch

"""
-------------------- INFO --------------------
1. this file output is a tuple: ( features, action label, card label )
2. features are a torch tensor holding the clip numpy array for all actions
3. action label is encoded and the map object is shown below in the create_dataset() fn
4. card label is of float type [0.0: no card, 1.0: warning, 3.0: yellow card, 5.0: red card]
5. shape of features is (number of actions, 2 clips, 224 width, 224 height, 3 rgb)

-------------------- HOW TO USE --------------------
1. Use create_dataset(batch_size)
2. specify batch_size to be number of actions you want (maxmimum is 2916)
3. the return type is a tuple of three tensors (features, actions, cards)
4. Example: 
features, action_label, card_label = create_dataset(2916)
"""

def center_crop(frame, size=224):
    h, w = frame.shape[:2]
    startx = w//2 - size//2
    starty = h//2 - size//2
    return frame[starty:starty+size, startx:startx+size]

def extract_frames(video_path):
    frames_arr = np.zeros((10, 224, 224, 3), dtype=np.float32)
    i = 0

    indices = np.round(np.linspace(60, 89, 10), decimals=0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    video_reader = cv2.VideoCapture(video_path)

    # --- DEBUGGING BLOCK START ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("cannot open video:", video_path)
        return np.zeros((10, 224, 224, 3), dtype=np.float32)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{video_path} -> frame_count: {frame_count}, indices: {indices}")
    # --- DEBUGGING BLOCK END ---

    for idx in indices:
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video_reader.read()

        if not success:
            break

        h, w = frame.shape[:2]
        scale = 256 / min(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frame = center_crop(frame, 224)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = rgb_frame.astype(np.float32) / 255.0
        rgb_frame = (rgb_frame - mean) / std
        frames_arr[i] = rgb_frame
        i += 1

    video_reader.release()
    return frames_arr

# outputs [ features - action - card ] tensors
def create_dataset(batch_size: int) -> tuple:

    if batch_size > 2916:
        print('ERROR in create_dataset(): Maximum number of actions is 2916')
        return 0, 0, 0

    r = batch_size // 2

    features = np.zeros((r, 2, 10, 224, 224, 3), dtype=np.float32)
    action_labels = np.zeros((r,), dtype=np.int32)
    card_labels = np.zeros((r,), dtype=np.int32)

    data = dataframe(TRAIN_PATH)
    idx = 0

    action_mapping = {
        'Standing tackling': 0,
        'Tackling': 1,
        'Challenge': 2,
        'Holding': 3,
        'Elbowing': 4,
        'High leg': 5,
        'Pushing': 6,
        'Dive' : 7
    }

    for i in range(0,batch_size,2):
        frames_view1 = extract_frames(data['clip_path'][i])
        frames_view2 = extract_frames(data['clip_path'][i+1])
        print(f"extracted frames for video of type : {data['Action_class'][i]} //// i = {i}")

        stacked = np.stack((frames_view1, frames_view2), axis=0)
        features[idx] = stacked
        action_labels[idx] = action_mapping[data['Action_class'][i]]
        card_labels[idx] = float(data['card'][i])
        idx += 1

    features_tensor = torch.from_numpy(features)
    action_labels_tensor = torch.from_numpy(action_labels)
    card_labels_tensor = torch.from_numpy(card_labels)
    return features_tensor, action_labels_tensor, card_labels_tensor
