import torch
from torch import nn
import numpy as np
import cv2
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

# preprocessing
def get_video_transform():
    """Get the official preprocessing transforms."""
    weights = MViT_V2_S_Weights.DEFAULT
    return weights.transforms()

global_transform = get_video_transform()

def frames_extract(video_path, frames_count=16):
    """
    Reads video, extracts 16 frames, and applies the transform.
    Returns: Tensor of shape (C, T, H, W)
    """
    empty_tensor = torch.zeros(3, frames_count, 224, 224)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return empty_tensor

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # extract frames from frame 60 to 90
    start_frame = 60
    end_frame = 90
    indices = np.round(np.linspace(start_frame, end_frame, frames_count), decimals=0)

    frames_list = []

    for idx in indices:
        if idx >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()

        if not success:
            if len(frames_list) > 0:
                frames_list.append(frames_list[-1])
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(rgb_frame)

    cap.release()

    if len(frames_list) == 0:
        return empty_tensor

    # padding
    while len(frames_list) < frames_count:
        frames_list.append(frames_list[-1])

    # stack frames: (T, H, W, C)
    video_data = np.stack(frames_list)
    
    # convert to tensor and permute to (T, C, H, W)
    video_tensor = torch.from_numpy(video_data).permute(0, 3, 1, 2)
    
    # apply transform (returns C, T, H, W)
    final_tensor = global_transform(video_tensor)

    return final_tensor

# batching
def batch_tensor(tensor, dim=1, squeeze=True):
    """Merge Batch and View dimensions."""
    if squeeze:
        return tensor.flatten(0, 1)
    return tensor

def unbatch_tensor(tensor, b, dim=1, unsqueeze=True):
    """Separate Batch and View dimensions."""
    if unsqueeze:
        return tensor.view(b, -1, tensor.size(-1))
    return tensor

# model architecture
class WeightedAggregate(nn.Module):
    def __init__(self, model, feat_dim, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feature_dim = feat_dim

        r1, r2 = -1, 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)
        
        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))

        aux = torch.matmul(aux, self.attention_weights)
        aux_t = aux.permute(0, 2, 1)
        prod = torch.bmm(aux, aux_t)
        relu_res = self.relu(prod)

        aux_sum = torch.sum(torch.reshape(relu_res, (B, V*V)).T, dim=0).unsqueeze(0)
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V*V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T

        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V))
        final_attention_weights = torch.sum(final_attention_weights, 1)

        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))
        output = torch.sum(output, 1)

        return output.squeeze(), final_attention_weights


class ViewMaxAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux

class ViewAvgAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux

class MVAggregate(nn.Module):
    def __init__(self, model, agr_type="max", feat_dim=768, lifting_net=nn.Sequential(), num_actions=8):
        super().__init__()
        self.agr_type = agr_type

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, num_actions) 
        )

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(model=model, lifting_net=lifting_net)
        else:
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)

    def forward(self, mvimages):
        pooled_view, attention = self.aggregation_model(mvimages)
        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)
        return pred_offence_severity, pred_action, attention

class MVNetwork(nn.Module):
    def __init__(self, net_name='mvit_v2_s', agr_type='max', lifting_net=nn.Sequential(), num_actions=8):
        super().__init__()
        
        self.net_name = net_name
        self.agr_type = agr_type
        self.lifting_net = lifting_net
        self.feat_dim = 768

        # load pretrained MViT model
        weights_model = MViT_V2_S_Weights.DEFAULT
        network = mvit_v2_s(weights=weights_model)
        
        # remove classification head
        network.head = nn.Sequential()

        self.mvnetwork = MVAggregate(
            model=network,
            agr_type=self.agr_type,
            feat_dim=self.feat_dim,
            lifting_net=self.lifting_net,
            num_actions=num_actions 
        )

    def forward(self, mvimages):
        return self.mvnetwork(mvimages)

# load model

# class labels
CARD_LABELS = ['No card', 'Yellow card', 'Red card', 'Yellow->Red card']
ACTION_LABELS = ['Tackling', 'Standing tackling', 'High leg', 'Holding', 'Pushing', 'Elbowing', 'Challenge', 'Dive']

def load_model(model_path='model_mvit.pth', num_actions=8):
    """Load the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MVNetwork(net_name='mvit_v2_s', agr_type='max', num_actions=num_actions)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model, device

def predict_single_view(video_path, model, device):
    """
    Run inference on a single video view.
    Returns: card prediction, action prediction, and their probabilities
    """
    # frame extraction
    video_tensor = frames_extract(video_path)  # Shape: (C, T, H, W)
    
    # batch and view dimensions: (1, 1, C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0).unsqueeze(0)
    video_tensor = video_tensor.to(device)
    
    with torch.no_grad():
        pred_offence, pred_action, attention = model(video_tensor)
    
    if pred_offence.dim() == 1:
        pred_offence = pred_offence.unsqueeze(0)
    if pred_action.dim() == 1:
        pred_action = pred_action.unsqueeze(0)
    
    card_probs = torch.softmax(pred_offence, dim=-1).cpu().numpy()
    action_probs = torch.softmax(pred_action, dim=-1).cpu().numpy()
    
    # flatten to 1d (optional)
    card_probs = card_probs.flatten()
    action_probs = action_probs.flatten()
    
    card_idx = np.argmax(card_probs)
    action_idx = np.argmax(action_probs)
    
    return {
        'card': CARD_LABELS[card_idx],
        'card_confidence': float(card_probs[card_idx]),
        'card_probs': {label: float(prob) for label, prob in zip(CARD_LABELS, card_probs)},
        'action': ACTION_LABELS[action_idx],
        'action_confidence': float(action_probs[action_idx]),
        'action_probs': {label: float(prob) for label, prob in zip(ACTION_LABELS, action_probs)}
    }

# usage
def predict(VIDEO_PATH):
    # Load model
    model, device = load_model('model_mvit.pth', num_actions=8)
    
    # Single view prediction
    video_path = VIDEO_PATH
    print('VIDEOPATH', VIDEO_PATH)
    result = predict_single_view(video_path, model, device)
    
    print("\n=== PREDICTION RESULTS ===")
    print(f"Card: {result['card']} (Confidence: {result['card_confidence']:.2%})")
    print(f"Action: {result['action']} (Confidence: {result['action_confidence']:.2%})")
    
    print("\n=== CARD PROBABILITIES ===")
    for label, prob in result['card_probs'].items():
        print(f"{label}: {prob:.2%}")
    
    print("\n=== ACTION PROBABILITIES ===")
    for label, prob in result['action_probs'].items():
        print(f"{label}: {prob:.2%}")
    
    return [f'{result['card']}: {result['card_confidence']:.2%}', f'{result['action']}: {result['action_confidence']:.2%}']