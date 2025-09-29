import argparse
import glob
# Standard library imports
import os
import sys
import math
import json
import random
import xml.etree.ElementTree as ET

# Third-party imports
import pytorch_kinematics as pk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.io import read_video
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    def summary(*args, **kwargs):
        print("\nWarning: torchinfo not installed. Model summary will not be displayed.")
        print("Install with: pip install torchinfo\n")
        return None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# protobuf
POSE_FEATURE_SIZE = 225  # 75 landmarks × 3 coordinates

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0fpose_data.proto\x12\x04pose"?\n\x08Landmark\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\x12\x12\n\nvisibility\x18\x04 \x01(\x02"\xa9\x01\n\tFrameData\x12\r\n\x05\x66rame\x18\x01 \x01(\x05\x12\n\n\x02ts\x18\x02 \x01(\x01\x12&\n\x0epose_landmarks\x18\x03 \x03(\x0b\x32\x0e.pose.Landmark\x12+\n\x13left_hand_landmarks\x18\x04 \x03(\x0b\x32\x0e.pose.Landmark\x12,\n\x14right_hand_landmarks\x18\x05 \x03(\x0b\x32\x0e.pose.Landmark"0\n\rFrameDataList\x12\x1f\n\x06\x66rames\x18\x01 \x03(\x0b\x32\x0f.pose.FrameDatab\x06proto3'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "pose_data_pb2", globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _LANDMARK._serialized_start = 25
    _LANDMARK._serialized_end = 88
    _FRAMEDATA._serialized_start = 91
    _FRAMEDATA._serialized_end = 260
    _FRAMEDATALIST._serialized_start = 262
    _FRAMEDATALIST._serialized_end = 310

this_module = sys.modules[__name__]
pose_data_pb2 = this_module
sys.modules["pose_data_pb2"] = this_module

class ProtobufProcessor:
    @staticmethod
    def load_protobuf_data(proto_file_path):
        """Load protobuf data from a file"""
        with open(proto_file_path, "rb") as f:
            data = pose_data_pb2.FrameDataList()
            data.ParseFromString(f.read())  # Convert binary data to protobuf format
        return data

    @staticmethod
    def transform_coordinates(point):
        """Transform coordinates from protobuf format to usable format"""
        x = getattr(point, "x", 0.0)
        y = getattr(point, "y", 0.0)
        z = getattr(point, "z", 0.0)
        return torch.tensor([[-x, -z, -y]])  # Flip the coordinate system for compatibility

    @staticmethod
    def extract_landmark_coordinates(frame, landmark_type, indices):
        """Extract and transform landmark coordinates from a specific frame"""
        transformed_landmarks = torch.empty((0, 3))
        landmarks = getattr(frame, landmark_type, [])

        for i in indices:
            if i < len(landmarks):
                transformed_landmarks = torch.cat((transformed_landmarks, ProtobufProcessor.transform_coordinates(landmarks[i])), dim=0)
            else:
                transformed_landmarks = torch.cat((transformed_landmarks, torch.zeros((1, 3))), dim=0) # Default value for missing landmarks

        return transformed_landmarks

    @staticmethod
    def compute_bone_vectors(landmarks):
        """
        landmarks: torch.Tensor of shape (21, 3)
        returns: torch.Tensor of shape (20, 3), each row is a bone vector
        """
        bone_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        bones = torch.stack([landmarks[j] - landmarks[i] for i, j in bone_connections])
        return bones  # shape: [20, 3]

    @staticmethod
    def angle(v1, v2, dim=-1, eps=1e-8):
        """
        Calculate the signed angle between two tensors in radians.
        
        Args:
            v1, v2: Input tensors of same shape
            dim: Dimension along which to compute the angle
            eps: Small value to avoid division by zero
        
        Returns:
            Signed angle in radians between -pi and pi
        """
        # Normalize the vectors
        v1_norm = F.normalize(v1, p=2, dim=dim, eps=eps)
        v2_norm = F.normalize(v2, p=2, dim=dim, eps=eps)
        
        # Dot product (cosine of angle)
        dot_product = (v1_norm * v2_norm).sum(dim=dim)
        
        # Cross product magnitude (sine of angle)
        cross_product = torch.norm(torch.cross(v1_norm, v2_norm, dim=dim), dim=dim)
        
        # Use atan2 to get signed angle
        angle = torch.atan2(cross_product, dot_product)
        
        return angle

    @staticmethod
    def normalize_hand_landmarks(landmarks, left_side):
        """
        Normalize all 10 body landmarks using shoulders as reference points.

        Args:
            landmarks (torch.Tensor): Shape (10, 3), 3D coordinates of body landmarks.
            left_side (bool): Whether the landmarks belong to the left arm.

        Returns:
            torch.Tensor: Normalized landmarks of shape (1, 10, 3)
        """
        if landmarks.shape[0] < 21:
            return torch.empty((1, 7, 3))  # Return zero if not enough landmarks

        # Flip x for left side
        if left_side:
            landmarks[:, 0] *= -1
            landmarks[:, 2] *= -1

        bones = ProtobufProcessor.compute_bone_vectors(landmarks)

        return torch.clamp(torch.tensor([
            (ProtobufProcessor.angle(bones[4], bones[16]) - ProtobufProcessor.angle(bones[5], bones[17]) + 0.2)/0.4,
            ProtobufProcessor.angle(bones[19], bones[16]) / math.pi,
            ProtobufProcessor.angle(bones[15], bones[12]) / math.pi,
            ProtobufProcessor.angle(bones[11], bones[8]) / math.pi,
            ProtobufProcessor.angle(bones[7], bones[4]) / math.pi,
            2 * ProtobufProcessor.angle(bones[2], bones[1]) / math.pi,
            2 * ProtobufProcessor.angle(bones[3], bones[2]) / math.pi
        ]), 0.0, 1.0).view(1, 7)

    @staticmethod
    def normalize_body_landmarks(landmarks, left_side):
        """Normalize body landmarks using shoulders as reference points"""
        if landmarks.shape[0] < 10:
            return torch.empty((1, 3, 3))  # Return zero if less than 2 landmarks
        
        origin = landmarks[1]#.clone().detach()
        # Compute the distance between the shoulder and elbow as the scale factor
        L = (torch.linalg.vector_norm(landmarks[1] - landmarks[3]) + torch.linalg.vector_norm(landmarks[3] - landmarks[5]))/2.0#.clone().detach()/2.0
        indices = [5, 7, 9]
        if left_side:
            indices = [4, 6, 8]
            landmarks[:, 0] *= -1
            origin = landmarks[0]#.clone().detach()
            # Compute the distance between the shoulder and elbow as the scale factor
            L = (torch.linalg.vector_norm(landmarks[0] - landmarks[2]) + torch.linalg.vector_norm(landmarks[4] - landmarks[2]))/2.0#.clone().detach()/2.0
        
        L = L if L > 0 else 1.0  # Prevent division by zero

        landmarks = (landmarks - origin) / L  # Wrist as reference

        # Normalize hand landmarks
        return landmarks[indices].unsqueeze(0)

def batch_vectors_to_6D(pose: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Convert two orthogonal vectors (u, v) on a plane to 6D rotation representation.
    Args:
        pose: Tensor of shape (B, N, 3, 3), first vector (e.g., palm's local x-axis).
        eps: Small value to avoid division by zero.
    Returns:
        6D representation of shape (B, N, 6).
    """
    # Ensure u and v are unit vectors (optional but recommended)
    root = pose[:,:,0]
    u = pose[:,:,2] - root
    u = u / (torch.norm(u, dim=-1, keepdim=True) + eps)
    v = pose[:,:,1] - root
    v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
    
    # Orthogonalize v w.r.t. u (Gram-Schmidt)
    v_ortho = v - (torch.sum(u * v, dim=-1, keepdim=True) * u)
    v_ortho = v_ortho / (torch.norm(v_ortho, dim=-1, keepdim=True) + eps)
    
    # Stack u and v_ortho to form 6D representation
    six_d = torch.stack([root, root+0.2*u, root+0.2*v_ortho], dim=-1)
    return six_d

def interpolate(tensor):
    tensor = tensor.clone().float()
    n = tensor.size(0)

    # Indices of all elements
    indices = torch.arange(n)

    # Mask for non-NaN values
    valid_mask = ~torch.isnan(tensor)
    
    if valid_mask.all():
        # no NaNs, just return tensor
        return tensor

    # If start or end are NaN, fill them with nearest valid value (forward/backward fill)
    if not valid_mask[0]:
        first_valid_index = valid_mask.nonzero()[0].item()
        tensor[:first_valid_index] = tensor[first_valid_index]
        valid_mask[:first_valid_index] = True

    if not valid_mask[-1]:
        last_valid_index = valid_mask.nonzero()[-1].item()
        tensor[last_valid_index+1:] = tensor[last_valid_index]
        valid_mask[last_valid_index+1:] = True

    # Now do linear interpolation between valid values
    valid_indices = indices[valid_mask]
    valid_values = tensor[valid_mask]

    # Interpolate manually for NaN positions
    for i in range(len(valid_indices) - 1):
        start_idx = valid_indices[i].item()
        end_idx = valid_indices[i+1].item()

        start_val = valid_values[i].item()
        end_val = valid_values[i+1].item()

        gap = end_idx - start_idx
        if gap > 1:
            # linear interpolation for the gap positions
            for j in range(start_idx + 1, end_idx):
                weight = (j - start_idx) / gap
                tensor[j] = (1 - weight) * start_val + weight * end_val

    return tensor

def fill_fingers(fingers):
    zero = torch.tensor([[0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
    fingers = torch.cat((zero, fingers, zero), dim=0)
    fingers = interpolate(fingers)
    return fingers[1:-1]

def fill_hand(hand):
    zero = torch.tensor([[[-0.2567, -0.3659, -0.2908],
                            [-0.3203, -0.4553, -0.4184],
                            [-0.4181, -0.5175, -0.2473]]])
    hand = torch.cat((zero, hand, zero), dim=0)
    hand = interpolate(hand)
    return hand[1:-1]

# finger model
class FingerModel(nn.Module):
    def __init__(self):
        super(FingerModel, self).__init__()

        # --- Encoder:
        self.project = nn.Linear(1, 3)

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),   # → (B, 8, 15, 10, 3)
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),  # → (B, 16, 8, 5, 2)
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, 2, 2)),
            nn.Flatten()  # → (B, 16*8*5*2)
        )

        latent_dim = 8

        # Latent layers (mean + logvar)
        self.fc_mu = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(16 * 8 * 2 * 2, latent_dim),
            nn.LeakyReLU(0.1)
        )

        self.fc_logvar = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(16 * 8 * 2 * 2, latent_dim),
            nn.LeakyReLU(0.1)
        )

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 16 * 8 * 5 * 2)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, 8, 5, 2)),
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 1, kernel_size=3, stride=1, padding=1)
        )

        # Final linear output: map back to (15 time steps, 6 values each)
        self.fc_output = nn.Sequential(
            nn.Linear(640, 16 * 7),
            nn.Sigmoid()
        )

    def forward(self, pose_input, deterministic=False):
        # pose_input: (B, 15, 10, 3)
        x = pose_input.unsqueeze(-1)         # (B, T, 7, 1)
        x = self.project(x)         # (B, T, 7, 3)
        x = x.unsqueeze(1)
        # --- Encoder
        h = self.encoder(x)

        embedding = self.fc_mu(h)

        return embedding

# hand model
class HandModel(nn.Module):
    def __init__(self):
        super(HandModel, self).__init__()

        # Temporal RNN (LSTM)
        self.encoder = nn.Sequential(
            #nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=(1, 1, 1)),  # (8, 15, 3, 3)
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),  # (8, 15, 3, 3)
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1), # (16, 8, 2, 2)
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, 2, 2)),
            nn.Flatten()
        )

        # Latent space
        self.fc_mu = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(16*8*2*2, 16),
            nn.LeakyReLU(0.1),
        )

        self.fc_logvar = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(16*8*2*2, 16),
            nn.LeakyReLU(0.1),
        )

        # Decoder parameters
        self.fc_decode = nn.Linear(16, 16 * 8 * 2 * 2)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, 8, 2, 2)),
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=(1, 0, 0)),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 1, kernel_size=3, stride=1, padding=1)
        )
        
        # transform 3x3 representation to 6d representation
        self.fc_output = nn.Sequential(
            nn.Linear(16 * 3 * 3, 16 * 6),
            nn.Sigmoid()
        )

    def forward(self, input_data, deterministic=False):
        batch_size, seq_len = input_data.shape[0], input_data.shape[1]

        # Flatten pose input: (B, T, 3, 3) -> (B, T, 9)
        x = input_data.permute(0, 3, 1, 2).unsqueeze(1)
        h = self.encoder(x)

        # Latent space
        embedding = self.fc_mu(h)

        return embedding

def sliding_chunks(tensor, window_size=16, step=4):
    print(tensor.shape)
    t = tensor.shape[0]
    num_chunks = (t - window_size) // step + 1  # ensures full coverage without padding
    chunks = torch.stack([tensor[i*step : i*step + window_size] for i in range(num_chunks)])
    return chunks

# loading models
model_f = FingerModel().to(device)
model_f.load_state_dict(torch.load("sign_language_finger_model_v1.0_10.pth", weights_only=True, map_location=torch.device('cpu')), strict=False)
model_f.eval()

model_h = HandModel().to(device)
model_h.load_state_dict(torch.load("sign_language_pose_model_v3.2_22.pth", weights_only=True, map_location=torch.device('cpu')), strict=False)
model_h.eval()

# path definition
input_folder = "../output"       # folder containing txt files
output_folder = "../encoded_signs"  # output folder
os.makedirs(output_folder, exist_ok=True)

# Iterate over files
for filename in os.listdir(input_folder):
    if filename.endswith(".pb"):
        print(filename)
        # data preperation
        input_path = os.path.join(input_folder, filename)
        
        proto_data = ProtobufProcessor.load_protobuf_data(input_path)

        right_pose_tensor = torch.empty([0, 3, 3], dtype=torch.float32)
        left_pose_tensor = torch.empty([0, 3, 3], dtype=torch.float32)
        right_finger_tensor = torch.empty([0, 7], dtype=torch.float32)
        left_finger_tensor = torch.empty([0, 7], dtype=torch.float32)

        for frame in proto_data.frames:
            pose_landmarks = ProtobufProcessor.extract_landmark_coordinates(frame, "pose_landmarks", range(10))
            selected_landmarks = ProtobufProcessor.normalize_body_landmarks(pose_landmarks, False)
            right_pose_tensor = torch.cat((right_pose_tensor, selected_landmarks), dim=0)
            selected_landmarks = ProtobufProcessor.normalize_body_landmarks(pose_landmarks, True)
            left_pose_tensor = torch.cat((left_pose_tensor, selected_landmarks), dim=0)

            right_finger_landmarks = ProtobufProcessor.extract_landmark_coordinates(frame, "right_hand_landmarks", range(21))
            selected_landmarks = ProtobufProcessor.normalize_hand_landmarks(right_finger_landmarks, False)
            right_finger_tensor = torch.cat((right_finger_tensor, selected_landmarks), dim=0)

            left_finger_landmarks = ProtobufProcessor.extract_landmark_coordinates(frame, "left_hand_landmarks", range(21))
            selected_landmarks = ProtobufProcessor.normalize_hand_landmarks(left_finger_landmarks, True)
            left_finger_tensor = torch.cat((left_finger_tensor, selected_landmarks), dim=0)

        num_frames = left_pose_tensor.shape[0]
        if num_frames < 16:
            print("Low frame count" + filename)
            padding = torch.empty([16 - num_frames, 3, 3 ]).to(device)
            left_pose_tensor = torch.cat((left_pose_tensor, padding), dim=0)
            right_pose_tensor = torch.cat((right_pose_tensor, padding), dim=0)
            padding = torch.empty([16 - num_frames, 7]).to(device)
            left_finger_tensor = torch.cat((left_finger_tensor, padding), dim=0)
            right_finger_tensor = torch.cat((right_finger_tensor, padding), dim=0)


        right_pose_tensor = right_pose_tensor.to(device)
        left_pose_tensor = left_pose_tensor.to(device)
        right_finger_tensor = right_finger_tensor.to(device)
        left_finger_tensor = left_finger_tensor.to(device)

        right_pose_tensor = batch_vectors_to_6D(right_pose_tensor)
        left_pose_tensor = batch_vectors_to_6D(left_pose_tensor)

        right_pose_tensor = fill_hand(right_pose_tensor)
        left_pose_tensor = fill_hand(left_pose_tensor)
        right_finger_tensor = fill_fingers(right_finger_tensor)
        left_finger_tensor = fill_fingers(left_finger_tensor)

        # feeding to model
        with torch.no_grad():
            right_hand_embedding = model_h(sliding_chunks(right_pose_tensor, 16, 4))
            left_hand_embedding = model_h(sliding_chunks(left_pose_tensor, 16, 4))
            right_finger_embedding = model_f(sliding_chunks(right_finger_tensor, 16, 4))
            left_finger_embedding = model_f(sliding_chunks(left_finger_tensor, 16, 4))
        embedding = torch.cat((right_hand_embedding, left_hand_embedding, right_finger_embedding, left_finger_embedding), dim=1)
        # Save everything to a JSON file
        output_path = os.path.join(output_folder, filename.replace(".pb", ".pt"))
        torch.save(embedding, output_path)
        print(" saved!")

print("\n✅ All files processed and saved in", output_folder)