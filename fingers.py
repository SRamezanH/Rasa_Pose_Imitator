#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
# Standard library imports
import os
import sys
import xml.etree.ElementTree as ET
import random

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
from soft_dtw import SoftDTW

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

mse_loss = nn.MSELoss()
dtw = SoftDTW(gamma=1.0, normalize=True) # just like nn.MSELoss()

# Constants
POSE_FEATURE_SIZE = 225  # 75 landmarks × 3 coordinates

# Protocol buffer definitions for pose data
# This is a key for unpacking protobuf files
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

# Make pose_data_pb2 importable
this_module = sys.modules[__name__]
pose_data_pb2 = this_module
sys.modules["pose_data_pb2"] = this_module

# Module for processing protobuf data
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
    def normalize_body_landmarks(landmarks, left_side):
        """
        Normalize all 10 body landmarks using shoulders as reference points.

        Args:
            landmarks (torch.Tensor): Shape (10, 3), 3D coordinates of body landmarks.
            left_side (bool): Whether the landmarks belong to the left arm.

        Returns:
            torch.Tensor: Normalized landmarks of shape (1, 10, 3)
        """
        if landmarks.shape[0] < 10:
            return torch.zeros((1, 7, 3))  # Return zero if not enough landmarks

        # Flip x for left side
        if left_side:
            landmarks[:, 0] *= -1
        origin = landmarks[0].clone().detach()  # left shoulder
        
        # Normalize all 10 landmarks using origin and scale
        normalized = landmarks - origin

        indices = [20, 16, 12, 8, 4, 17, 5]

        return normalized[indices].unsqueeze(0)  # Shape: (1, 10, 3)

def transform_points(data):
    """
    Transforms 3D points so that:
    - Origin stays at (0,0,0)
    - P1 (second last) becomes (1,0,0)
    - P2 (last) lies in positive y half of x-y plane
    Inputs:
        data: Tensor of shape (B, S, N+2, 3)
    Returns:
        Transformed data: Tensor of shape (B, S, N, 3)
    """
    # Split input
    P1 = data[..., -2, :]  # Shape: (B, S, 3)
    P2 = data[..., -1, :]  # Shape: (B, S, 3)
    points = data[..., :-2, :]  # Shape: (B, S, N, 3)

    # Step 1: Normalize P1 to unit vector
    v1 = P1  # (B, S, 3)
    v1_norm = torch.norm(v1, dim=-1, keepdim=True)  # (B, S, 1)
    v1_unit = v1 / (v1_norm + 1e-8)  # Avoid division by zero

    # Target direction is x-axis
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=data.device).expand_as(v1)

    # Compute axis of rotation (cross product) and angle (dot product)
    axis = torch.cross(v1_unit, x_axis, dim=-1)  # (B, S, 3)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    axis_unit = axis / (axis_norm + 1e-8)  # unit rotation axis

    dot = (v1_unit * x_axis).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(dot)  # (B, S, 1)

    # Rodrigues' rotation formula to build rotation matrix R1
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=data.device)  # (B, S, 3, 3)
    ax, ay, az = axis_unit.unbind(dim=-1)
    zero = torch.zeros_like(ax)

    K[..., 0, 1] = -az
    K[..., 0, 2] = ay
    K[..., 1, 0] = az
    K[..., 1, 2] = -ax
    K[..., 2, 0] = -ay
    K[..., 2, 1] = ax

    I = torch.eye(3, device=data.device).expand(*axis.shape[:-1], 3, 3)
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    K2 = K @ K

    R1 = I + sin * K + (1 - cos) * K2  # Shape: (B, S, 3, 3)

    # Apply R1 and scaling
    points = points / (v1_norm[..., None] + 1e-8)  # scale
    P2 = P2 / (v1_norm + 1e-8)

    # Apply R1 to points and P2
    points = torch.matmul(R1.unsqueeze(-3), points.unsqueeze(-1)).squeeze(-1)  # (B, S, N, 3)
    P2 = torch.matmul(R1, P2.unsqueeze(-1)).squeeze(-1)  # (B, S, 3)

    # Step 2: Rotate around x-axis to bring P2 into x-y plane (z=0)
    yz = P2[..., 1:]  # (B, S, 2)
    theta = torch.atan2(P2[..., 2], P2[..., 1])  # angle to rotate around x

    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)

    # Build rotation matrix R2 (x-axis rotation)
    R2 = torch.zeros(*theta.shape, 3, 3, device=data.device)
    R2[..., 0, 0] = 1
    R2[..., 1, 1] = cos_t
    R2[..., 1, 2] = -sin_t
    R2[..., 2, 1] = sin_t
    R2[..., 2, 2] = cos_t

    # Apply R2 to points
    points = torch.matmul(R2.unsqueeze(-3), points.unsqueeze(-1)).squeeze(-1)  # (B, S, N, 3)

    return (torch.nan_to_num(points, nan=0.0) + torch.tensor([-1.0, -0.5, 0.0])) / 2.0

# Dataset class for handling video and pose data
class PoseVideoDataset(Dataset):
    def __init__(self, pb_root_dir, video_root_dir, length=16):
        """
        Args:
            pb_root_dir (string): Directory with all the protobuf files
            video_root_dir (string): Directory with all the videos files
            length (int): total frames in a sample
        """
        self.pb_root_dir = pb_root_dir
        self.video_root_dir = video_root_dir
        self.length = length
        self.samples = []
        for pb_string in sorted(glob.glob(os.path.join(pb_root_dir, "*.pb"))):
            parts = os.path.basename(pb_string).replace('.pb', '').split('_')
            if parts[-1] == "fingers":
                video_name = os.path.join(video_root_dir, '_'.join(parts[:-3]) + ".mp4")
                start, end = map(int, parts[-3].split('-'))
                left = parts[-2].lower() == 'left'
                if end - start >= self.length:
                    for t in range(start, end-self.length+1, int(length/4)):
                        self.samples.append({
                            'pb_path': pb_string,
                            # 'video_path': video_name,
                            # 'video_start': t-1,
                            'pb_start': t-start,
                            'left': left
                        })
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): sample index
        """
        # Load video file
        sample = self.samples[idx]

        # Load video frames - no try/except as we've pre-validated
        #video_data = self._load_video(sample["video_path"], sample["video_start"], self.length)

        # Load pose data from protobuf
        pose_data = self._load_protobuf(sample["pb_path"], sample["pb_start"], self.length, sample["left"])

        return {"pose": pose_data}

    def _load_video(self, video_path, start, length):
        """
        Load and process video frames.

        Args:
            video_path (string): Path to the video file
            start (int): the start frame of video chunk
            length (int): the length of video chunk

        Returns:
            torch.Tensor: Tensor containing video frames with shape  [length, 3, 273, 210]
        """
        with torch.no_grad():
            #Open the video file
            video, _, _ = read_video(video_path, pts_unit="pts", output_format="TCHW")
            video = video[start:start+length]
            video_tensor = video.to(device, dtype=torch.float32)

            return torch.stack([self.clip_transform(frame) for frame in video_tensor])

    def _load_protobuf(self, pb_path, start, length, left):
        """
        Load and process protobuf file containing pose data.

        Args:
            pb_path (string): Path to the protobuf file
            start (int): the start frame of chunk
            length (int): the length of chunk
            left (bool): is it left hand data?

        Returns:
            torch.Tensor: Tensor containing pose data with shape [length, feature_size]
                         where feature_size is the total number of landmarks * 3 (x, y, z)
        """
        proto_data = ProtobufProcessor.load_protobuf_data(pb_path)

        pose_tensor = torch.empty([0, 7, 3], dtype=torch.float32)

        for frame in proto_data.frames[start:start+length]:
            pose_landmarks = ProtobufProcessor.extract_landmark_coordinates(
                frame,
                "left_hand_landmarks" if left else "right_hand_landmarks",
                range(21)
            )
            selected_landmarks = ProtobufProcessor.normalize_body_landmarks(pose_landmarks, left)
            pose_tensor = torch.cat((pose_tensor, selected_landmarks), dim=0)

        pose_tensor = transform_points(pose_tensor.unsqueeze(0)).squeeze()
        # Ensure we have exactly length frames
        num_frames = pose_tensor.shape[0]
        if num_frames < length:
            padding = torch.zeros([length - num_frames, 5, 3], dtype=torch.float32)
            pose_tensor = torch.cat((pose_tensor, padding), dim=0)
        elif num_frames > length:
            pose_tensor = pose_tensor[:length, :, :]

        return pose_tensor  # (length, 5, 3)


# Neural Network Model Definition

class PoseVideoCNNRNN(nn.Module):
    def __init__(self):
        super(PoseVideoCNNRNN, self).__init__()

        # --- Encoder:
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),   # → (B, 8, 15, 10, 3)
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),  # → (B, 16, 8, 5, 2)
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, 2, 2)),
            nn.Flatten()  # → (B, 16*8*5*2)
        )

        latent_dim = 16

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
        # self.fc_decode = nn.Linear(latent_dim, 16 * 8 * 5 * 2)

        # self.decoder = nn.Sequential(
        #     nn.Unflatten(1, (16, 8, 5, 2)),
        #     nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 1)),
        #     nn.ReLU(),
        #     nn.ConvTranspose3d(8, 1, kernel_size=3, stride=1, padding=1)
        # )

        # # Final linear output: map back to (15 time steps, 6 values each)
        # self.fc_output = nn.Sequential(
        #     nn.Linear(640, 16 * 7),
        #     nn.Sigmoid()
        # )

        self.hidden_size = 128
        self.num_layers = 2
        self.t = 16
        self.output_size = 7

        # Project the initial condition to the RNN's initial hidden state
        self.hidden_proj = nn.Linear(latent_dim, self.num_layers * self.hidden_size)
        self.cell_proj = nn.Linear(latent_dim, self.num_layers * self.hidden_size)
        
        # The core RNN cell (LSTM is chosen here)
        self.rnn = nn.LSTM(7, self.hidden_size, self.num_layers, batch_first=True)
        
        # The linear layer that maps the RNN output to the desired output size
        self.fc_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )

    def forward(self, pose_input, deterministic=False):
        # pose_input: (B, 15, 10, 3)
        # x = pose_input.permute(0, 4 - 1, 1, 2)  # → (B, 1, 15, 10, 3)
        x = pose_input.unsqueeze(1)
        # --- Encoder
        h = self.encoder(x)

        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)

        # --- Reparameterization
        embedding = mu
        if not deterministic:
            std = torch.exp(0.5 * logvar)
            embedding += torch.randn_like(std) * std

        # --- Decoder
        # h = self.fc_decode(embedding)
        # output = self.decoder(h)        # → (B, 1, 15, 10, 3)
        # output = output.squeeze(1)      # → (B, 15, 10, 3)

        # # --- Final linear mapping to 6D
        # flat = output.view(output.size(0), -1)       # → (B, 15×10×3)
        # out_6d = self.fc_output(flat)                # → (B, 15×6)
        # generated_sequence = out_6d.view(-1, 16, 7)               # → (B, 15, 6)

        batch_size = embedding.size(0)
        # 1. Initialize the hidden state using the condition vector
        # Project and reshape to (num_layers, batch_size, hidden_size)
        hidden_init = self.hidden_proj(embedding) # (batch, num_layers * hidden_size)
        hidden_init = hidden_init.view(batch_size, self.num_layers, self.hidden_size)
        hidden_init = hidden_init.permute(1, 0, 2) # (num_layers, batch, hidden_size)
        hidden_init = hidden_init.contiguous()
        
        cell_init = self.cell_proj(embedding) # (batch, num_layers * hidden_size)
        cell_init = cell_init.view(batch_size, self.num_layers, self.hidden_size)
        cell_init = cell_init.permute(1, 0, 2) # (num_layers, batch, hidden_size)
        cell_init = cell_init.contiguous()
        hidden = (hidden_init, cell_init) # Tuple for LSTM

        # 2. Prepare the first input (start token).
        # For the first step, we need an input. We can use a learned start token or zeros.
        # This is a learned parameter that gets optimized during training.
        if not hasattr(self, 'start_token'):
            self.start_token = nn.Parameter(torch.zeros(1, 1, self.output_size))
        # Expand the start token to match the batch size
        decoder_input = self.start_token.expand(batch_size, 1, self.output_size)

        # 3. Autoregressive generation loop
        outputs = []
        for _ in range(self.t):
            # decoder_input shape for each step: (batch, 1, input_size==7)
            rnn_out, hidden = self.rnn(decoder_input, hidden)
            # rnn_out shape: (batch, 1, hidden_size)

            # Predict the output for this time step
            out_step = self.fc_out(rnn_out) # (batch, 1, 7)
            outputs.append(out_step)

            # The output becomes the input for the next time step (teacher forcing is handled elsewhere)
            decoder_input = out_step.detach() # Use .detach() during inference to prevent backprop through the whole sequence

        # 4. Concatenate all time steps
        # Stack along the time dimension (dim=1)
        generated_sequence = torch.cat(outputs, dim=1) # (batch, t, 7)

        return generated_sequence, mu, logvar


class ForwardKinematics:
    def __init__(self, urdf_path):
        """
        Initialize a forward kinematics model using a URDF file.

        Args:
            urdf_path (str): Path to the URDF file.
        """
        self.urdf_path = urdf_path
        self.robot_chain = None
        self.all_joints = None
        self.joint_limits = {}

        # Define the joints used as input (7 in total)
        self.selected_joints = [
            "right_Finger_1_4",
            "right_Finger_1_4",
            "right_Finger_2_4",
            "right_Finger_3_4",
            "right_Finger_4_4",
            "thumb1",
            "thumb2"
        ]

        self.load_robot()

    def load_robot(self):
        """Load robot model from URDF file and extract joint limits."""
        with open(self.urdf_path, "r") as f:
            urdf_content = f.read()
        self.robot_chain = pk.build_chain_from_urdf(urdf_content)
        self.all_joints = self.robot_chain.get_joint_parameter_names()

        # Parse joint limits from URDF
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        for joint in root.findall("joint"):
            joint_name = joint.get("name")
            limit = joint.find("limit")
            if limit is not None and joint_name in self.all_joints:
                lower = float(limit.get("lower", "0"))
                upper = float(limit.get("upper", "0"))
                self.joint_limits[joint_name] = (lower, upper)

        # Get indices of selected joints
        self.selected_indices = [
            self.all_joints.index(j)
            for j in self.selected_joints
            if j in self.all_joints
        ]

        # Store joint limit tensors
        self.joint_lowers = torch.tensor([self.joint_limits[j][0] for j in self.selected_joints])
        self.joint_uppers = torch.tensor([self.joint_limits[j][1] for j in self.selected_joints])
        self.joint_ranges = self.joint_uppers - self.joint_lowers

        print("Kinematic chain initialized")
    
    def batch_forward_kinematics(self, batch_joint_values):
        """
        Compute forward kinematics for a batch of joint values.

        Args:
            batch_joint_values (torch.Tensor): Tensor of shape [B, T, 7] with normalized joint values.

        Returns:
            torch.Tensor: Tensor of shape [B, T, 10, 3] with normalized 3D positions of selected points.
        """
        batch_size, seq_len, _ = batch_joint_values.shape

        # Denormalize joint angles to actual values using limits
        denormalized = (batch_joint_values.cpu() * self.joint_ranges) + self.joint_lowers

        # Prepare full joint tensor with zeros for unused joints
        full_joints = torch.zeros((batch_size, seq_len, len(self.all_joints)))
        full_joints[:, :, self.selected_indices] = denormalized

        # Flatten for batch processing
        joints_flat = full_joints.view(-1, len(self.all_joints))
        fk_result = self.robot_chain.forward_kinematics(joints_flat)

        # Extract 7 key 3D positions
        wrist_pos = fk_result["right_Wrist"].get_matrix()[:, :3, 3]

        link_names = [
            "right_Finger_1_4",
            "right_Finger_2_4",
            "right_Finger_3_4",
            "right_Finger_4_4",
            #"thumb",
            "thumbband2",
            "right_Finger_1_1",
            "right_Finger_4_1"
        ]
        # Collect positions
        link_positions = [fk_result[k].get_matrix()[:, :3, 3] for k in link_names]

        # Normalize relative to shoulder position
        normalized = torch.stack([
            (pos - wrist_pos) for pos in link_positions
        ], dim=1)  # Shape: [B*T, 7, 3]

        normalized = transform_points(normalized.view(batch_size, seq_len, 7, 3).to(device))
        return normalized

def loss_fn(fk, pose_data, model_output, logvar, mu, gamma=1.0, normalize=True, lambda_kl=0.1, eps=1e-7):
    """
    Computes the loss between the predicted and actual pose data (no FK needed)

    Args:
        pose_data: Ground truth pose data [batch, 15, 3, 3]
        model_output: Model predictions [batch, 15, 6] => directly output from network in 6D form
        logvar: Log variance from the model
        mu: Mean from the model
        lambda_R: Weight for rotation loss
        lambda_kl: Weight for KL divergence loss
        lambda_vel: Weight for velocity loss
        eps: Small epsilon value to prevent division by zero

    Returns:
        Tuple of total loss and individual loss components
    """
    kine_output = fk.batch_forward_kinematics(model_output)

    pose_loss = mse_loss(kine_output, pose_data)

    batch_size, time_steps, n_series, dim = kine_output.shape
    
    # # Compute distance matrix
    # pred_expanded = kine_output.unsqueeze(2)  # (batch, time_pred, 1, n, dim)
    # target_expanded = pose_data.unsqueeze(1)    # (batch, 1, time_target, n, dim)
    # distance_matrix = torch.sum((pred_expanded - target_expanded) ** 2, dim=(-1, -2))
    
    # # Initialize cost matrix
    # cost_matrix = torch.zeros(batch_size, time_steps, time_steps, 
    #                          device=device)
    
    # # First row and column
    # cost_matrix[:, 0, 0] = distance_matrix[:, 0, 0]
    
    # # Fill first row
    # for j in range(1, time_steps):
    #     cost_matrix[:, 0, j] = cost_matrix[:, 0, j-1] + distance_matrix[:, 0, j]
    
    # # Fill first column
    # for i in range(1, time_steps):
    #     cost_matrix[:, i, 0] = cost_matrix[:, i-1, 0] + distance_matrix[:, i, 0]
    
    # # Fill the rest of the matrix
    # for i in range(1, time_steps):
    #     for j in range(1, time_steps):
    #         min_prev = torch.min(
    #             torch.stack([
    #                 cost_matrix[:, i-1, j],
    #                 cost_matrix[:, i, j-1],
    #                 cost_matrix[:, i-1, j-1]
    #             ], dim=1),
    #             dim=1
    #         )[0]
    #         cost_matrix[:, i, j] = min_prev + distance_matrix[:, i, j]
    
    # dtw_distances = cost_matrix[:, -1, -1]
    
    # dtw_loss = torch.mean(dtw_distances) / (2*time_steps)
    # dtw_loss = dtw(kine_output.view(batch_size, time_steps, -1), pose_data.view(batch_size, time_steps, -1)).mean()
    distances = torch.norm(kine_output - pose_data, dim=-1)
    max_loss = torch.max(distances)

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Combine all loss terms
    loss = pose_loss + lambda_kl * kl_loss
    return loss, pose_loss, max_loss, kl_loss

def lambda_scheduler(current_epoch, warmup_start=5, warmup_end=10, final_value=0.5):
    """Linear warmup for velocity loss coefficient"""
    if current_epoch < warmup_start:
        return 0.0
    elif warmup_start <= current_epoch <= warmup_end:
        # Linear interpolation between 0 and final_value
        progress = (current_epoch - warmup_start) / (warmup_end - warmup_start)
        return final_value * progress
    else:
        return final_value

def train_model(data_dir, test_dir, video_dir, urdf_path, num_epochs=10, batch_size=8, learning_rate=0.001):
    """
    Train the neural network model

    Args:
        data_dir (str): Path to the dataset directory
        data_dir (str): Path to the test dataset directory
        urdf_path (str): Path to the URDF file
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for the optimizer
    """
    print(f"\n Using device: {device}")
    if device.type == "cuda":
        cuda_id = torch.cuda.current_device()
        print(f"    - GPU: {torch.cuda.get_device_name(cuda_id)}")
        print(f"    - CUDA Capability: {torch.cuda.get_device_capability(cuda_id)}")
        print(f"    - Memory: {torch.cuda.get_device_properties(cuda_id).total_memory / (1024**3):.2f} GB")

    # Create dataset with validation to filter out corrupted videos
    dataset = PoseVideoDataset(data_dir, video_dir)
    test_dataset = PoseVideoDataset(test_dir, video_dir)
    
    if len(dataset) == 0:
        print("No valid videos found in the dataset. Please check your data files.")
        return

    # if len(test_dataset) == 0:
    #     print("No valid videos found in the test dataset. Please check your data files.")
    #     return

    print(f"Dataset loaded: {len(dataset)} valid samples")

    # Split dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    eval_size = int(0.2 * len(dataset))
    e = len(dataset) - train_size - eval_size
    train_dataset, eval_dataset, _ = random_split(dataset, [train_size, eval_size, e])

    # for i in random.sample(train_dataset.indices, 2):
    #     print(dataset.video_files[i])
    # for i in random.sample(eval_dataset.indices, 2):
    #     print(dataset.video_files[i])
    # for i in random.sample(test_dataset.indices, 2):
    #     print(dataset.video_files[i])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}, Test size: {len(test_dataset)}")

    if TORCHINFO_AVAILABLE:
            print("\nModel Architecture:")
            model = PoseVideoCNNRNN().to(device)
            # Get shape from a sample
            sample = dataset[0]
            #video_shape = sample["video"].shape
            pose_shape = sample["pose"].shape
            # Show model summary
            model_stats = summary(
                model,
                input_size=(batch_size,) + pose_shape, #input_size=(batch_size,) + video_shape,
                depth=4,
                device=device,
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                col_width=20,
                row_settings=["var_names", "depth"],
                verbose=0
                )
            if model_stats:
                print(f"\nTotal Parameters: {model_stats.total_params:,}")
                print(f"Trainable Parameters: {model_stats.trainable_params:,}")
                print(f"Non-trainable Parameters: {model_stats.total_params - model_stats.trainable_params:,}\n")

    # Create model, loss function, and optimizer
    model = PoseVideoCNNRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#geoopt.optim.RiemannianAdam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # Initialize forward kinematics
    fk = ForwardKinematics(urdf_path)

    # Import tqdm for progress bars
    from tqdm import tqdm
    import time

    # Track best model
    best_eval_loss = float('inf')
    overfit = 0
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)

    # Training loop
    for epoch in epoch_pbar:
        # Start time for this epoch
        start_time = time.time()

        # Training
        model.train()  # Set model to training mode
        total_loss = 0
        total_pose_loss = 0
        total_dtw_loss = 0
        total_kl_loss = 0

        # Create progress bar for batches
        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=False,
            position=1,
            total=len(train_loader)
        )

        lambda_kl = lambda_scheduler(epoch, warmup_start=5, warmup_end=15, final_value=0.1)#0.1 * (min(1, 0.005 * (epoch+1)))

        for i, batch in enumerate(batch_pbar):
            # video_data = batch["video"]  # Shape: [batch, 15, 3, 258, 196]
            pose_data = batch["pose"].to(device)  # Shape: [batch, 15, 3, 3] (ground truth)

            # model_output, mu, logvar = model(video_data)  # Output: [batch, 15, 26] (predicted)
            # pose data as input instead of video
            model_output, mu, logvar = model(pose_data)  # Output: [batch, 15, 6] (predicted)

            # Compute loss
            loss, pose_loss, dtw_loss, kl_loss = loss_fn(fk, pose_data, model_output, logvar, mu, lambda_kl = lambda_kl)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            # Update running loss
            loss_val = loss.item()
            pose_loss_val = pose_loss.item()
            dtw_loss_val = dtw_loss.item()
            kl_loss_val = kl_loss.item()

            total_loss += loss_val
            total_pose_loss += pose_loss_val
            total_dtw_loss += dtw_loss_val
            total_kl_loss += kl_loss_val

            # Update progress bar with current loss
            batch_pbar.set_postfix({
                'loss': f'{loss_val:.2f}',
                'pose_loss': f'{pose_loss_val:.2f}',
                'dtw_loss': f'{dtw_loss_val:.2f}',
                'kl_loss': f'{kl_loss_val:.2f}'
            })

        total_loss /= len(train_loader)
        total_pose_loss /= len(train_loader)
        total_dtw_loss /= len(train_loader)
        total_kl_loss /= len(train_loader)

        # Evaluation
        model.eval()  # Set model to evaluation mode
        eval_loss = 0
        eval_pose_loss = 0
        eval_dtw_loss = 0
        eval_kl_loss = 0

        # Progress bar for test batches
        eval_pbar = tqdm(
    	    eval_loader,
            desc="Evaluating",
            total=len(eval_loader),
            leave=False,
            position=1,
        )

        with torch.no_grad():  # No gradient computation
            for i, batch in enumerate(eval_pbar):
                # video_data = batch["video"]
                pose_data = batch["pose"].to(device)

                # Forward pass with pose data as input instead of video
                model_output, mu, logvar = model(pose_data)

                # Compute loss
                loss, pose_loss, dtw_loss, kl_loss = loss_fn(fk, pose_data, model_output, logvar, mu, lambda_kl = lambda_kl)

                loss_val = loss
                pose_loss_val = pose_loss
                dtw_loss_val = dtw_loss
                kl_loss_val = kl_loss

                eval_loss += loss_val
                eval_pose_loss += pose_loss_val
                eval_dtw_loss += dtw_loss_val
                eval_kl_loss += kl_loss_val

                # Update progress bar
                eval_pbar.set_postfix({
                    'loss': f'{loss_val:.2f}',
                    'pose_loss': f'{pose_loss_val:.2f}',
                    'dtw_loss': f'{dtw_loss_val:.2f}',
                    'kl_loss': f'{kl_loss_val:.2f}'
                })
        eval_loss /= len(eval_loader)
        eval_pose_loss /= len(eval_loader)
        eval_dtw_loss /= len(eval_loader)
        eval_kl_loss /= len(eval_loader)

        scheduler.step(eval_loss)

        # Check Overfitting
        if(eval_loss <= best_eval_loss):
            overfit = 0
            best_eval_loss = eval_loss
            # Save the trained model
            torch.save(model.state_dict(), f"sign_language_finger_model_v1.0_{epoch+1}_best.pth")
        else:
            overfit += 1
            torch.save(model.state_dict(), f"sign_language_finger_model_v1.0_{epoch+1}.pth")
            # if(overfit > 5):
            #     break

        epoch_time = time.time() - start_time

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train loss': f'{total_loss:.2f}',
            'eval loss': f'{eval_loss:.2f}',
            'time': f'{epoch_time:.1f}s'
        })

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s\n"
              f"Train Loss: {total_loss:.5f}, Pose Loss: {total_pose_loss:.5f}, dtw Loss: {total_dtw_loss:.5f}, "
              f"KL Loss: {total_kl_loss:.5f}\n"#, Vel Loss: {total_vel_loss:.5f}, dir Loss: {total_dir_loss:.5f}\n"
              f"Eval Loss: {eval_loss:.5f}, Pose Loss: {eval_pose_loss:.5f}, dtw Loss: {eval_dtw_loss:.5f}, "
              f"KL Loss: {eval_kl_loss:.5f}")#, Vel Loss: {eval_vel_loss:.5f}, dir Loss: {eval_dir_loss:.5f}")

    print(f"\nTraining completed in {num_epochs} epochs")

    # Evaluation on test data
    # model.load_state_dict(torch.load("/home/cedra/psl_project/sign_language_pose_model_v3.2_100.pth", weights_only=True))
    print("\nEvaluating model on test data...")
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    test_pose_loss = 0
    test_dtw_loss = 0
    test_kl_loss = 0

    # Progress bar for test batches
    test_pbar = tqdm(
        test_loader,
        desc="testing",
        total=len(test_loader),
        leave=False,
        position=1,
    )

    lambda_kl = 0.1
    lambda_max = 2.0

    with torch.no_grad():  # No gradient computation
        f = open("test.log", 'w')
        for i, batch in enumerate(test_pbar):
            # video_data = batch["video"]
            pose_data = batch["pose"].to(device)

            # Forward pass with pose data as input instead of video
            model_output, mu, logvar = model(pose_data)

            # Compute loss
            loss, pose_loss, dtw_loss, kl_loss = loss_fn(fk, pose_data, model_output, logvar, mu, lambda_kl = lambda_kl)

            loss_val = loss
            pose_loss_val = pose_loss
            dtw_loss_val = dtw_loss
            kl_loss_val = kl_loss

            f.write(str(batch["name"][0])+","+str(loss_val)+","+str(pose_loss_val)+","+str(dtw_loss_val)+","+str(kl_loss_val)+"\n")

            test_loss += loss_val
            test_pose_loss += pose_loss_val
            test_dtw_loss += dtw_loss_val
            test_kl_loss += kl_loss_val

            # Update progress bar
            test_pbar.set_postfix({
                'loss': f'{loss_val:.2f}',
                'pose_loss': f'{pose_loss_val:.2f}',
                'dtw_loss': f'{dtw_loss_val:.2f}',
                'kl_loss': f'{kl_loss_val:.2f}',
                # 'vel_loss': f'{vel_loss_val:.2f}',
                # 'dir_loss': f'{dir_loss_val:.2f}'
            })
    test_loss /= len(test_loader)
    test_pose_loss /= len(test_loader)
    test_dtw_loss /= len(test_loader)
    test_kl_loss /= len(test_loader)

    print(f"\nTest Loss: {test_loss:.5f}, Pose Loss: {test_pose_loss:.5f}, dtw Loss: {test_dtw_loss:.5f}, "
          f"KL Loss: {test_kl_loss:.5f}")

def main():
    """Main function to parse arguments and run the training or testing process"""

    # default values
    DEFAULT_DATA_DIR = "../dataset"
    DEFAULT_TEST_DIR = "../dataset/test"
    DEFAULT_VIDEO_DIR = "../1_clips"
    DEFAULT_URDF_PATH = "../rasa/hand.urdf"
    DEFAULT_NUM_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 64

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Sign Language Recognition using Pose Estimation"
    )
    parser.add_argument(
        "--data_dir", type=str, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--test_dir", type=str, help="Path to the test dataset directory"
    )
    parser.add_argument(
        "--video_dir", type=str, help="Path to the video dataset directory"
    )    
    parser.add_argument(
        "--urdf_path", type=str, help="Path to the robot URDF file (default: rasa/robot.urdf)",
    )
    parser.add_argument(
        "--num_epochs", type=int, help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training (default: 8)"
    )

    args = parser.parse_args()

    # Use command-line arguments if provided, otherwise use defaults
    data_dir = args.data_dir if args.data_dir is not None else DEFAULT_DATA_DIR
    test_dir = args.test_dir if args.test_dir is not None else DEFAULT_TEST_DIR
    video_dir = args.video_dir if args.video_dir is not None else DEFAULT_VIDEO_DIR
    urdf_path = args.urdf_path if args.urdf_path is not None else DEFAULT_URDF_PATH
    num_epochs = args.num_epochs if args.num_epochs is not None else DEFAULT_NUM_EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE

    print(f"Using data directory: {data_dir}")
    print(f"Using test directory: {test_dir}")
    print(f"Using video directory: {video_dir}")
    print(f"Using URDF path: {urdf_path}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")

    # Train model
    train_model(
        data_dir=data_dir,
        test_dir=test_dir,
        video_dir=video_dir,
        urdf_path=urdf_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

if __name__ == "__main__":
    main()
