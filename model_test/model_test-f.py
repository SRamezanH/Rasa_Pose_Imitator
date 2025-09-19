#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sign Language Recognition using Pose Estimation and Deep Learning.
This script processes video data and corresponding pose landmarks to train a model
for sign language recognition.
"""

import argparse
import glob
# Standard library imports
import os
import shutil
import sys
import math
import xml.etree.ElementTree as ET

# Third-party imports
import numpy as np
import pytorch_kinematics as pk
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_video
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
from torch.utils.data import DataLoader, Dataset, random_split

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
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            landmarks[:, 2] *= -1

        bones = ProtobufProcessor.compute_bone_vectors(landmarks)

        return torch.clamp(torch.tensor([
            (ProtobufProcessor.angle(bones[5], bones[17]) - ProtobufProcessor.angle(bones[5], bones[17]) + 0.2)/0.4,
            ProtobufProcessor.angle(bones[19], bones[16]) / math.pi,
            ProtobufProcessor.angle(bones[15], bones[12]) / math.pi,
            ProtobufProcessor.angle(bones[11], bones[8]) / math.pi,
            ProtobufProcessor.angle(bones[7], bones[4]) / math.pi,
            2 * ProtobufProcessor.angle(bones[2], bones[1]) / math.pi,
            2 * ProtobufProcessor.angle(bones[3], bones[2]) / math.pi
        ]), 0.0, 1.0).view(1, 7)
    
    @staticmethod
    def normalize_body_landmarks_for_viz(landmarks, left_side):
        """Normalize body landmarks using shoulders as reference points"""
        if landmarks.shape[0] < 10:
            return torch.zeros((1, 22, 3))  # Return zero if not enough landmarks

        # Flip x for left side
        if left_side:
            landmarks[:, 0] *= -1
        origin = landmarks[0].clone().detach()  # left shoulder
        
        # Normalize all 10 landmarks using origin and scale
        normalized = landmarks - origin

        indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 , 17, 5]
        
        return normalized[indices].unsqueeze(0)  # Shape: (1, 10, 3)

def scale_theta(points, scale=1.0):
    """
    Scale the angular component θ of a set of 3D points using cylindrical coordinates.
    
    Args:
        points: Tensor of shape (batch, t, n, 3) in Cartesian coordinates (x, y, z)
        theta_scale: Float, linear scale factor to apply to θ (angle)

    Returns:
        Tensor of shape (batch, t, n, 3), transformed back to Cartesian coordinates
    """
    x, y, z = points.unbind(-1)
    
    # Compute theta and scale it
    theta = torch.atan2(y, x)
    scale_expanded = scale.repeat(1, 1, points.size(2))
    theta_scaled = theta * scale_expanded
    
    # Compute new x, y coordinates
    rho = torch.sqrt(x**2 + y**2)
    x_scaled = rho * torch.cos(theta_scaled)
    y_scaled = rho * torch.sin(theta_scaled)
    
    return torch.stack([x_scaled, y_scaled, z], dim=-1)

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
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(v1)

    # Compute axis of rotation (cross product) and angle (dot product)
    axis = torch.cross(v1_unit, x_axis, dim=-1)  # (B, S, 3)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    axis_unit = axis / (axis_norm + 1e-8)  # unit rotation axis

    dot = (v1_unit * x_axis).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(dot)  # (B, S, 1)

    # Rodrigues' rotation formula to build rotation matrix R1
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=device)  # (B, S, 3, 3)
    ax, ay, az = axis_unit.unbind(dim=-1)
    zero = torch.zeros_like(ax)

    K[..., 0, 1] = -az
    K[..., 0, 2] = ay
    K[..., 1, 0] = az
    K[..., 1, 2] = -ax
    K[..., 2, 0] = -ay
    K[..., 2, 1] = ax

    I = torch.eye(3, device=device).expand(*axis.shape[:-1], 3, 3)
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
    R2 = torch.zeros(*theta.shape, 3, 3, device=device)
    R2[..., 0, 0] = 1
    R2[..., 1, 1] = cos_t
    R2[..., 1, 2] = -sin_t
    R2[..., 2, 1] = sin_t
    R2[..., 2, 2] = cos_t

    # Apply R2 to points
    points = torch.matmul(R2.unsqueeze(-3), points.unsqueeze(-1)).squeeze(-1)  # (B, S, N, 3)
    scale = (0.25*math.pi)/torch.acos((P2 * x_axis).sum(dim=-1, keepdim=True)/(torch.norm(P2, dim=-1, keepdim=True) + 1e-8))
    points = scale_theta(points, scale)

    return (torch.nan_to_num(points, nan=0.0) + torch.tensor([-1.0, -0.5, 0.0], device=device)) / 2.0

def _load_protobuf(pb_path, left):
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

    pose_tensor = torch.empty([0, 7], dtype=torch.float32)
    viz_tensor = torch.empty([0, 22, 3], dtype=torch.float32)

    for frame in proto_data.frames[0:16]:
        pose_landmarks = ProtobufProcessor.extract_landmark_coordinates(
            frame,
            "left_hand_landmarks" if left else "right_hand_landmarks",
            range(21)
        )

        selected_landmarks = ProtobufProcessor.normalize_body_landmarks(pose_landmarks, left)
        selected_landmarks_for_viz = ProtobufProcessor.normalize_body_landmarks_for_viz(pose_landmarks, left)

        pose_tensor = torch.cat((pose_tensor, selected_landmarks), dim=0)
        viz_tensor = torch.cat((viz_tensor, selected_landmarks_for_viz), dim=0)

    viz_tensor = transform_points(viz_tensor.unsqueeze(0).to(device)).squeeze()

    return pose_tensor, viz_tensor

class PoseVideoCNNRNN(nn.Module):
    def __init__(self):
        super(PoseVideoCNNRNN, self).__init__()

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

        latent_dim = 32

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

        # self.hidden_size = 128
        # self.num_layers = 2
        # self.t = 16
        # self.output_size = 7

        # # Project the initial condition to the RNN's initial hidden state
        # self.hidden_proj = nn.Linear(latent_dim, self.num_layers * self.hidden_size)
        # self.cell_proj = nn.Linear(latent_dim, self.num_layers * self.hidden_size)
        
        # # The core RNN cell (LSTM is chosen here)
        # self.rnn = nn.LSTM(7, self.hidden_size, self.num_layers, batch_first=True)
        
        # # The linear layer that maps the RNN output to the desired output size
        # self.fc_out = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.output_size),
        #     nn.Sigmoid()
        # )

    def forward(self, pose_input, deterministic=False):
        # pose_input: (B, 15, 10, 3)
        # x = pose_input.permute(0, 4 - 1, 1, 2)  # → (B, 1, 15, 10, 3)
        x = pose_input.unsqueeze(-1)         # (B, T, 7, 1)
        x = self.project(x)         # (B, T, 7, 3)
        x = x.unsqueeze(1)
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
        h = self.fc_decode(embedding)
        output = self.decoder(h)        # → (B, 1, 15, 10, 3)
        output = output.squeeze(1)      # → (B, 15, 10, 3)

        # # --- Final linear mapping to 6D
        flat = output.view(output.size(0), -1)       # → (B, 15×10×3)
        out_6d = self.fc_output(flat)                # → (B, 15×6)
        generated_sequence = out_6d.view(-1, 16, 7)               # → (B, 15, 6)

        # batch_size = embedding.size(0)
        # # 1. Initialize the hidden state using the condition vector
        # # Project and reshape to (num_layers, batch_size, hidden_size)
        # hidden_init = self.hidden_proj(embedding) # (batch, num_layers * hidden_size)
        # hidden_init = hidden_init.view(batch_size, self.num_layers, self.hidden_size)
        # hidden_init = hidden_init.permute(1, 0, 2) # (num_layers, batch, hidden_size)
        # hidden_init = hidden_init.contiguous()
        
        # cell_init = self.cell_proj(embedding) # (batch, num_layers * hidden_size)
        # cell_init = cell_init.view(batch_size, self.num_layers, self.hidden_size)
        # cell_init = cell_init.permute(1, 0, 2) # (num_layers, batch, hidden_size)
        # cell_init = cell_init.contiguous()
        # hidden = (hidden_init, cell_init) # Tuple for LSTM

        # # 2. Prepare the first input (start token).
        # # For the first step, we need an input. We can use a learned start token or zeros.
        # # This is a learned parameter that gets optimized during training.
        # if not hasattr(self, 'start_token'):
        #     self.start_token = nn.Parameter(torch.zeros(1, 1, self.output_size))
        # # Expand the start token to match the batch size
        # decoder_input = self.start_token.expand(batch_size, 1, self.output_size)

        # # 3. Autoregressive generation loop
        # outputs = []
        # for _ in range(self.t):
        #     # decoder_input shape for each step: (batch, 1, input_size==7)
        #     rnn_out, hidden = self.rnn(decoder_input, hidden)
        #     # rnn_out shape: (batch, 1, hidden_size)

        #     # Predict the output for this time step
        #     out_step = self.fc_out(rnn_out) # (batch, 1, 7)
        #     outputs.append(out_step)

        #     # The output becomes the input for the next time step (teacher forcing is handled elsewhere)
        #     decoder_input = out_step.detach() # Use .detach() during inference to prevent backprop through the whole sequence

        # # 4. Concatenate all time steps
        # # Stack along the time dimension (dim=1)
        # generated_sequence = torch.cat(outputs, dim=1) # (batch, t, 7)

        return generated_sequence, mu, logvar

class ForwardKinematics:
    def __init__(self, urdf_path):
        """
        Initialize a forward kinematics model using a URDF file

        Args:
            urdf_path (str): Path to the URDF file
        """
        self.urdf_path = urdf_path
        self.robot_chain = None
        self.all_joints = None
        self.joint_limits = {}
        self.selected_joints = [
            "right_Finger_1_1",
            "right_Finger_1_4",
            "right_Finger_2_4",
            "right_Finger_3_4",
            "right_Finger_4_4",
            "thumb1",
            "thumb2"
        ]
        # self.mimicjoints = {
        #     "right_Finger_1_1":["right_Finger_2_1", "right_Finger_3_1", "right_Finger_4_1"],
        #     "right_Finger_1_4":["right_Finger_1_3", "right_Finger_1_2"],
        #     "right_Finger_2_4":["right_Finger_2_3", "right_Finger_2_2"],
        #     "right_Finger_3_4":["right_Finger_3_3", "right_Finger_3_2"],
        #     "right_Finger_4_4":["right_Finger_4_3", "right_Finger_4_2"]
        # }

        self.A = torch.zeros(24, 7, device=device)
        self.A[6, 0] = 1.0
        self.A[7, 1] = 1.1
        self.A[8, 1] = 1.0
        self.A[9, 1] = 1.0
        self.A[10, 0] = 1.0
        self.A[11, 2] = 1.1
        self.A[12, 2] = 1.0
        self.A[13, 2] = 1.0
        self.A[14, 0] = -1.0
        self.A[15, 3] = 1.1
        self.A[16, 3] = 1.0
        self.A[17, 3] = 1.0
        self.A[18, 0] = -1.0
        self.A[19, 4] = 1.1
        self.A[20, 4] = 1.0
        self.A[21, 4] = 1.0
        self.A[22, 5] = 1.0
        self.A[23, 6] = 1.0

        self.load_robot()

    def load_robot(self):
        """Load robot model from URDF file and extract joint limits"""
        # Load URDF and build kinematic chain
        with open(self.urdf_path, "r") as f:
            urdf_content = f.read()
        self.robot_chain = pk.build_chain_from_urdf(urdf_content)
        self.all_joints = self.robot_chain.get_joint_parameter_names()

        # Extract joint limits
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        for joint in root.findall("joint"):
            joint_name = joint.get("name")
            limit = joint.find("limit")
            if limit is not None and joint_name in self.all_joints:
                lower = float(limit.get("lower", "0"))
                upper = float(limit.get("upper", "0"))
                self.joint_limits[joint_name] = (lower, upper)
        
        self.selected_indices = [
            self.all_joints.index(j) 
            for j in self.selected_joints 
            if j in self.all_joints
        ]

        # Parse URDF to get link hierarchy (simplified)
        self.link_parents = self._parse_urdf_hierarchy()

        # Convert to tensors in order of selected_joints
        self.joint_lowers = torch.tensor([self.joint_limits[j][0] for j in self.selected_joints])
        self.joint_uppers = torch.tensor([self.joint_limits[j][1] for j in self.selected_joints])
        self.joint_ranges = self.joint_uppers - self.joint_lowers

        print("Kinematic chain initialized")

    def batch_forward_kinematics(self, batch_joint_values):
        """
        Compute forward kinematics for a batch of joint values

        Args:
            batch_joint_values (torch.Tensor): Tensor with shape [batch_size, seq_len, num_joints]
                containing joint values

        Returns:
            torch.Tensor: Tensor with shape [batch_size, seq_len, num_links*3] containing 3D positions
                of important links
        """
        batch_size, seq_len, _ = batch_joint_values.shape

        denormalized = (batch_joint_values.cpu() * self.joint_ranges) + self.joint_lowers
        full_joints = torch.matmul(denormalized, self.A.T)
        # full_joints = torch.zeros((batch_size, seq_len, len(self.all_joints)))
        # full_joints[:, :, self.selected_indices] = denormalized
            
        joints_flat = full_joints.view(-1, len(self.all_joints))
        fk_result = self.robot_chain.forward_kinematics(joints_flat)
        
        # 4. Extract and normalize positions
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
    
    def forward_kinematics(self, batch_joint_values):
        """
        Compute forward kinematics for a batch of joint values

        Args:
            batch_joint_values (torch.Tensor): Tensor with shape [batch_size, seq_len, num_joints]
                containing joint values

        Returns:
            torch.Tensor: Tensor with shape [batch_size, seq_len, num_links*3] containing 3D positions
                of important links
        """
        batch_size, seq_len, _ = batch_joint_values.shape
        output_positions = []

        for b in range(batch_size):
            for t in range(seq_len):
                joint_values = batch_joint_values[b, t, :]
                denormalized = (joint_values.cpu() * self.joint_ranges) + self.joint_lowers
                full_joint_values = torch.matmul(denormalized, self.A.T)
                # full_joint_values = torch.zeros(len(self.all_joints))
                # for i, joint_name in enumerate(self.selected_joints):
                #     if joint_name in self.joint_limits:
                #         lower, upper = self.joint_limits[joint_name]
                #         denormalized_value = (joint_values[i] * (upper - lower)) + lower
                #         full_joint_values[self.all_joints.index(joint_name)] = denormalized_value
                #         # if joint_name in self.mimicjoints:
                #         #     for j in self.mimicjoints[joint_name]:
                #         #         full_joint_values[self.all_joints.index(j["name"])] = denormalized_value * j["mul"]
                output_positions.append(self.robot_chain.forward_kinematics(full_joint_values.unsqueeze(0)))

        return output_positions
    
    def _parse_urdf_hierarchy(self) -> Dict[str, str]:
        """
        Simplified URDF parser to get link parent-child relationships
        Returns dictionary of {child: parent}
        """
        # In practice, you'd want to use a proper URDF parser here
        # This is a simplified version that might need adjustment
        
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        hierarchy = {}
        for joint in root.findall('joint'):
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            hierarchy[child] = parent
        
        return hierarchy

    def visualize(self, fk_result, pose, title, path):
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.view_init(elev = 0, azim = 90)

        ax.set_box_aspect([1,1,1])
        # Clear previous plot
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Compute forward kinematics
        # Compute forward kinematics
        positions = {}
        origin = fk_result["right_Wrist"].get_matrix()[:, :3, 3]
        f11 = fk_result["right_Finger_1_1"].get_matrix()[:, :3, 3] - origin
        f41 = fk_result["right_Finger_4_1"].get_matrix()[:, :3, 3] - origin
        for name, tf in fk_result.items():
            if name not in ["right_Forearm_2", "right_Forearm_1", "right_Arm_2", "right_Arm_1", "right_Shoulder_2", "right_Shoulder_1"]:
                pos = torch.cat((tf.get_matrix()[:, :3, 3] - origin, f11, f41), dim=0)
                positions[name] = transform_points(pos.unsqueeze(0).unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()
        
        # Draw connections based on hierarchy
        for child, parent in self.link_parents.items():
            if parent in positions and child in positions:
                x = [positions[parent][0], positions[child][0]]
                y = [positions[parent][1], positions[child][1]]
                z = [positions[parent][2], positions[child][2]]
                ax.plot(x, y, z, 'b-', linewidth=2)
        # Draw joints
        # for name, pos in positions.items():
        # pose = pose *0.2
        for i in range(pose.size(0)):
            if i in [0,4,8,12,16]:
                x = [-0.5, pose[i,0]]
                y = [-0.25, pose[i,1]]
                z = [0, pose[i,2]]
            else:
                x = [pose[i-1,0], pose[i,0]]
                y = [pose[i-1,1], pose[i,1]]
                z = [pose[i-1,2], pose[i,2]]
            ax.plot(x, y, z, 'r-', linewidth=2)
        
        # Set plot limits
        all_pos = np.array(list(positions.values()))
        all_pos_2 = np.array(list(pose))
        max_range = 1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        ax.set_title(title)
        ax.view_init(elev = 0, azim = 90)
        # plt.draw()
        plt.savefig(os.path.join(path, f'front {title}.png'))
        ax.view_init(elev = 0, azim = 180)
        plt.savefig(os.path.join(path, f'right {title}.png'))
        ax.view_init(elev = 90, azim = 90)
        plt.savefig(os.path.join(path, f'up {title}.png'))

        # plt.pause(0.3)
    
    def animate_movement(self, batch_joint_values, pose, path):
        """
        Animate a sequence of joint angle configurations
        
        Args:
            joint_angle_sequence: List of dictionaries containing joint angles
        """
        positions = self.forward_kinematics(batch_joint_values)
        for i, fk in enumerate(positions):
            self.visualize(fk, pose[0,i], f"Pose {i+1} of {len(positions)}", path)
            # plt.pause(1.0)

def loss_fn(fk, pose_data, model_output, lambda_R=1.0, lambda_vel=10.0, eps=1e-7, single=False):
    """
    Computes the loss between the predicted and actual pose data (no FK needed)
    
    Args:
        pose_data: Ground truth pose data [batch, 15, 3, 3]
        model_output: Model predictions [batch, 15, 6] => directly output from network in 6D form
        lambda_R: Weight for rotation loss
        lambda_kl: Weight for KL divergence loss
        lambda_vel: Weight for velocity loss
        eps: Small epsilon value to prevent division by zero
        
    Returns:
        Tuple of total loss and individual loss components
    """
    pose_loss = mse_loss(model_output, pose_data)

    # distances = torch.norm(kine_output - pose_data, dim=-1)
    # max_loss = torch.max(distances)

    errors = torch.abs(model_output - pose_data)
    max_per_joint, _ = torch.max(
            errors.view(errors.size(0), errors.size(1), -1),
            dim=1  # Reduce across frames and coordinates
        )
    max_loss = torch.mean(max_per_joint)

    return pose_loss

def test_model(sample_path, urdf_path, model_path, output_path):
    """
    Test the neural network model
    """

    # model = PoseVideoCNNRNN().to(device)
    # model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')), strict=False)
    # model.eval()

    fk = ForwardKinematics(urdf_path)

    f = open(os.path.join(output_path, "results.log"), 'w')

    for sample in sample_path:
        left = os.path.basename(sample).replace('.pb', '').split('_')[-2].lower() == 'left'

        pose, viz = _load_protobuf(sample+".pb", left)
        pose = pose.unsqueeze(0).to(device)
        viz = viz.unsqueeze(0).to(device)
        
        # Forward pass through the model to get 6D representation
        # model_output, _, _ = model(pose, deterministic=True)
        model_output = pose

        f.write("------ "+sample+" ------\n")
        print("------ "+sample+" ------\n")
        
        for i in range(16):
            pose_loss = loss_fn(fk, pose[:,i].unsqueeze(0), model_output[:,i].unsqueeze(0), single=True)
            f.write(f"- frame {i} Pose Loss: {pose_loss:.5f}\n")
        
        pose_loss = loss_fn(fk, pose[:,:], model_output)
        f.write(f"\nPose Loss: {pose_loss:.5f}\n")
        print(f"\nPose Loss: {pose_loss:.5f}\n")

        path = os.path.join(output_path,"fig",sample.split("/")[-1])
        if os.path.exists(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(path)
        fk.animate_movement(model_output, viz.cpu(), path)

    # print("zipping...")
    # shutil.make_archive(output_path, 'zip', os.path.dirname(output_path) )

def main():
    """Main function to parse arguments and run the training or testing process"""
    
    samples = ["../dataset/test/Deafinno_6_574-600_left_fingers",
                "../dataset/test/Deafinno_6_737-758_left_fingers",
                "../dataset/test/Deafinno_6_738-759_right_fingers",
                "../dataset/test/Deafinno_1036_3-26_right_fingers",
                "../dataset/test/Deafinno_1036_147-239_left_fingers",
                "../dataset/test/yalda_3622_1-141_right_fingers",
                "../dataset/test/IRIB2_79_8517_204-234_right_fingers"]

    model_path = "sign_language_finger_model_v1.0_22.pth"

    urdf_path="../rasa/hand.urdf"

    output_path="model_test/"

    test_model(
        sample_path=samples,
        model_path=model_path,
        urdf_path=urdf_path,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
