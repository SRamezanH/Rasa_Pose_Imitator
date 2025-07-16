#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sign Language Recognition using Pose Estimation and Deep Learning.
This script processes pose data from protobuf files to test a model
for sign language recognition.
"""

import argparse
import glob

import os
import shutil
import sys
import xml.etree.ElementTree as ET


import numpy as np
import pytorch_kinematics as pk
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
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

device = torch.device("cpu")

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
        return torch.tensor([[-x*273.0/210.0, -z, -y]])  # Flip the coordinate system for compatibility

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
        """Normalize body landmarks using shoulders as reference points"""
        if landmarks.shape[0] < 10:
            return torch.zeros((1, 3, 3))  # Return zero if less than 2 landmarks
        
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

    @staticmethod
    def normalize_body_landmarks_for_viz(landmarks, left_side):
        """Normalize body landmarks using shoulders as reference points"""
        if landmarks.shape[0] < 10:
            return torch.zeros((1, 5, 3))  # Return zero if less than 2 landmarks

        origin = landmarks[1].clone().detach()
        # Compute the distance between the shoulder and elbow as the scale factor
        L = (torch.linalg.vector_norm(landmarks[1] - landmarks[3]) + torch.linalg.vector_norm(landmarks[3] - landmarks[5])).clone().detach()/2.0
        indices = [1, 3, 5, 7, 9]
        if left_side:
            indices = [0, 2, 4, 6, 8]
            # landmarks[:, 0] *= -1
            origin = landmarks[0].clone().detach()
            # Compute the distance between the shoulder and elbow as the scale factor
            L = (torch.linalg.vector_norm(landmarks[0] - landmarks[2]) + torch.linalg.vector_norm(landmarks[4] - landmarks[2])).clone().detach()/2.0

        L = L if L > 0 else 1.0  # Prevent division by zero

        landmarks = (landmarks - origin) / L # Wrist as reference

        # Normalize hand landmarks
        return landmarks[indices].unsqueeze(0)

def _load_protobuf(pb_path):
    """
    Load and process protobuf file containing pose data.

    Args:
        pb_path (string): Path to the protobuf file

    Returns:
        torch.Tensor: Tensor containing pose data with shape [15, feature_size]
                        where feature_size is the total number of landmarks * 3 (x, y, z)
    """
    proto_data = ProtobufProcessor.load_protobuf_data(pb_path)
    left_side = pb_path.endswith("_left.pb")

    pose_tensor = torch.empty([0, 3, 3], dtype=torch.float32)
    viz_tensor = torch.empty([0, 5, 3], dtype=torch.float32)

    for frame in proto_data.frames:
        # Extract and normalize landmarks for body, left hand, and right hand
        pose_landmarks = ProtobufProcessor.extract_landmark_coordinates(
            frame, "pose_landmarks", range(10)
        )  # Body landmarks
        
        # Normalize the extracted landmarks
        selected_landmarks = ProtobufProcessor.normalize_body_landmarks(pose_landmarks, left_side)
        selected_landmarks_for_viz = ProtobufProcessor.normalize_body_landmarks_for_viz(pose_landmarks, left_side)

        pose_tensor = torch.cat((pose_tensor, selected_landmarks), dim=0)
        viz_tensor = torch.cat((viz_tensor, selected_landmarks_for_viz), dim=0)

    pose_tensor = pose_tensor.to(device)

    # Ensure we have exactly 15 frames
    num_frames, feature_size, dim_size = pose_tensor.shape
    if num_frames < 15:
        print("Low frame count" + pb_path)
        padding = torch.zeros([15 - num_frames, feature_size, dim_size ]).to(device)
        pose_tensor = torch.cat((pose_tensor, padding), dim=0)
    elif num_frames > 15:
        pose_tensor = pose_tensor[:15, :, :]

    return batch_vectors_to_6D(pose_tensor), viz_tensor


# Neural Network Model Definition - Copied from model_test-v3.2.py
class PoseVideoCNNRNN(nn.Module):
    def __init__(self):
        super(PoseVideoCNNRNN, self).__init__()

        # Temporal RNN (LSTM)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),  # (8, 15, 3, 3)
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1), # (16, 8, 2, 2)
            nn.ReLU(),
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
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1),# output_padding=(1,1,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 1, kernel_size=3, stride=1, padding=1)
        )
        
        # transform 3x3 representation to 6d representation
        self.fc_output = nn.Sequential(
            nn.Linear(15 * 3 * 3, 15 * 6),
            nn.Sigmoid()
        )

    def forward(self, input_data, deterministic=False):
        batch_size, seq_len = input_data.shape[0], input_data.shape[1]

        # Handle both 6D input (B, T, 6) and 3D pose input (B, T, 3, 3)
        if len(input_data.shape) == 3:  # 6D input format (B, T, 6)
            # Reshape 6D input to 3x3 format for processing
            x = input_data.view(batch_size, seq_len, 2, 3)  # 6D -> (B, T, 2, 3)
            # Pad to make it (B, T, 3, 3) for compatibility with encoder
            padding = torch.zeros(batch_size, seq_len, 1, 3, device=input_data.device)
            x = torch.cat([x, padding], dim=2)  # (B, T, 3, 3)
        else:  # 3D pose input format (B, T, 3, 3)
            x = input_data
            
        # Process pose input: (B, T, 3, 3) -> (B, 1, T, 3, 3)
        x = x.permute(0, 3, 1, 2).unsqueeze(1)
        h = self.encoder(x)

        # Latent space
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)

        # Reparameterization trick
        embedding = mu
        if not deterministic:
            std = torch.exp(0.5 * logvar)
            embedding = embedding + torch.randn_like(std) * std

        h = self.fc_decode(embedding)
        output = self.decoder(h)
        output = output.squeeze(1)
        
        # transform to 6D representation
        output_flat = output.view(batch_size, -1)  # (B, 135)
        output_6d = self.fc_output(output_flat)  # (B, 90)
        output_6d = output_6d.view(batch_size, 15, 6)  # (B, 15, 6)

        return output_6d, mu, logvar

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
            "right_Shoulder_1",
            "right_Shoulder_2",
            "right_Shoulder_3",
            "right_Elbow_1",
            "right_Elbow_2",
            "right_Wrist"
        ]

        self.load_robot()
        self._precompute_body_references()

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

    def _precompute_body_references(self):
        """Precompute reference points and lengths using zero joint angles"""
        with torch.no_grad():
            zero_joints = torch.zeros(len(self.all_joints))
            fk_result = self.robot_chain.forward_kinematics(zero_joints)

            # Precompute reference points
            shoulder_pos = fk_result["right_Arm_1"].get_matrix()[:, :3, 3]
            forearm_pos = fk_result["right_Forearm_1"].get_matrix()[:, :3, 3]
            wrist_pos = fk_result["right_Wrist"].get_matrix()[:, :3, 3]

            # Calculate reference length
            self.L_ref = (torch.norm(shoulder_pos - forearm_pos) +
                          torch.norm(wrist_pos - forearm_pos)) / 2.0

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
        full_joints = torch.zeros((batch_size, seq_len, len(self.all_joints)))
        full_joints[:, :, self.selected_indices] = denormalized
        joints_flat = full_joints.view(-1, len(self.all_joints))
        fk_result = self.robot_chain.forward_kinematics(joints_flat)

        # 4. Extract and normalize positions
        shoulder_pos = fk_result["right_Arm_1"].get_matrix()[:, :3, 3]
        wrist_pos = fk_result["right_Wrist"].get_matrix()[:, :3, 3]
        finger1_pos = fk_result["right_Finger_1_1"].get_matrix()[:, :3, 3]
        finger4_pos = fk_result["right_Finger_4_1"].get_matrix()[:, :3, 3]

        normalized = torch.stack([
            (wrist_pos - shoulder_pos) / self.L_ref,
            (finger1_pos - shoulder_pos) / self.L_ref,
            (finger4_pos - shoulder_pos) / self.L_ref,
        ], dim=2).view(batch_size, seq_len, 3, 3)

        return batch_vectors_to_6D(normalized.to(device))

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
                full_joint_values = torch.zeros(len(self.all_joints))
                for i, joint_name in enumerate(self.selected_joints):
                    if joint_name in self.joint_limits:
                        lower, upper = self.joint_limits[joint_name]
                        denormalized_value = (joint_values[i] * (upper - lower)) + lower
                        full_joint_values[self.all_joints.index(joint_name)] = denormalized_value
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
        origin = fk_result["right_Arm_1"].get_matrix()[:, :3, 3].detach().numpy()
        for name, tf in fk_result.items():
            pos = tf.get_matrix()[:, :3, 3].detach().numpy()  # Extract translation component
            positions[name] = (pos - origin) / self.L_ref.detach().numpy()

        # Draw connections based on hierarchy
        for child, parent in self.link_parents.items():
            if parent in positions and child in positions:
                x = [positions[parent][0,0], positions[child][0,0]]
                y = [positions[parent][0,1], positions[child][0,1]]
                z = [positions[parent][0,2], positions[child][0,2]]
                ax.plot(x, y, z, 'b-', linewidth=2)
        # Draw joints
        # for name, pos in positions.items():
        # pose = pose *0.2
        dx = 0
        dy = 0
        dz = 0
        x = [dx + pose[0,0], dx + pose[1,0]]
        y = [dy + pose[0,1], dy + pose[1,1]]
        z = [dz + pose[0,2], dz + pose[1,2]]
        ax.plot(x, y, z, 'r-', linewidth=2)
        x = [dx + pose[1,0], dx + pose[2,0]]
        y = [dy + pose[1,1], dy + pose[2,1]]
        z = [dz + pose[1,2], dz + pose[2,2]]
        ax.plot(x, y, z, 'r-', linewidth=2)
        x = [dx + pose[2,0], dx + pose[3,0]]
        y = [dy + pose[2,1], dy + pose[3,1]]
        z = [dz + pose[2,2], dz + pose[3,2]]
        ax.plot(x, y, z, 'r-', linewidth=2)
        x = [dx + pose[2,0], dx + pose[4,0]]
        y = [dy + pose[2,1], dy + pose[4,1]]
        z = [dz + pose[2,2], dz + pose[4,2]]
        ax.plot(x, y, z, 'r-', linewidth=2)
        x = [dx + pose[3,0], dx + pose[4,0]]
        y = [dy + pose[3,1], dy + pose[4,1]]
        z = [dz + pose[3,2], dz + pose[4,2]]
        ax.plot(x, y, z, 'r-', linewidth=2)

        # Set plot limits
        all_pos = np.array(list(positions.values()))
        all_pos_2 = np.array(list(pose))
        max_range = max(np.max(np.abs(all_pos)), np.max(np.abs(all_pos_2))) * 1.1
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

        plt.close('all')

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
    six_d = torch.cat([u, v_ortho], dim=-1)  # Shape: (B, N, 6)
    return six_d

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
    # Position loss (e.g., difference in palm position)
    #  may select specific joints or axes; here we use the 1st vector of rotation matrix
    # pose_loss = mse_loss(model_output[:, :, 0], pose_data[:, :, 0])
    # pose_loss = mse_loss(model_output, pose_data)
    #errors = torch.abs(model_output - pose_data)

    # Convert pose_data to 6D representation
    kine_output = fk.batch_forward_kinematics(model_output)
    # calculate position loss using 6d representation
    errors = torch.abs(kine_output - pose_data)
    max_per_joint, _ = torch.max(
            errors.view(errors.size(0), errors.size(1), -1),
            dim=1  # Reduce across frames and coordinates
        )
    pose_loss = torch.mean(max_per_joint)

    return pose_loss

def _load_pose_chunks(pb_path):
    """
    Load and process pose data in overlapping 15-frame chunks.
    
    Args:
        pb_path (str): Path to the protobuf file
        
    Returns:
        List[torch.Tensor]: List of pose chunks with shape [15, 6]
    """
    pose_full, viz_full = _load_protobuf(pb_path)
    T = pose_full.shape[0]
    chunks = []
    
    # Create overlapping chunks: 0-14, 1-15, 2-16, etc.
    for start_frame in range(T - 14):  # Ensure we have at least 15 frames
        end_frame = start_frame + 15
        chunk = pose_full[start_frame:end_frame]
        
        # Add chunk if it has exactly 15 frames
        if chunk.shape[0] == 15:
            chunks.append(chunk)
    
    # Handle case where pose data has less than 15 frames
    if T < 15:
        print(f"Low frame count {pb_path}: {T} frames")
        padding = torch.zeros([15 - T, 6]).to(device)
        padded_pose = torch.cat((pose_full, padding), dim=0)
        chunks.append(padded_pose)
    
    return chunks, viz_full

def test_model_pose(pose_dir, urdf_path, model_path, output_path):
    """
    Test the neural network model using pose data input
    """
    with torch.no_grad():
        model = PoseVideoCNNRNN().to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        fk = ForwardKinematics(urdf_path)
        
        if isinstance(pose_dir, str) and os.path.isdir(pose_dir):
            pb_files = glob.glob(os.path.join(pose_dir, "*.pb"))
            pose_dir = [f[:-3] for f in pb_files]
        else:
            print("Invalid directory")
            return False
        
        f = open(os.path.join(output_path, "results.log"), 'w')
        
        for pose_base_path in pose_dir:
            pose_name = os.path.basename(pose_base_path)
            
            print(f"Processing pose data: {pose_name}")
            f.write(f"====== {pose_name} ======\n")
            
            pb_file = pose_base_path + ".pb"
            if not os.path.exists(pb_file):
                print(f"PB file not found!")
                f.write(f"PB file not found!")
                continue
            
            # Load pose chunks and visualization data
            pose_chunks, viz_full = _load_pose_chunks(pb_file)
            pose_full, _ = _load_protobuf(pb_file)  # Also load full pose data for ground truth

            # Create Hamming window for weighted aggregation
            window = torch.hamming_window(15, periodic=False, device=device)
            window = window.view(1, 15, 1)  # Shape: [1, 15, 1] for broadcasting

            # Initialize accumulation buffers for overlapping window aggregation
            T = pose_full.shape[0]
            full_output = torch.zeros(T, 6, device=device)
            full_weights = torch.zeros(T, device=device)
            frame_indices = torch.arange(T, device=device)
            
            # Process each pose chunk with overlapping windows
            for chunk_idx, pose_chunk in enumerate(pose_chunks):
                start_frame = chunk_idx
                end_frame = chunk_idx + 15  # 15 frames total (0-14)
                
                # Prepare pose tensor input for model
                pose_input = pose_chunk.unsqueeze(0).to(device)  # Add batch dimension
                
                # Forward pass through the model with deterministic inference
                model_output, _, _ = model(pose_input, deterministic=True)
                
                # Apply Hamming window weighting for smooth aggregation
                weighted_output = model_output.squeeze(0) * window.squeeze()  # Remove batch dim and apply window
                
                # Accumulate weighted results for overlapping regions
                chunk_mask = (frame_indices >= start_frame) & (frame_indices < end_frame)
                full_output[chunk_mask] += weighted_output
                full_weights[chunk_mask] += window.squeeze()
                
                # Calculate and log chunk-specific loss
                chunk_pose_loss = loss_fn(fk, pose_input, model_output)
                
                chunk_name = f"{pose_name}_{start_frame}_{end_frame-1}"
                f.write(f"Chunk {chunk_name}: Pose Loss: {chunk_pose_loss:.5f}\n")
                print(f"Chunk {chunk_name}: Pose Loss: {chunk_pose_loss:.5f}")
            
            # Normalize accumulated output by weights to handle overlapping regions
            full_weights = torch.clamp(full_weights, min=1e-8)  # Prevent division by zero
            final_output = full_output / full_weights.unsqueeze(1)
            
            # Calculate total aggregated loss using final output
            pose_loss = loss_fn(fk, pose_full.unsqueeze(0), final_output.unsqueeze(0))
            
            print(f"Total Pose Loss: {pose_loss.item():.6f}")
            f.write(f"Total Pose Loss: {pose_loss.item():.6f}\n")
            
            # Create output directory for visualizations
            output_dir = os.path.join(output_path, pose_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate visualizations using final aggregated output
            fk.animate_movement(final_output.unsqueeze(0), viz_full.unsqueeze(0), output_dir)
            
        f.close()
        print("Testing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pose-based sign language recognition model")
    parser.add_argument("--pose_dir", type=str, required=True, help="Directory containing pose protobuf files")
    parser.add_argument("--urdf_path", type=str, required=True, help="Path to URDF file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    test_model_pose(args.pose_dir, args.urdf_path, args.model_path, args.output_path)