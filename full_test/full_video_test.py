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
import xml.etree.ElementTree as ET

# Third-party imports
import numpy as np
import pytorch_kinematics as pk
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models.video as models
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

        origin = landmarks[1].clone().detach()
        # Compute the distance between the shoulder and elbow as the scale factor
        L = (torch.linalg.vector_norm(landmarks[1] - landmarks[3]) + torch.linalg.vector_norm(landmarks[3] - landmarks[5])).clone().detach()/2.0
        indices = [5, 7, 9]
        if left_side:
            indices = [4, 6, 8]
            landmarks[:, 0] *= -1
            origin = landmarks[0].clone().detach()
            # Compute the distance between the shoulder and elbow as the scale factor
            L = (torch.linalg.vector_norm(landmarks[0] - landmarks[2]) + torch.linalg.vector_norm(landmarks[4] - landmarks[2])).clone().detach()/2.0

        L = L if L > 0 else 1.0  # Prevent division by zero

        landmarks = (landmarks - origin) / L # Wrist as reference

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
        # selected_landmarks, h = ProtobufProcessor.normalize_body_landmarks(pose_landmarks, left_side)
        selected_landmarks = ProtobufProcessor.normalize_body_landmarks(pose_landmarks, left_side)
        selected_landmarks_for_viz = ProtobufProcessor.normalize_body_landmarks_for_viz(pose_landmarks, left_side)

        pose_tensor = torch.cat((pose_tensor, selected_landmarks), dim=0)
        viz_tensor = torch.cat((viz_tensor, selected_landmarks_for_viz), dim=0)
        # break

    pose_tensor = pose_tensor.to(device)

    # Ensure we have exactly 15 frames
    num_frames, feature_size, dim_size = pose_tensor.shape
    if num_frames < 15:
        print("Low frame count" + pb_path)
        padding = torch.zeros([15 - num_frames, feature_size, dim_size ]).to(device)
        pose_tensor = torch.cat((pose_tensor, padding), dim=0)

    return batch_vectors_to_6D(pose_tensor), viz_tensor

# def _load_video(video_path):
#     """
#     Load and process video frames using OpenCV.

#     Args:
#         video_path (string): Path to the video file

#     Returns:
#         torch.Tensor: Tensor containing video frames with shape  [15, 3, 273, 210]
#     """
#     # Open the video file
#     video, _, _ = read_video(video_path ,output_format="TCHW")
#     video_tensor = video.to(device)
#     video_tensor = video_tensor[:15, :3, :210, :273]
#     num_frames, ch, h, w = video_tensor.shape
#     if num_frames < 15:
#         print("Low frame count" + video_path)
#         padding = torch.zeros([15 - num_frames, ch, h, w ]).to(device)
#         video_tensor = torch.cat((video_tensor, padding), dim=0)

#     return video_tensor.permute(0,1,3,2) / 255.0


def _load_video_chunks(video_path):
    """
    load and process video frames (overlapping 15-frame chunks eg. 1...15 , 2...16 and so on).
    returjn:
        List[torch.Tensor]: List of tensors containing video chunks with shape [15, 3, 273, 210]
    """
    video, _, _ = read_video(video_path, output_format="TCHW")
    video_tensor = video.to(device)
    video_tensor = video_tensor[1:, :3, :210, :273]
    num_frames, ch, h, w = video_tensor.shape

    chunks = []

    #overlapping chunks: 1-15, 2-16, 3-17, etc.
    for start_frame in range(num_frames - 14):  # check if we have at least 15 frames
        end_frame = start_frame + 15 
        chunk = video_tensor[start_frame:end_frame]

        # if chunk has exactly 15 frames add it
        if chunk.shape[0] == 15:
            chunks.append(chunk.permute(0,1,3,2) / 255.0)

    # handle case where video has less than 15 frames
    if num_frames < 15:
        print(f"Low frame count {video_path}: {num_frames} frames")
        padding = torch.zeros([15 - num_frames, ch, h, w]).to(device)
        padded_video = torch.cat((video_tensor, padding), dim=0) 
        chunks.append(padded_video.permute(0,1,3,2) / 255.0)
    
    return chunks

class Adapter3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.down = nn.Conv3d(channels, channels//reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.up   = nn.Conv3d(channels//reduction, channels,     kernel_size=1, bias=False)
    def forward(self, x):
        return x + self.up(self.relu(self.down(x)))

# Neural Network Model Definition
class PoseVideoCNNRNN(nn.Module):
    def __init__(self):
        super(PoseVideoCNNRNN, self).__init__()
        
        # Backbone (frozen R3D-18)
        r3d = models.r3d_18(pretrained=True)
        for p in r3d.parameters():
            p.requires_grad = False

        self.backbone = nn.Sequential(*list(r3d.children())[:-2])

        # 2. Freeze all weights
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 3. Inject adapters into the *last* block of layer4
        layer4   = self.backbone[4]       # nn.Sequential of BasicBlocks
        last_blk = layer4[-1]        # the very last BasicBlock

        # 4. Find its conv2 and channel count
        conv2 = last_blk.conv2
        if isinstance(conv2, nn.Sequential):
            conv_layer = next(m for m in conv2 if isinstance(m, nn.Conv3d))
        else:
            conv_layer = conv2
        C_out = conv_layer.weight.shape[0]

        # 5. Create & attach adapter
        adapter = Adapter3D(C_out, reduction=16)
        last_blk.adapter = adapter

        # 6. Monkey-patch forward
        orig_fwd = last_blk.forward
        def new_forward(x, orig_fwd=orig_fwd, adapter=adapter):
            out = orig_fwd(x)
            return adapter(out)
        last_blk.forward = new_forward
        
        # Enhanced feature adapter (more channels + extra layer)
        self.feature_adapter = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=1),  # Increased channels
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024),  # Expanded layer
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, 512),  # Additional layer
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Preserved latent space layers (unchanged)
        self.fc_mu = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 16),
            nn.LeakyReLU(0.1),
        )
        self.fc_logvar = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 16),
            nn.LeakyReLU(0.1),
        )
        self.fc_decode = nn.Linear(16, 16 * 8 * 2 * 2)
        
        # Enhanced decoder (more channels + extra layer)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, 8, 2, 2)),
            nn.ConvTranspose3d(16, 128, kernel_size=3, stride=2, padding=1),  # Increased channels
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=1, padding=1),   # New layer
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1),    # Increased channels
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1),    # New layer
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1)      # Output layer
        )
        
        # Enhanced output projection (deeper architecture)
        self.fc_output = nn.Sequential(
            nn.Linear(29 * 5 * 5, 2048),  # Expanded layer
            nn.LeakyReLU(0.1),
            nn.Linear(2048, 1024),         # New layer
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 15 * 6),       # Final output layer (unchanged)
            nn.Sigmoid()                   # Preserved sigmoid
        )

    def forward(self, video_input, deterministic=False):
        B, C, T, H, W = video_input.shape
        
        # Input preprocessing (unchanged)
        x = video_input.permute(0, 2, 1, 3, 4)  
        x = x.reshape(-1, C, H, W)    
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        x = x.view(B, T, C, 112, 112)
        x = (x - 0.45) / 0.225
        
        # Feature extraction
        features = self.backbone(x)   
        h = self.feature_adapter(features)  
        
        # Latent space (unchanged)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        
        # Sampling
        embedding = mu
        if not deterministic:
            std = torch.exp(0.5 * logvar)
            embedding = embedding + torch.randn_like(std) * std
        
        # Decoding
        h = self.fc_decode(embedding)
        output = self.decoder(h)       
        output = output.squeeze(1)     
        
        # Output projection
        output_flat = output.reshape(B, -1)
        output_6d = self.fc_output(output_flat).view(B, 15, 6)
        
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
    six_d = torch.stack([root, root+0.2*u, root+0.2*v_ortho], dim=-1)
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

def test_model(video_dir, urdf_path, model_path, output_path):
    """
    Test the neural network model
    """
    with torch.no_grad():
        model = PoseVideoCNNRNN().to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        fk = ForwardKinematics(urdf_path)
        
        if isinstance(video_dir, str) and os.path.isdir(video_dir):
            video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
            video_dir = [f[:-4] for f in video_files]
        else:
            print("Invalid directory")
            return False
        
        f = open(os.path.join(output_path, "results.log"), 'w')
        
        for video_base_path in video_dir:
            video_name = os.path.basename(video_base_path)
            
            print(f"Processing video: {video_name}")
            f.write(f"====== {video_name} ======\n")
            
            video_file = video_base_path + ".mp4"
            if not os.path.exists(video_file):
                print(f"Video file not found!")
                f.write(f"Video file not found!")
                continue
            
            pb_file = video_base_path + ".pb"
            if not os.path.exists(pb_file):
                print(f"PB file not found!")
                f.write(f"PB file not found!")
                continue
            
            # new 
            video_chunks = _load_video_chunks(video_file)
            pose_full, viz_full = _load_protobuf(pb_file)

            # Create Hamming window (non-zero at edges to avoid division issues)
            window = torch.hamming_window(15, periodic=False, device=device)
            window = window.view(1, 15, 1)  # Shape: [15, 1] for broadcasting

            # Initialize accumulation buffers
            T = pose_full.shape[0]
            full_output = torch.zeros(T, 6, device=device)
            full_weights = torch.zeros(T, device=device)
            frame_indices = torch.arange(T, device=device)
            
            for chunk_idx, video_chunk in enumerate(video_chunks):
                start_frame = chunk_idx
                end_frame = chunk_idx + 15  # 15 frames total (0-14)
                
                pose_chunk = pose_full[start_frame:end_frame]
                
                video_input = video_chunk.unsqueeze(0).to(device)
                pose_input = pose_chunk.unsqueeze(0).to(device)
                
                model_output, _, _ = model(video_input, deterministic=True)

                weighted_chunks = model_output * window
                chunk_mask = (frame_indices >= start_frame) & (frame_indices < end_frame)
                full_output[chunk_mask] += weighted_chunks.view(-1, 6)
                full_weights[chunk_mask] += window.squeeze()
                
                pose_loss = loss_fn(fk, pose_input, model_output)
                
                chunk_name = f"{video_name}_{start_frame}_{end_frame}"
                f.write(f"Chunk {chunk_name}: Pose Loss: {pose_loss:.5f}\n")
                # print(f"Chunk {chunk_name}: Pose Loss: {pose_loss:.5f}")
                
                # viz_chunk = viz_full[start_frame:end_frame]
                # viz_input = viz_chunk.unsqueeze(0).to(device)
                # chunk_output_path = os.path.join(output_path, "fig", video_name, chunk_name)
                # if os.path.exists(chunk_output_path):
                #     for filename in os.listdir(chunk_output_path):
                #         file_path = os.path.join(chunk_output_path, filename)
                #         if os.path.isfile(file_path):
                #             os.remove(file_path)
                # else:
                #     os.makedirs(chunk_output_path)
                # fk.animate_movement(model_output, viz_input.cpu(), chunk_output_path)
            
            full_weights = torch.clamp(full_weights, min=1e-8)
            final_output = full_output / full_weights.unsqueeze(1)

            pose_loss = loss_fn(fk, pose_full.unsqueeze(0).to(device), final_output.unsqueeze(0))

            f.write(f"\n Total Pose Loss: {pose_loss:.5f}\n")
            print(f"\n Total Pose Loss: {pose_loss:.5f}")
            
            fig_output_path = os.path.join(output_path, "fig", video_name)
            if os.path.exists(fig_output_path):
                for filename in os.listdir(fig_output_path):
                    file_path = os.path.join(fig_output_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            else:
                os.makedirs(fig_output_path)
            fk.animate_movement(final_output.unsqueeze(0), viz_full.unsqueeze(0).cpu(), fig_output_path)
        f.close()
        print("Processing completed!")
        # print("zipping...")
        # shutil.make_archive(output_path, 'zip', os.path.dirname(output_path) )

def main():
    """Main function to parse arguments and run the testing process"""
    DEFAULT_NAME = "1_best"
    
    parser = argparse.ArgumentParser(
        description="Model Tester"
    )
    
    parser.add_argument(
        "--name", type=str, help="Saved Model Name: 34_best"
    )
    
    parser.add_argument(
        "--video_dir", type=str,
        help="path to video directory"
    )
    
    args = parser.parse_args()
    
    name = args.name if args.name is not None else DEFAULT_NAME
    model_path = "/home/cedra/psl_project/sign_language_pose_model_v3.1_"+name+".pth"
    
    urdf_path="/home/cedra/psl_project/rasa/hand.urdf"
    output_path="/home/cedra/psl_project/full_test"
    
    if not args.video_dir:
        video_dir = "/home/cedra/psl_project/clips/"
    else:
        video_dir = args.video_dir
    
    test_model(
        video_dir=video_dir,
        model_path=model_path,
        urdf_path=urdf_path,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
