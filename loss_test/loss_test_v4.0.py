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
import sys
import xml.etree.ElementTree as ET
import zipfile

# Third-party imports
import numpy as np
import pytorch_kinematics as pk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# Utility Functions
def extract_zip(zip_file_path, extract_to_path):
    """
    Extracts a zip file to the specified path

    Args:
        zip_file_path (str): Path to the zip file
        extract_to_path (str): Path to extract the zip file to
    """
    os.makedirs(extract_to_path, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)
    print(f"Files extracted to {extract_to_path}")


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
            return torch.zeros((1, 3)), torch.zeros((3, 3))  # Return zero if less than 2 landmarks
        
        origin = landmarks[1]#.clone().detach()
        # Compute the distance between the shoulder and elbow as the scale factor
        L = (torch.linalg.vector_norm(landmarks[1] - landmarks[3]) + torch.linalg.vector_norm(landmarks[3] - landmarks[5]))/2.0#.clone().detach()/2.0
        indices = [1, 3, 5, 7, 9]
        if left_side:
            indices = [0, 2, 4, 6, 8]
            landmarks[:, 0] *= -1
            origin = landmarks[0]#.clone().detach()
            # Compute the distance between the shoulder and elbow as the scale factor
            L = (torch.linalg.vector_norm(landmarks[0] - landmarks[2]) + torch.linalg.vector_norm(landmarks[4] - landmarks[2]))/2.0#.clone().detach()/2.0
        
        L = L if L > 0 else 1.0  # Prevent division by zero

        landmarks = (landmarks - origin) / L  # Wrist as reference

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
    human = torch.empty([0, 5, 3], dtype=torch.float32)

    for frame in proto_data.frames:
        # Extract and normalize landmarks for body, left hand, and right hand
        pose_landmarks = ProtobufProcessor.extract_landmark_coordinates(
            frame, "pose_landmarks", range(10)
        )  # Body landmarks
        
        # Normalize the extracted landmarks
        # selected_landmarks, h = ProtobufProcessor.normalize_body_landmarks(pose_landmarks, left_side)
        selected_landmarks = ProtobufProcessor.normalize_body_landmarks(pose_landmarks, left_side)

        pose_tensor = torch.cat((pose_tensor, selected_landmarks[:,2:]), dim=0)
        human = torch.cat((human, selected_landmarks), dim=0)
        # break

    pose_tensor = pose_tensor.to(device)

    # Ensure we have exactly 15 frames
    num_frames, feature_size, dim_size = pose_tensor.shape
    if num_frames < 15:
        print("Low frame count" + pb_path)
        padding = torch.zeros([15 - num_frames, feature_size, dim_size ]).to(device)
        pose_tensor = torch.cat((pose_tensor, padding), dim=0)
    elif num_frames > 15:
        pose_tensor = pose_tensor[:15, :, :]

    return pose_tensor, human


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
        output_positions = torch.zeros(
            batch_size, seq_len, 3, 3
        ).to(device)
        robot_positions = torch.zeros(
            batch_size, seq_len, 5, 3
        ).to(device)

        for b in range(batch_size):
            for t in range(seq_len):
                joint_values = batch_joint_values[b, t, :]
                full_joint_values = torch.zeros(len(self.all_joints))
                for i, joint_name in enumerate(self.selected_joints):
                    if joint_name in self.joint_limits:
                        lower, upper = self.joint_limits[joint_name]
                        denormalized_value = (joint_values[i] * (upper - lower)) + lower
                        full_joint_values[self.all_joints.index(joint_name)] = denormalized_value
                fk_result = self.robot_chain.forward_kinematics(
                    full_joint_values.unsqueeze(0)
                )

                body_L = (torch.linalg.vector_norm((fk_result["right_Shoulder_2"].get_matrix()[0, :3, 3]) - (fk_result["right_Forearm_1"].get_matrix()[0, :3, 3]))+torch.linalg.vector_norm((fk_result["right_Wrist"].get_matrix()[0, :3, 3]) - (fk_result["right_Forearm_1"].get_matrix()[0, :3, 3])))/2.0
                origin = (fk_result["right_Shoulder_2"].get_matrix()[0, :3, 3])
                
                output_positions[b, t, 0] = (fk_result["right_Wrist"].get_matrix()[0, :3, 3] - origin) / body_L
                output_positions[b, t, 1] = (fk_result["right_Finger_1_1"].get_matrix()[0, :3, 3] - origin) / body_L
                output_positions[b, t, 2] = (fk_result["right_Finger_4_1"].get_matrix()[0, :3, 3] - origin) / body_L

                robot_positions[b, t, 0] = (fk_result["right_Shoulder_2"].get_matrix()[0, :3, 3] - origin) / body_L
                robot_positions[b, t, 1] = (fk_result["right_Forearm_1"].get_matrix()[0, :3, 3] - origin) / body_L
                robot_positions[b, t, 2] = (fk_result["right_Wrist"].get_matrix()[0, :3, 3] - origin) / body_L
                robot_positions[b, t, 3] = (fk_result["right_Finger_1_1"].get_matrix()[0, :3, 3] - origin) / body_L
                robot_positions[b, t, 4] = (fk_result["right_Finger_4_1"].get_matrix()[0, :3, 3] - origin) / body_L

        return output_positions, robot_positions

def visualize(human, robot):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev = 0, azim = 90)

    ax.set_box_aspect([1,1,1])
    # Set plot limits
    m = max(torch.max(abs(robot)), torch.max(abs(human)))/2*1.5
    ax.set_xlim([-m, m])
    ax.set_ylim([-m, m])
    ax.set_zlim([-m, m])
    
    # Draw connections for human
    x = [human[0,0], human[1,0]]
    y = [human[0,1], human[1,1]]
    z = [human[0,2], human[1,2]]
    ax.plot(x, y, z, 'b-', linewidth=2)
    x = [human[1,0], human[2,0]]
    y = [human[1,1], human[2,1]]
    z = [human[1,2], human[2,2]]
    ax.plot(x, y, z, 'b-', linewidth=2)
    x = [human[2,0], human[3,0]]
    y = [human[2,1], human[3,1]]
    z = [human[2,2], human[3,2]]
    ax.plot(x, y, z, 'g-', linewidth=2)
    x = [human[2,0], human[4,0]]
    y = [human[2,1], human[4,1]]
    z = [human[2,2], human[4,2]]
    ax.plot(x, y, z, 'b-', linewidth=2)
    x = [human[3,0], human[4,0]]
    y = [human[3,1], human[4,1]]
    z = [human[3,2], human[4,2]]
    ax.plot(x, y, z, 'g-', linewidth=2)

    # Draw connections for robot
    x = [robot[0,0], robot[1,0]]
    y = [robot[0,1], robot[1,1]]
    z = [robot[0,2], robot[1,2]]
    ax.plot(x, y, z, 'r-', linewidth=2)
    x = [robot[1,0], robot[2,0]]
    y = [robot[1,1], robot[2,1]]
    z = [robot[1,2], robot[2,2]]
    ax.plot(x, y, z, 'r-', linewidth=2)
    x = [robot[2,0], robot[3,0]]
    y = [robot[2,1], robot[3,1]]
    z = [robot[2,2], robot[3,2]]
    ax.plot(x, y, z, 'k-', linewidth=2)
    x = [robot[2,0], robot[4,0]]
    y = [robot[2,1], robot[4,1]]
    z = [robot[2,2], robot[4,2]]
    ax.plot(x, y, z, 'r-', linewidth=2)
    x = [robot[3,0], robot[4,0]]
    y = [robot[3,1], robot[4,1]]
    z = [robot[3,2], robot[4,2]]
    ax.plot(x, y, z, 'k-', linewidth=2)
    
    plt.draw()
    plt.pause(100.0)

def compute_palm_quaternion(pose, eps=1e-8):
    """
    Compute palm orientation quaternion from wrist, little finger root, and index finger root positions.
    Args:
        wrist (torch.Tensor): [..., 3] wrist position
        little_root (torch.Tensor): [..., 3] little finger root position
        index_root (torch.Tensor): [..., 3] index finger root position
        eps (float): Small value to avoid division by zero
    Returns:
        quat (torch.Tensor): [..., 4] quaternion (w, x, y, z)
    """
    root = pose[:,0]
    little_root = pose[:,2] - root
    index_root = pose[:,1] - root
    original_shape = little_root.shape
    little_root = little_root.reshape(-1, 3)
    index_root = index_root.reshape(-1, 3)
    # Compute local axes (right, forward, up)
    right = F.normalize(index_root, dim=-1, eps=eps)  # X-axis
    temp_forward = little_root
    forward = F.normalize(temp_forward - torch.sum(temp_forward * right, dim=-1, keepdim=True) * right, 
                          dim=-1, eps=eps)  # Y-axis (orthogonal to X)
    up = torch.cross(right, forward, dim=-1)  # Z-axis

    # Construct rotation matrix [X, Y, Z]
    trace = right[:, 0] + forward[:, 1] + up[:, 2]
    quat = torch.zeros(little_root.shape[0], 4, device=device)

    # Case handling for numerical stability
    mask1 = trace > eps
    mask2 = (~mask1) & (right[:, 0] > forward[:, 1]) & (right[:, 0] > up[:, 2])
    mask3 = (~mask1) & (~mask2) & (forward[:, 1] > up[:, 2])
    mask4 = (~mask1) & (~mask2) & (~mask3)

    # Case 1: Standard computation
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2
        quat[mask1, 0] = 0.25 * s
        quat[mask1, 1] = (up[mask1, 1] - forward[mask1, 2]) / s
        quat[mask1, 2] = (right[mask1, 2] - up[mask1, 0]) / s
        quat[mask1, 3] = (forward[mask1, 0] - right[mask1, 1]) / s

    # Case 2: Right axis dominates
    if mask2.any():
        s = torch.sqrt(1.0 + right[mask2, 0] - forward[mask2, 1] - up[mask2, 2]) * 2
        quat[mask2, 0] = (up[mask2, 1] - forward[mask2, 2]) / s
        quat[mask2, 1] = 0.25 * s
        quat[mask2, 2] = (right[mask2, 1] + forward[mask2, 0]) / s
        quat[mask2, 3] = (right[mask2, 2] + up[mask2, 0]) / s

    # Case 3: Forward axis dominates
    if mask3.any():
        s = torch.sqrt(1.0 + forward[mask3, 1] - right[mask3, 0] - up[mask3, 2]) * 2
        quat[mask3, 0] = (right[mask3, 2] - up[mask3, 0]) / s
        quat[mask3, 1] = (right[mask3, 1] + forward[mask3, 0]) / s
        quat[mask3, 2] = 0.25 * s
        quat[mask3, 3] = (forward[mask3, 2] + up[mask3, 1]) / s

    # Case 4: Up axis dominates
    if mask4.any():
        s = torch.sqrt(1.0 + up[mask4, 2] - right[mask4, 0] - forward[mask4, 1]) * 2
        quat[mask4, 0] = (forward[mask4, 0] - right[mask4, 1]) / s
        quat[mask4, 1] = (right[mask4, 2] + up[mask4, 0]) / s
        quat[mask4, 2] = (forward[mask4, 2] + up[mask4, 1]) / s
        quat[mask4, 3] = 0.25 * s

    quat = F.normalize(quat, dim=-1, eps=eps)
    return quat.view(*original_shape[:-1], 4)

def batch_vectors_to_rotation(pose: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Convert batches of two in-plane vectors (u, v) to rotation matrices.
    
    Args:
        u: (B, 3) or (B, N, 3) tensor, primary axis (e.g., palm's local x-axis).
        v: (B, 3) or (B, N, 3) tensor, secondary axis (e.g., palm's local y-axis).
        eps: Small value to avoid division by zero.
    
    Returns:
        R: (B, 3, 3) or (B, N, 3, 3) rotation matrices.
    """
    root = pose[:,0]
    u = pose[:,2] - root
    v = pose[:,1] - root
    if u.dim() == 2:
        u, v = u.unsqueeze(1), v.unsqueeze(1)  # (B, 1, 3)
    
    # Step 1: Orthogonalize and normalize u and v
    e1 = u / (torch.norm(u, dim=-1, keepdim=True) + eps)  # (B, N, 3)
    v_perp = v - (torch.sum(e1 * v, dim=-1, keepdim=True) * e1)  # Gram-Schmidt
    e2 = v_perp / (torch.norm(v_perp, dim=-1, keepdim=True) + eps)  # (B, N, 3)
    
    # Step 2: Compute third axis (cross product)
    e3 = torch.cross(e1, e2, dim=-1)  # (B, N, 3)
    
    # Step 3: Stack to form rotation matrices
    R = torch.stack([e1, e2, e3], dim=-1)  # (B, N, 3, 3)
    
    return R.squeeze(1) if u.dim() == 3 else R  # Remove N if input was (B, 3)

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
    root = pose[:,0]
    u = pose[:,2] - root
    u = u / (torch.norm(u, dim=-1, keepdim=True) + eps)
    v = pose[:,1] - root
    v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
    
    # Orthogonalize v w.r.t. u (Gram-Schmidt)
    v_ortho = v - (torch.sum(u * v, dim=-1, keepdim=True) * u)
    v_ortho = v_ortho / (torch.norm(v_ortho, dim=-1, keepdim=True) + eps)
    
    # Stack u and v_ortho to form 6D representation
    six_d = torch.cat([u, v_ortho], dim=-1)
    return six_d

def palm_vectors(pose: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Convert two orthogonal vectors (u, v) on a plane to 6D rotation representation.
    Args:
        pose: Tensor of shape (B, N, 3, 3), first vector (e.g., palm's local x-axis).
    Returns:
        U, V ortho
    """
    # Ensure u and v are unit vectors (optional but recommended)
    root = pose[:,0]
    u = pose[:,2] - root
    u = u / (torch.norm(u, dim=-1, keepdim=True) + eps)
    v = pose[:,1] - root
    v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
    
    # Orthogonalize v w.r.t. u (Gram-Schmidt)
    v_ortho = v - (torch.sum(u * v, dim=-1, keepdim=True) * u)
    v_ortho = v_ortho / (torch.norm(v_ortho, dim=-1, keepdim=True) + eps)

    e = torch.cross(u, v_ortho, dim=-1)

    return u, v_ortho, e

def loss_fn(kine_output, pose_data, model_output, lambda_R = 1.0, lambda_vel = 0, eps: float = 1e-7):
    pose_loss = mse_loss(pose_data[:,0], kine_output[:,0])
    # print(pose_data[:,0])
    # print(kine_output[:,0])

    # velocity = torch.diff(model_output, dim=1)
    # vel_loss = torch.mean(velocity ** 2)

    # input_quat = compute_palm_quaternion(pose_data, eps=1e-8)
    # output_quat = compute_palm_quaternion(kine_output, eps=1e-8)
    # R_loss = torch.mean(1 - torch.abs(torch.sum(input_quat * output_quat, dim=-1)))  # 1 - |q1⋅q2|
    input_R = batch_vectors_to_rotation(pose_data, eps = eps)
    output_R = batch_vectors_to_rotation(kine_output, eps = eps)
    R = torch.matmul(output_R.transpose(-2, -1), input_R)
    trace = R.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    R_loss = torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps)).mean()
    # input_6d = batch_vectors_to_6D(pose_data, eps = eps)
    # output_6d = batch_vectors_to_6D(kine_output, eps = eps)
    # R_loss = torch.mean((output_6d - input_6d)**2)
    # u1, v1, e1 = palm_vectors(pose_data, eps = eps)
    # u2, v2, e2 = palm_vectors(kine_output, eps = eps)
    # R_loss = (3.0 - torch.cosine_similarity(u1, u2) - torch.cosine_similarity(v1, v2) - torch.cosine_similarity(e1, e2)).mean()
    
    loss = pose_loss + lambda_R * R_loss #+ lambda_vel * vel_loss

    return loss, pose_loss, R_loss#, vel_loss

def test_loss(sample_path, urdf_path, output):
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
    pose_data, human = _load_protobuf(sample_path)
    pose_data = pose_data[0,:,:].unsqueeze(0).to(device)
    human = human
    # print(pose_data)
    # print(human[0,:,:])

    # Forward pass
    fk = ForwardKinematics(urdf_path)
    kine_output, robot = fk.batch_forward_kinematics(output)
    # print(kine_output[0,:,:,:])
    # print(robot[0,0,:,:].squeeze())

    # Compute loss
    loss, pose_loss, quat_loss = loss_fn(kine_output[0,:,:,:], pose_data, output)
    print(f"Total Loss: {loss:.5f}\nPose Loss: {pose_loss:.5f}\nQuat Loss: {quat_loss:.5f}")

    visualize(human[0,:,:], robot[0,:,:,:].squeeze())


def main():
    """Main function to parse arguments and run the training or testing process"""
    
    sample_path = "/home/cedra/psl_project/loss test/Deafinno_1036_51-65_right.pb"#Deafinno_6_17-31_right.pb"

    joints = [[[0.75, 0.0, 0.8, 0.75, 0.45, 0.3]]]
    test_loss(
        sample_path=sample_path,
        urdf_path="/home/cedra/psl_project/rasa/hand.urdf",
        output=torch.tensor(joints)
    )

if __name__ == "__main__":
    main()
