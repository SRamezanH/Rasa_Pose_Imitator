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
        return -x, -z, -y  # Flip the coordinate system for compatibility

    @staticmethod
    def extract_landmark_coordinates(frame, landmark_type, indices):
        """Extract and transform landmark coordinates from a specific frame"""
        transformed_landmarks = []
        landmarks = getattr(frame, landmark_type, [])

        for i in indices:
            if i < len(landmarks):
                transformed_landmarks.append(
                    ProtobufProcessor.transform_coordinates(landmarks[i])
                )
            else:
                transformed_landmarks.append(
                    (0.0, 0.0, 0.0)
                )  # Default value for missing landmarks

        return transformed_landmarks

    @staticmethod
    def normalize_body_landmarks(landmarks):
        """Normalize body landmarks using shoulders as reference points"""
        if len(landmarks) < 2:
            return np.zeros((len(landmarks), 3))  # Return zero if less than 2 landmarks

        # Use the average of both shoulders as the origin
        origin = (np.array(landmarks[0]) + np.array(landmarks[1])) / 2
        # Compute the distance between the shoulders as the scale factor
        L = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[1]))
        L = L if L > 0 else 1.0  # Prevent division by zero


        # Normalize landmarks by subtracting the origin and dividing by scale factor
        return (np.array(landmarks) - origin) / L

    @staticmethod
    def normalize_hand_landmarks(landmarks):
        """Normalize hand landmarks using the wrist as the reference and middle fingertip for scale"""
        if len(landmarks) < 18:
            return np.zeros(
                (len(landmarks), 3)
            )  # Return zero if less than 18 landmarks

        origin = np.array(landmarks[0])  # Wrist as reference
        # Compute the distance between wrist and middle fingertip as scale factor
        L = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[17]))
        L = L if L > 0 else 1.0  # Prevent division by zero

        # Normalize hand landmarks
        return (np.array(landmarks) - origin) / L

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

    all_frames = []

    for frame in proto_data.frames:
        # Extract and normalize landmarks for body, left hand, and right hand
        pose_landmarks = ProtobufProcessor.extract_landmark_coordinates(
            frame, "pose_landmarks", range(6)
        )  # Body landmarks
        left_hand_landmarks = ProtobufProcessor.extract_landmark_coordinates(
            frame, "left_hand_landmarks", range(21)
        )  # Left hand landmarks
        right_hand_landmarks = ProtobufProcessor.extract_landmark_coordinates(
            frame, "right_hand_landmarks", range(21)
        )  # Right hand landmarks

        # Normalize the extracted landmarks
        normalized_pose = ProtobufProcessor.normalize_body_landmarks(pose_landmarks)
        normalized_left_hand = ProtobufProcessor.normalize_hand_landmarks(
            left_hand_landmarks
        )
        normalized_right_hand = ProtobufProcessor.normalize_hand_landmarks(
            right_hand_landmarks
        )

        # Flatten the normalized landmarks and append them to the list
        selected_landmarks = [
            normalized_pose[4],
            normalized_pose[5],  # Body landmarks (shoulders and upper body)
            normalized_left_hand[4],
            normalized_left_hand[5],
            normalized_left_hand[8],  # Left hand landmarks
            normalized_left_hand[12],
            normalized_left_hand[16],
            normalized_left_hand[17],
            normalized_left_hand[20],
            normalized_right_hand[4],
            normalized_right_hand[5],
            normalized_right_hand[8],  # Right hand landmarks
            normalized_right_hand[12],
            normalized_right_hand[16],
            normalized_right_hand[17],
            normalized_right_hand[20],
        ]

        all_frames.append(np.array(selected_landmarks).flatten())

    pose_tensor = torch.tensor(np.array(all_frames), dtype=torch.float32)

    # Ensure we have exactly 15 frames
    num_frames, feature_size = pose_tensor.shape
    if num_frames < 15:
        padding = torch.zeros(15 - num_frames, feature_size, dtype=torch.float32)
        pose_tensor = torch.cat([pose_tensor, padding], dim=0)
    elif num_frames > 15:
        pose_tensor = pose_tensor[:15, :]

    return pose_tensor


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
            "right_Wrist",
            "right_Finger_4_4",
            "right_Finger_3_4",
            "right_Finger_2_4",
            "right_Finger_1_1",
            "right_Finger_1_4",
            "thumb1",
            "thumb2",
            "left_Shoulder_1",
            "left_Shoulder_2",
            "left_Shoulder_3",
            "left_Elbow_1",
            "left_Elbow_2",
            "left_Wrist",
            "left_Finger_1_1",
            "left_Finger_1_4",
            "left_Finger_2_4",
            "left_Finger_3_4",
            "left_Finger_4_4",
            "left_Finger_5_1",
            "left_Finger_5_2",
        ]
        self.body_links = [
            "left_Wrist",
            "right_Wrist"
        ]
        self.left_links = [
            "left_Finger_5_2",
            "left_Finger_4_1",
            "left_Finger_4_4",
            "left_Finger_3_4",
            "left_Finger_2_4",
            "left_Finger_1_1",
            "left_Finger_1_4"
        ]
        self.right_links = [
            "thumbband2",
            "right_Finger_4_1",
            "right_Finger_4_4",
            "right_Finger_3_4",
            "right_Finger_2_4",
            "right_Finger_1_1",
            "right_Finger_1_4",
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
        output_positions = torch.zeros(
            batch_size, seq_len, (len(self.body_links) + len(self.left_links) + len(self.right_links)) * 3
        )

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
                body_origin = (fk_result["right_Shoulder_1"].get_matrix()[0, :3, 3] + fk_result["left_Shoulder_1"].get_matrix()[0, :3, 3]) / 2
                body_L = np.linalg.norm(np.array(fk_result["right_Shoulder_1"].get_matrix()[0, :3, 3]) - np.array(fk_result["left_Shoulder_1"].get_matrix()[0, :3, 3]))
                left_origin = fk_result["left_Wrist"].get_matrix()[0, :3, 3]
                left_L = np.linalg.norm(np.array(fk_result["left_Wrist"].get_matrix()[0, :3, 3]) - np.array(fk_result["left_Finger_1_1"].get_matrix()[0, :3, 3]))
                right_origin = fk_result["right_Wrist"].get_matrix()[0, :3, 3]
                right_L = np.linalg.norm(np.array(fk_result["right_Wrist"].get_matrix()[0, :3, 3]) - np.array(fk_result["right_Finger_1_1"].get_matrix()[0, :3, 3]))
                positions = [
                    (fk_result[link].get_matrix()[0, :3, 3] - body_origin) / body_L
                    for link in self.body_links
                ] + [
                    (fk_result[link].get_matrix()[0, :3, 3] - left_origin) / left_L
                    for link in self.left_links
                ] + [
                    (fk_result[link].get_matrix()[0, :3, 3] - right_origin) / right_L
                    for link in self.right_links
                ]
                # print(positions)
                
                output_positions[b, t, :] = torch.cat(positions)
                # print(fk_result["right_Shoulder_1"].get_matrix()[0, :3, 3])
                # print(fk_result["left_Shoulder_1"].get_matrix()[0, :3, 3])

        return output_positions

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pose_data = _load_protobuf(sample_path)[0,:].to(device)

    # Forward pass
    fk = ForwardKinematics(urdf_path)
    kine_output = fk.batch_forward_kinematics(output)[0,0,:].to(device)

    # Compute loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(kine_output, pose_data)
    loss_val = loss.item()

    print(f"Loss: {loss_val:.5f}\n")


def main():
    """Main function to parse arguments and run the training or testing process"""
    
    sample_path = "/home/cedra/psl_project/yalda_3822_40-54.pb"

    joints = [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    print("All zero:")
    test_loss(
        sample_path=sample_path,
        urdf_path="/home/cedra/psl_project/rasa/robot.urdf",
        output=torch.tensor(joints)
    )

    joints = [[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                1.0, 
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]
    print("All one:")
    test_loss(
        sample_path=sample_path,
        urdf_path="/home/cedra/psl_project/rasa/robot.urdf",
        output=torch.tensor(joints)
    )
    
    joints = [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.5, 
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    print("Right Fingers closed:")
    test_loss(
        sample_path=sample_path,
        urdf_path="/home/cedra/psl_project/rasa/robot.urdf",
        output=torch.tensor(joints)
    )
    
    joints = [[[0.7, 0.0, 0.25, 1.0, 0.5, 0.5,
                0.5, 
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    print("Right Arm+Fingers:")
    test_loss(
        sample_path=sample_path,
        urdf_path="/home/cedra/psl_project/rasa/robot.urdf",
        output=torch.tensor(joints)
    )

    joints = [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                0.5, 
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]
    print("Left Fingers Close:")
    test_loss(
        sample_path=sample_path,
        urdf_path="/home/cedra/psl_project/rasa/robot.urdf",
        output=torch.tensor(joints)
    )

    joints = [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                0.5, 0.0, 0.25, 0.0, 0.5, 0.5, 
                0.5, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    print("Left Arm:")
    test_loss(
        sample_path=sample_path,
        urdf_path="/home/cedra/psl_project/rasa/robot.urdf",
        output=torch.tensor(joints)
    )

    joints = [[[0.7, 0.0, 0.25, 1.0, 0.5, 0.5,
                0.5, 
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 
                0.5, 0.0, 0.25, 0.0, 0.5, 0.5, 
                0.5, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    print("All:")
    test_loss(
        sample_path=sample_path,
        urdf_path="/home/cedra/psl_project/rasa/robot.urdf",
        output=torch.tensor(joints)
    )

if __name__ == "__main__":
    main()
