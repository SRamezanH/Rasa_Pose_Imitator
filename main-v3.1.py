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
import pytorch_kinematics as pk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


# Dataset class for handling video and pose data
class PoseVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, validate_files=True):
        """
        Args:
            root_dir (string): Directory with all the videos and protobuf files
            transform (callable, optional): Optional transform to be applied on samples
            validate_files (bool): Whether to validate files during initialization
        """
        self.root_dir = root_dir
        self.transform = transform
        self.video_files = sorted(glob.glob(os.path.join(root_dir, "*.mp4")))
        self.valid_indices = []
        self.print_warnings = True
        
        if validate_files:
            print("Validating video files...")
            # Use tqdm if available for progress bar
            try:
                from tqdm import tqdm
                iterator = tqdm(enumerate(self.video_files), total=len(self.video_files), desc="Validating videos")
            except ImportError:
                iterator = enumerate(self.video_files)
                
            for i, video_path in iterator:
                pb_path = video_path.replace(".mp4", ".pb")
                is_valid = True
                
                # Check if video file can be opened
                try:
                    video_test, _, _ = read_video(video_path ,output_format="TCHW")
                    is_valid == video_test.shape[0] == 15
                    
                    # Check if protobuf file exists and can be loaded
                    if is_valid and not os.path.exists(pb_path):
                        if self.print_warnings:
                            print(f"Warning: Missing protobuf file for {video_path}")
                        is_valid = False
                    
                except Exception as e:
                    if self.print_warnings:
                        print(f"Warning: Error validating {video_path}: {str(e)}")
                    is_valid = False
                
                if is_valid:
                    self.valid_indices.append(i)
            
            print(f"Found {len(self.valid_indices)} valid videos out of {len(self.video_files)} total videos")
        else:
            # If not validating, assume all files are valid
            self.valid_indices = list(range(len(self.video_files)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual file index from valid_indices list
        file_idx = self.valid_indices[idx]
        # Load video file
        video_path = self.video_files[file_idx]

        # Derive protobuf path from video path
        pb_path = video_path.replace(".mp4", ".pb")

        # Load video frames - no try/except as we've pre-validated
        video_data = self._load_video(video_path)

        # Load pose data from protobuf
        pose_data = self._load_protobuf(pb_path)

        sample = {"video": video_data, "pose": pose_data}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_video(self, video_path):
        """
        Load and process video frames using OpenCV.

        Args:
            video_path (string): Path to the video file

        Returns:
            torch.Tensor: Tensor containing video frames with shape  [15, 3, 273, 210]
        """
        # Open the video file
        video, _, _ = read_video(video_path ,output_format="TCHW")
        video_tensor = video.to(device)
        video_tensor = video_tensor[:15, :3, :210, :273]
        num_frames, ch, w, h = video_tensor.shape
        if num_frames < 15:
            print("Low frame count" + video_path)
            padding = torch.zeros([15 - num_frames, ch, w, h ]).to(device)
            video_tensor = torch.cat((video_tensor, padding), dim=0)

        return video_tensor.permute(0,1,3,2) / 255.0

    def _load_protobuf(self, pb_path):
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

        for frame in proto_data.frames:
            # Extract and normalize landmarks for body, left hand, and right hand
            pose_landmarks = ProtobufProcessor.extract_landmark_coordinates(
                frame, "pose_landmarks", range(10)
            )  # Body landmarks
            
            # Normalize the extracted landmarks
            selected_landmarks = ProtobufProcessor.normalize_body_landmarks(pose_landmarks, left_side)

            pose_tensor = torch.cat((pose_tensor, selected_landmarks), dim=0)

        pose_tensor = pose_tensor.to(device)

        # Ensure we have exactly 15 frames
        num_frames, feature_size, dim_size = pose_tensor.shape
        if num_frames < 15:
            print("Low frame count" + pb_path)
            padding = torch.zeros([15 - num_frames, feature_size, dim_size ]).to(device)
            pose_tensor = torch.cat((pose_tensor, padding), dim=0)
        elif num_frames > 15:
            pose_tensor = pose_tensor[:15, :, :]

        return pose_tensor


# Neural Network Model Definition
class PoseVideoCNNRNN(nn.Module):
    def __init__(self):
        super(PoseVideoCNNRNN, self).__init__()

        # CNN for spatial feature extraction
        self.video_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((4, 4)),  # Global Average Pooling
        )
        for i in [0, 4, 8, 12]:
            nn.init.kaiming_normal_(self.video_cnn[i].weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.video_cnn[i].bias)

        # LSTM/GRU for temporal modeling
        self.temporal_rnn1 = nn.LSTM(
                input_size=256*4*4,
                hidden_size=128,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
        for name, param in self.temporal_rnn1.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Hidden-hidden weights (recurrent)
                nn.init.orthogonal_(param.data)  # Helps with long-term dependencies
            elif 'bias' in name:
                nn.init.zeros_(param.data)  # Biases typically zero-initialized
                # Optional: Initialize forget gate bias to 1 (helps with training)
                if 'bias_hh' in name:
                    param.data[128:256] = 1.0

        # self.L1 = nn.Sequential(
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(0.1),
        #     nn.Linear(32, 16),
        #     nn.LeakyReLU(0.1)
        # )
        # nn.init.kaiming_normal_(self.L1[2].weight, mode='fan_in', nonlinearity='relu')
        # nn.init.zeros_(self.L1[2].bias)

        # Latent space projection
        self.fc_mu = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
        )
        nn.init.kaiming_normal_(self.fc_mu[2].weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc_mu[2].bias)

        self.fc_logvar = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
        )
        nn.init.kaiming_normal_(self.fc_logvar[2].weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc_logvar[2].bias)
        
        # # Second LSTM/GRU layer for time series modeling
        # self.temporal_rnn2 = nn.LSTM(
        #         input_size=32,
        #         hidden_size=32,
        #         num_layers=2,
        #         batch_first=True,
        #         bidirectional=False,
        #     )
        # for name, param in self.temporal_rnn1.named_parameters():
        #     if 'weight_ih' in name:  # Input-hidden weights
        #         nn.init.xavier_uniform_(param.data)
        #     elif 'weight_hh' in name:  # Hidden-hidden weights (recurrent)
        #         nn.init.orthogonal_(param.data)  # Helps with long-term dependencies
        #     elif 'bias' in name:
        #         nn.init.zeros_(param.data)  # Biases typically zero-initialized
        #         # Optional: Initialize forget gate bias to 1 (helps with training)
        #         if 'bias_hh' in name:
        #             param.data[16:32] = 1.0

        # self.output_layer = nn.Sequential(
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(32, 6),
        #     nn.Sigmoid(),
        # )
        # nn.init.xavier_normal_(self.output_layer[1].weight)  # Xavier/Glorot initialization
        # nn.init.zeros_(self.output_layer[1].bias)  # Initialize bias to zeros
        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        self.gru = nn.GRU(256, 128, bidirectional=True, batch_first=True)
        self.attn = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False),
            nn.Softmax(dim=1)
        )
        self.joint_fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 6),
            nn.Sigmoid()
        )

    def forward(self, video, deterministic=False):
        batch_size, seq_len, c, h, w = video.size()

        # CNN feature extraction
        # print(video.mean(), video.std())
        video = video.view(batch_size*seq_len, c, h, w)
        video = self.video_cnn(video)
        # print(video.mean(), video.std())
        # video = video.squeeze()
        video = video.view(batch_size, seq_len, -1)

        # First RNN (LSTM/GRU) for temporal modeling
        #_, (rnn_out, _) = self.temporal_rnn1(video)
        _, (rnn_out, _) = self.temporal_rnn1(video)
        rnn_out = rnn_out.squeeze(0)
        # rnn_out = self.L1(rnn_out)

        # Latent space
        mu = self.fc_mu(rnn_out)
        logvar = torch.clamp(self.fc_logvar(rnn_out), min=-10, max=10)

        # reparameterization
        embedding = mu
        if not deterministic:
            std = torch.exp(0.5 * logvar)
            embedding += torch.rand_like(std) * std

        # Repeat vector to 15x32
        # embedding = embedding.view(2, batch_size, 32)
        # decoder_input = torch.zeros(batch_size, seq_len, 32).to(device)

        # # Second RNN for time series modeling
        # rnn_out2, _ = self.temporal_rnn2(decoder_input, (embedding, embedding))  # Shape: [batch_size, 15, 64]

        # # Output layer to get final 15x26 sequence
        # output = self.output_layer(rnn_out2)  # Shape: [batch_size, 15, 26]
        # output = torch.clamp(output, 0, 1)  # Apply clamp to ensure outputs are between 0 and 1

        h = self.fc(embedding).unsqueeze(1).repeat(1, 15, 1)
        gru_out, _ = self.gru(h)                  # [B, seq_len, 512]
        w = self.attn(gru_out)                    # [B, seq_len, 1]
        c = torch.sum(w * gru_out, dim=1, keepdim=True)
        output = self.joint_fc(gru_out + c)       # [B, seq_len, joint_dim=6]

        return output, mu, logvar


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
            batch_size, seq_len, 3, 3
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

                body_L = (torch.linalg.vector_norm((fk_result["right_Shoulder_2"].get_matrix()[0, :3, 3]) - (fk_result["right_Forearm_1"].get_matrix()[0, :3, 3])) +
                          torch.linalg.vector_norm((fk_result["right_Wrist"].get_matrix()[0, :3, 3]) - (fk_result["right_Forearm_1"].get_matrix()[0, :3, 3]))) / 2
                origin = fk_result["right_Shoulder_2"].get_matrix()[0, :3, 3]

                output_positions[b, t, 0] = (fk_result["right_Wrist"].get_matrix()[0, :3, 3] - origin) / body_L
                output_positions[b, t, 1] = (fk_result["right_Finger_1_1"].get_matrix()[0, :3, 3] - origin) / body_L
                output_positions[b, t, 2] = (fk_result["right_Finger_4_1"].get_matrix()[0, :3, 3] - origin) / body_L

        return output_positions


def test_dataset(data_dir):
    """
    Test the dataset loader by loading a sample batch

    Args:
        data_dir (str): Path to the dataset directory
    """
    print(f"Testing PoseVideoDataset with data directory: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        return

    video_files = [f for f in os.listdir(data_dir) if f.endswith(".mp4")]
    if not video_files:
        print(f"Error: No video files found in '{data_dir}'.")
        return

    print(f"Found {len(video_files)} video files.")

    pb_files = [f for f in os.listdir(data_dir) if f.endswith(".pb")]
    print(f"Found {len(pb_files)} protobuf files.")

    # Create dataset with validation (it will automatically detect and skip corrupted files)
    try:
        print("\nCreating dataset and validating files...")
        dataset = PoseVideoDataset(root_dir=data_dir, validate_files=False)
        
        if len(dataset) == 0:
            print(f"❌ No valid videos found in the dataset. Please check your data files.")
            return
            
        print(f"✅ Successfully created dataset with {len(dataset)} valid samples.")
        
        # Calculate what percentage of videos are valid
        valid_ratio = len(dataset) / len(video_files) * 100
        print(f"📊 {valid_ratio:.1f}% of videos are valid and will be used for training/testing.")
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return

    try:
        print("\nTesting access to first sample...")
        sample = dataset[0]
        video_data = sample["video"]
        pose_data = sample["pose"]

        print("\nSample data:")
        print(f"  Video shape: {video_data.shape}")
        print(f"  Pose shape: {pose_data.shape}")

        print(f"  Video data range: [{video_data.min():.4f}, {video_data.max():.4f}]")
        print(f"  Pose data range: [{pose_data.min():.4f}, {pose_data.max():.4f}]")
        print(f"✅ Successfully accessed sample data.")

    except Exception as e:
        print(f"❌ Error accessing sample: {e}")
        return

    try:
        print("\nTesting DataLoader...")
        batch_size = min(4, len(dataset))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print("\nTesting batch loading:")
        from tqdm import tqdm
        
        # Load a few batches with progress bar
        max_batches = min(3, len(dataloader))
        for i, batch in enumerate(tqdm(dataloader, desc="Loading batches", total=max_batches)):
            videos = batch["video"]
            poses = batch["pose"]

            print(f"\nBatch {i+1}:")
            print(f"  Video shape: {videos.shape}")
            print(f"  Pose shape: {poses.shape}")

            if i >= max_batches - 1:
                break

        print("\n✅ DataLoader test successful!")

    except Exception as e:
        print(f"❌ Error witis noth DataLoader: {e}")
        return

    print("\n✅ All tests passed successfully!")

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
    six_d = torch.cat([u, v_ortho], dim=-1)
    return six_d

def loss_fn(fk, pose_data, model_output, logvar, mu, lambda_R=1.0, lambda_kl=0.1, lambda_vel=10.0, 
          lambda_motion=2.0, lambda_rel_vel=1.5, lambda_traj=2.0, eps=1e-7):
    """
    Computes the loss between the predicted and actual pose data
    
    Args:
        fk: Forward kinematics model
        pose_data: Ground truth pose data [batch, 15, 3, 3]
        model_output: Model predictions [batch, 15, 6]
        logvar: Log variance from the model
        mu: Mean from the model
        lambda_R: Weight for rotation loss
        lambda_kl: Weight for KL divergence loss
        lambda_vel: Weight for velocity loss
        lambda_motion: Weight for motion encouragement loss
        lambda_rel_vel: Weight for relative velocity loss
        lambda_traj: Weight for trajectory loss
        eps: Small epsilon value to prevent division by zero
        
    Returns:
        Tuple of total loss and individual loss components
    """
    kine_output = fk.batch_forward_kinematics(model_output)  # [batch, 15, 48]
    
    # Position loss - penalizes differences in hand position
    pose_loss = mse_loss(pose_data[:,:,0], kine_output[:,:,0])

    # KL divergence loss for the VAE component
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Velocity loss - matches the velocity profiles
    velocity_in = torch.diff(pose_data[:,:,0], dim=1)  # [batch, 14, 3]
    velocity_out = torch.diff(kine_output[:,:,0], dim=1)  # [batch, 14, 3]
    vel_loss = 30.0 * 15.0 * mse_loss(velocity_in, velocity_out)

    # Rotation loss - penalizes differences in hand orientation
    input_6d = batch_vectors_to_6D(pose_data, eps=eps)
    output_6d = batch_vectors_to_6D(kine_output, eps=eps)
    R_loss = torch.mean((output_6d - input_6d)**2)
    
    # NEW LOSS TERMS
    
    # Motion encouragement loss - penalizes minimal movement (encourages the hand to move)
    # Calculate the total distance moved across frames
    total_movement_out = torch.sum(torch.norm(velocity_out, dim=2), dim=1)  # [batch]
    total_movement_in = torch.sum(torch.norm(velocity_in, dim=2), dim=1)  # [batch]
    # Penalize when output movement is less than input movement
    motion_loss = torch.mean(torch.relu(total_movement_in - total_movement_out))

    # Relative velocity loss - ensures the relative velocities between joints are similar
    # Calculate relative velocities between hand points (wrist to finger1, wrist to finger2)
    rel_vel_in_1 = torch.diff(pose_data[:,:,1] - pose_data[:,:,0], dim=1)  # [batch, 14, 3]
    rel_vel_in_2 = torch.diff(pose_data[:,:,2] - pose_data[:,:,0], dim=1)  # [batch, 14, 3]
    rel_vel_out_1 = torch.diff(kine_output[:,:,1] - kine_output[:,:,0], dim=1)  # [batch, 14, 3]
    rel_vel_out_2 = torch.diff(kine_output[:,:,2] - kine_output[:,:,0], dim=1)  # [batch, 14, 3]
    
    # Compute loss on relative velocities
    rel_vel_loss = mse_loss(rel_vel_in_1, rel_vel_out_1) + mse_loss(rel_vel_in_2, rel_vel_out_2)

    # Trajectory loss - ensures the shape of the movement path is similar
    # Compute normalized trajectory directions
    traj_dir_in = F.normalize(velocity_in, p=2, dim=2, eps=eps)  # [batch, 14, 3]
    traj_dir_out = F.normalize(velocity_out, p=2, dim=2, eps=eps)  # [batch, 14, 3]
    
    # Compute cosine similarity between directions (1 is perfect alignment)
    # Convert to loss by taking 1 - similarity
    cos_sim = torch.sum(traj_dir_in * traj_dir_out, dim=2)  # [batch, 14]
    traj_loss = torch.mean(1.0 - cos_sim)
    
    # Combine all loss terms with their respective weights
    loss = (pose_loss + 
            lambda_R * R_loss + 
            lambda_kl * kl_loss + 
            lambda_vel * vel_loss +
            lambda_motion * motion_loss +
            lambda_rel_vel * rel_vel_loss +
            lambda_traj * traj_loss)

    return loss, pose_loss, R_loss, kl_loss, vel_loss, motion_loss, rel_vel_loss, traj_loss

def train_model(data_dir, test_dir, urdf_path, num_epochs=10, batch_size=8, learning_rate=0.01):
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
    dataset = PoseVideoDataset(root_dir=data_dir, validate_files=False)
    # test_dataset = PoseVideoDataset(root_dir=test_dir, validate_files=False)
    
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
    test_size = 0 # int(0.15 * len(dataset))
    e = len(dataset) - train_size - eval_size - test_size
    train_dataset, eval_dataset, test_dataset, _ = random_split(dataset, [train_size, eval_size, test_size, e])

    for i in random.sample(train_dataset.indices, 2):
        print(dataset.video_files[i])
    for i in random.sample(eval_dataset.indices, 2):
        print(dataset.video_files[i])
    # for i in random.sample(test_dataset.indices, 2):
    #     print(dataset.video_files[i])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}, Test size: {len(test_dataset)}")

    if TORCHINFO_AVAILABLE:
            print("\nModel Architecture:")
            model = PoseVideoCNNRNN().to(device)
            # Get shape from a sample
            sample = dataset[0]
            video_shape = sample["video"].shape
            # Show model summary
            model_stats = summary(
                model,
                input_size=(batch_size,) + video_shape,
                depth=4,
                device=device,
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                col_width=20,
                row_settings=["var_names", "depth"],
                verbose=1
                )
            if model_stats:
                print(f"\nTotal Parameters: {model_stats.total_params:,}")
                print(f"Trainable Parameters: {model_stats.trainable_params:,}")
                print(f"Non-trainable Parameters: {model_stats.total_params - model_stats.trainable_params:,}\n")

    # Create model, loss function, and optimizer
    lambda_kl = 0.1

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
        total_R_loss = 0
        total_kl_loss = 0
        total_vel_loss = 0
        total_motion_loss = 0
        total_rel_vel_loss = 0
        total_traj_loss = 0
        
        # Create progress bar for batches
        batch_pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs}", 
            leave=False, 
            position=1,
            total=len(train_loader)
        )        
        for i, batch in enumerate(batch_pbar):
            video_data = batch["video"]  # Shape: [batch, 15, 3, 258, 196]
            pose_data = batch["pose"]  # Shape: [batch, 15, 48] (ground truth)

            # Forward pass
            model_output, mu, logvar = model(video_data)  # Output: [batch, 15, 26] (predicted)

            # Compute loss
            loss, pose_loss, R_loss, kl_loss, vel_loss, motion_loss, rel_vel_loss, traj_loss = loss_fn(fk, pose_data, model_output, logvar, mu, lambda_kl = lambda_kl)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update running loss
            loss_val = loss.item()
            pose_loss_val = pose_loss.item()
            R_loss_val = R_loss.item()
            kl_loss_val = kl_loss.item()
            vel_loss_val = vel_loss.item()
            motion_loss_val = motion_loss.item()
            rel_vel_loss_val = rel_vel_loss.item()
            traj_loss_val = traj_loss.item()

            total_loss += loss_val
            total_pose_loss += pose_loss_val
            total_R_loss += R_loss_val
            total_kl_loss += kl_loss_val
            total_vel_loss += vel_loss_val
            total_motion_loss += motion_loss_val
            total_rel_vel_loss += rel_vel_loss_val
            total_traj_loss += traj_loss_val

            # Update progress bar with current loss
            batch_pbar.set_postfix({
                'loss': f'{loss_val:.2f}', 
                'pose_loss': f'{pose_loss_val:.2f}', 
                'R_loss': f'{R_loss_val:.2f}', 
                'kl_loss': f'{kl_loss_val:.2f}', 
                'vel_loss': f'{vel_loss_val:.2f}',
                'motion_loss': f'{motion_loss_val:.2f}',
                'rel_vel_loss': f'{rel_vel_loss_val:.2f}',
                'traj_loss': f'{traj_loss_val:.2f}'
            })
        
        total_loss /= len(train_loader)
        total_pose_loss /= len(train_loader)
        total_R_loss /= len(train_loader)
        total_kl_loss /= len(train_loader)
        total_vel_loss /= len(train_loader)
        total_motion_loss /= len(train_loader)
        total_rel_vel_loss /= len(train_loader)
        total_traj_loss /= len(train_loader)

        # Evaluation
        model.eval()  # Set model to evaluation mode
        eval_loss = 0
        eval_pose_loss = 0
        eval_R_loss = 0
        eval_kl_loss = 0
        eval_vel_loss = 0
        eval_motion_loss = 0
        eval_rel_vel_loss = 0
        eval_traj_loss = 0
    
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
                video_data = batch["video"]
                pose_data = batch["pose"]

                # Forward pass
                model_output, mu, logvar = model(video_data)

                # Compute loss
                loss, pose_loss, R_loss, kl_loss, vel_loss, motion_loss, rel_vel_loss, traj_loss = loss_fn(fk, pose_data, model_output, logvar, mu, lambda_kl = lambda_kl)
                
                loss_val = loss
                pose_loss_val = pose_loss
                R_loss_val = R_loss
                kl_loss_val = kl_loss
                vel_loss_val = vel_loss
                motion_loss_val = motion_loss
                rel_vel_loss_val = rel_vel_loss
                traj_loss_val = traj_loss

                eval_loss += loss_val
                eval_pose_loss += pose_loss_val
                eval_R_loss += R_loss_val
                eval_kl_loss += kl_loss_val
                eval_vel_loss += vel_loss_val
                eval_motion_loss += motion_loss_val
                eval_rel_vel_loss += rel_vel_loss_val
                eval_traj_loss += traj_loss_val

                # Update progress bar
                eval_pbar.set_postfix({
                    'loss': f'{loss_val:.2f}', 
                    'pose_loss': f'{pose_loss_val:.2f}', 
                    'R_loss': f'{R_loss_val:.2f}', 
                    'kl_loss': f'{kl_loss_val:.2f}', 
                    'vel_loss': f'{vel_loss_val:.2f}',
                    'motion_loss': f'{motion_loss_val:.2f}',
                    'rel_vel_loss': f'{rel_vel_loss_val:.2f}',
                    'traj_loss': f'{traj_loss_val:.2f}'
                })
        eval_loss /= len(eval_loader)
        eval_pose_loss /= len(eval_loader)
        eval_R_loss /= len(eval_loader)
        eval_kl_loss /= len(eval_loader)
        eval_vel_loss /= len(eval_loader)
        eval_motion_loss /= len(eval_loader)
        eval_rel_vel_loss /= len(eval_loader)
        eval_traj_loss /= len(eval_loader)

        scheduler.step(eval_loss)

        # Check Overfitting
        if(eval_loss <= best_eval_loss):
            overfit = 0
            best_eval_loss = eval_loss
            # Save the trained model
            torch.save(model.state_dict(), f"sign_language_pose_model_v3.1_{epoch+1}_best.pth")
        else:
            overfit += 1
            torch.save(model.state_dict(), f"sign_language_pose_model_v3.1_{epoch+1}.pth")
            # if(overfit > 5):
            #     break

        epoch_time = time.time() - start_time

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train loss': f'{total_loss:.2f}',
            'eval loss': f'{eval_loss:.2f}',
            'time': f'{epoch_time:.1f}s'
        })

        lambda_kl *= 0.5

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s\n" 
              f"Train Loss: {total_loss:.5f}, Pose Loss: {total_pose_loss:.5f}, R Loss: {total_R_loss:.5f}, "
              f"KL Loss: {total_kl_loss:.5f}, Vel Loss: {total_vel_loss:.5f}, "
              f"Motion Loss: {total_motion_loss:.5f}, Rel Vel Loss: {total_rel_vel_loss:.5f}, Traj Loss: {total_traj_loss:.5f}\n"
              f"Eval Loss: {eval_loss:.5f}, Pose Loss: {eval_pose_loss:.5f}, R Loss: {eval_R_loss:.5f}, "
              f"KL Loss: {eval_kl_loss:.5f}, Vel Loss: {eval_vel_loss:.5f}, "
              f"Motion Loss: {eval_motion_loss:.5f}, Rel Vel Loss: {eval_rel_vel_loss:.5f}, Traj Loss: {eval_traj_loss:.5f}")
        
    print(f"\nTraining completed in {num_epochs} epochs")

    # # Evaluation on test data
    # print("\nEvaluating model on test data...")
    # model.eval()  # Set model to evaluation mode
    # test_loss = 0
    # test_pose_loss = 0
    # test_R_loss = 0
    # test_kl_loss = 0
    # test_vel_loss = 0
    
    # # Progress bar for test batches
    # test_pbar = tqdm(
    #     test_loader, 
    #     desc="Testing", 
    #     total=len(test_loader)
    # )

    # with torch.no_grad():  # No gradient computation
    #     for i, batch in enumerate(test_pbar):
    #         video_data = batch["video"]
    #         pose_data = batch["pose"]

    #         # Forward pass
    #         model_output, mu, logvar = model(video_data)
            
    #         # Compute Loss
    #         loss, pose_loss, R_loss, kl_loss, vel_loss = loss_fn(fk, pose_data, model_output, logvar, mu)
                
    #         loss_val = loss
    #         test_loss += loss_val
    #         test_pose_loss += pose_loss
    #         test_R_loss += R_loss
    #         test_kl_loss += kl_loss
    #         test_vel_loss += vel_loss

    #         # Update progress bar
    #         test_pbar.set_postfix({
    #             'loss': f'{test_loss/(i+1):.2f}', 
    #             'pose_loss': f'{test_pose_loss/(i+1):.2f}', 
    #             'R_loss': f'{test_R_loss/(i+1):.2f}', 
    #             'kl_loss': f'{test_kl_loss/(i+1):.2f}', 
    #             'vel_loss': f'{test_vel_loss/(i+1):.2f}'
    #         })

    # test_loss /= len(test_loader)
    # test_pose_loss /= len(test_loader)
    # test_R_loss /= len(test_loader)
    # test_kl_loss /= len(test_loader)
    # test_vel_loss /= len(test_loader)
    # print(f"\nFinal Test Loss: {test_loss:.5f}, Pose Loss: {test_pose_loss:.5f}, R Loss: {test_R_loss:.5f}, KL Loss: {test_kl_loss:.5f}, Vel Loss: {test_vel_loss:.5f}")


def main():
    """Main function to parse arguments and run the training or testing process"""

    # default values
    DEFAULT_DATA_DIR = "/home/cedra/psl_project/5_dataset"
    TEST_DATA_DIR = "/home/cedra/psl_project/5_dataset/test"
    DEFAULT_URDF_PATH = "/home/cedra/psl_project/rasa/hand.urdf"
    DEFAULT_NUM_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_TEST_ONLY = False
    DEFAULT_EXTRACT_ZIP = None

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Sign Language Recognition using Pose Estimation"
    )
    parser.add_argument(
        "--data_dir", type=str, help="Path to the dataset directory (default: dataset)"
    )
    parser.add_argument(
        "--test_dir", type=str, help="Path to the test dataset directory (default: dataset)"
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        help="Path to the robot URDF file (default: rasa/robot.urdf)",
    )
    parser.add_argument(
        "--num_epochs", type=int, help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only test the dataset without training",
    )
    parser.add_argument(
        "--extract_zip", type=str, help="Extract a zip file to the dataset directory"
    )

    args = parser.parse_args()

    # Use command-line arguments if provided, otherwise use defaults
    data_dir = args.data_dir if args.data_dir is not None else DEFAULT_DATA_DIR
    test_dir = args.test_dir if args.test_dir is not None else TEST_DATA_DIR
    urdf_path = args.urdf_path if args.urdf_path is not None else DEFAULT_URDF_PATH
    num_epochs = args.num_epochs if args.num_epochs is not None else DEFAULT_NUM_EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE
    test_only = args.test_only if args.test_only else DEFAULT_TEST_ONLY
    zip_file_path = (
        args.extract_zip if args.extract_zip is not None else DEFAULT_EXTRACT_ZIP
    )

    print(f"Using data directory: {data_dir}")
    print(f"Using test directory: {test_dir}")
    print(f"Using URDF path: {urdf_path}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Test only mode: {test_only}")
    if zip_file_path:
        print(f"Extracting zip file: {zip_file_path}")

    # Extract zip file if specified
    if zip_file_path:
        extract_zip(zip_file_path, os.path.dirname(data_dir))

    # Test dataset only
    if test_only:
        test_dataset(data_dir)
    else:
        # Train model
        train_model(
            data_dir=data_dir,
            test_dir=test_dir,
            urdf_path=urdf_path,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

if __name__ == "__main__":
    main()
