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
import math
import random
import xml.etree.ElementTree as ET
import zipfile

# Third-party imports
import pytorch_kinematics as pk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.io import read_video
from torchvision.models import mobilenet_v3_small
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=15):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Neural Network Model Definition
class PoseVideoCNNRNN(nn.Module):
    def __init__(self):
        super(PoseVideoCNNRNN, self).__init__()

        # CNN for spatial feature extraction
        self.cnn = mobilenet_v3_small( pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        # f = (273, 210)
        # inp = torch.randn(1,3,*f)
        # print(self.cnn(inp).flatten(1).shape[-1]) #576
        self.pos_embedding = nn.Parameter(torch.randn(1,15,576))

        # LSTM/GRU for temporal modeling
        encodder = nn.TransformerEncoderLayer(
            d_model=576,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
            dropout=0.1,
        )

        self.encodder = nn.TransformerEncoder(encodder, 3)
        
        self.L1 = nn.Sequential(
            nn.Linear(576, 32),
            nn.ReLU(),
        )

        # Latent space projection
        self.fc_mu = nn.Linear(32, 6)
        self.fc_logvar = nn.Linear(32, 6)
        
        self.L2 = nn.Linear(6, 32)
        # Second LSTM/GRU layer for time series modeling
        self.pos_encoder = PositionalEncoding(32)
        decoder = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder, 3)

        # Final TimeDistributed Dense layer
        self.output_layer = nn.Linear(32, 6)

    def forward(self, video, deterministic=False):
        batch_size, seq_len, c, h, w = video.size()

        # CNN feature extraction
        # print(video.mean(), video.std())
        video = video.view(batch_size*seq_len, c, h, w)
        video = self.cnn(video)
        # print(video.mean(), video.std())
        video = video.flatten()
        video = video.view(batch_size, seq_len, -1)
        x = video + self.pos_embedding

        # First RNN (LSTM/GRU) for temporal modeling
        #_, (rnn_out, _) = self.temporal_rnn1(video)
        x = self.encodder(x)
        x = x.mean(dim=1)
        x = self.L1(x)

        # Latent space
        mu = self.fc_mu(x)
        logvar = torch.clamp(self.fc_logvar(x), min=-10, max=10)

        # reparameterization
        embedding = mu
        if not deterministic:
            std = torch.exp(0.5 * logvar)
            embedding += torch.rand_like(std) * std

        # expanding
        embedding = self.L2(embedding)
        
        # Repeat vector to 15x32
        repeated = embedding.unsqueeze(1).repeat(1, 15, 1)  # Shape: [batch_size, 15, 32]

        # Second RNN for time series modeling
        x = self.pos_encoder(repeated)
        x = self.transformer(x)

        # Output layer to get final 15x26 sequence
        output = self.output_layer(x)  # Shape: [batch_size, 15, 26]
        output = torch.clamp(output, 0, 1)  # Apply clamp to ensure outputs are between 0 and 1


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

def loss_fn(fk, pose_data, model_output, logvar, mu, lambda_R = 1.0, lambda_kl = 1.0, lambda_vel = 0.1, eps: float = 1e-7):
    kine_output = fk.batch_forward_kinematics(model_output)  # [batch, 15, 48]
    pose_loss = mse_loss(pose_data[:,:,0], kine_output[:,:,0])

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    velocity_in = torch.diff(pose_data[:,:,0], dim=1)
    velocity_out = torch.diff(kine_output[:,:,0], dim=1)
    vel_loss = mse_loss(velocity_in, velocity_out)

    input_6d = batch_vectors_to_6D(pose_data, eps = eps)
    output_6d = batch_vectors_to_6D(kine_output, eps = eps)
    R_loss = torch.mean((output_6d - input_6d)**2)
    # l1_loss = nn.L1Loss()
    # R_loss = l1_loss(input_6d, output_6d)

    
    loss = pose_loss + lambda_R * R_loss + lambda_kl * kl_loss + lambda_vel * vel_loss

    return loss, pose_loss, R_loss, kl_loss, vel_loss

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
    train_size = int(0.7 * len(dataset))
    eval_size = int(0.15 * len(dataset))
    test_size = int(0.15 * len(dataset))
    e = len(dataset) - train_size - eval_size - test_size
    train_dataset, eval_dataset, test_dataset, _ = random_split(dataset, [train_size, eval_size, test_size, e])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for i in random.sample(train_dataset.indices, 2):
        print(dataset.video_files[i])
    for i in random.sample(eval_dataset.indices, 2):
        print(dataset.video_files[i])
    for i in random.sample(test_dataset.indices, 2):
        print(dataset.video_files[i])

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
    model = PoseVideoCNNRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

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
    lambda_kl = 1.0
    
    # Training loop
    for epoch in epoch_pbar:
        torch.cuda.empty_cache()
        model.train()  # Set model to training mode
        total_loss = 0
        total_pose_loss = 0
        total_quat_loss = 0
        total_kl_loss = 0
        total_vel_loss = 0
        
        # Create progress bar for batches
        batch_pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs}", 
            leave=False, 
            position=1,
            total=len(train_loader)
        )

        lambda_kl = 0.1#torch.max(torch.tensor(0.1), torch.tensor(lambda_kl * (1.0 - epoch/10))).item()
        
        # Start time for this epoch
        start_time = time.time()
        
        for i, batch in enumerate(batch_pbar):
            video_data = batch["video"]  # Shape: [batch, 15, 3, 258, 196]
            pose_data = batch["pose"]  # Shape: [batch, 15, 48] (ground truth)

            # Forward pass
            model_output, mu, logvar = model(video_data)  # Output: [batch, 15, 26] (predicted)

            # Compute loss
            loss, pose_loss, quat_loss, kl_loss, vel_loss = loss_fn(fk, pose_data, model_output, logvar, mu, lambda_kl=lambda_kl)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update running loss
            loss_val = loss.item()
            total_loss += loss_val
            total_pose_loss += pose_loss.item()
            total_quat_loss += quat_loss.item()
            total_kl_loss += kl_loss.item()
            total_vel_loss += vel_loss.item()

            # Update progress bar with current loss
            batch_pbar.set_postfix({
                'loss': f'{total_loss/(i+1):.2f}', 
                'pose_loss': f'{total_pose_loss/(i+1):.2f}', 
                'quat_loss': f'{total_quat_loss/(i+1):.2f}', 
                'kl_loss': f'{total_kl_loss/(i+1):.2f}', 
                'vel_loss': f'{total_vel_loss/(i+1):.2f}'
            })
        
        scheduler.step()
        
        total_loss /= len(train_loader)
        total_pose_loss /= len(train_loader)
        total_quat_loss /= len(train_loader)
        total_kl_loss /= len(train_loader)
        total_vel_loss /= len(train_loader)

        # Evaluation
        model.eval()  # Set model to evaluation mode
        eval_loss = 0
        eval_pose_loss = 0
        eval_quat_loss = 0
        eval_kl_loss = 0
        eval_vel_loss = 0
    
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
                loss, pose_loss, quat_loss, kl_loss, vel_loss = loss_fn(fk, pose_data, model_output, logvar, mu)
                
                loss_val = loss.item()
                eval_loss += loss_val
                eval_pose_loss += pose_loss
                eval_quat_loss += quat_loss
                eval_kl_loss += kl_loss
                eval_vel_loss += vel_loss

                # Update progress bar
                eval_pbar.set_postfix({
                    'loss': f'{eval_loss/(i+1):.2f}', 
                    'pose_loss': f'{eval_pose_loss/(i+1):.2f}', 
                    'quat_loss': f'{eval_quat_loss/(i+1):.2f}', 
                    'kl_loss': f'{eval_kl_loss/(i+1):.2f}', 
                    'vel_loss': f'{eval_vel_loss/(i+1):.2f}'
                })

        epoch_time = time.time() - start_time

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train loss': f'{total_loss:.2f}',
            'eval loss': f'{eval_loss:.2f}',
            'time': f'{epoch_time:.1f}s'
        })

        eval_loss /= len(eval_loader)
        eval_pose_loss /= len(eval_loader)
        eval_quat_loss /= len(eval_loader)
        eval_kl_loss /= len(eval_loader)
        eval_vel_loss /= len(eval_loader)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s\nTrain Loss: {total_loss:.5f}, Pose Loss: {total_pose_loss:.5f}, Quat Loss: {total_quat_loss:.5f}, KL Loss: {total_kl_loss:.5f}, Vel Loss: {total_vel_loss:.5f}\nEval Loss: {eval_loss:.5f}, Pose Loss: {eval_pose_loss:.5f}, Quat Loss: {eval_quat_loss:.5f}, KL Loss: {eval_kl_loss:.5f}, Vel Loss: {eval_vel_loss:.5f}")
        
        # Check Overfitting
        if(eval_loss <= best_eval_loss):
            overfit = 0
            best_eval_loss = eval_loss
            # Save the trained model
            torch.save(model.state_dict(), f"sign_language_pose_model_v4.0_{epoch+1}_best.pth")
        else:
            overfit += 1
            torch.save(model.state_dict(), f"sign_language_pose_model_v4.0_{epoch+1}.pth")
            # if(overfit > 5):
            #     break
        
    print(f"\nTraining completed in {num_epochs} epochs")

    # Evaluation on test data
    print("\nEvaluating model on test data...")
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    test_pose_loss = 0
    test_quat_loss = 0
    test_kl_loss = 0
    test_vel_loss = 0
    
    # Progress bar for test batches
    test_pbar = tqdm(
        test_loader, 
        desc="Testing", 
        total=len(test_loader)
    )

    with torch.no_grad():  # No gradient computation
        for i, batch in enumerate(test_pbar):
            video_data = batch["video"]
            pose_data = batch["pose"]

            # Forward pass
            model_output, mu, logvar = model(video_data)
            
            # Compute Loss
            loss, pose_loss, quat_loss, kl_loss, vel_loss = loss_fn(fk, pose_data, model_output, logvar, mu)
                
            loss_val = loss.item()
            test_loss += loss_val
            test_pose_loss += pose_loss
            test_quat_loss += quat_loss
            test_kl_loss += kl_loss
            test_vel_loss += vel_loss

            # Update progress bar
            test_pbar.set_postfix({
                'loss': f'{test_loss/(i+1):.2f}', 
                'pose_loss': f'{test_pose_loss/(i+1):.2f}', 
                'quat_loss': f'{test_quat_loss/(i+1):.2f}', 
                'kl_loss': f'{test_kl_loss/(i+1):.2f}', 
                'vel_loss': f'{test_vel_loss/(i+1):.2f}'
            })

    test_loss /= len(test_loader)
    test_pose_loss /= len(test_loader)
    test_quat_loss /= len(test_loader)
    test_kl_loss /= len(test_loader)
    test_vel_loss /= len(test_loader)
    print(f"\nFinal Test Loss: {test_loss:.5f}, Pose Loss: {test_pose_loss:.5f}, Quat Loss: {test_quat_loss:.5f}, KL Loss: {test_kl_loss:.5f}, Vel Loss: {test_vel_loss:.5f}")


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
