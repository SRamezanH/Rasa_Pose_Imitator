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

from transformers import CLIPVisionModel, VideoMAEModel

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
        # Frame preprocessor for CLIP
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize for CLIP
            transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],
                                 std=[0.2686, 0.2613, 0.2758])
        ])
        for pb_string in sorted(glob.glob(os.path.join(pb_root_dir, "*.pb"))):
            parts = os.path.basename(pb_string).replace('.pb', '').split('_')
            if not parts[-1] == "fingers":
                video_name = os.path.join(video_root_dir, '_'.join(parts[:-2]) + ".mp4")
                start, end = map(int, parts[-2].split('-'))
                left = parts[-1].lower() == 'left'
                for t in range(start, end-self.length+1, int(length/2)):
                    self.samples.append({
                        'pb_path': pb_string,
                        'video_path': video_name,
                        'video_start': t-1,
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
        video_data = self._load_video(sample["video_path"], sample["video_start"], self.length)

        # Load pose data from protobuf
        pose_data = self._load_protobuf(sample["pb_path"], sample["pb_start"], self.length, sample["left"])

        return {"video": video_data, "pose": pose_data}

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

        pose_tensor = torch.empty([0, 3, 3], dtype=torch.float32)

        for frame in proto_data.frames[start:start+length]:
            # Extract and normalize landmarks
            pose_landmarks = ProtobufProcessor.extract_landmark_coordinates(frame, "pose_landmarks", range(10))
            
            # Normalize the extracted landmarks
            selected_landmarks = ProtobufProcessor.normalize_body_landmarks(pose_landmarks, left)

            pose_tensor = torch.cat((pose_tensor, selected_landmarks), dim=0)

        pose_tensor = pose_tensor.to(device)

        # Ensure we have exactly 15 frames
        num_frames, feature_size, dim_size = pose_tensor.shape
        if num_frames < length:
            print("Low frame count" + pb_path)
            padding = torch.zeros([length - num_frames, feature_size, dim_size ]).to(device)
            pose_tensor = torch.cat((pose_tensor, padding), dim=0)
        elif num_frames > length:
            pose_tensor = pose_tensor[:length, :, :]

        return batch_vectors_to_6D(pose_tensor)


# Neural Network Model Definition

class PoseVideoVAE(nn.Module):
    def __init__(self):
        super(PoseVideoVAE, self).__init__()

        # Load CLIP vision model
        self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        # Load VideoMAE model
        self.videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large")
        
        # Freeze both models
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.videomae_model.eval()
        for param in self.videomae_model.parameters():
            param.requires_grad = False

        # Get output dimensions
        clip_output_dim = self.clip_model.config.hidden_size  # Typically 1024
        videomae_output_dim = self.videomae_model.config.hidden_size  # Typically 1024
        combined_dim = clip_output_dim + videomae_output_dim

        # Latent space
        self.encoder = nn.Linear(clip_output_dim + videomae_output_dim, 16*8*2*2)

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

        # Decoder parameters
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

    def forward(self, x, deterministic=False):
        batch_size, time, channels, height, width = x.shape

        with torch.no_grad():
            # Reshape for CLIP: (B*T, C, H, W)
            spatial_frames = x.view(-1, channels, height, width)  # (B*T, C, H, W)
            # Get CLIP features            
            clip_outputs = self.clip_model(spatial_frames)
        
            # Use pooled output or mean of last hidden states
            if hasattr(clip_outputs, 'pooler_output') and clip_outputs.pooler_output is not None:
                clip_features = clip_outputs.pooler_output
            else:
                clip_features = clip_outputs.last_hidden_state.mean(dim=1)
            
            # Reshape back to (B, T, feature_dim) and average over time
            clip_features = clip_features.view(batch_size, time, -1)
            clip_features = clip_features.mean(dim=1)  # (B, feature_dim)

            # === VideoMAE ===
            videomae_outputs = self.videomae_model(x)
        
            # Use mean of last hidden states across time dimension
            videomae_features = videomae_outputs.last_hidden_state.mean(dim=1)

            # === Combine + Project ===
            combined = torch.cat([clip_features, videomae_features], dim=1)

        # Latent space
        h = self.encoder(combined)
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
        output_flat = output.view(batch_size, -1)
        output_6d = self.fc_output(output_flat)
        output_6d = output_6d.view(batch_size, 16, 6)

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

def loss_fn(fk, pose_data, model_output, logvar, mu, lambda_max=1.0, lambda_kl=0.1, lambda_vel=10.0, eps=1e-7):
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

    # calculate position loss using 6d representation
    pose_loss = 0#mse_loss(kine_output, pose_data)
    errors = torch.abs(kine_output - pose_data)
    max_per_joint, _ = torch.max(
            errors.view(errors.size(0), errors.size(1), -1),
            dim=1  # Reduce across frames and coordinates
        )
    max_loss = torch.mean(max_per_joint)

    # KL divergence loss for VAE regularization
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Combine all loss terms
    loss = pose_loss + lambda_kl * kl_loss + lambda_max * max_loss

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
    train_size = int(0.4 * len(dataset))
    eval_size = int(0.1 * len(dataset))
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
            model = PoseVideoVAE().to(device)
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
                verbose=0
                )
            if model_stats:
                print(f"\nTotal Parameters: {model_stats.total_params:,}")
                print(f"Trainable Parameters: {model_stats.trainable_params:,}")
                print(f"Non-trainable Parameters: {model_stats.total_params - model_stats.trainable_params:,}\n")

    # Create model, loss function, and optimizer
    model = PoseVideoVAE().to(device)
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
        total_max_loss = 0
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
        lambda_max = 1.0#lambda_scheduler(epoch, warmup_start=5, warmup_end=15, final_value=1.0)

        for i, batch in enumerate(batch_pbar):
            video_data = batch["video"]  # Shape: [batch, 15, 3, 258, 196]
            pose_data = batch["pose"]  # Shape: [batch, 15, 3, 3] (ground truth)
            
            model_output, mu, logvar = model(video_data)  # Output: [batch, 15, 26] (predicted)

            # Compute loss
            loss, pose_loss, max_loss, kl_loss = loss_fn(fk, pose_data, model_output, logvar, mu, lambda_kl = lambda_kl, lambda_max=lambda_max)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            # Update running loss
            loss_val = loss.item()
            pose_loss_val = 0#pose_loss.item()
            max_loss_val = max_loss.item()
            kl_loss_val = kl_loss.item()
            
            total_loss += loss_val
            total_pose_loss += pose_loss_val
            total_max_loss += max_loss_val
            total_kl_loss += kl_loss_val
            
            # Update progress bar with current loss
            batch_pbar.set_postfix({
                'loss': f'{loss_val:.2f}', 
                'pose_loss': f'{pose_loss_val:.2f}', 
                'max_loss': f'{max_loss_val:.2f}', 
                'kl_loss': f'{kl_loss_val:.2f}'
            })
        
        total_loss /= len(train_loader)
        total_pose_loss /= len(train_loader)
        total_max_loss /= len(train_loader)
        total_kl_loss /= len(train_loader)

        # Evaluation
        model.eval()  # Set model to evaluation mode
        eval_loss = 0
        eval_pose_loss = 0
        eval_max_loss = 0
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
                video_data = batch["video"]
                pose_data = batch["pose"]

                # Forward pass with pose data as input instead of video
                model_output, mu, logvar = model(video_data)

                # Compute loss
                loss, pose_loss, max_loss, kl_loss = loss_fn(fk, pose_data, model_output, logvar, mu, lambda_kl = lambda_kl, lambda_max=lambda_max)
                
                loss_val = loss
                pose_loss_val = pose_loss
                max_loss_val = max_loss
                kl_loss_val = kl_loss

                eval_loss += loss_val
                eval_pose_loss += pose_loss_val
                eval_max_loss += max_loss_val
                eval_kl_loss += kl_loss_val

                # Update progress bar
                eval_pbar.set_postfix({
                    'loss': f'{loss_val:.2f}', 
                    'pose_loss': f'{pose_loss_val:.2f}', 
                    'max_loss': f'{max_loss_val:.2f}', 
                    'kl_loss': f'{kl_loss_val:.2f}'
                })
        eval_loss /= len(eval_loader)
        eval_pose_loss /= len(eval_loader)
        eval_max_loss /= len(eval_loader)
        eval_kl_loss /= len(eval_loader)

        scheduler.step(eval_loss)

        # Check Overfitting
        if(eval_loss <= best_eval_loss):
            overfit = 0
            best_eval_loss = eval_loss
            # Save the trained model
            torch.save(model.state_dict(), f"sign_language_vae_{epoch+1}_best.pth")
        else:
            overfit += 1
            torch.save(model.state_dict(), f"sign_language_vae_{epoch+1}.pth")
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
              f"Train Loss: {total_loss:.5f}, Pose Loss: {total_pose_loss:.5f}, max Loss: {total_max_loss:.5f}, "
              f"KL Loss: {total_kl_loss:.5f}\n"#, Vel Loss: {total_vel_loss:.5f}, dir Loss: {total_dir_loss:.5f}\n"
              f"Eval Loss: {eval_loss:.5f}, Pose Loss: {eval_pose_loss:.5f}, max Loss: {eval_max_loss:.5f}, "
              f"KL Loss: {eval_kl_loss:.5f}")#, Vel Loss: {eval_vel_loss:.5f}, dir Loss: {eval_dir_loss:.5f}")
        
    print(f"\nTraining completed in {num_epochs} epochs")

    # Evaluation on test data
    # model.load_state_dict(torch.load("/home/cedra/psl_project/sign_language_pose_model_v3.2_100.pth", weights_only=True))
    print("\nEvaluating model on test data...")
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    test_pose_loss = 0
    test_max_loss = 0
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
    lambda_max = 1.0

    with torch.no_grad():  # No gradient computation
        f = open("test.log", 'w')
        for i, batch in enumerate(test_pbar):
            video_data = batch["video"]
            pose_data = batch["pose"]

            # Forward pass with pose data as input instead of video
            model_output, mu, logvar = model(video_data)

            # Compute loss
            loss, pose_loss, max_loss, kl_loss = loss_fn(fk, pose_data, model_output, logvar, mu, lambda_kl = lambda_kl, lambda_max=lambda_max)
            
            loss_val = loss
            pose_loss_val = pose_loss
            max_loss_val = max_loss
            kl_loss_val = kl_loss

            f.write(str(batch["name"][0])+","+str(loss_val)+","+str(pose_loss_val)+","+str(max_loss_val)+","+str(kl_loss_val)+"\n")

            test_loss += loss_val
            test_pose_loss += pose_loss_val
            test_max_loss += max_loss_val
            test_kl_loss += kl_loss_val

            # Update progress bar
            test_pbar.set_postfix({
                'loss': f'{loss_val:.2f}', 
                'pose_loss': f'{pose_loss_val:.2f}', 
                'max_loss': f'{max_loss_val:.2f}', 
                'kl_loss': f'{kl_loss_val:.2f}', 
                # 'vel_loss': f'{vel_loss_val:.2f}', 
                # 'dir_loss': f'{dir_loss_val:.2f}'
            })
    test_loss /= len(test_loader)
    test_pose_loss /= len(test_loader)
    test_max_loss /= len(test_loader)
    test_kl_loss /= len(test_loader)

    print(f"\nTest Loss: {test_loss:.5f}, Pose Loss: {test_pose_loss:.5f}, max Loss: {test_max_loss:.5f}, "
          f"KL Loss: {test_kl_loss:.5f}")

def main():
    """Main function to parse arguments and run the training or testing process"""

    # default values
    DEFAULT_DATA_DIR = "../dataset"
    DEFAULT_TEST_DIR = "../dataset/test"
    DEFAULT_VIDEO_DIR = "../1_clips"
    DEFAULT_URDF_PATH = "../rasa/hand.urdf"
    DEFAULT_NUM_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 16

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
