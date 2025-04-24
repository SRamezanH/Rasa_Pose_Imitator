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
import time
from tqdm import tqdm


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


# Dataset class for handling pose data
class PoseDataset(Dataset):
    def __init__(self, root_dir, transform=None, validate_files=True):
        self.root_dir = root_dir
        self.transform = transform
        self.pb_files = sorted(glob.glob(os.path.join(root_dir, "*.pb")))
        self.valid_indices = []
        
        if validate_files:
            print("Validating pose files...")
            try:
                from tqdm import tqdm
                iterator = tqdm(enumerate(self.pb_files), total=len(self.pb_files), desc="Validating files")
            except ImportError:
                iterator = enumerate(self.pb_files)
                
            for i, pb_path in iterator:
                is_valid = True
                
                try:
                    ProtobufProcessor.load_protobuf_data(pb_path)
                except Exception as e:
                    print(f"Warning: Error validating {pb_path}: {str(e)}")
                    is_valid = False
                
                if is_valid:
                    self.valid_indices.append(i)
            
            print(f"Found {len(self.valid_indices)} valid files out of {len(self.pb_files)} total files")
        else:
            self.valid_indices = list(range(len(self.pb_files)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        file_idx = self.valid_indices[idx]
        pb_path = self.pb_files[file_idx]
        pose_data = self._load_protobuf(pb_path)
        return pose_data

    def _load_protobuf(self, pb_path):
        proto_data = ProtobufProcessor.load_protobuf_data(pb_path)
        left_side = pb_path.endswith("_left.pb")

        pose_tensor = torch.empty([0, 3, 3], dtype=torch.float32)

        for frame in proto_data.frames:
            pose_landmarks = ProtobufProcessor.extract_landmark_coordinates(
                frame, "pose_landmarks", range(10)
            )
            
            selected_landmarks = ProtobufProcessor.normalize_body_landmarks(pose_landmarks, left_side)
            pose_tensor = torch.cat((pose_tensor, selected_landmarks), dim=0)

        pose_tensor = pose_tensor.to(device)

        num_frames, feature_size, dim_size = pose_tensor.shape
        if num_frames < 15:
            padding = torch.zeros([15 - num_frames, feature_size, dim_size]).to(device)
            pose_tensor = torch.cat((pose_tensor, padding), dim=0)
        elif num_frames > 15:
            pose_tensor = pose_tensor[:15, :, :]

        return pose_tensor


# 2. Forward Kinematics for generator
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



# 3. Generator: noise -> joint_seq -> fk positions
class MotionGenerator(nn.Module):
    def __init__(self, latent_dim=100, joint_dim=6, seq_len=15, urdf_path=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.joint_dim = joint_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )
        self.gru = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.attn = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1, bias=False),
            nn.Softmax(dim=1)
        )
        self.joint_fc = nn.Linear(512, joint_dim)
        self.fk = ForwardKinematics(urdf_path)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim]
        Returns:
            poses: [B, seq_len, 3, 3]
        """
        h = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        gru_out, _ = self.gru(h)                  # [B, seq_len, 512]
        w = self.attn(gru_out)                    # [B, seq_len, 1]
        c = torch.sum(w * gru_out, dim=1, keepdim=True)
        joints = self.joint_fc(gru_out + c)       # [B, seq_len, joint_dim=6]
        poses = self.fk.batch_forward_kinematics(joints)  # [B, seq_len, 6_joints, 3]
        # reshape to [B, seq_len, 3,3] if only 3x3 needed per frame:
        # Here select first 3 links to form 3x3?
        # Or assume joint_dim==9 and fk returns [B,T,3,3].
        return poses
        


class MotionDiscriminator(nn.Module):
    def __init__(self, seq_len=15):
        super().__init__()
        self.frame_net = nn.Sequential(
            nn.Linear(3 * 3, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2)
        )
        self.seq_net = nn.Sequential(
            nn.Linear(32 * seq_len, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, 3, 3]
        Returns:
            validity: [B, 1]
        """
        B, T, _, _ = x.shape
        f = x.view(B * T, 9)
        f = self.frame_net(f)
        f = f.view(B, T * 32)
        return self.seq_net(f)

    def extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frame-level features for FM loss
        Args:
            x: [B, seq_len, 3, 3]
        Returns:
            frame_features: [B*T, 32]
        """
        B, T, _, _ = x.shape
        x = x.view(B * T, 9)
        return self.frame_net(x)

def train_motion_gan(
    data_dir,
    test_dir,
    urdf_path,
    num_epochs,
    batch_size,
    learning_rate=1e-4,
    latent_dim=100,
    train_split=0.8,
    eval_split=0.2,
    model_prefix="motion_gan"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"  - GPU: {props.name}")
        print(f"  - CUDA Capability: {props.major}.{props.minor}")
        print(f"  - Total Memory: {props.total_memory/1024**3:.1f} GB")

    dataset = PoseDataset(root_dir=data_dir, validate_files=True)
    N = len(dataset)
    if N == 0:
        print("❌ No valid .pb samples found.")
        return
    print(f"Loaded dataset: {N} samples")

    train_n = int(train_split * N)
    eval_n = N - train_n
    train_ds, eval_ds = random_split(dataset, [train_n, eval_n])
    print(f"Train size: {train_n}, Eval size: {eval_n}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    G = MotionGenerator(latent_dim=latent_dim, joint_dim=6, seq_len=15, urdf_path=urdf_path).to(device)
    D = MotionDiscriminator(seq_len=15).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion = torch.nn.BCELoss()

    best_eval_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        G.train()
        D.train()
        epoch_start = time.time()

        running_D_loss = 0.0
        running_G_loss = 0.0

        for real in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            real = real.to(device)
            B = real.size(0)
            valid = torch.ones(B, 1, device=device)
            fake = torch.zeros(B, 1, device=device)

            # Train Discriminator
            optD.zero_grad()
            real_loss = criterion(D(real), valid)
            z = torch.randn(B, latent_dim, device=device)
            fake_motions = G(z)
            fake_loss = criterion(D(fake_motions.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optD.step()

            # Train Generator
            optG.zero_grad()
            z = torch.randn(B, latent_dim, device=device)
            gen_motions = G(z)
            g_adv_loss = criterion(D(gen_motions), valid)

            # Feature matching loss
            real_feat = D.extract_frame_features(real)
            fake_feat = D.extract_frame_features(gen_motions)
            fm_loss = F.mse_loss(fake_feat.mean(0), real_feat.mean(0))
            

            total_g_loss = g_adv_loss + 0.1 * fm_loss
            total_g_loss.backward()
            optG.step()

            running_D_loss += d_loss.item()
            running_G_loss += total_g_loss.item()

        avg_D_loss = running_D_loss / len(train_loader)
        avg_G_loss = running_G_loss / len(train_loader)

        # Eval
        D.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for real in eval_loader:
                real = real.to(device)
                pred = D(real)
                eval_loss += criterion(pred, torch.ones_like(pred)).item()
        avg_eval_loss = eval_loss / len(eval_loader)

        print(f"\nEpoch {epoch}/{num_epochs} — D_loss: {avg_D_loss:.4f}, G_loss: {avg_G_loss:.4f}, Eval_D: {avg_eval_loss:.4f}, Time: {time.time()-epoch_start:.1f}s")

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save(G.state_dict(), f"{model_prefix}_G_best.pth")
            torch.save(D.state_dict(), f"{model_prefix}_D_best.pth")

    print("✅ Training complete.")

    if test_dir:
        print(f"\nTesting test dataset at '{test_dir}':")
        test_dataset(test_dir)

# 6. Utility: test dataset loader
def test_dataset(data_dir: str):
    print(f"Testing PoseDataset with directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"Error: '{data_dir}' does not exist.")
        return

    pb_files = [f for f in os.listdir(data_dir) if f.endswith('.pb')]
    if not pb_files:
        print(f"Error: no .pb files in '{data_dir}'")
        return
    print(f"Found {len(pb_files)} .pb files")

    try:
        dataset = PoseDataset(root_dir=data_dir, validate_files=False)
        n = len(dataset)
        print(f"Loaded dataset with {n} samples ({n/len(pb_files)*100:.1f}% valid)")
    except Exception as e:
        print("❌ Error creating dataset:", e)
        return

    try:
        sample = dataset[0]
        print("Sample shape:", sample.shape, "range:", sample.min().item(), sample.max().item())
    except Exception as e:
        print("❌ Error accessing sample:", e)
        return

    try:
        loader = DataLoader(dataset, batch_size=min(4,n), shuffle=True)
        from tqdm import tqdm
        for i, batch in enumerate(tqdm(loader, total=min(3,len(loader)), desc="Batches")):
            print(f"Batch {i+1} shape:", batch.shape)
            if i>=2: break
        print("✅ DataLoader OK")
    except Exception as e:
        print("❌ DataLoader error:", e)
        return

    print("✅ All dataset tests passed.")


# Example usage:
# test_dataset('/path/to/pb_dir')
# ds = PoseDataset('/path/to/pb_dir')
# loader = DataLoader(ds, batch_size=32, shuffle=True)
# G = MotionGenerator(latent_dim=100, joint_dim=6, seq_len=15, urdf_path='robot.urdf')
# D = MotionDiscriminator(seq_len=15)
# trainer = MotionGANTrainer(G, D)
# trainer.train(loader, epochs=100)


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
        train_motion_gan(
            data_dir=data_dir,
            test_dir=test_dir,
            urdf_path=urdf_path,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

if __name__ == "__main__":
    main()
