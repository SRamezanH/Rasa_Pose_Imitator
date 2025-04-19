# Sign Language Recognition Using Pose Estimation

This project implements a sign language recognition system using pose estimation and deep learning. The system extracts pose landmarks from videos and uses a neural network model to classify sign language gestures.

## Prerequisites

- Python 3.11.11 recommended (becuase of colab)
- URDF file for the robot model (placed in the `rasa` folder)

## Setup Instructions

### 1. Create Virtual Environment and Install Dependencies

Run the setup script to create a virtual environment and install all required packages:

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a virtual environment named `psl_venv`
- Install all required dependencies

### 2. Activate the Virtual Environment

```bash
source psl_venv/bin/activate
```

### 3. Prepare the Dataset

The project expects your sign language videos and protobuf files to be organized in a directory structure:

```
dataset/
├── video1.mp4
├── video1.pb
├── video2.mp4
├── video2.pb
└── ...
```

Each `.mp4` file should have a corresponding `.pb` file with the same base name.

### 4. Extract URDF Files

If you have the robot model in a zip file named `rasa.zip`, extract it:

```bash
mkdir -p rasa
unzip rasa.zip -d rasa
```

## Running the Project

The script can be run without any command-line arguments, as it now uses default values for all parameters. The default values are:
- `data_dir`: "dataset" (looks for data in a folder named "dataset" in the current directory)
- `urdf_path`: "rasa/robot.urdf" (looks for the URDF file in the rasa folder)
- `num_epochs`: 10
- `batch_size`: 8
- `test_only`: False
- `extract_zip`: None

### Basic Usage

Simply run:
```bash
python main.py
```

This will:
1. Validate all videos in the specified data directory, skipping any corrupted files
2. Train the model using only valid videos
3. Evaluate the model on a test set
4. Save the trained model to "sign_language_model.pth"

### Command-line Parameters

You can still override the default values using these command-line arguments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data_dir` | string | "dataset" | Path to the dataset directory |
| `--urdf_path` | string | 'rasa/robot.urdf' | Path to the robot URDF file |
| `--num_epochs` | integer | 10 | Number of training epochs |
| `--batch_size` | integer | 8 | Batch size for training |
| `--test_only` | flag | False | Only test the dataset without training; includes comprehensive video validation |
| `--extract_zip` | string | None | Extract a zip file to the dataset directory |

### Examples

Basic training and evaluation with default values (includes automatic video validation):
```bash
python main.py
```

Specifying a custom dataset directory:
```bash
python main.py --data_dir /path/to/your/dataset
```

Test the dataset loading without training (useful to check video validation status):
```bash
python main.py --test_only
```

Full training with custom parameters:
```bash
python main.py --data_dir /path/to/your/dataset --urdf_path /path/to/urdf --num_epochs 20 --batch_size 16
```

Extract a zip file and train:
```bash
python main.py --data_dir /path/to/dataset --extract_zip /path/to/dataset.zip
```

## Project Structure

- `main.py`: The main script containing dataset processing, model definition, and training/evaluation code
- `requirements.txt`: List of required Python packages
- `setup.sh`: Script to set up the virtual environment
- `rasa/`: Directory containing the robot URDF model

## How It Works

1. **Data Processing and Validation**: 
   - The system performs upfront validation of all videos and protobuf files during dataset initialization.
   - Videos that cannot be opened, have no readable frames, or are missing corresponding protobuf files are automatically excluded.
   - Only valid videos are used for training and evaluation, completely skipping corrupted files.
   - A summary of valid and invalid files is displayed when loading the dataset.

2. **Feature Extraction**: CNN extracts spatial features from video frames.

3. **Temporal Modeling**: LSTM/GRU layers model the temporal relationships.

4. **Forward Kinematics**: The model's output is fed into a forward kinematics model to generate final pose predictions.

## Dataset Validation

The dataset validation process occurs automatically when creating a `PoseVideoDataset` instance:

- Each video file is checked to ensure it can be opened and at least one frame can be read
- The existence of the corresponding protobuf file is verified
- Only videos that pass all validation checks are included in the dataset
- The validation results are printed, showing:
  - Total number of videos found
  - Number of valid videos after validation
  - Percentage of videos that are valid

You can disable validation by setting `validate_files=False` when creating the dataset, but this is not recommended as it may lead to errors during training if corrupted files are encountered.

When running with `--test_only`, the validation statistics will be displayed, helping you identify potential issues with your dataset before starting a lengthy training process.

## Troubleshooting

If you encounter issues with pytorch-kinematics:
- Make sure numpy version is 1.23.5
- Check that the URDF file path is correctly specified in the code

Make sure numpy version is 1.23.5
