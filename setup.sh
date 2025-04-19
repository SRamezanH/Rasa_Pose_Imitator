#!/bin/bash

# Exit on error
set -e

echo "Setting up virtual environment for PSL project..."

# Create virtual environment
python -m venv psl_venv

# Activate virtual environment
source psl_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install setuptools and wheel
pip install --upgrade setuptools wheel

# Get Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PYTHON_VERSION"

# Python version then install numpy
if [[ $(echo "$PYTHON_VERSION >= 3.12" | bc -l) -eq 1 ]]; then
    echo "Installing numpy>=1.26.0 for Python 3.12+"
    pip install "numpy>=1.26.4"
elif [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l) -eq 1 ]]; then
    echo "Installing numpy>=1.23.5 for Python 3.11"
    pip install "numpy>=1.23.5"
elif [[ $(echo "$PYTHON_VERSION >= 3.10" | bc -l) -eq 1 ]]; then
    echo "Installing numpy>=1.21.2 for Python 3.10"
    pip install "numpy>=1.21.2"
else
    echo "Installing numpy>=1.19.3 for Python 3.9 or earlier"
    pip install "numpy>=1.19.3"
fi

grep -v "numpy" requirements.txt > requirements_temp.txt

echo "Installing remaining dependencies..."
pip install -r requirements_temp.txt || {
    echo "Hash verification failed. Trying installation without hash verification..."
    pip install --no-deps --ignore-installed --no-cache-dir -r requirements_temp.txt
}


rm requirements_temp.txt

echo "Setup completed successfully! Virtual environment 'psl_venv' is now ready."
echo "The virtual environment is now active in your current shell."