#!/bin/bash

# This script creates a virtual environment named 'adversarial' using python3/pip3

# Print Python version
echo "Python version:"
python3 --version

# Create the virtual environment
echo "Creating 'adversarial' virtual environment..."
python3 -m venv adversarial

# Check if environment was created successfully
if [ ! -d "adversarial" ]; then
    echo "Failed to create virtual environment. Trying with virtualenv..."
    pip3 install --user virtualenv
    python3 -m virtualenv adversarial
    
    # Check again if environment was created
    if [ ! -d "adversarial" ]; then
        echo "Error: Failed to create virtual environment. Exiting."
        exit 1
    fi
fi

# Determine activation script location
if [ -f "adversarial/bin/activate" ]; then
    ACTIVATE_SCRIPT="adversarial/bin/activate"
else
    echo "Error: Could not find activation script for virtual environment."
    exit 1
fi

echo "Activating virtual environment..."
source "$ACTIVATE_SCRIPT"

# Verify we're in the virtual environment
echo "Using pip from: $(which pip)"

echo "Installing required packages..."
# Install PyTorch
pip install torch torchvision torchaudio
# Install RobustBench
pip install robustbench
# Install AutoAttack
pip install git+https://github.com/fra31/auto-attack
# Install tqdm
pip install tqdm

echo "Creating test script..."
cat > test_installation.py << 'EOF'
import torch
import robustbench
from robustbench.model_zoo.cifar10 import linf
from robustbench.utils import load_model
from robustbench.data import load_cifar10
from autoattack import AutoAttack
from tqdm import tqdm

print("PyTorch version:", torch.__version__)
print("RobustBench version:", robustbench.__version__)
print("All packages imported successfully!")

# Try to load a small CIFAR10 dataset
try:
    x_test, y_test = load_cifar10(n_examples=10)
    print("Successfully loaded CIFAR10 data with shape:", x_test.shape)
except Exception as e:
    print("Error loading CIFAR10 data:", e)

print("Installation test complete!")
EOF

echo ""
echo "Setup complete! To use the environment:"
echo "1. Activate it with: source $ACTIVATE_SCRIPT"
echo "2. Test the installation with: python test_installation.py"
echo ""
echo "To deactivate the environment when done: deactivate"
