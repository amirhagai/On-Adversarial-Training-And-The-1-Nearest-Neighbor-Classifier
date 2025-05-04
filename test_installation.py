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
