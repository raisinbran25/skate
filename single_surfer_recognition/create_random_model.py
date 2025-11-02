import torch
import torch.nn as nn
import os
import sys

# --- 1. GLOBAL CONFIGURATION (Must match the architecture) ---
DIM = 416
S = 13
B = 3
OUTPUT_DEPTH = B * 5 # 15 channels
MODEL_SAVE_PATH = 'single_surfer_recognition/random_model.pth'

# Set device (Needed for the torch.load/save context)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL AND INITIALIZATION DEFINITIONS ---

def initialize_weights(model):
    """Initializes the weights and biases of the model randomly (Kaiming/He initialization)."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Apply Kaiming (He) initialization suitable for ReLU/LeakyReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # Initialize BatchNorm weights and biases
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))

class SurferDetector(nn.Module):
    def __init__(self, output_depth):
        super().__init__()
        # Backbone Definition (Matching the training script)
        self.features = nn.Sequential(
            ConvBlock(3, 16, 3, 2, 1), ConvBlock(16, 32, 3, 2, 1),
            ConvBlock(32, 64, 3, 2, 1), ConvBlock(64, 128, 3, 2, 1),
            ConvBlock(128, 256, 3, 2, 1), ConvBlock(256, 256, 3, 1, 1),
        )
        self.head = nn.Conv2d(256, output_depth, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.features(x)
        return self.head(x)

# --- 3. MAIN SAVE FUNCTION ---

def save_random_model(save_path):
    """Instantiates the model, initializes weights randomly, and saves the state dictionary."""
    
    # 1. Create Model Instance
    model = SurferDetector(OUTPUT_DEPTH).to(DEVICE)
    
    # 2. Apply Random Initialization
    # This step produces the "untrained" model with random weights and biases
    initialize_weights(model) 
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 3. Save the State Dictionary
    try:
        torch.save(model.state_dict(), save_path)
        print(f"✅ Successfully created and saved untrained model weights to:")
        print(f"   {save_path}")
        print("Model is ready for loading by the evaluation script.")
    except Exception as e:
        print(f"❌ ERROR saving model: {e}")
        sys.exit(1)

# --- Run the Script ---
if __name__ == "__main__":
    save_random_model(MODEL_SAVE_PATH)
