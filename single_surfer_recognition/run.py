import image_collection
import yolo_labels
# import student_training # problematic
import create_random_model
# import testing_model # problematic
import visualize_boxes

# option to add more training/testing images
def image_collection_function():
    image_collection.interactive_screen_recorder(
        image_collection.DIM, 
        image_collection.FPS_TARGET, 
        image_collection.MONITOR_REGION,
        image_collection.OUTPUT_FOLDERS,
        image_collection.IMG_EXTENSION,
        image_collection.START_KEY,
        image_collection.STOP_KEY
    )

# generating training teacher labels
def yolo_training_labels_function():
    yolo_labels.generate_teacher_labels_csv(
        yolo_labels.TRAINING_IMAGE_DIR, 
        yolo_labels.TRAINING_OUTPUT_CSV, 
        yolo_labels.TARGET_CLASS_ID, 
        yolo_labels.CONF_THRESHOLD
    )

#testing labels
def yolo_testing_labels_function():
    yolo_labels.generate_teacher_labels_csv(
        yolo_labels.TESTING_IMAGE_DIR, 
        yolo_labels.TESTING_OUTPUT_CSV, 
        yolo_labels.TARGET_CLASS_ID, 
        yolo_labels.CONF_THRESHOLD
    )

# create a random model with path 'single_surfer_recognition/random_model.pth'
def random_model_function():
    create_random_model.save_random_model(
        create_random_model.MODEL_SAVE_PATH
    )

# train a student model using the teacher training labels (PROBLEMATIC)
# def student_training_function():
#    student_training.train_model()

# apply the student/random models to the testing folder (PROBLEMATIC)
# def apply_models_function():
#    testing_model.run()

# visualize the results
def visualise_function():
    visualize_boxes.visualize_labels()

prompts = {

    "enter nothing to begin the screen recording, or 1 to skip: " : image_collection_function,
    "enter nothing to generate teacher TRAINING labels, or 1 to skip: " : yolo_training_labels_function,
    "enter nothing to generate teacher TESTING labels, or 1 to skip: " : yolo_testing_labels_function,
    "enter nothing to create a random model, or 1 to skip: " : random_model_function
    #"enter nothing to train a student model using the teacher labels, or 1 to skip: " : student_training_function,
    #"enter nothing to apply the models to testing images, or 1 to skip: " : apply_models_function,
    #"enter nothing to visualize the results, or 1 to skip: " : visualise_function
    
}

for key, value in prompts.items():
    if input(key) == "":
        value()

input("enter nothing to train a student model using the teacher labels: ")

# entire student training model below.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
import math
import sys

# --- 1. GLOBAL CONFIGURATION ---

# Training and Architecture Parameters
DIM = 416                      # Input image size (DIM x DIM)
S = 13                         # Grid size (S x S)
B = 3                          # Number of Anchor Boxes
C = 1                          # Number of Classes (Surfer only)
OUTPUT_DEPTH = B * 5           # 3 * (4 coords + 1 confidence) = 15

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
LAMBDA_LOC = 50                # Weight for GIoU Loss (Localization)
LAMBDA_CONF = 1                # Weight for BCE Loss (Confidence)
MODEL_SAVE_PATH = 'single_surfer_recognition/student_model.pth'

# Data Paths (Must match the paths used in the capture/label scripts)
IMAGE_DIR = "single_surfer_recognition/training_photos"
LABEL_CSV = "single_surfer_recognition/teacher_train_labels.csv"

# Optional: List of filenames to include in training (e.g., ["00001.jpg", "00003.jpg"])
# Leave this array empty [] to use NONE of the images.
SELECTED_FILES = [] 

# uncomment to train using first 2500 images
for i in range(len(os.listdir(f"{IMAGE_DIR}"))):
    # FIX: Append the .jpg extension to match the actual file names
    SELECTED_FILES.append(f"{i:05d}.jpg")

# Normalized Anchor Box Priors [w, h]
ANCHORS = torch.tensor([
    [0.10, 0.30],  # Tall/Thin
    [0.25, 0.50],  # Medium
    [0.40, 0.20]   # Wide/Prone
], dtype=torch.float32)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. UTILITY FUNCTIONS ---

def iou_giou(boxes1, boxes2, is_giou=True):
    """Calculates IoU and GIoU loss. Boxes in [x_c, y_c, w, h] format."""
    
    def box_cwh_to_xyxy(boxes):
        x1 = boxes[..., 0] - boxes[..., 2] / 2
        y1 = boxes[..., 1] - boxes[..., 3] / 2
        x2 = boxes[..., 0] + boxes[..., 2] / 2
        y2 = boxes[..., 1] + boxes[..., 3] / 2
        return torch.stack((x1, y1, x2, y2), dim=-1)

    box1 = box_cwh_to_xyxy(boxes1)
    box2 = box_cwh_to_xyxy(boxes2)

    # Intersection coordinates
    x1_inter = torch.max(box1[..., 0], box2[..., 0])
    y1_inter = torch.max(box1[..., 1], box2[..., 1])
    x2_inter = torch.min(box1[..., 2], box2[..., 2])
    y2_inter = torch.min(box1[..., 3], box2[..., 3])
    
    # Intersection area
    intersection = (x2_inter - x1_inter).clamp(0) * (y2_inter - y1_inter).clamp(0)
    
    # Area of boxes
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = area1 + area2 - intersection

    iou = intersection / (union + 1e-6)
    
    if not is_giou:
        return iou

    # Smallest enclosing box C coordinates
    x1_c = torch.min(box1[..., 0], box2[..., 0])
    y1_c = torch.min(box1[..., 1], box2[..., 1])
    x2_c = torch.max(box1[..., 2], box2[..., 2])
    y2_c = torch.max(box1[..., 3], box2[..., 3])
    
    area_c = (x2_c - x1_c) * (y2_c - y1_c)
    
    # GIoU
    giou = iou - (area_c - union) / (area_c + 1e-6)
    
    # GIoU Loss
    return 1 - giou

def decode_predictions(predictions, anchors, S):
    """
    Decodes raw network output (tx, ty, tw, th, tc) into normalized box coordinates 
    (xc, yc, w, h, p_obj).
    """
    N = predictions.shape[0] 
    # Reshape to (N, S, S, B, 5)
    predictions = predictions.view(N, B, 5, S, S).permute(0, 3, 4, 1, 2).contiguous()

    # Raw outputs
    t_xy = predictions[..., 0:2] 
    t_wh = predictions[..., 2:4] 
    t_c = predictions[..., 4:5] 

    # Grid Offsets (cx, cy)
    grid_y, grid_x = torch.meshgrid(torch.arange(S, device=DEVICE), torch.arange(S, device=DEVICE), indexing='ij')
    grid_xy = torch.stack((grid_x, grid_y), dim=-1).float().view(1, S, S, 1, 2) 

    # Decode Coordinates: (cx + sigmoid(tx)) / S
    pred_xy = (grid_xy + torch.sigmoid(t_xy)) / S
    
    # Decode Dimensions: pw * exp(tw)
    pred_wh = anchors.to(DEVICE) * torch.exp(t_wh)
    
    # Decode Confidence: sigmoid(tc)
    pred_conf = torch.sigmoid(t_c)

    # (N, S, S, B, [xc, yc, w, h, p_obj])
    return torch.cat([pred_xy, pred_wh, pred_conf], dim=-1)

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

# --- 3. CUSTOM MODEL DEFINITION ---

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
        
        # Simple Sequential Backbone (Downsampling to 13x13 grid)
        self.features = nn.Sequential(
            # 416x416 -> 208x208
            ConvBlock(3, 16, 3, 2, 1),
            # 208x208 -> 104x104
            ConvBlock(16, 32, 3, 2, 1),
            # 104x104 -> 52x52
            ConvBlock(32, 64, 3, 2, 1),
            # 52x52 -> 26x26
            ConvBlock(64, 128, 3, 2, 1),
            # 26x26 -> 13x13
            ConvBlock(128, 256, 3, 2, 1),
            # Final Feature Map Processing
            ConvBlock(256, 256, 3, 1, 1), # No stride, just deeper features
        )
        
        # Prediction Head (1x1 Convolution to reduce channels to OUTPUT_DEPTH=15)
        # K=1, S=1 ensures no change to the 13x13 spatial size
        self.head = nn.Conv2d(256, output_depth, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.features(x)
        predictions = self.head(x)
        return predictions

# --- 4. CUSTOM DATASET ---

class SurferDataset(Dataset):
    def __init__(self, image_dir, label_csv, dim, selected_files=None, transform=None):
        self.image_dir = image_dir
        self.dim = dim
        self.transform = transform
        
        try:
            self.labels_df = pd.read_csv(label_csv)
        except FileNotFoundError:
            print(f"FATAL ERROR: Label CSV not found at {label_csv}")
            sys.exit(1)
            
        # Filter files based on user selection. 
        if selected_files:
            # If the list is NOT empty, filter the dataframe to include only selected files.
            self.labels_df = self.labels_df[self.labels_df['filename'].isin(selected_files)].reset_index(drop=True)
        else:
            # If the list IS empty, set the DataFrame to be empty (0 rows).
            self.labels_df = self.labels_df.iloc[0:0] 
        
        self.filenames = self.labels_df['filename'].tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        
        # Load image (PIL for compatibility with torchvision transforms)
        image = Image.open(image_path).convert("RGB")
        
        # Get normalized label [xc, yc, w, h, conf]
        row = self.labels_df.iloc[idx]
        x_c, y_c, w, h = row[['x_center', 'y_center', 'width', 'height']].values.astype(np.float32)
        confidence = row['confidence']
        
        # --- Create Ground Truth Tensor (Target) ---
        
        # Target tensor shape: (S, S, B, 5) -> [p_obj, xc, yc, w, h] for positive anchor
        target = torch.zeros(S, S, B, 5) 

        # If the teacher model found a surfer (conf > 0)
        if confidence > 0.0:
            # 1. Determine the Responsible Grid Cell (cx, cy)
            grid_x_float = x_c * S
            grid_y_float = y_c * S
            
            # The integer part is the grid cell index
            grid_x = int(grid_x_float)
            grid_y = int(grid_y_float)
            
            # 2. Find the Best Anchor Box
            max_iou = -1
            best_anchor_idx = 0
            
            # Compare the ground truth box dimensions (w, h) against the fixed anchors
            for anchor_idx, anchor in enumerate(ANCHORS):
                # Calculate IoU between ground truth and anchor *relative to the same center*
                iou = iou_giou(torch.tensor([[0.5, 0.5, w, h]]), 
                               torch.tensor([[0.5, 0.5, anchor[0].item(), anchor[1].item()]]), 
                               is_giou=False)
                
                if iou.item() > max_iou:
                    max_iou = iou.item()
                    best_anchor_idx = anchor_idx
            
            # 3. Assign Target
            # Set target confidence to 1 (Surfer is Present)
            target[grid_y, grid_x, best_anchor_idx, 0] = 1.0 
            
            # Set box coordinates (these are the normalized YOLO coordinates)
            target[grid_y, grid_x, best_anchor_idx, 1:5] = torch.tensor([x_c, y_c, w, h])

        # Apply transformations (e.g., ToTensor, Normalization)
        if self.transform:
            image = self.transform(image)
        
        # Return image and target, correctly permuted for the loss function
        return image, target.permute(3, 0, 1, 2).contiguous() # (5, S, S, B)

# --- 5. CUSTOM LOSS MODULE ---

class YoloLoss(nn.Module):
    def __init__(self, anchors, S, LAMBDA_LOC, LAMBDA_CONF):
        super().__init__()
        self.anchors = anchors
        self.S = S
        self.lambda_loc = LAMBDA_LOC
        self.lambda_conf = LAMBDA_CONF
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets):
        
        N = predictions.shape[0]
        # predictions -> (N, B, 5, S, S) -> (N, S, S, B, 5)
        raw_preds = predictions.view(N, B, 5, S, S).permute(0, 3, 4, 1, 2)
        
        decoded_boxes = decode_predictions(predictions, self.anchors, self.S)
        
        # target_mask: 1 where object is present, 0 otherwise (N, S, S, B)
        target_mask = targets[..., 0] 
        # target_box: Ground truth box [xc, yc, w, h] (N, S, S, B, 4)
        target_box = targets[..., 1:5] 

        # --- A. Localization Loss (GIoU) ---
        
        # Only consider predictions from the responsible anchor/cell
        predicted_pos_boxes = decoded_boxes[target_mask == 1][..., :4] 
        target_pos_boxes = target_box[target_mask == 1]
        
        loc_loss = torch.tensor(0.0, device=DEVICE)
        if predicted_pos_boxes.numel() > 0:
            giou_loss = iou_giou(predicted_pos_boxes, target_pos_boxes, is_giou=True)
            loc_loss = self.lambda_loc * torch.mean(giou_loss)
        
        # --- B. Confidence Loss (BCE) ---
        
        # Target confidence is 1 where target_mask is 1, 0 otherwise
        target_conf = target_mask.float()
        
        # Predicted confidence (raw logit before sigmoid)
        pred_conf_logit = raw_preds[..., 4]
        
        # BCE Loss (sum over batch and all 507 possible locations)
        conf_loss = self.lambda_conf * self.bce(pred_conf_logit, target_conf)

        # --- C. Total Loss ---
        total_loss = loc_loss + conf_loss
        
        return total_loss, loc_loss, conf_loss


# --- 6. MAIN TRAINING FUNCTION ---

def train_model():
    print(f"Using device: {DEVICE}")

    # 1. Setup Data Transforms and DataLoader
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = SurferDataset(IMAGE_DIR, LABEL_CSV, DIM, SELECTED_FILES, transform)
    
    
    if len(dataset) == 0:
        print("Dataset is empty. Check paths and ensure SELECTED_FILES array is populated.")
        return
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Loaded {len(dataset)} images into DataLoader.")

    # 2. Initialize Model, Loss, and Optimizer
    model = SurferDetector(OUTPUT_DEPTH).to(DEVICE)
    
    # Initialize weights randomly, achieving the goal of an untrained model.
    initialize_weights(model) 
    
    criterion = YoloLoss(ANCHORS, S, LAMBDA_LOC, LAMBDA_CONF)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # 3. Training Loop
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE)
            # Targets are permuted back to (N, B, S, S, 5) for easier processing in loss function
            targets = targets.permute(0, 2, 3, 4, 1).to(DEVICE) 
            
            # Forward pass
            predictions = model(images) 
            
            # Calculate total loss
            loss, loc_loss, conf_loss = criterion(predictions, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} (Loc: {loc_loss.item():.4f} Conf: {conf_loss.item():.4f})")

        avg_epoch_loss = total_epoch_loss / len(train_loader)
        print(f"\n--- Epoch {epoch+1} Finished | Avg Loss: {avg_epoch_loss:.4f} ---")

        # 4. Save Checkpoint

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"*** Model saved: New best average loss: {best_loss:.4f} ***\n")

    print(f"\nTraining finished! Best model saved to {MODEL_SAVE_PATH}")


# --- Run the Script ---
if __name__ == "__main__":
    train_model()

# end student_training here.

input("enter nothing to apply the models to testing images: ")

# entire student testing model below.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
import sys
import math

# --- 1. CONFIGURATION ---

# Training and Architecture Parameters (MUST match the trained model)
DIM = 416                      # Input image size (DIM x DIM)
S = 13                         # Grid size (S x S)
B = 3                          # Number of Anchor Boxes
OUTPUT_DEPTH = B * 5           # 3 * (4 coords + 1 confidence) = 15

# Hyperparameters (Must match training if calculating loss)
LAMBDA_LOC = 50                
LAMBDA_CONF = 1                
BATCH_SIZE = 16

# Paths and Files
IMAGE_DIR = "single_surfer_recognition/testing_photos" # Assuming test images are here 
LABEL_CSV = "single_surfer_recognition/teacher_labels.csv" # Ground truth labels for loss calculation

INPUT_OUTPUT = {    # model files, corresponding csv output names
    "single_surfer_recognition/student_model.pth" : "single_surfer_recognition/student_labels.csv",
    "single_surfer_recognition/random_model.pth" : "single_surfer_recognition/random_labels.csv"
}

# List of filenames to use for evaluation (e.g., test set files). 
# Leave this array empty [] to use NONE of the images.
SELECTED_FILES = [] 
for i in range(len(os.listdir(f"{IMAGE_DIR}"))):
    SELECTED_FILES.append(f"{i:05d}.jpg")

# Normalized Anchor Box Priors [w, h]
ANCHORS = torch.tensor([
    [0.10, 0.30],  
    [0.25, 0.50],  
    [0.40, 0.20]   
], dtype=torch.float32)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. UTILITY FUNCTIONS (Reused from Trainer) ---

def iou_giou(boxes1, boxes2, is_giou=True):
    """Calculates IoU and GIoU loss. Boxes in [x_c, y_c, w, h] format."""
    def box_cwh_to_xyxy(boxes):
        x1 = boxes[..., 0] - boxes[..., 2] / 2
        y1 = boxes[..., 1] - boxes[..., 3] / 2
        x2 = boxes[..., 0] + boxes[..., 2] / 2
        y2 = boxes[..., 1] + boxes[..., 3] / 2
        return torch.stack((x1, y1, x2, y2), dim=-1)

    box1 = box_cwh_to_xyxy(boxes1)
    box2 = box_cwh_to_xyxy(boxes2)

    x1_inter = torch.max(box1[..., 0], box2[..., 0])
    y1_inter = torch.max(box1[..., 1], box2[..., 1])
    x2_inter = torch.min(box1[..., 2], box2[..., 2])
    y2_inter = torch.min(box1[..., 3], box2[..., 3])
    
    intersection = (x2_inter - x1_inter).clamp(0) * (y2_inter - y1_inter).clamp(0)
    
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = area1 + area2 - intersection

    iou = intersection / (union + 1e-6)
    
    if not is_giou: return iou

    x1_c = torch.min(box1[..., 0], box2[..., 0])
    y1_c = torch.min(box1[..., 1], box2[..., 1])
    x2_c = torch.max(box1[..., 2], box2[..., 2])
    y2_c = torch.max(box1[..., 3], box2[..., 3])
    
    area_c = (x2_c - x1_c) * (y2_c - y1_c)
    
    giou = iou - (area_c - union) / (area_c + 1e-6)
    return 1 - giou

def decode_predictions(predictions, anchors, S):
    """Decodes raw network output into normalized box coordinates."""
    N = predictions.shape[0] 
    predictions = predictions.view(N, B, 5, S, S).permute(0, 3, 4, 1, 2).contiguous()

    t_xy = predictions[..., 0:2] 
    t_wh = predictions[..., 2:4] 
    t_c = predictions[..., 4:5] 

    grid_y, grid_x = torch.meshgrid(torch.arange(S, device=DEVICE), torch.arange(S, device=DEVICE), indexing='ij')
    grid_xy = torch.stack((grid_x, grid_y), dim=-1).float().view(1, S, S, 1, 2) 

    pred_xy = (grid_xy + torch.sigmoid(t_xy)) / S
    pred_wh = anchors.to(DEVICE) * torch.exp(t_wh)
    pred_conf = torch.sigmoid(t_c)

    return torch.cat([pred_xy, pred_wh, pred_conf], dim=-1)

def apply_nms(decoded_boxes, iou_threshold=0.5, conf_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) and selects the single best box 
    from the 507 possible predictions.
    
    Returns: [x_c, y_c, w, h, confidence] of the best box, or zeros if none found.
    """
    # Reshape all 507 predictions into a list: (507, 5)
    all_preds = decoded_boxes.view(-1, 5) 
    
    # Filter by Confidence Threshold
    conf_mask = all_preds[:, 4] >= conf_threshold
    filtered_preds = all_preds[conf_mask]
    
    if filtered_preds.size(0) == 0:
        return torch.zeros(5, device=DEVICE) # Return zero tensor if no box found

    # Convert [x_c, y_c, w, h] to [x_min, y_min, x_max, y_max] for NMS utility
    boxes_xyxy = filtered_preds[:, :4].clone()
    boxes_xyxy[:, 0] = filtered_preds[:, 0] - filtered_preds[:, 2] / 2 # x_min
    boxes_xyxy[:, 1] = filtered_preds[:, 1] - filtered_preds[:, 3] / 2 # y_min
    boxes_xyxy[:, 2] = filtered_preds[:, 0] + filtered_preds[:, 2] / 2 # x_max
    boxes_xyxy[:, 3] = filtered_preds[:, 1] + filtered_preds[:, 3] / 2 # y_max
    
    # Get scores and apply NMS
    scores = filtered_preds[:, 4]
    
    # PyTorch NMS utility (takes xyxy)
    indices = torch.ops.torchvision.nms(boxes_xyxy, scores, iou_threshold)
    
    # We only need the single best box for this single-surfer model
    best_index = indices[0]
    best_prediction = filtered_preds[best_index]
    
    return best_prediction.cpu().numpy()


# --- 3. MODEL AND LOSS DEFINITIONS (Reused from Trainer) ---

# Defined again so the script is self-contained and loads weights correctly.

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
        self.features = nn.Sequential(
            ConvBlock(3, 16, 3, 2, 1), ConvBlock(16, 32, 3, 2, 1),
            ConvBlock(32, 64, 3, 2, 1), ConvBlock(64, 128, 3, 2, 1),
            ConvBlock(128, 256, 3, 2, 1), ConvBlock(256, 256, 3, 1, 1),
        )
        self.head = nn.Conv2d(256, output_depth, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.features(x)
        return self.head(x)

class YoloLoss(nn.Module):
    def __init__(self, anchors, S, LAMBDA_LOC, LAMBDA_CONF):
        super().__init__()
        self.anchors = anchors
        self.S = S
        self.lambda_loc = LAMBDA_LOC
        self.lambda_conf = LAMBDA_CONF
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
    def forward(self, predictions, targets):
        N = predictions.shape[0]
        raw_preds = predictions.view(N, B, 5, S, S).permute(0, 3, 4, 1, 2)
        decoded_boxes = decode_predictions(predictions, self.anchors, self.S)
        target_mask = targets[..., 0] 
        target_box = targets[..., 1:5] 
        loc_loss = torch.tensor(0.0, device=DEVICE)
        predicted_pos_boxes = decoded_boxes[target_mask == 1][..., :4] 
        target_pos_boxes = target_box[target_mask == 1]
        if predicted_pos_boxes.numel() > 0:
            giou_loss = iou_giou(predicted_pos_boxes, target_pos_boxes, is_giou=True)
            loc_loss = self.lambda_loc * torch.mean(giou_loss)
        target_conf = target_mask.float()
        pred_conf_logit = raw_preds[..., 4]
        conf_loss = self.lambda_conf * self.bce(pred_conf_logit, target_conf)
        total_loss = loc_loss + conf_loss
        return total_loss, loc_loss, conf_loss


# --- 4. CUSTOM DATASET (Evaluation Version) ---

class SurferEvalDataset(Dataset):
    def __init__(self, image_dir, LABEL_CSV, dim, selected_files=None, transform=None):
        self.image_dir = image_dir
        self.dim = dim
        self.transform = transform
        
        try:
            self.labels_df = pd.read_csv(LABEL_CSV)
        except FileNotFoundError:
            print(f"FATAL ERROR: Label CSV not found at {LABEL_CSV}")
            sys.exit(1)
            
        if not selected_files:
             # If list is empty, ensure the dataframe is empty
             self.labels_df = self.labels_df.iloc[0:0]
        else:
             # Filter dataframe by selected files
             self.labels_df = self.labels_df[self.labels_df['filename'].isin(selected_files)].reset_index(drop=True)
        
        if self.labels_df.empty:
            print("FATAL ERROR: SELECTED_FILES list is empty or invalid. Cannot run evaluation.")
            sys.exit(1)
            
        self.filenames = self.labels_df['filename'].tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        
        image = Image.open(image_path).convert("RGB")
        
        # We need the index to retrieve the correct label row later
        original_idx = self.labels_df.index[self.labels_df['filename'] == filename].tolist()[0]
        
        # Create Ground Truth Target Tensor for loss calculation
        row = self.labels_df.iloc[idx]
        x_c, y_c, w, h = row[['x_center', 'y_center', 'width', 'height']].values.astype(np.float32)
        confidence = row['confidence']
        target = torch.zeros(S, S, B, 5) 

        if confidence > 0.0:
            grid_x, grid_y = int(x_c * S), int(y_c * S)
            best_anchor_idx = 0
            max_iou = -1
            
            # Find the best anchor for target assignment
            for anchor_idx, anchor in enumerate(ANCHORS):
                iou = iou_giou(torch.tensor([[0.5, 0.5, w, h]]), 
                               torch.tensor([[0.5, 0.5, anchor[0].item(), anchor[1].item()]]), is_giou=False)
                if iou.item() > max_iou:
                    max_iou = iou.item()
                    best_anchor_idx = anchor_idx
            
            target[grid_y, grid_x, best_anchor_idx, 0] = 1.0 # p=1
            target[grid_y, grid_x, best_anchor_idx, 1:5] = torch.tensor([x_c, y_c, w, h])

        if self.transform:
            image = self.transform(image)
        
        # Return image, filename, and the ground truth target
        return image, filename, target.permute(3, 0, 1, 2).contiguous()


# --- 5. EVALUATION FUNCTION ---

def evaluate_model(model_filename, output_csv):
    print(f"Using device: {DEVICE}")

    # 1. Setup DataLoader
    from torchvision import transforms
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = SurferEvalDataset(IMAGE_DIR, LABEL_CSV, DIM, SELECTED_FILES, transform)
    
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Loaded {len(dataset)} images for evaluation.")

    # 2. Initialize Model and Load Weights
    model = SurferDetector(OUTPUT_DEPTH).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_filename, map_location=DEVICE))
        print(f"✅ Successfully loaded trained weights from {model_filename}.")
    except FileNotFoundError:
        print(f"❌ Error: Model weights file not found at {model_filename}. Cannot evaluate.")
        sys.exit(1)
    
    # Set model to evaluation mode
    model.eval()
    criterion = YoloLoss(ANCHORS, S, LAMBDA_LOC, LAMBDA_CONF)
    
    total_loss = 0.0
    inference_records = []
    
    print("Starting inference and loss calculation...")

    with torch.no_grad(): # Disable gradient calculation for faster inference
        for batch_images, batch_filenames, batch_targets in eval_loader:
            batch_images = batch_images.to(DEVICE)
            
            # 1. Forward Pass
            predictions = model(batch_images)
            
            # 2. Loss Calculation
            batch_targets_loss = batch_targets.permute(0, 2, 3, 4, 1).to(DEVICE)
            loss, loc_loss, conf_loss = criterion(predictions, batch_targets_loss)
            total_loss += loss.item() * batch_images.size(0) # Weight loss by batch size

            # 3. Prediction Decoding and NMS (Box Selection)
            decoded_boxes = decode_predictions(predictions, ANCHORS, S)
            
            # Iterate through individual images in the batch to apply NMS
            for i, filename in enumerate(batch_filenames):
                # NMS selects the single best box: [x_c, y_c, w, h, confidence]
                best_box = apply_nms(decoded_boxes[i].unsqueeze(0)) 
                
                # Append the result to records
                inference_records.append([filename] + best_box.tolist())

    # --- Final Output ---
    avg_loss = total_loss / len(dataset)
    print("\n--- EVALUATION RESULTS ---")
    print(f"Total Images Evaluated: {len(dataset)}")
    print(f"Average Total Loss (Lower is Better): {avg_loss:.4f}")
    print("--------------------------")
    
    # Save predictions to CSV
    df_pred = pd.DataFrame(inference_records, columns=['filename', 'x_center', 'y_center', 'width', 'height', 'confidence'])
    df_pred.to_csv(output_csv, index=False)
    
    print(f"✅ Predictions saved to: {output_csv}")

def run():
    # Ensure torchvision's NMS utility is accessible
    try:
        # Check for torchvision import needed for NMS (often pre-installed with PyTorch for CUDA)
        import torchvision 
    except ImportError:
        print("❌ FATAL ERROR: This script requires 'torchvision' for NMS. Please install it.")
        sys.exit(1)

    for key, value in INPUT_OUTPUT.items():
        input("press enter to continue: ")
        evaluate_model(key, value)
    


# --- Run the Script ---
if __name__ == "__main__":
    run()

# end testing model here.

# visualize function below.

input("enter nothing to visualize the results")

visualise_function()
    
# end
print("thank you for trying this code!")