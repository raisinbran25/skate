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