import torch
import torch.nn as nn
import argparse
import os
import glob
import json
from PIL import Image
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision import models, transforms
from huggingface_hub import PyTorchModelHubMixin
from sklearn.metrics import accuracy_score

# 1. Drop in your exact custom model class
class CelebAResNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 40)

    def forward(self, x):
        return self.resnet(x)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def evaluate_fid(real_dir, fake_dir, device):
    print(f"\n--- Calculating FID on {device.upper()} ---")
    
    fid_device = "cpu" if device == "mps" else device
    
    fid_value = calculate_fid_given_paths(
        paths=[real_dir, fake_dir],
        batch_size=50,
        device=fid_device,
        dims=2048 
    )
    print(f"FID Score: {fid_value:.4f}")
    return float(fid_value)

def evaluate_accuracy(fake_dir, hf_classifier_path, device):
    print(f"\n--- Calculating y-Accuracy using {hf_classifier_path} ---")
    
    conditions_path = os.path.join(fake_dir, "conditions.pt")
    parent_dir = os.path.dirname(os.path.normpath(fake_dir))
    alt_conditions_path = os.path.join(parent_dir, "conditions.pt")
    
    if os.path.exists(conditions_path):
        target_path = conditions_path
    elif os.path.exists(alt_conditions_path):
        target_path = alt_conditions_path
    else:
        print("No conditions.pt found! Skipping accuracy calculation (likely unconditional).")
        return None, None
        
    y_true = torch.load(target_path, map_location=device)
    
    # 2. Load your model using the Mixin's from_pretrained method
    model = CelebAResNet.from_pretrained(hf_classifier_path).to(device)
    model.eval()

    # 3. Use the exact transforms from your training script
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_paths = sorted(glob.glob(os.path.join(fake_dir, "*.png")))
    if len(image_paths) != len(y_true):
        print(f"Mismatch Error: Found {len(image_paths)} images but {len(y_true)} condition vectors.")
        return None, None

    batch_size = 64
    all_preds = []

    print("Running generated images through the classifier...")
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            
            batch_tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(img_transform(img))
            
            pixel_values = torch.stack(batch_tensors).to(device)
            
            logits = model(pixel_values)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.append(preds)

    y_pred = torch.cat(all_preds, dim=0)

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    overall_acc = accuracy_score(y_true_np, y_pred_np) 
    elementwise_acc = (y_true_np == y_pred_np).mean()  
    
    print("\n--- Accuracy Results ---")
    print(f"Exact Match Accuracy (All 40 correct): {overall_acc * 100:.2f}%")
    print(f"Element-wise Attribute Accuracy:     {elementwise_acc * 100:.2f}%")
    
    return float(overall_acc), float(elementwise_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake-dir", type=str, required=True, help="Directory containing generated pngs")
    parser.add_argument("--real-dir", type=str, required=True, help="Path to your local CelebA 'real_images' folder")
    parser.add_argument("--classifier", type=str, required=True, help="Hugging Face repo id for your ResNet")
    args = parser.parse_args()
    
    device = get_device()
    
    fid_score = evaluate_fid(args.real_dir, args.fake_dir, device)
    overall_acc, elementwise_acc = evaluate_accuracy(args.fake_dir, args.classifier, device)
    
    results = {
        "fid_score": fid_score
    }
    
    if overall_acc is not None:
        results["exact_match_accuracy"] = overall_acc
        results["elementwise_accuracy"] = elementwise_acc
        
    parent_dir = os.path.dirname(os.path.normpath(args.fake_dir))
    log_path = os.path.join(parent_dir, "evaluation_results.json")
    
    with open(log_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nMetrics successfully saved to: {log_path}")