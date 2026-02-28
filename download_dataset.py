import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from dataset import create_dataloader 

def main():
    output_dir = "./data/local_celeba"
    img_dir = os.path.join(output_dir, "real_images")
    os.makedirs(img_dir, exist_ok=True)

    print("Loading dataset from Hugging Face...")
    # Use shuffle=False to ensure the images and attributes align perfectly in order
    loader = create_dataloader(
        root="./data/hf_cache",
        split="train",
        image_size=64,
        batch_size=500,
        augment=False,
        shuffle=False, 
        from_hub=True,
        repo_name="electronickale/cmu-10799-celeba64-subset"
    )

    all_attributes = []
    img_count = 0

    print(f"Saving images to {img_dir}...")
    for batch_x, batch_y in tqdm(loader):
        # Unnormalize [-1, 1] to [0, 1] for saving
        samples = torch.clamp((batch_x + 1.0) / 2.0, 0.0, 1.0)
        
        for i in range(samples.size(0)):
            save_image(samples[i], os.path.join(img_dir, f"{img_count:06d}.png"))
            img_count += 1
        
        all_attributes.append(batch_y.cpu())

    # Concatenate and save all attributes for the generator to use
    all_attributes_tensor = torch.cat(all_attributes, dim=0)
    torch.save(all_attributes_tensor, os.path.join(output_dir, "attributes.pt"))
    
    print("\nExtraction Complete!")
    print(f"Saved {img_count} real images for FID.")
    print(f"Saved attributes tensor of shape {all_attributes_tensor.shape}.")

if __name__ == "__main__":
    main()