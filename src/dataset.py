import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class MultiCancerDataset(Dataset):
    """
    A custom PyTorch Dataset to load the Multi-Cancer images for fine-tuning.
    It automatically discovers classes from folder names.
    """
    def __init__(self, root_dir: str, transform: transforms.Compose = None):
        self.root_dir = root_dir
        
        # 1. Define the image transformations
        # We use the standard ResNet50/ImageNet transforms
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
        # 2. Discover classes (e.g., 'brain_tumor', 'breast_malignant')
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 3. Create a list of all samples (image_path, label_index)
        self.samples = []
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(root_dir, class_name)
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
                    
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes.")
        print(f"Classes: {self.class_to_idx}")

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Fetches one sample (image + label) at the given index.
        """
        # 1. Get the path and label
        img_path, label = self.samples[idx]
        
        # 2. Load the image
        # .convert('RGB') is crucial for grayscale X-rays
        image = Image.open(img_path).convert('RGB')
        
        # 3. Apply transformations
        image = self.transform(image)
        
        return image, label

# ---------------------------------------------------------
# SCRIPT ENTRY POINT (Test the dataset loader)
# ---------------------------------------------------------
if __name__ == "__main__":
    # IMPORTANT: Point this to your actual multicancer data folder
    DATA_DIR = "data/multicancer" 
    
    if os.path.exists(DATA_DIR):
        print("Testing MultiCancerDataset...")
        dataset = MultiCancerDataset(root_dir=DATA_DIR)
        
        # Check if it loaded correctly
        print(f"Total samples: {len(dataset)}")
        
        # Try to get one item
        if len(dataset) > 0:
            image_tensor, label_index = dataset[0]
            
            # Find the class name from the index
            label_name = dataset.classes[label_index]
            
            print(f"\n--- Sample 0 ---")
            print(f"Image Tensor Shape: {image_tensor.shape}") # Should be [3, 224, 224]
            print(f"Label Index: {label_index}")
            print(f"Label Name: {label_name}")
    else:
        print(f"Test skipped. Data directory not found: {DATA_DIR}")