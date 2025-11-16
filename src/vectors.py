import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

class ImageEmbedder:
    """
    Uses a pre-trained ResNet50 to extract visual features from medical images.
    This acts as the 'Eyes' of the RAG system.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"🧠 Initializing ResNet50 Feature Extractor on {self.device}...")
        
        # 1. Load Pre-trained Model (The "Knowledge")
        # weights='DEFAULT' loads the best available ImageNet weights
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 2. Surgical Removal of the Classification Head
        # We don't want the final layer that says "Cat" or "Dog".
        # We want the layer *before* that (the features).
        # ResNet50's last layer is 'fc'. We replace it with an Identity layer.
        self.model.fc = nn.Identity()
        
        # 3. Set to Eval Mode
        # Crucial for inference! Turns off Dropout/BatchNorm updates.
        self.model.eval()
        self.model.to(self.device)
        
        # 4. Define the Preprocessing Pipeline (Standard ImageNet stats)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),      # Standard ResNet input size
            transforms.ToTensor(),              # Convert PIL Image to Tensor
            transforms.Normalize(               # Normalize based on ImageNet mean/std
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])



    def vectorize(self, image_path: str) -> np.ndarray:
        """
        Reads an image path -> Preprocesses -> Passes through CNN -> Returns Vector.
        """
        try:
            # Open image (ensure RGB, X-rays are often grayscale)
            img = Image.open(image_path).convert('RGB')
            
            # Apply transforms -> Add batch dimension (1, 3, 224, 224)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Inference (No Gradients needed = Faster)
            with torch.no_grad():
                embedding = self.model(img_tensor)
                
            # Convert tensor to flat numpy array (Size: 2048)
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"⚠️ Error vectorizing {image_path}: {e}")
            # Return zero vector of size 2048 as fallback (avoids crashing)
            return np.zeros(2048)

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# TEXT EMBEDDER (The "Language Brain")
# ---------------------------------------------------------
class TextEmbedder:
    """
    Uses a HuggingFace Transformer to convert medical text (Q&A) into vectors.
    We use 'all-MiniLM-L6-v2' - the industry standard for speed/accuracy balance.
    """
    def __init__(self, device: str = None):
        # Auto-detect GPU if not manually specified
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"📚 Initializing Text Embedder (MiniLM) on {device}...")
        
        # Load the pre-trained model from HuggingFace
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    def vectorize(self, text: str) -> np.ndarray:
        """
        Takes a string -> Returns a 384-dimensional vector.
        """
        try:
            # sentence-transformers handles tokenization and pooling automatically
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"⚠️ Error vectorizing text: {e}")
            # Return zero vector of size 384 (MiniLM output size)
            return np.zeros(384)

# ---------------------------------------------------------
# MAIN BLOCK (Test Both)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Test Image Embedder
    print("--- Testing Vision ---")
    img_embedder = ImageEmbedder()
    dummy_img = Image.new('RGB', (100, 100), color = 'red')
    dummy_img.save("test.jpg")
    v_img = img_embedder.vectorize("test.jpg")
    print(f"Image Vector Shape: {v_img.shape}") # Should be (2048,)
    os.remove("test.jpg")

    # 2. Test Text Embedder
    print("\n--- Testing Language ---")
    txt_embedder = TextEmbedder()
    v_txt = txt_embedder.vectorize("Patient has severe chest pain.")
    print(f"Text Vector Shape: {v_txt.shape}") # Should be (384,)
    print(f"First 5 values: {v_txt[:5]}")