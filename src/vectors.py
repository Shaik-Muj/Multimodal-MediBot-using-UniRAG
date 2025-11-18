import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# --- Removed MODEL_PATH and NUM_CLASSES constants ---

# ---------------------------------------------------------
# REFACTORED IMAGE EMBEDDER (The "Generalist Brain")
# ---------------------------------------------------------
class ImageEmbedder:
    """
    Uses a GENERIC, pre-trained ResNet50 to extract visual features.
    This is a "Generalist" brain.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"🧠 Loading GENERIC ResNet50 Feature Extractor on {self.device}...")
        
        # 1. Load the model ARCHITECTURE with pre-trained ImageNet weights
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 2. --- THE CRITICAL HOOK ---
        # We replace the final 1000-class layer with an "Identity" layer.
        # This makes the model output the 2048-feature vector directly.
        self.model.fc = nn.Identity()
        
        # 3. Set to Eval Mode
        self.model.eval()
        self.model.to(self.device)
        
        # 4. Define the Preprocessing Pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def vectorize(self, image_path: str) -> np.ndarray:
        """
        Reads an image path -> Preprocesses -> Passes through CNN -> Returns Vector.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                # The model now directly outputs the (1, 2048) feature vector
                embedding = self.model(img_tensor)
                
            # Flatten to (2048,)
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"⚠️ Error vectorizing {image_path}: {e}")
            return np.zeros(2048)

# ---------------------------------------------------------
# TEXT EMBEDDER (No Change)
# ---------------------------------------------------------
class TextEmbedder:
    """
    Uses a HuggingFace Transformer to convert medical text (Q&A) into vectors.
    """
    def __init__(self, device: str = None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📚 Initializing Text Embedder (MiniLM) on {device}...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    def vectorize(self, text: str) -> np.ndarray:
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"⚠️ Error vectorizing text: {e}")
            return np.zeros(384)

# ---------------------------------------------------------
# UPDATED MAIN BLOCK (Test Both)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Test Image Embedder
    print("--- Testing Vision (Generic) ---")
    img_embedder = ImageEmbedder(device="cuda" if torch.cuda.is_available() else "cpu")
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