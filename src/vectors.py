import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# CONFIGURATION FOR THE CUSTOM MODEL
# ---------------------------------------------------------
MODEL_PATH = "src/unirag_cnn.pth"
NUM_CLASSES = 3 # The number of classes we trained on (brain, breast, kidney)

# ---------------------------------------------------------
# REFACTORED IMAGE EMBEDDER (The "Specialist Brain")
# ---------------------------------------------------------
class ImageEmbedder:
    """
    Uses OUR FINE-TUNED ResNet50 to extract visual features.
    This is the "Specialist" brain that understands medical scans.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"🧠 Loading OUR FINE-TUNED ResNet50 Feature Extractor on {self.device}...")
        
        # 1. Load the model ARCHITECTURE
        self.model = models.resnet50(weights=None) # No pre-trained weights
        
        # 2. Re-create the final layer so the architecture matches
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        
        # 3. Load OUR fine-tuned weights
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}. Did src/train.py run successfully?")
            
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        
        # 4. Set to Eval Mode
        self.model.eval()
        self.model.to(self.device)
        
        # 5. Define the Preprocessing Pipeline (MUST be same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 6. --- THE CRITICAL HOOK ---
        # We don't want the final 3-class "prediction".
        # We want the 2048-feature vector *before* that layer.
        # We "hook" the 'avgpool' layer, which is right before the final 'fc' layer.
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])


    def vectorize(self, image_path: str) -> np.ndarray:
        """
        Reads an image path -> Preprocesses -> Passes through CNN -> Returns Vector.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                # Pass through the model *up to the avgpool layer*
                features = self.feature_extractor(img_tensor)
                
            # Flatten the (Batch, 2048, 1, 1) tensor to (2048,)
            embedding = features.cpu().numpy().flatten()
            return embedding
            
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
    print("--- Testing Vision (Fine-Tuned) ---")
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