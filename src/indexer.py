import os
import faiss
import numpy as np
import pickle
from tqdm import tqdm
# --- MultiCancerLoader is GONE ---
from loaders import MedQuADLoader, ROCOLoader
from vectors import TextEmbedder, ImageEmbedder
import torch 

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Define our data paths
MEDQUAD_PATH = "data/medquad/medquad.csv"
ROCO_CSV_PATH = "data/roco/radiologytraindata.csv"
ROCO_IMG_DIR = "data/roco/images"
# --- MULTICANCER_DIR is GONE ---

# Define where the final "Knowledge Base" will be saved
STORE_DIR = "vector_store"
TEXT_INDEX_FILE = os.path.join(STORE_DIR, "text_knowledge.faiss")
TEXT_MAP_FILE = os.path.join(STORE_DIR, "text_map.pkl")
IMAGE_INDEX_FILE = os.path.join(STORE_DIR, "image_knowledge.faiss")
IMAGE_MAP_FILE = os.path.join(STORE_DIR, "image_map.pkl")
IMAGE_TEXT_INDEX_FILE = os.path.join(STORE_DIR, "image_text.faiss") 
IMAGE_TEXT_MAP_FILE = os.path.join(STORE_DIR, "image_text_map.pkl") 

# ---------------------------------------------------------
# THE INDEXER CLASS
# ---------------------------------------------------------
class VectorIndexer:
    def __init__(self, device: str = "cuda"):
        # Initialize our processing tools
        self.text_embedder = TextEmbedder(device=device)
        self.image_embedder = ImageEmbedder(device=device)
        self.loaders = {
            "medquad": MedQuADLoader(),
            "roco": ROCOLoader(),
            # --- "multicancer" is GONE ---
        }
        
        os.makedirs(STORE_DIR, exist_ok=True)

    def _create_and_save_index(self, vectors: np.ndarray, records: list, index_path: str, map_path: str):
        """
        Helper function to create, populate, and save a FAISS index and its metadata map.
        (No changes to this function)
        """
        if vectors.size == 0 or len(records) == 0:
            print(f"No vectors found for {index_path}, skipping.")
            return

        print(f"Building {index_path}...")
        
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        
        print(f"Saving Index to {index_path}...")
        faiss.write_index(index, index_path)
        
        print(f"Saving Metadata Map to {map_path}...")
        metadata_map = {i: r for i, r in enumerate(records)}
        with open(map_path, 'wb') as f:
            pickle.dump(metadata_map, f)

    def build_indexes(self):
        """
        Main function: Loads all data, vectorizes it (on GPU), 
        and saves THREE separate FAISS indexes (on CPU).
        """
        print("--- Starting Phase 2: Building Knowledge Base ---")
        
        # 1. Load all data sources
        medquad_records = self.loaders["medquad"].load(MEDQUAD_PATH)
        roco_records = self.loaders["roco"].load(ROCO_CSV_PATH, ROCO_IMG_DIR)
        # --- cancer_records is GONE ---
        
        # 2. Process and Index Text (MedQuAD)
        if medquad_records:
            print("\nVectorizing Text (MedQuAD)...")
            text_vectors = np.array([self.text_embedder.vectorize(r.content) for r in tqdm(medquad_records)])
            self._create_and_save_index(text_vectors, medquad_records, TEXT_INDEX_FILE, TEXT_MAP_FILE)
        else:
            print("\nNo MedQuAD records found. Skipping text index.")
        
        # --- Combined Image Data (Now only ROCO) ---
        image_records = roco_records
        
        # 3. Process and Index *Images* (ROCO only)
        if image_records:
            print("\nVectorizing *Images* (ROCO only)...")
            image_vectors = np.array([self.image_embedder.vectorize(r.image_path) for r in tqdm(image_records)])
            self._create_and_save_index(image_vectors, image_records, IMAGE_INDEX_FILE, IMAGE_MAP_FILE)
        else:
            print("\nNo image records found. Skipping image index.")
            
        # 4. Process and Index *Image Text* (ROCO Captions only)
        if image_records:
            print("\nVectorizing *Image Text* (ROCO Captions only)...")
            image_text_vectors = np.array([self.text_embedder.vectorize(r.content) for r in tqdm(image_records)])
            self._create_and_save_index(image_text_vectors, image_records, IMAGE_TEXT_INDEX_FILE, IMAGE_TEXT_MAP_FILE)
        else:
            print("\nNo image records found. Skipping image text index.")
                
        print("\n--- ✅ Knowledge Base Build Complete ---")

# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    # Detect GPU for Embedders, but FAISS will remain on CPU
    indexer = VectorIndexer(device="cuda" if torch.cuda.is_available() else "cpu")
    indexer.build_indexes()