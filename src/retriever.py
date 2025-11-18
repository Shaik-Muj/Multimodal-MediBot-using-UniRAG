import faiss
import pickle
import numpy as np
import os
import torch
from typing import List, Optional
from vectors import TextEmbedder, ImageEmbedder
from loaders import MedicalRecord

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
STORE_DIR = "vector_store"
TEXT_INDEX_FILE = os.path.join(STORE_DIR, "text_knowledge.faiss")
TEXT_MAP_FILE = os.path.join(STORE_DIR, "text_map.pkl")
IMAGE_INDEX_FILE = os.path.join(STORE_DIR, "image_knowledge.faiss")
IMAGE_MAP_FILE = os.path.join(STORE_DIR, "image_map.pkl")
IMAGE_TEXT_INDEX_FILE = os.path.join(STORE_DIR, "image_text.faiss")
IMAGE_TEXT_MAP_FILE = os.path.join(STORE_DIR, "image_text_map.pkl")

class UniRAGRetriever:
    def __init__(self, device: str = "cuda"):
        print("🤖 Initializing UniRAG Hybrid Retriever...")
        
        self.text_embedder = TextEmbedder(device=device)
        self.image_embedder = ImageEmbedder(device=device)
        
        print("Loading Text Index (MedQuAD)...")
        self.text_index = faiss.read_index(TEXT_INDEX_FILE)
        
        print("Loading Image-Visual Index (ROCO/Cancer)...")
        self.image_index = faiss.read_index(IMAGE_INDEX_FILE)
        
        print("Loading Image-Text Index (Captions/Labels)...")
        self.image_text_index = faiss.read_index(IMAGE_TEXT_INDEX_FILE)
        
        print("Loading Metadata Maps...")
        with open(TEXT_MAP_FILE, 'rb') as f:
            self.text_map = pickle.load(f)
        with open(IMAGE_MAP_FILE, 'rb') as f:
            self.image_map = pickle.load(f)
            
        print("✅ Hybrid Retriever is Ready.")

    def search_text(self, query: str, k: int = 3) -> List[MedicalRecord]:
        print(f"Searching text for: '{query}'...")
        query_vector = self.text_embedder.vectorize(query).astype('float32')
        query_vector = np.expand_dims(query_vector, axis=0) 
        
        distances, indices = self.text_index.search(query_vector, k)
        
        results = []
        for i in indices[0]:
            if i in self.text_map:
                results.append(self.text_map[i])
            
        return results

    def _fuse_results(self, visual_indices: List[int], text_indices: List[int], k: int = 60) -> List[int]:
        """
        Fuses results using WEIGHTED Reciprocal Rank Fusion (Weighted RRF).
        We give Text Search 3x more weight than Visual Search to prevent hallucinations.
        """
        scores = {}
        
        # 1. Weight Configuration
        VISUAL_WEIGHT = 1.0
        TEXT_WEIGHT = 3.0  # Text is 3x more important
        
        # 2. Score Visual Results
        for rank, item_id in enumerate(visual_indices):
            if item_id not in scores:
                scores[item_id] = 0
            scores[item_id] += (VISUAL_WEIGHT / (k + rank))
            
        # 3. Score Text Results
        for rank, item_id in enumerate(text_indices):
            if item_id not in scores:
                scores[item_id] = 0
            scores[item_id] += (TEXT_WEIGHT / (k + rank))
                
        # 4. Sort
        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        return [item_id for item_id, score in sorted_results]

    def search_hybrid(self, image_path: str, text_query: Optional[str] = None, k: int = 5) -> List[MedicalRecord]:
        print(f"Searching for similar images to: '{image_path}'...")
        
        # 1. Visual Search
        img_vector = self.image_embedder.vectorize(image_path).astype('float32')
        img_vector = np.expand_dims(img_vector, axis=0)
        vis_distances, vis_indices = self.image_index.search(img_vector, k * 10) 
        visual_list = vis_indices[0].tolist() # Convert to list

        # 2. Text Search
        text_list = []
        if text_query:
            print(f"Searching for images related to: '{text_query}'...")
            txt_vector = self.text_embedder.vectorize(text_query).astype('float32')
            txt_vector = np.expand_dims(txt_vector, axis=0)
            txt_distances, txt_indices = self.image_text_index.search(txt_vector, k * 10)
            text_list = txt_indices[0].tolist() # Convert to list

        # 3. Fuse (Passing separate lists now)
        fused_indices = self._fuse_results(visual_list, text_list)
        
        # 4. Retrieve Results
        results = []
        for i in fused_indices[:k]:
            if i in self.image_map:
                results.append(self.image_map[i])
                
        return results

if __name__ == "__main__":
    # Test block
    retriever = UniRAGRetriever(device="cuda" if torch.cuda.is_available() else "cpu")
    # TEST_IMAGE_PATH = "data/multicancer/brain_tumor/brain_tumor_0149.jpg"
    # if os.path.exists(TEST_IMAGE_PATH):
    #    results = retriever.search_hybrid(TEST_IMAGE_PATH, "brain tumor mri", k=3)
    #    for r in results:
    #        print(r.content)