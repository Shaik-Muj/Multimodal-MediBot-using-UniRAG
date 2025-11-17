import faiss
import pickle
import numpy as np
import os
import torch
from typing import List, Optional
from vectors import TextEmbedder, ImageEmbedder
from loaders import MedicalRecord # We re-use this data class

# ---------------------------------------------------------
# CONFIGURATION (Now loading all 6 files)
# ---------------------------------------------------------
STORE_DIR = "vector_store"
TEXT_INDEX_FILE = os.path.join(STORE_DIR, "text_knowledge.faiss")
TEXT_MAP_FILE = os.path.join(STORE_DIR, "text_map.pkl")
IMAGE_INDEX_FILE = os.path.join(STORE_DIR, "image_knowledge.faiss") # 2048-dim
IMAGE_MAP_FILE = os.path.join(STORE_DIR, "image_map.pkl")
IMAGE_TEXT_INDEX_FILE = os.path.join(STORE_DIR, "image_text.faiss") # 384-dim
IMAGE_TEXT_MAP_FILE = os.path.join(STORE_DIR, "image_text_map.pkl")

class UniRAGRetriever:
    """
    Handles loading all FAISS indexes and performing hybrid searches.
    """
    def __init__(self, device: str = "cuda"):
        print("🤖 Initializing UniRAG Hybrid Retriever...")
        
        # 1. Load Embedders
        self.text_embedder = TextEmbedder(device=device)
        self.image_embedder = ImageEmbedder(device=device)
        
        # 2. Load FAISS Indexes
        print("Loading Text Index (MedQuAD)...")
        self.text_index = faiss.read_index(TEXT_INDEX_FILE)
        
        print("Loading Image-Visual Index (ROCO/Cancer)...")
        self.image_index = faiss.read_index(IMAGE_INDEX_FILE)
        
        print("Loading Image-Text Index (Captions/Labels)...")
        self.image_text_index = faiss.read_index(IMAGE_TEXT_INDEX_FILE)
        
        # 3. Load Metadata Maps
        print("Loading Metadata Maps...")
        with open(TEXT_MAP_FILE, 'rb') as f:
            self.text_map = pickle.load(f)
        with open(IMAGE_MAP_FILE, 'rb') as f:
            self.image_map = pickle.load(f)
        # Note: image_text_map is identical to image_map, so we can just re-use it
        # to save memory.
            
        print("✅ Hybrid Retriever is Ready.")

    def search_text(self, query: str, k: int = 3) -> List[MedicalRecord]:
        """
        Searches the Text (MedQuAD) Knowledge Base. (No change here)
        """
        print(f"Searching text for: '{query}'...")
        query_vector = self.text_embedder.vectorize(query).astype('float32')
        query_vector = np.expand_dims(query_vector, axis=0) 
        
        distances, indices = self.text_index.search(query_vector, k)
        
        results = []
        for i in indices[0]:
            if i in self.text_map:
                results.append(self.text_map[i])
            
        return results

    def _fuse_results(self, search_lists: List[List[int]], k: int = 60) -> List[int]:
        """
        Fuses multiple search result lists using Reciprocal Rank Fusion (RRF).
        'k' is a constant, 60 is a common default.
        """
        scores = {}
        
        # Go through each search list (e.g., [visual_results, text_results])
        for rank_list in search_lists:
            # Go through each item's index (ID) in the list
            for rank, item_id in enumerate(rank_list):
                if item_id not in scores:
                    scores[item_id] = 0
                # Add the RRF score
                scores[item_id] += 1.0 / (k + rank)
                
        # Sort by the new fused score, highest first
        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        # Return just the item IDs
        return [item_id for item_id, score in sorted_results]

    def search_hybrid(self, image_path: str, text_query: Optional[str] = None, k: int = 5) -> List[MedicalRecord]:
        """
        Searches the Image Knowledge Base using both Visual and (optional) Text queries.
        """
        search_lists_to_fuse = []
        
        # --- 1. Visual Search (Always happens) ---
        print(f"Searching for similar images to: '{image_path}'...")
        img_vector = self.image_embedder.vectorize(image_path).astype('float32')
        img_vector = np.expand_dims(img_vector, axis=0)
        
        # We search for more (k*5) to give the fusion algorithm more to work with
        vis_distances, vis_indices = self.image_index.search(img_vector, k * 5)
        search_lists_to_fuse.append(vis_indices[0])

        # --- 2. Text Search (Optional) ---
        if text_query:
            print(f"Searching for images related to: '{text_query}'...")
            txt_vector = self.text_embedder.vectorize(text_query).astype('float32')
            txt_vector = np.expand_dims(txt_vector, axis=0)
            
            txt_distances, txt_indices = self.image_text_index.search(txt_vector, k * 5)
            search_lists_to_fuse.append(txt_indices[0])

        # --- 3. Fuse Results ---
        fused_indices = self._fuse_results(search_lists_to_fuse)
        
        # --- 4. Retrieve Top-K Final Results ---
        results = []
        for i in fused_indices[:k]: # Get the Top-K from the fused list
            if i in self.image_map:
                results.append(self.image_map[i])
                
        return results

# ---------------------------------------------------------
# SCRIPT ENTRY POINT (Test the new hybrid search)
# ---------------------------------------------------------
if __name__ == "__main__":
    import torch
    
    retriever = UniRAGRetriever(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Test 1: Text Search (No change) ---
    print("\n--- Testing Text Search (MedQuAD) ---")
    text_results = retriever.search_text("What is pneumonia?")
    for res in text_results:
        print(f"  Found: {res.content[:70]}...") 
        
    # --- Test 2: Hybrid Image Search (The New Test) ---
    # IMPORTANT: Change this path to one of your actual test images!
    TEST_IMAGE_PATH = "C:/Users/mesmh/OneDrive/Desktop/images.jpg" # <-- CHANGE ME
    
    if os.path.exists(TEST_IMAGE_PATH):
        print(f"\n--- Testing HYBRID Search ---")
        # Here we provide both the image and a text hint
        hybrid_results = retriever.search_hybrid(
            image_path=TEST_IMAGE_PATH,
            text_query="brain scan mri glioma tumor", # <-- The optional text hint
            k=3
        )
        for res in hybrid_results:
            print(f"  Found Match: {res.metadata['type']} ({res.image_path})")
            print(f"     Caption: {res.content[:70]}...")
    else:
        print(f"\nSkipping image test. Put a real path in TEST_IMAGE_PATH to run.")