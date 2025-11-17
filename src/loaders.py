import os
import pandas as pd
from typing import List, Optional, Dict

# ---------------------------------------------------------
# 1. THE UNIFIED DATA OBJECT (No Change)
# ---------------------------------------------------------
class MedicalRecord:
    """
    The atom of our system. 
    Every text or image we load becomes a 'MedicalRecord'.
    """
    def __init__(self, content: str, metadata: Dict, image_path: Optional[str] = None):
        self.content = content      # Text info (Q&A or Diagnosis Label)
        self.metadata = metadata    # {'source': 'medquad', 'type': 'brain_tumor', ...}
        self.image_path = image_path # Path to .jpg (if applicable)

    def __repr__(self):
        return f"<MedicalRecord source={self.metadata.get('source')} content={self.content[:30]}...>"


# ---------------------------------------------------------
# 2. THE TEXT LOADER (MedQuAD) (No Change)
# ---------------------------------------------------------
class MedQuADLoader:
    """
    Ingests the Text Knowledge Base.
    Strategy: 1 Row = 1 Fact.
    """
    def load(self, csv_path: str) -> List[MedicalRecord]:
        records = []
        print(f"📚 Loading MedQuAD text from {csv_path}...")
        
        try:
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                # Use lowercase to match CSV headers
                q = row.get('question', '')
                a = row.get('answer', '')
                
                if pd.notna(q) and pd.notna(a):
                    full_text = f"Question: {q}\nAnswer: {a}"
                    
                    records.append(MedicalRecord(
                        content=full_text,
                        metadata={"source": "medquad", "type": "text_knowledge"}
                    ))
            print(f"✅ Loaded {len(records)} text records.")
            
        except Exception as e:
            print(f"❌ Error loading MedQuAD: {e}")
            
        return records


# ---------------------------------------------------------
# 3. THE CAPTION LOADER (ROCO) (No Change)
# ---------------------------------------------------------
class ROCOLoader:
    """
    Ingests Radiology Images + Captions.
    Strategy: Link 'name' in CSV to the file in the folder.
    """
    def load(self, csv_path: str, images_dir: str) -> List[MedicalRecord]:
        records = []
        print(f"☢️ Loading ROCO radiology samples from {images_dir}...")

        try:
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                img_filename = row.get('name') # Use the 'name' column directly
                caption = row.get('caption')
                
                if not img_filename or not caption:
                    continue # Skip rows with missing data
                
                full_path = os.path.join(images_dir, img_filename)
                
                if os.path.exists(full_path):
                    records.append(MedicalRecord(
                        content=f"Radiology Caption: {caption}",
                        image_path=full_path,
                        metadata={"source": "roco", "type": "radiology"}
                    ))
                    
            print(f"✅ Loaded {len(records)} ROCO image records.")
            
        except Exception as e:
            print(f"❌ Error loading ROCO: {e}")
            
        return records


# ---------------------------------------------------------
# 4. THE CLASSIFICATION LOADER (Multi-Cancer) (Refactored)
# ---------------------------------------------------------
class MultiCancerLoader:
    """
    Ingests images from labeled folders.
    Strategy: Folder Name (e.g., 'brain_tumor') = The Diagnosis Label.
    """
    def load(self, root_dir: str) -> List[MedicalRecord]:
        records = []
        print(f"🔬 Loading Multi-Cancer samples from {root_dir}...")

        for label_folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, label_folder)
            
            if os.path.isdir(folder_path):
                # The folder name IS the diagnosis
                diagnosis = label_folder 
                
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(folder_path, img_file)
                        
                        # --- REFACTORED ---
                        # Removed the "heuristic" caption.
                        # The content is now just the diagnosis label.
                        # We will retrieve based on visual similarity and use the 
                        # metadata to display the diagnosis in the UI.
                        records.append(MedicalRecord(
                            content=diagnosis,
                            image_path=full_path,
                            metadata={"source": "multicancer", "type": diagnosis}
                        ))

        print(f"✅ Loaded {len(records)} cancer reference images.")
        return records