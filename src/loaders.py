import os
import pandas as pd
from typing import List, Optional, Dict

# ---------------------------------------------------------
# 1. THE UNIFIED DATA OBJECT
# ---------------------------------------------------------
class MedicalRecord:
    """
    The atom of our system. 
    Every text or image we load becomes a 'MedicalRecord'.
    """
    def __init__(self, content: str, metadata: Dict, image_path: Optional[str] = None):
        self.content = content      # Text info (Q&A or Image Caption)
        self.metadata = metadata    # {'source': 'medquad', 'type': 'radiology', ...}
        self.image_path = image_path # Path to .jpg (if applicable)

    def __repr__(self):
        return f"<MedicalRecord source={self.metadata.get('source')} content={self.content[:30]}...>"


# ---------------------------------------------------------
# 2. THE TEXT LOADER (MedQuAD)
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
# 3. THE CAPTION LOADER (ROCO)
# ---------------------------------------------------------
class ROCOLoader:
    """
    Ingests Radiology Images + Captions.
    This is now our ONLY image source.
    """
    def load(self, csv_path: str, images_dir: str) -> List[MedicalRecord]:
        records = []
        print(f"☢️ Loading ROCO radiology samples from {images_dir}...")

        try:
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                img_filename = row.get('name') 
                caption = row.get('caption')
                
                if not img_filename or not caption:
                    continue 
                
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