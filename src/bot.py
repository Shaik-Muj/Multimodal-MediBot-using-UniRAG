import os
import torch
from typing import List, Dict, Tuple, Optional
from retriever import UniRAGRetriever
from loaders import MedicalRecord
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import gc

# --- OPTIMIZATION 1: Enable TF32 for Ampere GPUs ---
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------
# 1. THE "GENERATION" BRAIN (Streaming Edition)
# ---------------------------------------------------------
class LLMGenerator:
    def __init__(self, device: str = "cuda"):
        print("💬 Initializing Conversational LLM (Streaming Mode)...")
        
        # Use Qwen 0.5B (Tiny but smart)
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print(f"⏳ Loading {model_id} on {device}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )
            print(f"✅ Model loaded successfully on: {self.model.device}")
        except Exception as e:
            print(f"⚠️ GPU loading failed ({e}). Falling back to CPU.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )

    def _build_prompt(self, query: str, context: str, history: List[Tuple[str, str]]) -> str:
        messages = []
        
        system_msg = (
            "You are an empathetic, expert Medical AI Assistant. "
            "Your goal is to explain medical imaging findings to a user in plain English, based on similar cases found in a reference database.\n\n"
            "GUIDELINES:\n"
            "1. **Tone:** Be professional, calm, and conversational. Do not sound like a robot.\n"
            "2. **Synthesis:** Weave matches into a story. Example: 'This scan shows a mass in the kidney. In similar cases, features like [Feature A] often point towards [Condition].'\n"
            "3. **Safety First:** Never give a definitive diagnosis (e.g., 'You have cancer'). Use hedging language: 'This appearance is often associated with...' or 'This warrants clinical investigation for...'\n"
            "4. **Anatomy Check:** If the user says 'Kidney' but the data discusses 'Lung', ignore the irrelevant data.\n"
            "5. **Actionable Advice:** End with what a doctor typically does next (e.g., biopsy, specialist referral).\n\n"
            "--- RETRIEVED SIMILAR CASES ---\n"
            f"{context}\n"
            "--- END CASES ---"
        )
        messages.append({"role": "system", "content": system_msg})
        
        recent_history = history[-6:] 
        
        for role, message in recent_history:
            qwen_role = "assistant" if role == "bot" else "user"
            messages.append({"role": qwen_role, "content": message})
            
        messages.append({"role": "user", "content": query})
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt
        
    def generate_stream(self, query: str, context: str, history: List[Tuple[str, str]]):
        """
        Yields tokens one by one for the typing animation.
        """
        prompt_str = self._build_prompt(query, context, history)
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            inputs, 
            streamer=streamer, 
            max_new_tokens=500, 
            do_sample=True, 
            temperature=0.3,
            use_cache=False
        )
        
        # Run generation in a separate thread so we can yield tokens immediately
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text

# ---------------------------------------------------------
# 2. THE BOT (Refactored for Streaming)
# ---------------------------------------------------------
class MediRAGBot:
    def __init__(self, device: str = "cuda"):
        self.retriever = UniRAGRetriever(device=device)
        self.generator = LLMGenerator(device=device)
        self.chat_history: List[Tuple[str, str]] = []
        print("\n--- 🩺 MediRAG Agent is Online ---")

    def _format_context(self, records: List[MedicalRecord], is_image_query: bool = False) -> str:
        if not records:
            return "No relevant information found."
        
        context_str = ""
        for i, record in enumerate(records):
            content = record.content
            if len(content) > 500:
                content = content[:500] + "... (truncated)"
            
            if is_image_query:
                context_str += f"Fact {i+1}: A visual match was found.\n"
                context_str += f"  - Diagnosis/Label: {record.metadata.get('type', 'N/A')}\n"
                context_str += f"  - Source: {record.metadata.get('source', 'N/A')}\n"
                if record.metadata.get('source') == 'roco':
                    context_str += f"  - Caption: {content}\n"
            else:
                context_str += f"Fact {i+1}: {content}\n"
        return context_str

    def text_query_stream(self, query: str):
        """
        Generator function that yields text chunks.
        """
        print(f"\n[User Query]: {query}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # 1. Retrieve
        retrieved_records = self.retriever.search_text(query, k=3)
        context = self._format_context(retrieved_records, is_image_query=False)
        
        # 2. Generate (Stream)
        full_response = ""
        for chunk in self.generator.generate_stream(query, context, self.chat_history):
            full_response += chunk
            yield chunk
        
        # 3. Update Memory (Only after full generation)
        self.chat_history.append(("user", query))
        self.chat_history.append(("bot", full_response))
        
    def image_query_stream(self, image_path: str, text_hint: Optional[str] = None):
        """
        Generator function that yields text chunks for image queries.
        """
        print(f"\n[User Image Query]: {image_path}")
        
        retrieved_records = self.retriever.search_hybrid(
            image_path=image_path,
            text_query=text_hint,
            k=3
        )
        
        context = self._format_context(retrieved_records, is_image_query=True)
        
        if text_hint:
            summary_query = f"I've uploaded an image description '{text_hint}'. The system found these matches. Summarize the findings."
        else:
            summary_query = f"I've uploaded an image. The system found these matches. Summarize the findings."

        full_response = ""
        for chunk in self.generator.generate_stream(summary_query, context, self.chat_history):
            full_response += chunk
            yield chunk

        log_message = f"[User uploaded image: {os.path.basename(image_path)}]"
        if text_hint:
            log_message += f" [With hint: {text_hint}]"
        self.chat_history.append(("user", log_message))
        self.chat_history.append(("bot", full_response))

# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    bot = MediRAGBot()
    print("Bot ready. Run via Streamlit.")