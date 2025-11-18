import os
import torch
from typing import List, Dict, Tuple, Optional
from retriever import UniRAGRetriever
from loaders import MedicalRecord
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- OPTIMIZATION 1: Enable TF32 for Ampere GPUs (RTX 30xx series) ---
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------
# 1. THE "GENERATION" BRAIN (Turbo Edition)
# ---------------------------------------------------------
class LLMGenerator:
    def __init__(self, device: str = "cuda"):
        print("💬 Initializing Conversational LLM (Turbo Mode)...")
        
        # --- OPTIMIZATION 2: Use the 0.5B Model ---
        # This model is 3x smaller than the 1.5B version.
        # It fits easily in VRAM and generates text much faster.
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

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=500,
        )

    def _build_prompt(self, query: str, context: str, history: List[Tuple[str, str]]) -> str:
        messages = []
        
        system_msg = (
            "You are a helpful medical AI assistant. "
            "Synthesize the provided facts to answer the user's question. "
            "Be concise. If the answer isn't in the facts, say so.\n\n"
            "--- RELEVANT FACTS ---\n"
            f"{context}\n"
            "--- END FACTS ---"
        )
        messages.append({"role": "system", "content": system_msg})
        
        # --- OPTIMIZATION 3: Truncate History ---
        # Only keep the last 3 turns (6 messages) to keep the prompt short and fast.
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
        
    def generate(self, query: str, context: str, history: List[Tuple[str, str]]) -> str:
        prompt = self._build_prompt(query, context, history)
        
        generation_args = {
            "return_full_text": False,
            "temperature": 0.3,
            "do_sample": True,
            "use_cache": False,
        }
        
        try:
            output = self.pipeline(prompt, **generation_args)
            response = output[0]['generated_text'].strip()
        except Exception as e:
            print(f"LLM Generation failed: {e}")
            response = "I'm sorry, I encountered an error. Please try again."
            
        return response

# ---------------------------------------------------------
# 2. THE BOT (No major logic changes)
# ---------------------------------------------------------
class UniRAGBot:
    def __init__(self, device: str = "cuda"):
        self.retriever = UniRAGRetriever(device=device)
        self.generator = LLMGenerator(device=device)
        self.chat_history: List[Tuple[str, str]] = []
        print("\n--- 🤖 UniRAG Conversational Agent is Online ---")

    def _format_context(self, records: List[MedicalRecord], is_image_query: bool = False) -> str:
        """
        Formats retrieved records, truncating them to prevent context overflow.
        """
        if not records:
            return "No relevant information found."
        
        context_str = ""
        for i, record in enumerate(records):
            # --- OPTIMIZATION: TRUNCATE CONTENT ---
            # Limit each fact to 500 characters. 
            # This keeps the prompt small and the model fast.
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

    def text_query(self, query: str) -> str:
        print(f"\n[User Query]: {query}")
        
        retrieved_records = self.retriever.search_text(query, k=3)
        context = self._format_context(retrieved_records, is_image_query=False)
        
        bot_response = self.generator.generate(
            query=query, 
            context=context, 
            history=self.chat_history 
        )
        
        self.chat_history.append(("user", query))
        self.chat_history.append(("bot", bot_response))
        
        print(f"[Bot Response]:\n{bot_response}")
        return bot_response

    def image_query(self, image_path: str, text_hint: Optional[str] = None) -> str:
        print(f"\n[User Image Query]: {image_path}")
        if text_hint:
            print(f"[User Text Hint]: {text_hint}")

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

        bot_response = self.generator.generate(
            query=summary_query, 
            context=context, 
            history=self.chat_history
        )

        log_message = f"[User uploaded image: {os.path.basename(image_path)}]"
        if text_hint:
            log_message += f" [With hint: {text_hint}]"
        self.chat_history.append(("user", log_message))
        self.chat_history.append(("bot", bot_response))
        
        print(f"[Bot Response]:\n{bot_response}")
        return bot_response

# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bot = UniRAGBot(device=device)
    
    # Quick sanity check
    bot.text_query("What is pneumonia?")