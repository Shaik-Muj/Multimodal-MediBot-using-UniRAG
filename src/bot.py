import os
import torch
from typing import List, Dict, Tuple, Optional
from retriever import UniRAGRetriever
from loaders import MedicalRecord
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------------------------------------
# 1. THE NEW "GENERATION" BRAIN
# ---------------------------------------------------------
class LLMGenerator:
    """
    This class is the "G" in RAG.
    It loads a local LLM to synthesize conversational answers.
    """
    def __init__(self, device: str = "cuda"):
        print("💬 Initializing Conversational LLM...")
        
        # We use a small, fast, and capable model
        # Phi-3 is excellent. If you have less VRAM, 
        # you can use 'microsoft/phi-2' or 'gemma:2b'
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        try:
            # Try to load with 8-bit quantization to save VRAM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device,
                load_in_8bit=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"8-bit loading failed ({e}). Trying 16-bit. This will use more VRAM.")
            # Fallback to 16-bit
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
            )

        # Create a HuggingFace Pipeline for easy text generation
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def _build_prompt(self, query: str, context: str, history: List[Tuple[str, str]]) -> str:
        """
        Builds the full prompt for the LLM, including history and RAG context.
        """
        
        # 1. Format the chat history
        history_str = ""
        for role, message in history:
            history_str += f"<|{role}|>\n{message}<|end|>\n"
            
        # 2. Create the system prompt
        system_prompt = (
            "<|system|>\n"
            "You are a helpful medical AI assistant. You must answer the user's "
            "question based *only* on the provided facts. "
            "If the facts do not contain the answer, say 'I'm sorry, my knowledge base "
            "does not contain that information.' Do not make up information. "
            "Be concise and conversational.\n\n"
            "--- RELEVANT FACTS ---\n"
            f"{context}"
            "--- END FACTS ---<|end|>\n"
        )
        
        # 3. Combine history and the new query
        # This is the Phi-3 instruction format
        final_prompt = system_prompt + history_str + f"<|user|>\n{query}<|end|>\n<|assistant|>\n"
        return final_prompt
        
    def generate(self, query: str, context: str, history: List[Tuple[str, str]]) -> str:
        """
        Generates a conversational response.
        """
        prompt = self._build_prompt(query, context, history)
        
        # Generation settings
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.3,
            "do_sample": True,
        }
        
        try:
            output = self.pipeline(prompt, **generation_args)
            response = output[0]['generated_text']
        except Exception as e:
            print(f"LLM Generation failed: {e}")
            response = "I'm sorry, I encountered an error while generating a response."
            
        return response

# ---------------------------------------------------------
# 2. THE REFACTORED BOT (Now a "Conversational Agent")
# ---------------------------------------------------------
class UniRAGBot:
    def __init__(self, device: str = "cuda"):
        # 1. Initialize Retriever (The "R")
        self.retriever = UniRAGRetriever(device=device)
        
        # 2. Initialize Generator (The "G")
        self.generator = LLMGenerator(device=device)
        
        # 3. Initialize Chat Memory
        self.chat_history: List[Tuple[str, str]] = []
        
        print("\n--- 🤖 UniRAG Conversational Agent is Online ---")

    def _format_context(self, records: List[MedicalRecord]) -> str:
        if not records:
            return "No relevant information found."
        
        context_str = ""
        for i, record in enumerate(records):
            context_str += f"Fact {i+1}: {record.content}\n"
        return context_str

    def text_query(self, query: str) -> str:
        """
        Handles a text query with the full RAG pipeline.
        """
        print(f"\n[User Query]: {query}")
        
        # 1. Retrieve (The "R")
        retrieved_records = self.retriever.search_text(query, k=3)
        context = self._format_context(retrieved_records)
        
        # 2. Generate (The "G")
        # We pass the user's query, the facts we found, and the *past* history
        bot_response = self.generator.generate(
            query=query, 
            context=context, 
            history=self.chat_history # This is how it understands "this"
        )
        
        # 3. Update memory
        self.chat_history.append(("user", query))
        self.chat_history.append(("assistant", bot_response))
        
        print(f"[Bot Response]:\n{bot_response}")
        return bot_response

    # --- This function is now just for retrieval ---
    # The LLM doesn't see the images, it just sees our report.
    def image_query(self, image_path: str, text_hint: Optional[str] = None) -> str:
        print(f"\n[User Image Query]: {image_path}")
        if text_hint:
            print(f"[User Text Hint]: {text_hint}")

        retrieved_records = self.retriever.search_hybrid(
            image_path=image_path,
            text_query=text_hint,
            k=3
        )
        
        # We don't use the LLM for this part, we just report the facts.
        if not retrieved_records:
            bot_response = "I could not find any visually or semantically similar cases in the database."
        else:
            bot_response = "Based on a hybrid visual and text search, I found these similar cases:\n\n"
            for i, record in enumerate(retrieved_records):
                bot_response += f"Match {i+1}:\n"
                bot_response += f"  - Diagnosis/Label: {record.metadata.get('type', 'N/A')}\n"
                bot_response += f"  - Source: {record.metadata.get('source', 'N/A')}\n"
                if record.metadata.get('source') == 'roco':
                    bot_response += f"  - Caption: {record.content}\n\n"

        # Update memory
        log_message = f"[User uploaded image: {image_path}]"
        if text_hint:
            log_message += f" [With hint: {text_hint}]"
        self.chat_history.append(("user", log_message))
        self.chat_history.append(("assistant", bot_response))
        
        print(f"[Bot Response]:\n{bot_response}")
        return bot_response

# ---------------------------------------------------------
# SCRIPT ENTRY POINT (Test the new conversational logic)
# ---------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bot = UniRAGBot(device=device)
    
    print("\n--- Test 1: Initial Question ---")
    bot.text_query("I think I have typhoid fever... can you tell me the symptoms?")
    
    print("\n--- Test 2: Follow-up Question ---")
    bot.text_query("why does this happen")