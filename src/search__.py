import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore

load_dotenv()

# ---- Local LLM backends ------------------------------------------------------

class LocalLLM:
    """Unified interface for local LLM backends."""
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        raise NotImplementedError


class OllamaLLM(LocalLLM):
    """Uses the local Ollama server."""
    def __init__(self, model: str = "phi3", host: Optional[str] = None):
        import ollama
        self.ollama = ollama
        self.model = model
        # Allow custom host via OLLAMA_HOST (e.g., http://localhost:11434)
        self.host = host or os.getenv("OLLAMA_HOST", None)
        if self.host:
            # The python client reads OLLAMA_HOST env var
            os.environ["OLLAMA_HOST"] = self.host

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        # Chat format tends to steer instruction-following better
        resp = self.ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": max_new_tokens,
                # You can tweak these:
                "temperature": 0.2,
                "top_p": 0.95,
            },
        )
        return resp["message"]["content"].strip()


class HFTransformersLLM(LocalLLM):
    """Runs a HF model locally (CPU or GPU)."""
    def __init__(
        self,
        model_id: str = "microsoft/phi-3-mini-4k-instruct",
        device: Optional[str] = None,
        load_4bit: bool = True,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        quant_cfg = None
        if load_4bit:
            quant_cfg = BitsAndBytesConfig(load_in_4bit=True)

        # Auto device map keeps it simple across CPU/GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device is None else None,
            quantization_config=quant_cfg,
            torch_dtype=torch.float16 if self._has_cuda() else torch.float32,
        )
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        if device is not None:
            self.model.to(device)

    def _has_cuda(self) -> bool:
        import torch
        return torch.cuda.is_available()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # deterministic summarization
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text.strip()

# ---- RAG wrapper -------------------------------------------------------------

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        backend: str = "ollama",                # "ollama" or "hf"
        model_name: str = "phi3",               # Ollama model name OR HF model_id
        hf_4bit: bool = True,
    ):
        """
        backend="ollama" -> model_name like: "phi3", "mistral", "llama3", "gemma", "tinyllama"
        backend="hf"     -> model_name is a HF repo id, e.g. "microsoft/phi-3-mini-4k-instruct"
        """
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or (if missing) build FAISS index
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path  = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # Choose local LLM backend
        if backend.lower() == "ollama":
            self.llm: LocalLLM = OllamaLLM(model=model_name)
            print(f"[INFO] Using Ollama model: {model_name}")
        elif backend.lower() == "hf":
            self.llm = HFTransformersLLM(model_id=model_name, load_4bit=hf_4bit)
            print(f"[INFO] Using HF Transformers model: {model_name}")
        else:
            raise ValueError("backend must be 'ollama' or 'hf'")

    def _build_prompt(self, query: str, context: str) -> str:
        # Compact instruction to keep small models on track
        return (
            "You are a concise assistant for retrieval-augmented summarization.\n"
            "Given the context, answer the user query in a brief, factual way. "
            "Cite key terms from context when relevant. If unsure, say so.\n\n"
            f"Query:\n{query}\n\nContext:\n{context}\n\nAnswer:"
        )

    def search_and_summarize(self, query: str, top_k: int = 5, max_new_tokens: int = 256) -> str:
        results: List[Dict[str, Any]] = self.vectorstore.query(query, top_k=top_k)
        texts = [r.get("metadata", {}).get("text", "") for r in results if r.get("metadata")]
        # Basic dedupe / trunc
        uniq = []
        seen = set()
        for t in texts:
            t2 = t.strip()
            if t2 and t2 not in seen:
                seen.add(t2)
                uniq.append(t2)
        # Keep context reasonable for small models
        context = "\n\n---\n\n".join(uniq)[:8000]  # ~8k chars cap

        if not context:
            return "No relevant documents found."

        prompt = self._build_prompt(query, context)
        return self.llm.generate(prompt, max_new_tokens=max_new_tokens)


# ---- Example usage -----------------------------------------------------------

if __name__ == "__main__":
    # Choose one:
    # A) Local Ollama, using the model you pulled (e.g., 'phi3', 'mistral', 'llama3', 'gemma')
    rag = RAGSearch(backend="ollama", model_name="phi3")

    # B) Local HF Transformers
    # rag = RAGSearch(backend="hf", model_name="microsoft/phi-3-mini-4k-instruct", hf_4bit=True)

    query = "What is the attention mechanism?"
    summary = rag.search_and_summarize(query, top_k=3, max_new_tokens=256)
    print("Summary:", summary)
