import faiss
import numpy as np
import os
import torch
from typing import List, Optional
from sentence_transformers import SentenceTransformer

class ResearchRetriever:
    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-8B', device='cuda'):
        """
        A FAISS retriever compatible with H100, A100, and MIG instances.
        Safe for lower VRAM environments while retaining Flash Attention speedups where available.
        """
        print(f"🚀 Initializing Retriever on {device.upper()} with {model_name}...")
        self.device = device
        self.model_name = model_name
        
        # 1. Load Embedding Model
        # H100/A100 OPTIMIZATION: Flash Attention 2
        # We verify Compute Capability >= 8.0 (Ampere/Hopper) to ensure compatibility.
        # This speeds up the encoding phase significantly.
        model_args = {"trust_remote_code": True}
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("⚡ Flash Attention 2 enabled (Ampere/Hopper detected).")
                model_args["model_kwargs"] = {"attn_implementation": "flash_attention_2"}
            else:
                print("⚠️ Flash Attention 2 disabled (GPU too old). Using standard attention.")

        self.model = SentenceTransformer(model_name, device=device, **model_args)
        
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"   Model Dimension: {self.dimension}")
        
        # 2. Initialize GPU Resources
        try:
            self.res = faiss.StandardGpuResources().
        except AttributeError:
            print("⚠️ Error: FAISS-GPU is not installed or CUDA is missing.")
            self.res = None

        self.index = None

    def encode(self, texts: List[str], batch_size: int = 64, is_query: bool = False) -> np.ndarray:
        """
        Encodes text into normalized float32 vectors.
        Uses native 'prompt_name' for Qwen models.
        """
        encode_kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": True,
            "convert_to_numpy": True,
            "normalize_embeddings": True
        }

        # Qwen3 Instruction Handling (Native API)
        if is_query and ("Qwen" in self.model_name or "Qwen" in self.model.tokenizer.name_or_path):
            # Qwen3 models have a "query" prompt registered in their config
            encode_kwargs["prompt_name"] = "query"
            
        # BGE Instruction Handling (Manual Fallback)
        elif is_query and "bge" in self.model_name:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

        embeddings = self.model.encode(texts, **encode_kwargs)
        return embeddings.astype('float32')

    def build_index(self, texts: List[str], nlist: int = 4096, m: Optional[int] = None, nbits: int = 8):
        """
        Builds an IVF-PQ index on GPU.
        
        Args:
            nlist: Number of clusters. 4096 is a safe default for 1M-10M docs.
            m:     Product Quantization sub-vectors. 
                   Dynamic default ensures optimal accuracy based on model dimension.
        """
        # Smart 'm' calculation
        # H100/High-End Optimization: We prefer larger 'm' (16-32 dims per sub-vector)
        # because high-bandwidth memory can handle the larger codebooks easily.
        if m is None:
            if self.dimension % 16 == 0:
                m = int(self.dimension / 16)
            elif self.dimension % 8 == 0:
                m = int(self.dimension / 8)
            else:
                m = 64 # Fallback
            print(f"Auto-configured compression: m={m} (Sub-vector dim: {self.dimension//m})")

        print(f"\n🏗️  Building Index for {len(texts)} documents...")
        
        # 1. Quantizer (Inner Product for Cosine Similarity)
        quantizer = faiss.IndexFlatIP(self.dimension)
        
        # 2. IVF-PQ Structure
        cpu_index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        
        # 3. Move to GPU
        print("Moving index to GPU...")
        self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
        
        # 4. Train & Add
        # Batch size 32 is safe for 8B models on A100/H100.
        # If using a very small MIG slice, consider lowering this to 16 manually.
        embeddings = self.encode(texts, batch_size=32) 
        
        print(f"Training on {len(embeddings)} vectors...")
        self.index.train(embeddings)
        
        print("Adding vectors...")
        self.index.add(embeddings)
        
        print(f"✅ Index Ready. {self.index.ntotal} vectors indexed.")

    def save_index(self, filepath: str):
        """Saves index to disk (Move to CPU first)."""
        if not self.index: return
        print(f"💾 Saving index to {filepath}...")
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(cpu_index, filepath)
        print("   Save complete.")

    def load_index(self, filepath: str):
        """Loads index and moves to GPU."""
        if not os.path.exists(filepath):
            print("⚠️ File not found.")
            return
        print(f"DTO Loading index from {filepath}...")
        cpu_index = faiss.read_index(filepath)
        self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
        print(f"✅ Loaded {self.index.ntotal} vectors to GPU.")

    def search(self, query: str, k: int = 10, nprobe: int = 64):
        """
        Search with Qwen3-8B.
        nprobe=64 gives high recall. On H100 this is instant.
        On lower tier cards, you might lower nprobe to 32 if latency is too high (>50ms).
        """
        if not self.index: return None, None

        faiss.GpuParameterSpace().set_index_parameter(self.index, "nprobe", nprobe)
        
        # is_query=True triggers the Qwen instruction prefix via prompt_name
        q_emb = self.encode([query], batch_size=1, is_query=True)
        
        scores, indices = self.index.search(q_emb, k)
        return scores[0], indices[0]

if __name__ == "__main__":
    # Example Usage
    retriever = ResearchRetriever(model_name='Qwen/Qwen3-Embedding-8B')
    
    # 2. Dummy Data 
    corpus = [
        "The capital of France is Paris.",
        "Deep learning utilizes neural networks.",
        "Qwen3 is a large language model based embedding model.",
        "NVIDIA H100 is a powerful GPU for AI.",
        "Information retrieval is the science of searching for information."
    ] * 100

    # 3. Build
    if not os.path.exists("qwen_msmarco.index"):
        retriever.build_index(corpus, nlist=128) # m is auto-calculated
        retriever.save_index("qwen_msmarco.index")
    else:
        retriever.load_index("qwen_msmarco.index")

    # 4. Search
    query = "powerful graphics card"
    scores, ids = retriever.search(query, k=3)
    
    print(f"\n🔎 Query: {query}")
    for s, i in zip(scores, ids):
        print(f"   Score: {s:.4f} | ID: {i}")