import os

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class FaissDB:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B", device="cuda"):
        """
        A FAISS retriever compatible with H100, A100, and MIG instances.
        Optimized for high-throughput indexing on single-GPU setups.
        """
        print(f"🚀 Initializing Retriever on {device.upper()} with {model_name}...")
        self.device = device
        self.model_name = model_name

        model_args = {"trust_remote_code": True}

        # Check for Flash Attention support
        # if torch.cuda.is_available():
        #     major, _ = torch.cuda.get_device_capability()
        #     if major >= 8:
        #         print("   ⚡ Flash Attention 2 enabled (Ampere/Hopper detected).")
        #         model_args["model_kwargs"] = {"attn_implementation": "flash_attention_2"}

        self.model = SentenceTransformer(model_name, device=device, **model_args)
        self.model.eval()

        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"   Model Dimension: {self.dimension}")

        # Initialize GPU Resources
        try:
            self.res = faiss.StandardGpuResources()
        except AttributeError:
            print("⚠️ Error: FAISS-GPU is not installed or CUDA is missing.")
            self.res = None

        self.index = None

    def encode(self, texts: list[str], batch_size: int = 64, is_query: bool = False) -> np.ndarray:
        """
        Encodes text into normalized float32 vectors.
        Uses native 'prompt_name' for Qwen models.
        """
        encode_kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": False,  # We handle progress bar in outer loops usually
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }

        if is_query and ("Qwen" in self.model_name or "Qwen" in self.model.tokenizer.name_or_path):
            encode_kwargs["prompt_name"] = "query"
        elif is_query and "bge" in self.model_name:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

        embeddings = self.model.encode(texts, **encode_kwargs)
        return embeddings.astype("float32")

    def build_index(self, texts: list[str], nlist: int = 4096, m: int | None = None, nbits: int = 8):
        """
        Builds an IVF-PQ index on GPU using optimized streaming.
        Avoids OOM and CPU bottlenecks by processing data in chunks.
        """
        # 1. Smart 'm' calculation
        if m is None:
            if self.dimension == 1024:
                m = 32
            elif self.dimension % 32 == 0:
                m = 32
            elif self.dimension % 16 == 0:
                m = 16
            else:
                m = 8
            print(f"   Auto-configured compression: m={m} (Sub-vector dim: {self.dimension // m})")

        print(f"\n🏗️  Building Index for {len(texts)} documents...")

        # 2. Configure Index
        quantizer = faiss.IndexFlatIP(self.dimension)
        cpu_index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

        # Fix for Shared Memory Limit
        co = faiss.GpuClonerOptions()
        co.useFloat16LookupTables = True
        co.useFloat16 = True  # Use float16 for storage to save VRAM

        print("   Moving index to GPU...")
        try:
            self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index, co)
        except RuntimeError:
            print("⚠️ Shared Memory limit hit. Retrying with m=32...")
            cpu_index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, 32, nbits, faiss.METRIC_INNER_PRODUCT)
            self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index, co)

        # 3. OPTIMIZED PIPELINE
        # We use a custom Dataset + DataLoader with num_workers > 0
        # This runs tokenization in background processes, feeding the GPU constantly.

        class StringDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        # Helper to tokenize inside worker processes
        def collate_fn(batch):
            return self.model.tokenize(batch)

        dataset = StringDataset(texts)

        # Training Phase (using a random subset)
        train_size = min(len(texts), 200_000)
        print(f"   Training Index on random sample of {train_size} documents...")

        # Use a smaller batch size to keep latency low and pipeline smooth
        train_loader = DataLoader(
            dataset,
            batch_size=16,
            sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=train_size),
            num_workers=4,  # 4 CPU cores for tokenization
            collate_fn=collate_fn,
            prefetch_factor=2,
            persistent_workers=True,
        )

        # Collect training vectors
        train_vectors = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Encoding Train Set"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                emb = self.model(batch)["sentence_embedding"]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                train_vectors.append(emb)

        train_vectors = torch.cat(train_vectors).cpu().numpy().astype("float32")
        self.index.train(train_vectors)
        del train_vectors  # Free RAM

        # 4. Indexing Phase (Streaming Add)
        # We process the full dataset linearly and add to index on the fly
        print("   Adding all vectors to index (Streaming)...")
        full_loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            prefetch_factor=2,
        )

        with torch.no_grad():
            for batch in tqdm(full_loader, desc="Indexing"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                emb = self.model(batch)["sentence_embedding"]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

                # Add directly to GPU index
                self.index.add(emb.cpu().numpy().astype("float32"))

        print(f"✅ Index Ready. {self.index.ntotal} vectors indexed.")

    def save_index(self, filepath: str):
        if not self.index:
            return
        print(f"💾 Saving index to {filepath}...")
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(cpu_index, filepath)
        print("   Save complete.")

    def load_index(self, filepath: str):
        if not os.path.exists(filepath):
            return
        print(f"DTO Loading index from {filepath}...")
        cpu_index = faiss.read_index(filepath)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        co.useFloat16LookupTables = True
        self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index, co)
        print(f"✅ Loaded {self.index.ntotal} vectors.")

    def search(self, queries: list[str], k: int = 10, nprobe: int = 64, batch_size: int = 64):
        if not self.index:
            return None, None
        faiss.GpuParameterSpace().set_index_parameter(self.index, "nprobe", nprobe)
        q_embs = self.encode(queries, batch_size=batch_size, is_query=True)
        scores, indices = self.index.search(q_embs, k)
        return scores, indices


if __name__ == "__main__":
    # Example Usage
    retriever = FaissDB(model_name="Qwen/Qwen3-Embedding-0.6B")

    # 2. Dummy Data
    corpus = [
        "The capital of France is Paris.",
        "Deep learning utilizes neural networks.",
        "Qwen3 is a large language model based embedding model.",
        "NVIDIA H100 is a powerful GPU for AI.",
        "Information retrieval is the science of searching for information.",
    ] * 100

    # 3. Build
    if not os.path.exists("qwen_msmarco.db"):
        retriever.build_index(corpus, nlist=128)  # m is auto-calculated
        retriever.save_index("qwen_msmarco.db")
    else:
        retriever.load_index("qwen_msmarco.db")

    # 4. Search
    query = ["powerful graphics card"]
    scores, ids = retriever.search(query, k=3)

    print(f"\n🔎 Query: {query}")
    for s, i in zip(scores, ids):
        print(f"   Score: {s} | ID: {i % 5}")
