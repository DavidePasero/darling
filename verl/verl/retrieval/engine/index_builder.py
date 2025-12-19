import os
import json
import pickle
import numpy as np
import torch
import faiss
import subprocess
import shutil
from pathlib import Path
from typing import Literal, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from verl.retrieval.engine.document_dataset import BeirAdapter

class StringDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class IndexBuilder:
    def __init__(self, index_type: Literal["faiss", "bm25"], output_dir: Path, verbose: bool = True):
        self.index_type = index_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def build_from_beir(
        self,
        beir_dataset_path: str,
        embedding_model: Optional[str] = None,
        faiss_nlist: int = 4096,
        faiss_m: int = 32,
        batch_size: int = 128,
        bm25_threads: int = 8,
        device: str = "cuda",
        max_docs: Optional[int] = None
    ):
        if self.verbose:
            print(f"Loading BEIR dataset: {beir_dataset_path}")
        
        # Load Data
        adapter = BeirAdapter(data_path=beir_dataset_path, split="train")
        dataset = adapter.to_unified()
        

        if max_docs is not None:
            if self.verbose:
                print(f"Limiting to {max_docs} documents (maximizing relevance coverage)...")
            
            import csv

            best_docs = []
            qrels_dir = Path(beir_dataset_path) / "qrels"
            
            if qrels_dir.exists():
                for qrel_file in qrels_dir.glob("*.tsv"):
                    if self.verbose:
                        print(f"Loading relevance info from {qrel_file}")
                    
                    with open(qrel_file, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                        header = next(reader, None)
                        
                        qid_idx, doc_idx, score_idx = 0, 1, 2
                        if header:
                            try:
                                qid_idx = header.index("query-id")
                                doc_idx = header.index("corpus-id")
                                score_idx = header.index("score")
                            except ValueError:
                                pass
                        
                        for row in reader:
                            if len(row) < 3:
                                continue
                            doc_id = row[doc_idx]
                            try:
                                score = float(row[score_idx])
                            except ValueError:
                                score = 0
                                
                            if score > 0:
                                best_docs.append(doc_id)
            
            if self.verbose:
                print(f"Selected {len(best_docs)} documents.")
                
            corpus_ids = list(set(best_docs))

            all_ids = list(range(len(dataset.corpus)))
            while True:
                start = 0
                if len(corpus_ids) >= max_docs:
                    break
                corpus_ids += all_ids[start:max_docs - len(corpus_ids)]
                corpus_ids = list(set(corpus_ids))
                start += max_docs - len(corpus_ids)

            # if len(corpus_ids) < max_docs:
            #     for k in range(len(dataset.corpus.keys())):
            #         if len(corpus_ids)%100000 == 0:
            #             print(f"Selected {len(corpus_ids)} documents.")
            #         if len(corpus_ids) >= max_docs:
            #             break
            #         if k not in corpus_ids:
            #             corpus_ids.append(k)

            corpus_texts = {k: dataset.corpus[k] for k in corpus_ids}
        
        else:
            if self.verbose:
                print("No max_docs specified, using all documents")
            corpus_ids = list(dataset.corpus.keys())
            corpus_texts = dataset.corpus

        if self.verbose:
            print(f"Loaded {len(corpus_texts)} documents")

        if self.index_type == "faiss":
            if not embedding_model:
                raise ValueError("embedding_model is required for FAISS index")
            
            self._build_faiss_index(
                corpus_texts, corpus_ids, embedding_model, 
                faiss_nlist, faiss_m, batch_size, device
            )
        
        elif self.index_type == "bm25":
            self._build_bm25_index(
                dataset_unified.corpus, corpus_ids, bm25_threads
            )
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

    def _build_faiss_index(self, corpus_texts, corpus_ids, embedding_model, nlist, m, batch_size, device):
        print(f"🚀 Initializing Model: {embedding_model} on {device}")
        
        model = SentenceTransformer(embedding_model, device=device, trust_remote_code=True)
        
        # --- FIX 1: FORCE TRUNCATION ---
        # Qwen defaults to huge context. We clamp it to 512 to prevent OOM.
        model.max_seq_length = 512 
        model.eval()
        
        dimension = model.get_sentence_embedding_dimension()
        print(f"   Model Context Limit: {model.max_seq_length} tokens")
        print(f"   Model Dimension: {dimension}")

        if dimension % m != 0:
            raise ValueError(f"Dimension ({dimension}) must be divisible by m ({m}).")

        try:
            res = faiss.StandardGpuResources()
            # Reduce temp memory overhead
            res.setTempMemory(512 * 1024 * 1024) 
        except AttributeError:
            print("⚠️ FAISS-GPU not found.")
            return

        print(f"   Configuring Index (d={dimension}, nlist={nlist}, m={m})")
        quantizer = faiss.IndexFlatIP(dimension)
        cpu_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        co.useFloat16LookupTables = True

        print("   Moving index configuration to GPU...")
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)

        dataset_obj = StringDataset(corpus_texts)
        def collate_fn(batch):
            return model.tokenize(batch)

        # 1. Train Phase
        train_size = min(len(corpus_texts), 50000)
        print(f"   Training Index on random sample of {train_size} docs...")
        
        train_loader = DataLoader(
            dataset_obj,
            batch_size=batch_size,
            sampler=torch.utils.data.RandomSampler(dataset_obj, replacement=True, num_samples=train_size),
            num_workers=4,
            collate_fn=collate_fn,
            persistent_workers=True,
            prefetch_factor=2
        )

        train_vectors = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Encoding Train"):
                batch = {k: v.to(device) for k, v in batch.items()}
                emb = model(batch)["sentence_embedding"]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                
                # --- FIX 2: OFFLOAD TO CPU IMMEDIATELY ---
                # We move embeddings to CPU RAM to save VRAM for the next batch
                train_vectors.append(emb.cpu())
        
        # Concat on CPU
        train_vectors_np = torch.cat(train_vectors).numpy().astype("float32")
        
        # FAISS GPU train can handle CPU inputs (it copies internally)
        gpu_index.train(train_vectors_np)
        
        # Cleanup
        del train_vectors, train_vectors_np
        torch.cuda.empty_cache()

        # 2. Add Phase (Streaming)
        print(f"   Indexing {len(corpus_texts)} documents...")
        full_loader = DataLoader(
            dataset_obj, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            collate_fn=collate_fn,
            prefetch_factor=2
        )

        with torch.no_grad():
            for batch in tqdm(full_loader, desc="Indexing"):
                batch = {k: v.to(device) for k, v in batch.items()}
                emb = model(batch)["sentence_embedding"]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                
                # Add directly to GPU index
                gpu_index.add(emb.cpu().numpy().astype("float32"))

        print("   Moving index to CPU for saving...")
        final_cpu_index = faiss.index_gpu_to_cpu(gpu_index)
        
        index_path = self.output_dir / "faiss_index.faiss"
        id_mapping_path = self.output_dir / "id_mapping.pkl"
        
        faiss.write_index(final_cpu_index, str(index_path))
        with open(id_mapping_path, 'wb') as f:
            pickle.dump(corpus_ids, f)
            
        print(f"✅ FAISS Index built at: {self.output_dir}")

    def _build_bm25_index(self, corpus_texts, corpus_ids, threads):
        # (Same as before)
        jsonl_dir = self.output_dir / "corpus_jsonl"
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        jsonl_file = jsonl_dir / "corpus.jsonl"
        index_path = self.output_dir / "bm25_index"
        id_mapping_path = self.output_dir / "id_mapping.pkl"

        print(f"   Converting {len(corpus_ids)} docs to Pyserini JSONL format...")
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for doc_id in corpus_ids:
                doc = {"id": doc_id, "contents": corpus_texts[doc_id]}
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        print(f"   Running Pyserini/Lucene Indexer (Threads: {threads})...")
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(jsonl_dir),
            "--index", str(index_path),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(threads),
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]

        subprocess.run(cmd, check=True)
        shutil.rmtree(jsonl_dir)
        with open(id_mapping_path, 'wb') as f:
            pickle.dump(corpus_ids, f)
        print(f"✅ BM25 Index built at: {index_path}")