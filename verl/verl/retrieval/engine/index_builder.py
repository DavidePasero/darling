import sys
import json
import pickle
import subprocess
from pathlib import Path
from typing import List, Literal, Optional
import numpy as np
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from verl.retrieval.engine.document_dataset import BeirAdapter


class IndexBuilder:

    def __init__(
        self,
        index_type: Literal["faiss", "bm25"],
        output_dir: Path,
        verbose: bool = True
    ):
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
        bm25_threads: int = 8,
        batch_size: int = 128,
        device: str = "cuda"
    ):
        if self.verbose:
            print("=" * 80)
            print(f"Building {self.index_type.upper()} Index from BEIR Dataset")
            print("=" * 80)
            print(f"Dataset: {beir_dataset_path}")
            print(f"Output: {self.output_dir}")

        adapter = BeirAdapter(data_path=beir_dataset_path, split="train")
        dataset = adapter.to_unified()

        corpus_ids = list(dataset.corpus.keys())
        corpus_texts = dataset.corpus

        if self.verbose:
            print(f"Loaded {len(corpus_texts)} documents")

        if self.index_type == "faiss":
            if embedding_model is None:
                raise ValueError("embedding_model must be provided for FAISS index")

            self._build_faiss_index(
                corpus_texts=corpus_texts,
                corpus_ids=corpus_ids,
                embedding_model=embedding_model,
                nlist=faiss_nlist,
                m=faiss_m,
                batch_size=batch_size,
                device=device
            )

        elif self.index_type == "bm25":
            self._build_bm25_index(
                corpus_texts=corpus_texts,
                corpus_ids=corpus_ids,
                threads=bm25_threads
            )

        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        if self.verbose:
            print("\n" + "=" * 80)
            print("Index Build Complete")
            print("=" * 80)

    def _build_faiss_index(
        self,
        corpus_texts: dict,
        corpus_ids: list,
        embedding_model: str,
        nlist: int,
        m: int,
        batch_size: int,
        device: str
    ):
        embeddings_cache_path = self.output_dir / "embeddings_cache.npy"

        if embeddings_cache_path.exists():
            if self.verbose:
                print(f"\nLoading cached embeddings from: {embeddings_cache_path}")
            embeddings = np.load(embeddings_cache_path)
            if self.verbose:
                print(f"Loaded embeddings shape: {embeddings.shape}")
        else:
            if self.verbose:
                print(f"\nLoading embedding model: {embedding_model}")

            model = SentenceTransformer(
                embedding_model,
                device=device,
                trust_remote_code=True
            )
            model.eval()

            if self.verbose:
                print(f"Encoding {len(corpus_texts)} documents with batch size {batch_size}")

            corpus_text_list = [corpus_texts[doc_id] for doc_id in corpus_ids]
            embeddings = []

            for i in tqdm(range(0, len(corpus_text_list), batch_size), disable=not self.verbose):
                batch = corpus_text_list[i:i + batch_size]
                with torch.no_grad():
                    batch_embeddings = model.encode(
                        batch,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                embeddings.append(batch_embeddings)

            embeddings = np.vstack(embeddings).astype('float32')

            if self.verbose:
                print(f"\nSaving embeddings cache to: {embeddings_cache_path}")
            np.save(embeddings_cache_path, embeddings)

        if self.verbose:
            print(f"Encoded embeddings shape: {embeddings.shape}")
            print(f"\nBuilding FAISS index:")
            print(f"  Dimension: {embeddings.shape[1]}")
            print(f"  Num docs: {embeddings.shape[0]}")
            print(f"  IVF nlist: {nlist}")
            print(f"  PQ m: {m}")

        dimension = embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)

        if self.verbose:
            print("Training FAISS index...")

        use_gpu = device == "cuda" and torch.cuda.is_available()
        if use_gpu:
            if self.verbose:
                print("Using GPU for training")
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.train(embeddings)
            index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            if self.verbose:
                print("Using CPU for training")
            index.train(embeddings)

        if self.verbose:
            print("Adding vectors to index...")
        index.add(embeddings)

        index_path = self.output_dir / "faiss_index.faiss"
        id_mapping_path = self.output_dir / "id_mapping.pkl"

        if self.verbose:
            print(f"\nSaving index to: {index_path}")
        faiss.write_index(index, str(index_path))

        if self.verbose:
            print(f"Saving ID mapping to: {id_mapping_path}")
        with open(id_mapping_path, 'wb') as f:
            pickle.dump(corpus_ids, f)

    def _build_bm25_index(
        self,
        corpus_texts: dict,
        corpus_ids: list,
        threads: int
    ):
        jsonl_file = self.output_dir / "corpus.jsonl"
        index_path = self.output_dir / "index"
        id_mapping_path = self.output_dir / "id_mapping.pkl"

        if self.verbose:
            print(f"\nConverting corpus to Pyserini JSONL format")
            print(f"Output: {jsonl_file}")

        num_docs = 0
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for doc_id in corpus_ids:
                doc = {
                    "id": doc_id,
                    "contents": corpus_texts[doc_id]
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                num_docs += 1

        if self.verbose:
            print(f"Wrote {num_docs} documents to {jsonl_file}")
            print(f"\nBuilding Pyserini/Lucene index")
            print(f"Index: {index_path}")
            print(f"Threads: {threads}")

        index_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(jsonl_file.parent),
            "--index", str(index_path),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(threads),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw"
        ]

        if self.verbose:
            print(f"\nRunning command:")
            print(" ".join(cmd))
            print()

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        if self.verbose:
            print(result.stdout)
            if result.stderr:
                print("Stderr:", result.stderr)

        if self.verbose:
            print(f"\nSaving ID mapping to: {id_mapping_path}")
        with open(id_mapping_path, 'wb') as f:
            pickle.dump(corpus_ids, f)

        jsonl_file.unlink()
        if self.verbose:
            print(f"Removed intermediate JSONL file: {jsonl_file}")