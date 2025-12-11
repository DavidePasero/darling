from collections import defaultdict
import os
import torch
from verl import DataProto

from verl.retrieval.engine.retriever import FaissRetriever
from verl.retrieval.engine.bm25_retriever import Bm25Retriever
from verl.retrieval.engine.base_retriever import BaseRetriever
from verl.retrieval.engine.document_dataset import UnifiedDataset, BeirAdapter


class RetrievalRewardManager:
    """
    Reward manager for retrieval tasks supporting both FAISS and BM25.
    Supports NDCG, Recall, Precision, and Hit metrics.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score,
        reward_fn_key="data_source",
        retriever_type="faiss",
        # FAISS-specific parameters
        faiss_index_path=None,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        # BM25-specific parameters
        bm25_index_path=None,
        bm25_k1=0.9,
        bm25_b=0.4,
        # Common parameters
        id_mapping_path=None,
        beir_dataset_path=None,
        qrels_file="qrels/train.tsv",
        quality_method="ndcg",
        k=10,
        device="cuda",
        **kwargs
    ):
        """
        Initialize retrieval reward manager.

        Args:
            tokenizer: Tokenizer for decoding responses
            num_examine: Number of examples to print during training
            compute_score: Reward computation function (not used)
            reward_fn_key: Key to access data source in non_tensor_batch
            retriever_type: Type of retriever ('faiss' or 'bm25')
            faiss_index_path: Path to FAISS index file (for FAISS retriever)
            embedding_model: HuggingFace model for embeddings (for FAISS retriever)
            bm25_index_path: Path to BM25/Pyserini index directory (for BM25 retriever)
            bm25_k1: BM25 k1 parameter (default: 0.9)
            bm25_b: BM25 b parameter (default: 0.4)
            id_mapping_path: Path to pickle file with ID mapping (optional)
            beir_dataset_path: Path to BEIR dataset directory
            qrels_file: Relative path to qrels file (default: "qrels/train.tsv")
            quality_method: Metric to use ('ndcg', 'recall', 'precision', 'hit')
            k: Top-k for retrieval
            device: Device for models (FAISS only)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.quality_method = quality_method
        self.k = k
        self.device = device
        self.retriever_type = retriever_type.lower()

        if beir_dataset_path is None:
            raise ValueError("beir_dataset_path must be provided")

        print("=" * 80)
        print("Initializing RetrievalRewardManager")
        print("=" * 80)
        print(f"Retriever type: {self.retriever_type}")

        # Initialize retriever based on type
        if self.retriever_type == "faiss":
            if faiss_index_path is None:
                raise ValueError("faiss_index_path must be provided for FAISS retriever")

            self.retriever = FaissRetriever(
                faiss_index_path=faiss_index_path,
                embedding_model=embedding_model,
                id_mapping_path=id_mapping_path,
                device=device,
                verbose=True
            )

        elif self.retriever_type == "bm25":
            if bm25_index_path is None:
                raise ValueError("bm25_index_path must be provided for BM25 retriever")

            self.retriever = Bm25Retriever(
                index_path=bm25_index_path,
                k1=bm25_k1,
                b=bm25_b,
                id_mapping_path=id_mapping_path,
                verbose=True
            )

        else:
            raise ValueError(f"Unknown retriever_type: {retriever_type}. Must be 'faiss' or 'bm25'")

        adapter = BeirAdapter(data_path=beir_dataset_path, split="train")
        self.doc_dataset: UnifiedDataset = adapter.to_unified()

        print(f"Loaded {len(self.doc_dataset.qrels)} queries with relevance labels")
        print(f"Quality reward: {quality_method}@{k}")
        print(f"Index size: {self.retriever.get_index_size()} documents")
        print("=" * 80)
        print()

    def verify(self, data):
        """
        Extract and decode responses, then compute retrieval rewards.

        Args:
            data: DataProto object

        Returns:
            List of reward scores
        """
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        uid = data.non_tensor_batch["uid"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        scores, indices = self.retriever.search(responses_str, k=self.k, nprobe=64)
        retrieved_doc_ids = self.retriever.map_indices_to_ids(indices)
        
        rewards = self.doc_dataset.compute_rewards_batch(
            query_uids=uid,
            retrieved_doc_ids_batch=retrieved_doc_ids.tolist(),
            method=self.quality_method,
            k=self.k
        )


        debug_log = os.environ.get("DEBUG_LOG", "0") == "1"
        if debug_log:
            print("\n" + "=" * 100)
            print("DEBUG: RETRIEVAL REWARD COMPUTATION")
            print("=" * 100)
            
            # Check for UID mismatches
            missing_uids = []
            for uid_val in uid:
                if uid_val not in self.doc_dataset.queries:
                    missing_uids.append(uid_val)
            
            if missing_uids:
                print(f"\n⚠️  WARNING: {len(missing_uids)}/{len(uid)} UIDs not found in BEIR dataset!")
                print(f"Missing UIDs (first 5): {missing_uids[:5]}")
                print(f"\nAvailable query IDs in BEIR dataset (first 10):")
                available_qids = list(self.doc_dataset.queries.keys())[:10]
                for qid in available_qids:
                    print(f"  - {qid}")
                print(f"\nTotal queries in BEIR dataset: {len(self.doc_dataset.queries)}")
                print("\n💡 TIP: Your training data UIDs must match the query IDs in your BEIR dataset.")
                print("   Check that your parquet file's 'uid' column matches the BEIR queries.jsonl '_id' field.\n")
            
            for i in range(len(data)):
                query_uid = uid[i]
                response_str = responses_str[i]
                retrieved_docs = retrieved_doc_ids[i].tolist() if hasattr(retrieved_doc_ids[i], 'tolist') else retrieved_doc_ids[i]
                reward = rewards[i]
                
                # Get original query from dataset
                original_query = self.doc_dataset.queries.get(query_uid, "N/A")
                
                # Get ground truth relevant docs
                relevant_docs = self.doc_dataset.get_relevant_docs(query_uid)
                
                print(f"\n{'─' * 100}")
                print(f"Sample {i+1}/{len(data)}")
                print(f"{'─' * 100}")
                print(f"UID: {query_uid}")
                
                if original_query == "N/A":
                    print(f"⚠️  WARNING: UID '{query_uid}' not found in BEIR dataset!")
                    print(f"   This UID will receive a reward of 0.0")
                
                print(f"\nOriginal Query: {original_query}")
                print(f"\nGenerated Response (Rewritten Query):")
                print(f"  {response_str}")
                print(f"\nRetrieved Documents (Top-{self.k}):")
                
                for rank, doc_id in enumerate(retrieved_docs[:self.k], 1):
                    doc_text = self.doc_dataset.corpus.get(doc_id, "N/A")
                    # Truncate long documents
                    doc_text_truncated = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                    is_relevant = "✓ RELEVANT" if doc_id in relevant_docs else "✗ Not relevant"
                    print(f"  [{rank}] {doc_id} {is_relevant}")
                    print(f"      {doc_text_truncated}")
                
                print(f"\nGround Truth Relevant Docs: {relevant_docs[:10]}" + ("..." if len(relevant_docs) > 10 else ""))
                print(f"Num Relevant Docs: {len(relevant_docs)}")
                print(f"\n{self.quality_method.upper()}@{self.k} Reward: {reward:.4f}")
            
            print("\n" + "=" * 100)
            print(f"Batch Summary: {len(data)} samples, Mean Reward: {sum(rewards)/len(rewards):.4f}")
            if missing_uids:
                print(f"⚠️  {len(missing_uids)} samples had missing UIDs and received 0.0 reward")
            print("=" * 100 + "\n")

        return rewards

    def __call__(self, data: DataProto, return_dict=False):
        """
        Main entry point for reward computation.

        Args:
            data: DataProto object containing batch data
            return_dict: Whether to return dict with extra info

        Returns:
            Reward tensor or dict with reward_tensor and reward_extra_info
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch.get(self.reward_fn_key, ['retrieval'] * len(data))

        scores = self.verify(data)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            reward = scores[i]
            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(
                    data.batch["responses"][i][:length],
                    skip_special_tokens=True
                )
                prompt_str = self.tokenizer.decode(
                    data.batch["prompts"][i],
                    skip_special_tokens=True
                )
                uid_val = data.non_tensor_batch["uid"][i]

                print("=" * 80)
                print(f"Retrieval Reward Example - {data_source}")
                print(f"UID: {uid_val}")
                print(f"[prompt] {prompt_str[:200]}...")
                print(f"[response] {response_str[:200]}...")
                print(f"[{self.quality_method}@{self.k}] {reward:.4f}")
                print("=" * 80)
                print()

                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info
            }
        else:
            return reward_tensor