"""
Retrieval-based reward computation for query rewriting with diversity.

This module implements DARLING-style rewards for information retrieval:
- Quality Reward: NDCG@k or Recall@k from FAISS retrieval
- Diversity Reward: Partition-based semantic diversity
- Combination: quality × normalized_diversity (multiplicative)

Reuses existing infrastructure:
- verl.retrieval.engine.vectordatabase.FaissDB for retrieval
- verl.retrieval.reward.reward_manager.RewardManager for quality scoring
"""

import asyncio
import aiohttp
import os
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


# ============================================================================
# Quality Reward Computation (using existing FaissDB + RewardManager)
# ============================================================================

def compute_quality_rewards(
    queries: List[str],
    uid: List[str],
    faiss_db,  # FaissDB instance
    reward_manager,
    id_mapping: List[int],
    labels_dict: Dict[str, List[int]],
) -> List[float]:
    """
    Compute retrieval quality rewards using existing FAISS infrastructure.

    Args:
        queries: List of rewritten query strings
        uid: List of user/query IDs (for label lookup)
        faiss_db: FaissDB instance (already loaded with index)
        reward_manager: RewardManager instance (configured with method='ndcg' or 'recall')
        id_mapping: List mapping FAISS indices to real document IDs
        labels_dict: Dict mapping uid -> list of relevant document IDs

    Returns:
        List of quality scores (one per query)
    """
    # Retrieve top-k documents for all queries
    k = reward_manager.k
    scores, indices = faiss_db.search(queries, k=k, nprobe=64)

    #TODO: batch this!
    quality_rewards = []
    for i, (query_indices, u) in enumerate(zip(indices, uid)):
        # Get ground truth labels for this query
        labels = labels_dict.get(u, [])

        if not labels:
            # No ground truth - assign neutral reward
            quality_rewards.append(0.0)
            continue

        # Compute reward using existing RewardManager
        # Note: compute_rewards expects shape (1, k) for batch
        reward = reward_manager.compute_rewards(
            indices_batch=query_indices[np.newaxis, :],  # Shape: (1, k)
            labels=labels,
            id_mapping=id_mapping
        )[0]  # Extract single reward

        quality_rewards.append(reward)

    return quality_rewards


# ============================================================================
# Diversity Reward Computation - Local DeBERTa
# ============================================================================

class LocalPartitionClassifier:
    """
    Local DeBERTa classifier for semantic equivalence.
    Follows verl/verl/utils/reward_score/diversity_rewards.py:218-304
    """

    def __init__(
        self,
        model_name: str = 'microsoft/deberta-v3-large',
        device: str = 'cuda',
        threshold: float = 0.5
    ):
        """
        Args:
            model_name: HuggingFace model name for sequence classification
            device: Device to run model on
            threshold: Similarity threshold (>= threshold means same partition)
        """
        self.device = device
        self.threshold = threshold

        print(f"Loading local partition classifier: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def classify_pair(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are semantically equivalent.

        Returns:
            True if semantically equivalent (same partition)
        """
        # Tokenize with [CLS] text1 [SEP] text2 [SEP] format
        inputs = self.tokenizer(
            text1,
            text2,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # Assuming binary classification: [different, same]
            # If model outputs similarity score, use index 1
            similarity_score = probs[0, 1].item()

        return similarity_score >= self.threshold


def compute_diversity_local(
    queries: List[str],
    uid: List[str],
    model_name: str = 'microsoft/deberta-v3-large',
    threshold: float = 0.5,
    device: str = 'cuda'
) -> List[float]:
    """
    Compute partition-based diversity using local DeBERTa classifier.

    Follows the exact pattern from diversity_rewards.py:partition()

    Args:
        queries: List of query strings
        uid: List of user IDs (for grouping)
        model_name: HuggingFace model for semantic classification
        threshold: Similarity threshold
        device: Device for model

    Returns:
        List of diversity scores: distinct_count / (total - 1)
    """
    # Initialize classifier
    classifier = LocalPartitionClassifier(model_name, device, threshold)

    # Group queries by UID
    uid_to_indices = defaultdict(list)
    for idx, u in enumerate(uid):
        uid_to_indices[u].append(idx)

    # Initialize diversity rewards
    diversity_rewards = [0.0] * len(queries)

    # Process each UID group
    for u, indices in uid_to_indices.items():
        if len(indices) <= 1:
            # Single query in group - neutral diversity
            diversity_rewards[indices[0]] = 1.0
            continue

        # Union-Find data structure for partitioning
        parent = list(range(len(indices)))

        def find(x):
            """Find with path compression"""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            """Union two sets"""
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Pairwise comparison
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i = indices[i]
                idx_j = indices[j]

                # Check if semantically equivalent
                if classifier.classify_pair(queries[idx_i], queries[idx_j]):
                    union(i, j)

        # Build partitions
        partitions = defaultdict(list)
        for i in range(len(indices)):
            root = find(i)
            partitions[root].append(indices[i])

        # Calculate diversity scores
        total = len(indices)
        for partition_group in partitions.values():
            group_size = len(partition_group)
            distinct_count = total - group_size

            for idx in partition_group:
                diversity_rewards[idx] = distinct_count / (total - 1)

    # Apply floor (following DARLING pattern)
    diversity_rewards = [max(r, 0.1) for r in diversity_rewards]

    return diversity_rewards


async def classify_pair_vllm(
    text1: str,
    text2: str,
    server_cfg: Dict,
    session: aiohttp.ClientSession,
    threshold: float = 0.5,
    max_len: int = 4096
) -> bool:
    """
    Classify pair using VLLM server.
    From partition_reward_vllm_serve.py:210-252
    """

    # Format input for classifier: [CLS] s1 [SEP] s2 [SEP]
    prompt = f"[CLS] {text1} [SEP] {text2} [SEP]"

    # Prepare request
    url = f"{server_cfg['url']}/classify"
    payload = {
        "model": server_cfg['model'],
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
    }

    # Send request with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    similarity_score = result.get('score', 0.0)
                    return similarity_score >= threshold
                elif attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to classify pair after {max_retries} attempts: {e}")
                return False  # Default to different
            await asyncio.sleep(0.1 * (attempt + 1))

    return False


async def partition_async_vllm(
    queries: List[str],
    uid: List[str],
    vllm_servers: List[Dict],
    threshold: float = 0.5
) -> Dict[str, List[List[int]]]:
    """
    Async partition computation using VLLM servers.
    From partition_reward_vllm_serve.py:partition_async()

    Args:
        queries: List of query strings
        uid: List of user IDs
        vllm_servers: List of {"url": ..., "model": ...}
        threshold: Similarity threshold

    Returns:
        Dict mapping uid -> list of partition groups (each group is list of indices)
    """
    # Group by UID
    uid_to_indices = defaultdict(list)
    for idx, u in enumerate(uid):
        uid_to_indices[u].append(idx)

    # Assign each UID to a server (round-robin)
    uid_to_server = {}
    for i, u in enumerate(uid_to_indices.keys()):
        uid_to_server[u] = vllm_servers[hash(u) % len(vllm_servers)]

    # Create HTTP session
    async with aiohttp.ClientSession() as session:
        # Process each UID group
        uid_partitions = {}

        for u, indices in uid_to_indices.items():
            if len(indices) <= 1:
                uid_partitions[u] = [[idx] for idx in indices]
                continue

            # Union-Find for this UID
            parent = list(range(len(indices)))

            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            # Async pairwise comparisons
            server_cfg = uid_to_server[u]
            tasks = []

            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    task = classify_pair_vllm(
                        queries[indices[i]],
                        queries[indices[j]],
                        server_cfg,
                        session,
                        threshold
                    )
                    tasks.append((i, j, task))

            # Await all comparisons
            results = await asyncio.gather(*[task for _, _, task in tasks])

            # Union equivalent pairs
            for (i, j, _), is_equivalent in zip(tasks, results):
                if is_equivalent:
                    union(i, j)

            # Build partitions
            partitions = defaultdict(list)
            for i in range(len(indices)):
                root = find(i)
                partitions[root].append(indices[i])

            uid_partitions[u] = list(partitions.values())

    return uid_partitions


async def compute_diversity_vllm_async(
    queries: List[str],
    uid: List[str],
    vllm_servers: List[Dict],
    threshold: float = 0.5
) -> List[float]:
    """
    Compute partition-based diversity using VLLM servers (async).

    Args:
        queries: List of query strings
        uid: List of user IDs
        vllm_servers: List of {"url": ..., "model": ...}
        threshold: Similarity threshold

    Returns:
        List of diversity scores
    """
    # Get partitions
    uid_partitions = await partition_async_vllm(queries, uid, vllm_servers, threshold)

    # Group by UID for total count
    uid_to_indices = defaultdict(list)
    for idx, u in enumerate(uid):
        uid_to_indices[u].append(idx)

    # Calculate diversity scores
    diversity_rewards = [0.0] * len(queries)

    for u, partitions in uid_partitions.items():
        total = len(uid_to_indices[u])

        for partition_group in partitions:
            group_size = len(partition_group)
            distinct_count = total - group_size

            for idx in partition_group:
                diversity_rewards[idx] = distinct_count / (total - 1) if total > 1 else 1.0

    # Apply floor
    diversity_rewards = [max(r, 0.1) for r in diversity_rewards]

    return diversity_rewards


def compute_diversity_vllm(
    queries: List[str],
    uid: List[str],
    vllm_servers: List[Dict],
    threshold: float = 0.5
) -> List[float]:
    """
    Synchronous wrapper for async VLLM diversity computation.
    """
    return asyncio.run(compute_diversity_vllm_async(queries, uid, vllm_servers, threshold))


# ============================================================================
# Main Retrieval Reward Function (Quality × Diversity)
# ============================================================================

def retrieval_reward(
    data_source: List[str],
    solution_str: List[str],  # Rewritten queries
    ground_truth: Optional[List[str]],  # Not used directly
    uid: List[str],  # Query IDs for label lookup
    prompts: List[str],  # Original queries (not used)
    log_probs: Optional[List[float]] = None,  # Not used
    correctness: Optional[List[bool]] = None,  # Not used
    # Required FAISS infrastructure
    faiss_db=None,
    reward_manager=None,
    id_mapping: List[int] = None,
    labels_dict: Dict[str, List[int]] = None,
    # Diversity configuration
    diversity_enabled: bool = True,
    diversity_method: str = 'local',  # 'local' or 'vllm'
    deberta_model: str = 'microsoft/deberta-v3-large',
    vllm_servers: List[Dict] = None,
    threshold: float = 0.5,
    device: str = 'cuda',
    **kwargs
) -> List[float]:
    """
    Main retrieval reward function following DARLING's multiplicative combination.

    This function should be registered as:
        reward_model.custom_diversity_function.path=retrieval_rewards.py
        reward_model.custom_diversity_function.name=retrieval_reward

    Args:
        data_source: Dataset source names (not used)
        solution_str: Rewritten query strings
        ground_truth: Not used
        uid: Query IDs for looking up ground truth labels
        prompts: Original queries (not used)
        log_probs: Not used
        correctness: Not used
        faiss_db: FaissDB instance (loaded)
        reward_manager: RewardManager instance (configured with method/k)
        id_mapping: List mapping FAISS indices to real doc IDs
        labels_dict: Dict mapping uid -> relevant doc IDs
        diversity_enabled: Whether to compute diversity
        diversity_method: 'local' (DeBERTa) or 'vllm' (server)
        deberta_model: Model name for local diversity
        vllm_servers: Server configs for VLLM diversity
        threshold: Similarity threshold
        device: Device for local models

    Returns:
        List of final rewards: quality × normalized_diversity
    """
    # Validate inputs
    if faiss_db is None:
        raise ValueError("faiss_db must be provided")
    if reward_manager is None:
        raise ValueError("reward_manager must be provided")
    if id_mapping is None:
        raise ValueError("id_mapping must be provided")
    if labels_dict is None:
        raise ValueError("labels_dict must be provided")

    # Step 1: Compute quality rewards
    quality_rewards = compute_quality_rewards(
        queries=solution_str,
        uid=uid,
        faiss_db=faiss_db,
        reward_manager=reward_manager,
        id_mapping=id_mapping,
        labels_dict=labels_dict
    )

    # Step 2: Compute diversity rewards if enabled
    if diversity_enabled:
        if diversity_method == 'local':
            diversity_rewards = compute_diversity_local(
                queries=solution_str,
                uid=uid,
                model_name=deberta_model,
                threshold=threshold,
                device=device
            )
        elif diversity_method == 'vllm':
            if vllm_servers is None:
                raise ValueError("vllm_servers must be provided when diversity_method='vllm'")
            diversity_rewards = compute_diversity_vllm(
                queries=solution_str,
                uid=uid,
                vllm_servers=vllm_servers,
                threshold=threshold
            )
        else:
            raise ValueError(f"Unknown diversity_method: {diversity_method}")
    else:
        # No diversity - use neutral (1.0)
        diversity_rewards = [1.0] * len(solution_str)

    # Step 3: Normalize diversity per UID group (following DARLING pattern)
    uid_to_indices = defaultdict(list)
    for idx, u in enumerate(uid):
        uid_to_indices[u].append(idx)

    norm_diversity = [0.0] * len(diversity_rewards)
    for u, indices in uid_to_indices.items():
        div_vals = [diversity_rewards[i] for i in indices]
        max_div = max(div_vals) if div_vals else 1.0

        for idx in indices:
            norm_diversity[idx] = diversity_rewards[idx] / max_div if max_div > 0 else 1.0

    # Step 4: Multiplicative combination (DARLING pattern)
    final_rewards = [q * d for q, d in zip(quality_rewards, norm_diversity)]

    # Step 5: Apply floor for stability (optional, following partition_reward pattern)
    final_rewards = [max(r, 0.0) for r in final_rewards]

    return final_rewards