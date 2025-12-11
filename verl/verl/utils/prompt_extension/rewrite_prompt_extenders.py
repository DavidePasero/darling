from verl.utils.prompt_extension.base_prompt_extension import BasePromptExtender


class RewritePromptExtender(BasePromptExtender):
    _REWRITE_SYSTEM_PROMPT = "You are a query rewriting assistant. Your task is to rewrite user queries to make them more effective for document retrieval."
    _REWRITE_USER_PROMPT = "Rewrite the following query:"
    _NAME = "rewrite"
    
    def extend_prompt(self, prompt: str) -> str:
        return [
            {"role": "system", "content": self._REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": self._REWRITE_USER_PROMPT + "\n" + prompt}
        ]

    def __repr__(self) -> str:
        return f"RewritePromptExtender: {self._REWRITE_SYSTEM_PROMPT}\n{self._REWRITE_USER_PROMPT}"

class SemanticRewritePromptExtender(BasePromptExtender):
    _NAME = "semantic_rewrite"

    _REWRITE_SYSTEM_PROMPT = (
        "You are an expert in semantic query reformulation. Enhance the query using "
        "meaningful concepts and synonyms. Consider generating several candidates "
        "if helpful. Only return the rewritten queries with "
        "no explanations, no lists, and no additional text. Be creative and add menaingful extensions.."
    )
    _REWRITE_USER_PROMPT = "Rewrite the following query in a semantically enriched way. Only return the extended query. The existing query is:"

    _NAME = "semantic"

    def extend_prompt(self, prompt: str) -> str:
        return [
            {"role": "system", "content": self._REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": self._REWRITE_USER_PROMPT + "\n" + prompt},
        ]

    def __repr__(self) -> str:
        return f"SemanticRewritePromptExtender: {self._REWRITE_SYSTEM_PROMPT}\n{self._REWRITE_USER_PROMPT}"


class MinimalRewritePromptExtender(BasePromptExtender):
    _NAME = "minimal_rewrite"

    _REWRITE_SYSTEM_PROMPT = (
        "Rewrite the user's query. ONLY return the rewritten query—no lists, no reasoning."
    )
    _REWRITE_USER_PROMPT = "Query:"
    _NAME = "minimal"

    def extend_prompt(self, prompt: str) -> str:
        return [
            {"role": "system", "content": self._REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": self._REWRITE_USER_PROMPT + " " + prompt},
        ]

    def __repr__(self) -> str:
        return f"MinimalRewritePromptExtender: {self._REWRITE_SYSTEM_PROMPT}\n{self._REWRITE_USER_PROMPT}"


class ContextAwareRewritePromptExtender(BasePromptExtender):
    REWRITE_SYSTEM_PROMPT = (
        "Rewrite the query to be clearer, more specific, and more effective for retrieval. "
        "If the input is complex, you may consider multiple rewrite options."
        "Only return the rewritten queries with "
        "no explanations, no lists, and no additional text. Be creative and add menaingful extensions."    )
    _REWRITE_USER_PROMPT = "Rewrite the following query in a more context-aware way:"
    _NAME = "context_aware"

    def extend_prompt(self, prompt: str) -> str:
        return [
            {"role": "system", "content": self._REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": self._REWRITE_USER_PROMPT + "\n" + prompt},
        ]

    def __repr__(self) -> str:
        return f"ContextAwareRewritePromptExtender: {self._REWRITE_SYSTEM_PROMPT}\n{self._REWRITE_USER_PROMPT}"

class BM25RewritePromptExtender(BasePromptExtender):
    _REWRITE_SYSTEM_PROMPT = (
        "Rewrite the query optimized for BM25 retrieval. Produce a keyword-rich version "
        "with relevant synonyms. You may internally generate several alternatives if the "
        "query is long. Only return the rewritten queries with "
        "no explanations, no lists, and no additional text. Be creative and add menaingful extensions."
    )
    _REWRITE_USER_PROMPT = (
        "Rewrite the following query as BM25-optimized keywords:"
    )
    _NAME = "bm25"

    def extend_prompt(self, prompt: str) -> str:
        return [
            {"role": "system", "content": self._REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": self._REWRITE_USER_PROMPT + "\n" + prompt},
        ]

    def __repr__(self) -> str:
        return f"BM25RewritePromptExtender: {self._REWRITE_SYSTEM_PROMPT}\n{self._REWRITE_USER_PROMPT}"

class DenseVectorRewritePromptExtender(BasePromptExtender):
    _REWRITE_SYSTEM_PROMPT = (
        "Rewrite the query for dense vector semantic search. Make it explicit, unambiguous, "
        "and semantically rich. If helpful, consider multiple internal rewrites. Only return the rewritten queries with "
        "no explanations, no lists, and no additional text. Be creative and add menaingful extensions."
    )
    _REWRITE_USER_PROMPT = (
        "Rewrite the following query for improved semantic embedding quality:"
    )
    _NAME = "dense_vector"

    def extend_prompt(self, prompt: str) -> str:
        return [
            {"role": "system", "content": self._REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": self._REWRITE_USER_PROMPT + "\n" + prompt},
        ]

    def __repr__(self) -> str:
        return f"DenseVectorRewritePromptExtender: {self._REWRITE_SYSTEM_PROMPT}\n{self._REWRITE_USER_PROMPT}"


class NoOpPromptExtender(BasePromptExtender):
    _NAME = "noop_rewrite"

    _REWRITE_SYSTEM_PROMPT = (
        "Return the query exactly as given. Do not modify it. ONLY return the query text."
    )
    _REWRITE_USER_PROMPT = "Query:"

    def extend_prompt(self, prompt: str):
        return [
            {"role": "system", "content": self._REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": self._REWRITE_USER_PROMPT + " " + prompt},
        ]
