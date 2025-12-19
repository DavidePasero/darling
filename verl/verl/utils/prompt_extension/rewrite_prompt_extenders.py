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
    _REWRITE_SYSTEM_PROMPT = (
        "You are an expert in semantic query reformulation. "
        "Your task is to enhance user queries by expanding them with meaningful "
        "concepts, entities, and synonyms that improve retrieval performance."
    )
    _REWRITE_USER_PROMPT = (
        "Please provide a semantically enriched rewrite of the following query:"
    )
    _NAME = "semantic"

    def extend_prompt(self, prompt: str) -> str:
        return [
            {"role": "system", "content": self._REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": self._REWRITE_USER_PROMPT + "\n" + prompt},
        ]

    def __repr__(self) -> str:
        return f"SemanticRewritePromptExtender: {self._REWRITE_SYSTEM_PROMPT}\n{self._REWRITE_USER_PROMPT}"


class MinimalRewritePromptExtender(BasePromptExtender):
    _REWRITE_SYSTEM_PROMPT = "Rewrite the user's query."
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
    _REWRITE_SYSTEM_PROMPT = (
        "You are a context-aware query rewriting assistant. "
        "Rewrite user queries so they become clearer, more specific, and more effective "
        "for information retrieval, while strictly preserving the user’s intent."
    )
    _REWRITE_USER_PROMPT = (
        "Rewrite the following query in a clearer and more retrieval-oriented way:"
    )
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
        "You rewrite user queries specifically for BM25 retrieval. "
        "BM25 benefits from keyword-rich queries. "
        "Rewrite the query by extracting the essential keywords, adding useful synonyms, "
        "and removing unnecessary words while preserving intent."
    )
    _REWRITE_USER_PROMPT = (
        "Rewrite the query as BM25-optimized keywords (with optional synonyms):"
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
        "You rewrite queries for dense vector semantic search. "
        "Dense search works best when the query expresses clear intent, context, and meaning. "
        "Rewrite the query to make it more explicit, semantically rich, and unambiguous, "
        "without altering the user's intent."
    )
    _REWRITE_USER_PROMPT = (
        "Rewrite the following query to improve semantic embedding quality:"
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
    _NAME = "no_op"
    
    def extend_prompt(self, prompt: str):
        return [
            {"role": "user", "content": prompt}
        ]

    def __repr__(self) -> str:
        return "NoOpPromptExtender (wraps prompt in a single user message)"
