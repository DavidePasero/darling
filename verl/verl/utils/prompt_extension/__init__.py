from verl.utils.prompt_extension.rewrite_prompt_extenders import *

from verl.utils.prompt_extension.base_prompt_extension import BasePromptExtender

EXTENDER_REGISTRY = {
    RewritePromptExtender._NAME: RewritePromptExtender,
    SemanticRewritePromptExtender._NAME: SemanticRewritePromptExtender,
    MinimalRewritePromptExtender._NAME: MinimalRewritePromptExtender,
    ContextAwareRewritePromptExtender._NAME: ContextAwareRewritePromptExtender,
    NoOpPromptExtender._NAME: NoOpPromptExtender,
    DenseVectorRewritePromptExtender._NAME: DenseVectorRewritePromptExtender,
    BM25RewritePromptExtender._NAME: BM25RewritePromptExtender,
}

__all__ = ["EXTENDER_REGISTRY"]