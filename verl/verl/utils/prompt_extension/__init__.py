from .rewrite_prompt_extenders import *

from .base_prompt_extension import BasePromptExtender

EXTENDER_REGISTRY = {}
for extender in dir():
    if extender.endswith("PromptExtender"):
        EXTENDER_REGISTRY[extender._NAME] = extender


__all__ = ["EXTENDER_REGISTRY"]