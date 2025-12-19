from abc import ABC, abstractmethod


class BasePromptExtender(ABC):
    @abstractmethod
    def extend_prompt(self, prompt: str) -> str:
        """Extend the given prompt with additional information.

        Args:
            prompt (str): The original prompt to be extended.

        Returns:
            str: The extended prompt.
        """
        pass

    def __call__(self, prompt: str) -> str:
        return self.extend_prompt(prompt)