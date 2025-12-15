import os
import time
import logging
import random
import string
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

load_dotenv()

class APIClient:
    """
    Client for interacting with LLM API endpoints (OpenAI or other).
    Mimics eqbench usage: we have 'test' vs 'judge' model_type references.
    """

    def __init__(self, model_type=None, request_timeout=240, max_retries=3, retry_delay=5):
        self.model_type = model_type or "default"

        # Override with hardcoded values
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv('JUDGE_API_KEY', None)

        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", request_timeout))
        self.max_retries = int(os.getenv("MAX_RETRIES", max_retries))
        self.retry_delay = int(os.getenv("RETRY_DELAY", retry_delay))

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.request_timeout,
            max_retries=0  # We'll handle retries manually
        )

        logging.debug(f"Initialized {self.model_type} API client with URL: {base_url}")

    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 8096, include_seed=True, min_p = 0.1) -> str:
        """
        Generic chat-completion style call.  We allow an optional random seed block.
        """
        messages = [{"role": "user", "content": prompt}]

        # Optionally add random seed block as a system message for judging tasks.
        # This allows us to get variation between iterations without using temp > 0 which compromises judging performance.
        # The reason for doing this is to understand *judging* variance from the same inputs, i.e. when
        # using --redo-judging. In most use cases you won't need to worry about this and can leave it disabled.
        if False:
            if include_seed:
                seed_lines = [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=80)) for _ in range(5)
                ]
                random_seed_block = (
                    "<RANDOM SEED PLEASE IGNORE>\n" +
                    "\n".join(seed_lines) +
                    "\n</RANDOM SEED>"
                )
                messages = [{"role": "system", "content": random_seed_block}] + messages

        for attempt in range(self.max_retries):
            try:
                # Build kwargs for the API call
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }

                # Handle model-specific parameters
                if model == 'o3':
                    # o3 has special requirements via the openai api
                    kwargs['max_completion_tokens'] = max_tokens
                    kwargs['temperature'] = 1
                else:
                    kwargs['max_tokens'] = max_tokens

                    # Only use min_p for non-o3 models if specified
                    if min_p is not None:
                        # Note: min_p may not be supported by all models
                        # If your test model doesn't support min_p, you may need to
                        # disable this or use a provider like openrouter
                        kwargs['min_p'] = min_p

                # Make the API call using OpenAI client
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                # Strip out any <think> blocks if the model yields that
                if '<think>' in content and "</think>" in content:
                    post_think = content.find('</think>') + len("</think>")
                    content = content[post_think:]
                if '<reasoning>' in content and "</reasoning>" in content:
                    post_think = content.find('</reasoning>') + len("</reasoning>")
                    content = content[post_think:]

                return content

            except APITimeoutError:
                logging.warning(f"Request timed out on attempt {attempt+1}/{self.max_retries}")
            except RateLimitError as e:
                logging.warning(f"Rate limit hit on attempt {attempt+1}/{self.max_retries}. Backing off.")
                logging.error(e)
                time.sleep(self.retry_delay * (attempt + 1))
                continue
            except APIError as e:
                logging.error(f"API error on attempt {attempt+1}/{self.max_retries}: {str(e)}")
            except Exception as e:
                logging.error(f"Unexpected error on attempt {attempt+1}/{self.max_retries}: {str(e)}")

            time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to generate text after {self.max_retries} attempts")
