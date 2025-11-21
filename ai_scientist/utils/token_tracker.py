from functools import wraps
from typing import Dict, Optional, List
from collections import defaultdict
import asyncio
from datetime import datetime
import logging


class TokenTracker:
    def __init__(self):
        """
        Token counts for prompt, completion, reasoning, and cached.
        Reasoning tokens are included in completion tokens.
        Cached tokens are included in prompt tokens.
        Also tracks prompts, responses, and timestamps.
        We assume we get these from the LLM response, and we don't count
        the tokens by ourselves.
        """
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)

        self.MODEL_PRICES = {
            "gpt-4o-2024-11-20": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-08-06": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-05-13": {  # this ver does not support cached tokens
                "prompt": 5.0 / 1000000,  # $5.00 per 1M tokens
                "completion": 15 / 1000000,  # $15.00 per 1M tokens
            },
            "gpt-4o-mini-2024-07-18": {
                "prompt": 0.15 / 1000000,  # $0.15 per 1M tokens
                "cached": 0.075 / 1000000,  # $0.075 per 1M tokens
                "completion": 0.6 / 1000000,  # $0.60 per 1M tokens
            },
            "gpt-5.1-2025-11-13": {
                "prompt": 1.25 / 1000000,  # $3.00 per 1M tokens
                "cached": 0.125 / 1000000,  # $1.50 per 1M tokens
                "completion": 10.0 / 1000000,  # $12.00 per 1M tokens
                },
            "gpt-5.1": {
                "prompt": 1.25 / 1000000,  # $3.00 per 1M tokens
                "cached": 0.125 / 1000000,  # $1.50 per 1M tokens
                "completion": 10.0 / 1000000,  # $12.00 per 1M tokens
                },
            "gpt-5-mini-2025-08-07": {
                "prompt": 0.25 / 1000000,  # $1.20 per 1M tokens
                "cached": 0.025 / 1000000,  # $0.60 per 1M tokens
                "completion": 2.0 / 1000000,  # $4.80 per 1M tokens
            },
            "gpt-5-mini": {
                "prompt": 0.25 / 1000000,  # $1.20 per 1M tokens
                "cached": 0.025 / 1000000,  # $0.60 per 1M tokens
                "completion": 2.0 / 1000000,  # $4.80 per 1M tokens
            },
            "gpt-5-nano-2025-08-07": {
                "prompt": 0.05 / 1000000,  # $0.50 per 1M tokens
                "cached": 0.005 / 1000000,  # $0.25 per 1M tokens
                "completion": 0.4 / 1000000,  # $2.00 per 1M tokens
            }
        }

    def add_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ):
        self.token_counts[model]["prompt"] += prompt_tokens
        self.token_counts[model]["completion"] += completion_tokens
        self.token_counts[model]["reasoning"] += reasoning_tokens
        self.token_counts[model]["cached"] += cached_tokens

    def add_interaction(
        self,
        model: str,
        system_message: str,
        prompt: str,
        response: str,
        timestamp: datetime,
    ):
        """Record a single interaction with the model."""
        self.interactions[model].append(
            {
                "system_message": system_message,
                "prompt": prompt,
                "response": response,
                "timestamp": timestamp,
            }
        )

    def get_interactions(self, model: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get all interactions, optionally filtered by model."""
        if model:
            return {model: self.interactions[model]}
        return dict(self.interactions)

    def reset(self):
        """Reset all token counts and interactions."""
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)
        # self._encoders = {}

    def calculate_cost(self, model: str) -> float:
        """Calculate the cost for a specific model based on token usage."""
        if model not in self.MODEL_PRICES:
            logging.warning(f"Price information not available for model {model}")
            return 0.0

        prices = self.MODEL_PRICES[model]
        tokens = self.token_counts[model]

        # Calculate cost for prompt and completion tokens
        if "cached" in prices:
            prompt_cost = (tokens["prompt"] - tokens["cached"]) * prices["prompt"]
            cached_cost = tokens["cached"] * prices["cached"]
        else:
            prompt_cost = tokens["prompt"] * prices["prompt"]
            cached_cost = 0
        completion_cost = tokens["completion"] * prices["completion"]

        return prompt_cost + cached_cost + completion_cost

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        # return dict(self.token_counts)
        """Get summary of token usage and costs for all models."""
        summary = {}
        for model, tokens in self.token_counts.items():
            summary[model] = {
                "tokens": tokens.copy(),
                "cost (USD)": self.calculate_cost(model),
            }
        return summary


# Global token tracker instance
token_tracker = TokenTracker()


def track_token_usage(func):
    def _extract_usage(result):
        """Normalize usage fields from Chat Completions or Responses objects."""
        usage = getattr(result, "usage", None)
        if usage is None:
            return None
        # Responses API naming
        if hasattr(usage, "input_tokens") or hasattr(usage, "output_tokens"):
            prompt_tokens = getattr(usage, "input_tokens", 0)
            completion_tokens = getattr(usage, "output_tokens", 0)
            reasoning_tokens = (
                getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0)
            )
            cached_tokens = getattr(
                getattr(usage, "input_tokens_details", None), "cached_tokens", 0
            )
            return prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens
        # Legacy Chat Completions naming
        if hasattr(usage, "prompt_tokens") or hasattr(usage, "completion_tokens"):
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            reasoning_tokens = getattr(
                getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0
            )
            cached_tokens = getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
            )
            return prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens
        return None

    def _extract_text(result):
        """Pull assistant text from either Responses or ChatCompletion payloads."""
        if hasattr(result, "choices"):
            choice = result.choices[0] if result.choices else None
            if choice and hasattr(choice, "message"):
                return choice.message.content
        if hasattr(result, "output"):
            for item in getattr(result, "output", []):
                if getattr(item, "type", None) == "message":
                    for content_item in getattr(item, "content", []):
                        if getattr(content_item, "type", None) == "output_text":
                            return content_item.text
                elif getattr(item, "type", None) == "output_text":
                    return getattr(item, "text", None)
        return None

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt")
        system_message = kwargs.get("system_message")
        if not prompt and not system_message:
            raise ValueError(
                "Either 'prompt' or 'system_message' must be provided for token tracking"
            )

        logging.info("args: ", args)
        logging.info("kwargs: ", kwargs)

        result = await func(*args, **kwargs)
        model = getattr(result, "model", None)
        timestamp = getattr(result, "created", None) or getattr(result, "created_at", None)

        if not model or not timestamp:
            return result

        usage_values = _extract_usage(result)
        if usage_values:
            prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens = usage_values
            token_tracker.add_tokens(
                model,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                cached_tokens,
            )
            # Add interaction details
            token_tracker.add_interaction(
                model,
                system_message,
                prompt,
                _extract_text(result),
                timestamp,
            )
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt")
        system_message = kwargs.get("system_message")
        if not prompt and not system_message:
            raise ValueError(
                "Either 'prompt' or 'system_message' must be provided for token tracking"
            )
        result = func(*args, **kwargs)
        model = getattr(result, "model", None)
        timestamp = getattr(result, "created", None) or getattr(result, "created_at", None)
        logging.info("args: ", args)
        logging.info("kwargs: ", kwargs)

        if model and timestamp:
            usage_values = _extract_usage(result)
            if usage_values:
                prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens = usage_values
                token_tracker.add_tokens(
                    model,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    cached_tokens,
                )
                # Add interaction details
                token_tracker.add_interaction(
                    model,
                    system_message,
                    prompt,
                    _extract_text(result),
                    timestamp,
                )
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
