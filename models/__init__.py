"""
LLM client management and prompt templates.

This module provides:
- Unified LLM client interfaces for OpenAI and Anthropic
- Prompt templates for each agent
- Model configuration and selection
- Token usage tracking
"""

from .llm_client import (
    get_llm_client,
    LLMClient,
    OpenAIClient,
    AnthropicClient,
    count_tokens
)
from .prompts import (
    DRAFTER_SYSTEM_PROMPT,
    SAFETY_GUARDIAN_SYSTEM_PROMPT,
    CLINICAL_CRITIC_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    get_drafter_user_prompt,
    get_safety_user_prompt,
    get_critic_user_prompt
)

__all__ = [
    "get_llm_client",
    "LLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "count_tokens",
    "DRAFTER_SYSTEM_PROMPT",
    "SAFETY_GUARDIAN_SYSTEM_PROMPT",
    "CLINICAL_CRITIC_SYSTEM_PROMPT",
    "SUPERVISOR_SYSTEM_PROMPT",
    "get_drafter_user_prompt",
    "get_safety_user_prompt",
    "get_critic_user_prompt",
]
