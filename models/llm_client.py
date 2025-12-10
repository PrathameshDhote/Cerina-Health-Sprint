"""
Unified LLM client interface supporting multiple providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
import tiktoken

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from config import settings
from utils.logger import logger


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger
    
    @abstractmethod
    def invoke(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> LLMResponse:
        """
        Invoke the LLM with messages.
        
        Args:
            messages: List of messages
            **kwargs: Additional arguments
            
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
    def stream(
        self,
        messages: List[BaseMessage],
        **kwargs
    ):
        """
        Stream responses from the LLM.
        
        Args:
            messages: List of messages
            **kwargs: Additional arguments
            
        Yields:
            Response chunks
        """
        pass


class OpenAIClient(LLMClient):
    """OpenAI LLM client implementation."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model: Model identifier (default from settings)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: OpenAI API key (default from settings)
        """
        model = model or settings.openai_model
        super().__init__(model, temperature, max_tokens)
        
        self.api_key = api_key or settings.openai_api_key
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize LangChain client
        self.client = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key
        )
        
        self.logger.info(f"OpenAI client initialized with model: {self.model}")
    
    def invoke(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> LLMResponse:
        """
        Invoke OpenAI model.
        
        Args:
            messages: List of messages
            **kwargs: Additional arguments
            
        Returns:
            LLMResponse object
        """
        try:
            # Override temperature if provided
            temp = kwargs.pop('temperature', self.temperature)
            max_tok = kwargs.pop('max_tokens', self.max_tokens)
            
            # Create temporary client with custom params if needed
            if temp != self.temperature or max_tok != self.max_tokens:
                client = ChatOpenAI(
                    model=self.model,
                    temperature=temp,
                    max_tokens=max_tok,
                    api_key=self.api_key
                )
            else:
                client = self.client
            
            # Invoke
            response = client.invoke(messages, **kwargs)
            
            # Extract token usage if available
            tokens_used = None
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                tokens_used = usage.get('total_tokens')
            
            return LLMResponse(
                content=response.content,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=getattr(response, 'finish_reason', None),
                metadata=getattr(response, 'response_metadata', {})
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI invocation error: {str(e)}")
            raise
    
    def stream(
        self,
        messages: List[BaseMessage],
        **kwargs
    ):
        """
        Stream responses from OpenAI.
        
        Args:
            messages: List of messages
            **kwargs: Additional arguments
            
        Yields:
            Response chunks
        """
        try:
            for chunk in self.client.stream(messages, **kwargs):
                yield chunk.content
                
        except Exception as e:
            self.logger.error(f"OpenAI streaming error: {str(e)}")
            raise


class AnthropicClient(LLMClient):
    """Anthropic (Claude) LLM client implementation."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        """
        Initialize Anthropic client.
        
        Args:
            model: Model identifier (default from settings)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: Anthropic API key (default from settings)
        """
        model = model or settings.anthropic_model
        super().__init__(model, temperature, max_tokens)
        
        self.api_key = api_key or settings.anthropic_api_key
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        
        # Initialize LangChain client
        self.client = ChatAnthropic(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key
        )
        
        self.logger.info(f"Anthropic client initialized with model: {self.model}")
    
    def invoke(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> LLMResponse:
        """
        Invoke Anthropic model.
        
        Args:
            messages: List of messages
            **kwargs: Additional arguments
            
        Returns:
            LLMResponse object
        """
        try:
            # Override temperature if provided
            temp = kwargs.pop('temperature', self.temperature)
            max_tok = kwargs.pop('max_tokens', self.max_tokens)
            
            # Create temporary client with custom params if needed
            if temp != self.temperature or max_tok != self.max_tokens:
                client = ChatAnthropic(
                    model=self.model,
                    temperature=temp,
                    max_tokens=max_tok,
                    api_key=self.api_key
                )
            else:
                client = self.client
            
            # Invoke
            response = client.invoke(messages, **kwargs)
            
            # Extract token usage if available
            tokens_used = None
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('usage', {})
                tokens_used = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
            
            return LLMResponse(
                content=response.content,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=None,
                metadata=getattr(response, 'response_metadata', {})
            )
            
        except Exception as e:
            self.logger.error(f"Anthropic invocation error: {str(e)}")
            raise
    
    def stream(
        self,
        messages: List[BaseMessage],
        **kwargs
    ):
        """
        Stream responses from Anthropic.
        
        Args:
            messages: List of messages
            **kwargs: Additional arguments
            
        Yields:
            Response chunks
        """
        try:
            for chunk in self.client.stream(messages, **kwargs):
                yield chunk.content
                
        except Exception as e:
            self.logger.error(f"Anthropic streaming error: {str(e)}")
            raise


# Client Factory

def get_llm_client(
    provider: Optional[Literal["openai", "anthropic"]] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> LLMClient:
    """
    Get an LLM client based on provider.
    
    Args:
        provider: LLM provider (defaults to settings)
        model: Model identifier (defaults to settings)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        LLMClient instance
        
    Raises:
        ValueError: If provider is unsupported
    """
    provider = provider or settings.primary_llm_provider
    
    if provider == "openai":
        return OpenAIClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider == "anthropic":
        return AnthropicClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# Token Counting Utilities

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text for a given model.
    
    Args:
        text: Text to count tokens for
        model: Model identifier
        
    Returns:
        Token count
    """
    try:
        # Map model names to encoding
        encoding_map = {
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "claude": "cl100k_base",  # Approximation
        }
        
        # Get encoding name
        encoding_name = encoding_map.get(model, "cl100k_base")
        
        # Get encoding
        encoding = tiktoken.get_encoding(encoding_name)
        
        # Count tokens
        tokens = encoding.encode(text)
        return len(tokens)
        
    except Exception as e:
        logger.warning(f"Token counting failed: {str(e)}. Returning character count / 4 as approximation.")
        return len(text) // 4


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4-turbo"
) -> float:
    """
    Estimate cost for API call.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model identifier
        
    Returns:
        Estimated cost in USD
    """
    # Pricing as of December 2024 (approximate)
    pricing = {
        "gpt-4-turbo": {
            "input": 0.01 / 1000,   # $0.01 per 1K input tokens
            "output": 0.03 / 1000,  # $0.03 per 1K output tokens
        },
        "gpt-4": {
            "input": 0.03 / 1000,
            "output": 0.06 / 1000,
        },
        "gpt-3.5-turbo": {
            "input": 0.0005 / 1000,
            "output": 0.0015 / 1000,
        },
        "claude-3-5-sonnet": {
            "input": 0.003 / 1000,
            "output": 0.015 / 1000,
        },
        "claude-3-opus": {
            "input": 0.015 / 1000,
            "output": 0.075 / 1000,
        },
    }
    
    # Get pricing for model
    model_pricing = pricing.get(model, pricing["gpt-4-turbo"])
    
    # Calculate cost
    input_cost = input_tokens * model_pricing["input"]
    output_cost = output_tokens * model_pricing["output"]
    
    return input_cost + output_cost


# Usage tracking

class TokenUsageTracker:
    """Track token usage across requests."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.requests_by_agent = {}
    
    def record_usage(
        self,
        agent: str,
        input_tokens: int,
        output_tokens: int
    ):
        """
        Record token usage for an agent.
        
        Args:
            agent: Agent identifier
            input_tokens: Input tokens used
            output_tokens: Output tokens used
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        
        if agent not in self.requests_by_agent:
            self.requests_by_agent[agent] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "requests": 0
            }
        
        self.requests_by_agent[agent]["input_tokens"] += input_tokens
        self.requests_by_agent[agent]["output_tokens"] += output_tokens
        self.requests_by_agent[agent]["requests"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary of usage stats
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_requests": self.total_requests,
            "by_agent": self.requests_by_agent,
            "estimated_cost": estimate_cost(
                self.total_input_tokens,
                self.total_output_tokens,
                settings.openai_model if settings.primary_llm_provider == "openai" else settings.anthropic_model
            )
        }
    
    def reset(self):
        """Reset all counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.requests_by_agent = {}


# Global tracker instance
usage_tracker = TokenUsageTracker()
