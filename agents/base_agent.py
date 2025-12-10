"""
Base agent class providing common functionality for all agents.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel

from config import settings
from utils.logger import logger


class AgentResponse(BaseModel):
    """Structured response from an agent."""
    
    agent_name: str
    content: str
    reasoning: str
    confidence: float  # 0.0 to 1.0
    suggestions: list[str] = []
    flags: list[str] = []
    metadata: Dict[str, Any] = {}
    timestamp: datetime = datetime.now()


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Provides common functionality:
    - LLM client management
    - Logging and monitoring
    - Response structuring
    - Error handling
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent identifier
            role: Agent's role description
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.name = name
        self.role = role
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger
        
        self.logger.info(f"Initialized {self.name} with role: {self.role}")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        Must be implemented by subclasses.
        
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    def process(self, state: Any) -> AgentResponse:
        """
        Process the current state and return agent response.
        Must be implemented by subclasses.
        
        Args:
            state: Current protocol state
            
        Returns:
            AgentResponse with results
        """
        pass
    
    def _log_action(self, action: str, details: Optional[Dict[str, Any]] = None):
        """
        Log agent action for monitoring and debugging.
        
        Args:
            action: Action description
            details: Additional details to log
        """
        log_data = {
            "agent": self.name,
            "action": action,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            log_data.update(details)
        
        self.logger.info(f"[{self.name}] {action}", extra=log_data)
    
    def _create_response(
        self,
        content: str,
        reasoning: str,
        confidence: float,
        suggestions: list[str] = None,
        flags: list[str] = None,
        metadata: Dict[str, Any] = None
    ) -> AgentResponse:
        """
        Create a structured agent response.
        
        Args:
            content: Main response content
            reasoning: Explanation of decision
            confidence: Confidence score (0.0-1.0)
            suggestions: List of improvement suggestions
            flags: List of issues or concerns
            metadata: Additional metadata
            
        Returns:
            Structured AgentResponse
        """
        return AgentResponse(
            agent_name=self.name,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            suggestions=suggestions or [],
            flags=flags or [],
            metadata=metadata or {},
            timestamp=datetime.now()
        )
