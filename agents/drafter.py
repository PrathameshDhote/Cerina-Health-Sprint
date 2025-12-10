"""
CBT Drafter Agent - Primary content generator for CBT protocols.
"""

from datetime import datetime
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from config import settings
from utils.logger import logger
from models.prompts import DRAFTER_SYSTEM_PROMPT, get_drafter_user_prompt
from .base_agent import BaseAgent, AgentResponse


class CBTDrafterAgent(BaseAgent):
    """
    CBT Drafter Agent responsible for generating therapeutic exercise content.
    
    Specializes in:
    - Creating structured CBT exercises
    - Exposure hierarchy design
    - Cognitive reframing techniques
    - Homework assignments
    - Safety considerations
    """
    
    def __init__(self):
        super().__init__(
            name="CBT_Drafter",
            role="Clinical Content Generator",
            temperature=settings.drafter_temperature,
            max_tokens=3000
        )
        
        # Initialize LLM client
        if settings.primary_llm_provider == "openai":
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=settings.openai_api_key
            )
        else:
            self.llm = ChatAnthropic(
                model=settings.anthropic_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=settings.anthropic_api_key
            )
        
        self.logger.info(f"CBT Drafter initialized with {settings.primary_llm_provider}")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the drafter agent."""
        return DRAFTER_SYSTEM_PROMPT
    
    def process(self, state: Any) -> AgentResponse:
        """
        Generate CBT protocol content based on user intent and previous feedback.
        
        Args:
            state: Current protocol state
            
        Returns:
            AgentResponse with generated protocol draft
        """
        self._log_action("Starting draft generation", {
            "user_intent": state.user_intent,
            "iteration": state.iteration_count
        })
        
        try:
            # Build context from previous iterations
            feedback_context = state.get_context_for_revision()
            
            # Create user prompt using centralized function
            user_prompt = get_drafter_user_prompt(
                user_intent=state.user_intent,
                current_draft=state.current_draft if state.iteration_count > 0 else None,
                feedback_context=feedback_context if feedback_context else None,
                iteration=state.iteration_count
            )
            
            # Generate content
            messages = [
                SystemMessage(content=self.get_system_prompt()),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            draft_content = response.content
            
            # Evaluate quality of generated content
            confidence = self._evaluate_draft_quality(draft_content)
            
            # Create structured response
            agent_response = self._create_response(
                content=draft_content,
                reasoning=f"Generated CBT protocol for: {state.user_intent}. Iteration {state.iteration_count + 1}.",
                confidence=confidence,
                suggestions=[],
                metadata={
                    "word_count": len(draft_content.split()),
                    "has_structure": self._check_structure(draft_content),
                    "iteration": state.iteration_count + 1
                }
            )
            
            self._log_action("Draft generation completed", {
                "confidence": confidence,
                "word_count": len(draft_content.split())
            })
            
            return agent_response
            
        except Exception as e:
            self.logger.error(f"Error in draft generation: {str(e)}")
            raise
    
    def _evaluate_draft_quality(self, draft: str) -> float:
        """
        Evaluate the quality of generated draft.
        
        Args:
            draft: Generated draft content
            
        Returns:
            Confidence score (0.0-1.0)
        """
        score = 0.0
        
        # Check for required sections
        required_sections = [
            "session overview",
            "exposure",
            "cognitive",
            "homework",
            "safety"
        ]
        
        draft_lower = draft.lower()
        sections_present = sum(1 for section in required_sections if section in draft_lower)
        score += (sections_present / len(required_sections)) * 0.5
        
        # Check length (should be substantial)
        word_count = len(draft.split())
        if word_count >= 400:
            score += 0.3
        elif word_count >= 200:
            score += 0.15
        
        # Check for empathetic language
        empathy_indicators = [
            "understand", "normal", "common", "challenging", 
            "progress", "compassion", "acknowledge", "gentle"
        ]
        empathy_count = sum(1 for word in empathy_indicators if word in draft_lower)
        score += min(empathy_count / 5, 0.2)
        
        return min(score, 1.0)
    
    def _check_structure(self, draft: str) -> bool:
        """
        Check if draft has proper structure with headers.
        
        Args:
            draft: Draft content
            
        Returns:
            True if structured properly
        """
        # Look for markdown headers or numbered sections
        has_headers = any(line.startswith("#") or line.startswith("##") for line in draft.split("\n"))
        has_numbers = any(line.strip().startswith(("1.", "2.", "3.")) for line in draft.split("\n"))
        
        return has_headers or has_numbers
