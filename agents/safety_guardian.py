"""
Safety Guardian Agent - Validates content for safety and liability risks.
"""

from datetime import datetime
from typing import Any, List, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from config import settings
from utils.logger import logger
from models.prompts import SAFETY_GUARDIAN_SYSTEM_PROMPT, get_safety_user_prompt
from .base_agent import BaseAgent, AgentResponse


class SafetyGuardianAgent(BaseAgent):
    """
    Safety Guardian Agent responsible for identifying risks and safety concerns.
    
    Specializes in:
    - Self-harm and suicide risk detection
    - Inappropriate medical advice identification
    - Liability concerns
    - Emergency protocol validation
    - Contraindications detection
    """
    
    def __init__(self):
        super().__init__(
            name="Safety_Guardian",
            role="Risk Assessment and Safety Validation",
            temperature=settings.safety_temperature,  # Lower temp for consistency
            max_tokens=1500
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
        
        self.logger.info("Safety Guardian initialized")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the safety guardian agent."""
        return SAFETY_GUARDIAN_SYSTEM_PROMPT
    
    def process(self, state: Any) -> AgentResponse:
        """
        Analyze the current draft for safety concerns.
        
        Args:
            state: Current protocol state
            
        Returns:
            AgentResponse with safety assessment
        """
        self._log_action("Starting safety validation", {
            "draft_length": len(state.current_draft) if state.current_draft else 0,
            "iteration": state.iteration_count
        })
        
        try:
            # Create user prompt using centralized function
            user_prompt = get_safety_user_prompt(
                user_intent=state.user_intent,
                draft=state.current_draft
            )
            
            # Analyze draft for safety issues
            messages = [
                SystemMessage(content=self.get_system_prompt()),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            safety_assessment = response.content
            
            # Parse assessment
            parsed_assessment = self._parse_assessment(safety_assessment)
            
            # Determine flags
            flags = self._extract_flags(parsed_assessment)
            
            # Create response
            agent_response = self._create_response(
                content=safety_assessment,
                reasoning=f"Conducted safety review of draft. Found {len(flags)} concerns.",
                confidence=parsed_assessment.get("confidence", 0.8),
                flags=flags,
                suggestions=parsed_assessment.get("recommendations", []),
                metadata={
                    "safety_rating": parsed_assessment.get("rating", "NEEDS_REVIEW"),
                    "high_severity_issues": sum(1 for f in flags if "HIGH" in f),
                    "iteration": state.iteration_count
                }
            )
            
            self._log_action("Safety validation completed", {
                "rating": parsed_assessment.get("rating"),
                "flags_count": len(flags)
            })
            
            return agent_response
            
        except Exception as e:
            self.logger.error(f"Error in safety validation: {str(e)}")
            raise
    
    def _parse_assessment(self, assessment: str) -> Dict[str, Any]:
        """Parse the safety assessment into structured format."""
        parsed = {
            "rating": "NEEDS_REVIEW",
            "confidence": 0.7,
            "issues": [],
            "recommendations": []
        }
        
        assessment_lower = assessment.lower()
        
        # Extract rating
        if "overall safety rating" in assessment_lower:
            if "safe" in assessment_lower and "unsafe" not in assessment_lower:
                parsed["rating"] = "SAFE"
            elif "unsafe" in assessment_lower:
                parsed["rating"] = "UNSAFE"
            else:
                parsed["rating"] = "NEEDS_REVISION"
        
        # Extract issues (look for HIGH/MEDIUM/LOW markers)
        lines = assessment.split("\n")
        for line in lines:
            line_lower = line.lower()
            if any(severity in line_lower for severity in ["high", "medium", "low"]):
                if any(keyword in line_lower for keyword in ["issue", "concern", "risk", "problem"]):
                    parsed["issues"].append(line.strip())
        
        # Extract recommendations
        in_recommendations = False
        for line in lines:
            if "recommendation" in line.lower():
                in_recommendations = True
                continue
            if in_recommendations and line.strip() and line.strip().startswith(("-", "•", "*")):
                parsed["recommendations"].append(line.strip().lstrip("-•* "))
        
        # Extract confidence if mentioned
        if "confidence" in assessment_lower:
            import re
            confidence_match = re.search(r'confidence[:\s]+(0?\.\d+|\d+(?:\.\d+)?)', assessment_lower)
            if confidence_match:
                try:
                    parsed["confidence"] = float(confidence_match.group(1))
                    if parsed["confidence"] > 1.0:
                        parsed["confidence"] = parsed["confidence"] / 100.0
                except:
                    pass
        
        return parsed
    
    def _extract_flags(self, parsed_assessment: Dict[str, Any]) -> List[str]:
        """Extract safety flags from parsed assessment."""
        flags = []
        
        # Add rating-based flag
        rating = parsed_assessment.get("rating", "NEEDS_REVIEW")
        if rating != "SAFE":
            flags.append(f"SAFETY_RATING: {rating}")
        
        # Add issue-based flags
        for issue in parsed_assessment.get("issues", []):
            flags.append(issue)
        
        return flags
