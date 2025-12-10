"""
Clinical Critic Agent - Evaluates therapeutic quality and empathy.
"""

from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from config import settings
from utils.logger import logger
from models.prompts import CLINICAL_CRITIC_SYSTEM_PROMPT, get_critic_user_prompt
from .base_agent import BaseAgent, AgentResponse


class ClinicalCriticAgent(BaseAgent):
    """
    Clinical Critic Agent responsible for quality assessment.
    
    Specializes in:
    - Clinical accuracy evaluation
    - Empathy and tone assessment
    - Therapeutic alliance considerations
    - Evidence-based practice validation
    - Accessibility and clarity review
    """
    
    def __init__(self):
        super().__init__(
            name="Clinical_Critic",
            role="Quality Assessment and Clinical Review",
            temperature=settings.critic_temperature,
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
        
        self.logger.info("Clinical Critic initialized")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the clinical critic agent."""
        return CLINICAL_CRITIC_SYSTEM_PROMPT
    
    def process(self, state: Any) -> AgentResponse:
        """
        Evaluate the current draft for clinical quality and empathy.
        
        Args:
            state: Current protocol state
            
        Returns:
            AgentResponse with quality assessment
        """
        self._log_action("Starting clinical quality review", {
            "draft_length": len(state.current_draft) if state.current_draft else 0,
            "iteration": state.iteration_count
        })
        
        try:
            # Create user prompt using centralized function
            user_prompt = get_critic_user_prompt(
                user_intent=state.user_intent,
                draft=state.current_draft
            )
            
            # Perform quality assessment
            messages = [
                SystemMessage(content=self.get_system_prompt()),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            quality_assessment = response.content
            
            # Parse assessment
            parsed_assessment = self._parse_assessment(quality_assessment)
            
            # Extract suggestions
            suggestions = self._extract_suggestions(parsed_assessment)
            
            # Determine if revisions needed
            flags = self._determine_flags(parsed_assessment)
            
            # Create response
            agent_response = self._create_response(
                content=quality_assessment,
                reasoning=f"Clinical quality review completed. Overall score: {parsed_assessment.get('overall_score', 'N/A')}/10",
                confidence=parsed_assessment.get("confidence", 0.85),
                suggestions=suggestions,
                flags=flags,
                metadata={
                    "overall_score": parsed_assessment.get("overall_score", 0),
                    "empathy_score": parsed_assessment.get("empathy_score", 0),
                    "recommendation": parsed_assessment.get("recommendation", "REVIEW"),
                    "individual_scores": parsed_assessment.get("individual_scores", {}),
                    "iteration": state.iteration_count
                }
            )
            
            self._log_action("Clinical review completed", {
                "overall_score": parsed_assessment.get("overall_score"),
                "recommendation": parsed_assessment.get("recommendation")
            })
            
            return agent_response
            
        except Exception as e:
            self.logger.error(f"Error in clinical review: {str(e)}")
            raise
    
    def _parse_assessment(self, assessment: str) -> Dict[str, Any]:
        """Parse the quality assessment into structured format."""
        import re
        
        parsed = {
            "overall_score": 0,
            "empathy_score": 0.0,
            "individual_scores": {},
            "strengths": [],
            "improvements": [],
            "recommendation": "REVIEW",
            "confidence": 0.85
        }
        
        # Extract overall score
        overall_match = re.search(r'overall quality score[:\s]+(\d+(?:\.\d+)?)', assessment.lower())
        if overall_match:
            try:
                parsed["overall_score"] = float(overall_match.group(1))
            except:
                pass
        
        # Extract empathy score
        empathy_match = re.search(r'empathy score[:\s]+(0?\.\d+|\d+(?:\.\d+)?)', assessment.lower())
        if empathy_match:
            try:
                empathy_val = float(empathy_match.group(1))
                parsed["empathy_score"] = empathy_val if empathy_val <= 1.0 else empathy_val / 10.0
            except:
                pass
        
        # Extract individual scores
        criteria = ["clinical accuracy", "empathy & tone", "empathy", "clarity", "therapeutic alliance", "completeness", "engagement"]
        for criterion in criteria:
            criterion_match = re.search(rf'{criterion}[:\s]+(\d+(?:\.\d+)?)', assessment.lower())
            if criterion_match:
                try:
                    parsed["individual_scores"][criterion] = float(criterion_match.group(1))
                except:
                    pass
        
        # Extract recommendation
        if "approve" in assessment.lower() and "request" not in assessment.lower():
            parsed["recommendation"] = "APPROVE"
        elif "minor" in assessment.lower() and "revision" in assessment.lower():
            parsed["recommendation"] = "REQUEST_MINOR_REVISIONS"
        elif "major" in assessment.lower() and "revision" in assessment.lower():
            parsed["recommendation"] = "REQUEST_MAJOR_REVISIONS"
        
        # Extract strengths and improvements
        lines = assessment.split("\n")
        in_strengths = False
        in_improvements = False
        
        for line in lines:
            line_lower = line.lower()
            
            if "strength" in line_lower:
                in_strengths = True
                in_improvements = False
                continue
            elif "improvement" in line_lower or "area for" in line_lower:
                in_improvements = True
                in_strengths = False
                continue
            
            if line.strip() and line.strip().startswith(("-", "•", "*", "1", "2", "3", "4", "5")):
                cleaned_line = line.strip().lstrip("-•*123456789. ")
                if in_strengths and cleaned_line:
                    parsed["strengths"].append(cleaned_line)
                elif in_improvements and cleaned_line:
                    parsed["improvements"].append(cleaned_line)
        
        return parsed
    
    def _extract_suggestions(self, parsed_assessment: Dict[str, Any]) -> List[str]:
        """Extract actionable suggestions from assessment."""
        suggestions = []
        
        # Add improvements as suggestions
        for improvement in parsed_assessment.get("improvements", []):
            suggestions.append(improvement)
        
        # Add score-based suggestions
        individual_scores = parsed_assessment.get("individual_scores", {})
        for criterion, score in individual_scores.items():
            if score < 7:
                suggestions.append(f"Improve {criterion} (current score: {score}/10)")
        
        return suggestions
    
    def _determine_flags(self, parsed_assessment: Dict[str, Any]) -> List[str]:
        """Determine if quality issues require flagging."""
        flags = []
        
        overall_score = parsed_assessment.get("overall_score", 0)
        empathy_score = parsed_assessment.get("empathy_score", 0)
        recommendation = parsed_assessment.get("recommendation", "REVIEW")
        
        # Flag low overall quality
        if overall_score < 6:
            flags.append(f"LOW_QUALITY: Overall score {overall_score}/10")
        
        # Flag low empathy
        if empathy_score < 0.6:
            flags.append(f"LOW_EMPATHY: Empathy score {empathy_score}")
        
        # Flag revision requests
        if recommendation in ["REQUEST_MINOR_REVISIONS", "REQUEST_MAJOR_REVISIONS"]:
            flags.append(f"QUALITY_REVISION_NEEDED: {recommendation}")
        
        # Flag specific criteria issues
        individual_scores = parsed_assessment.get("individual_scores", {})
        for criterion, score in individual_scores.items():
            if score < 5:
                flags.append(f"POOR_{criterion.upper().replace(' ', '_')}: Score {score}/10")
        
        return flags
