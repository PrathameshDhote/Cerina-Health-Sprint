"""
Supervisor Agent - Orchestrates workflow and makes routing decisions.
"""

from datetime import datetime
from typing import Any, Literal

from config import settings
from utils.logger import logger
from models.prompts import SUPERVISOR_SYSTEM_PROMPT
from .base_agent import BaseAgent, AgentResponse


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent responsible for workflow orchestration.
    
    Specializes in:
    - Routing decisions (which agent runs next)
    - Iteration management
    - Quality gate evaluation
    - Human-in-loop halt decisions
    - Final approval logic
    """
    
    def __init__(self):
        super().__init__(
            name="Supervisor",
            role="Workflow Orchestrator and Decision Maker",
            temperature=settings.supervisor_temperature,
            max_tokens=1000
        )
        
        self.max_iterations = settings.max_agent_iterations
        self.quality_threshold = 7.5  # Minimum quality score to halt
        self.logger.info(f"Supervisor initialized with max iterations: {self.max_iterations}")
        self.logger.info(f"Quality threshold set to: {self.quality_threshold}")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the supervisor agent."""
        return SUPERVISOR_SYSTEM_PROMPT
    
    def process(self, state: Any) -> AgentResponse:
        """
        Not used - supervisor uses decide_next_action instead.
        
        Supervisor is rule-based and doesn't call LLMs.
        """
        raise NotImplementedError("Supervisor uses decide_next_action method instead of process")
    
    def decide_next_action(
        self, 
        state: Any
    ) -> Literal["run_drafter", "run_safety", "run_critic", "halt_for_human", "finalize", "max_iterations_reached"]:
        """
        Decide the next action in the workflow based on current state.
        
        This implements rule-based routing logic without LLM calls.
        
        Args:
            state: Current protocol state
            
        Returns:
            Next action to take
        """
        self._log_action("Evaluating workflow state", {
            "iteration": state.iteration_count,
            "has_draft": bool(state.current_draft),
            "safety_flags": len(state.scratchpad.get("safety_flags", [])),
            "critic_flags": len(state.scratchpad.get("critic_feedback", []))
        })
        
        # Check max iterations
        if state.iteration_count >= self.max_iterations:
            self._log_action("Max iterations reached", {"iterations": state.iteration_count})
            self._record_decision(state, "max_iterations_reached", "Maximum iteration limit reached")
            return "max_iterations_reached"
        
        # First iteration - always run drafter
        if state.iteration_count == 0:
            self._log_action("First iteration - routing to drafter")
            self._record_decision(state, "run_drafter", "Initial draft generation")
            return "run_drafter"
        
        # If no draft exists, run drafter
        if not state.current_draft:
            self._log_action("No draft exists - routing to drafter")
            self._record_decision(state, "run_drafter", "Generate missing draft")
            return "run_drafter"
        
        # Check if safety validation is needed
        if not self._has_recent_safety_check(state):
            self._log_action("Safety check needed")
            self._record_decision(state, "run_safety", "Safety validation required")
            return "run_safety"
        
        # Check if quality review is needed
        if not self._has_recent_quality_check(state):
            self._log_action("Quality check needed")
            self._record_decision(state, "run_critic", "Quality review required")
            return "run_critic"
        
        # Check for blocking safety issues
        safety_flags = state.scratchpad.get("safety_flags", [])
        if safety_flags:
            latest_flags = [f for f in safety_flags if isinstance(f, dict)]
            if latest_flags:
                latest_flag = latest_flags[-1]
                # Only revise on HIGH severity or UNSAFE rating
                flag_str = str(latest_flag)
                if "UNSAFE" in flag_str or "HIGH" in flag_str:
                    self._log_action("Critical safety issues - requesting revision")
                    self._record_decision(state, "run_drafter", "Address critical safety concerns")
                    return "run_drafter"
        
        # Check for quality issues requiring revision - IMPROVED LOGIC
        critic_feedback = state.scratchpad.get("critic_feedback", [])
        if critic_feedback:
            latest_feedback = [f for f in critic_feedback if isinstance(f, dict)]
            if latest_feedback:
                latest = latest_feedback[-1]
                recommendation = latest.get("recommendation", "")
                overall_score = latest.get("overall_score", 0)
                
                # Log the quality assessment
                self._log_action("Quality assessment", {
                    "score": overall_score,
                    "threshold": self.quality_threshold,
                    "recommendation": recommendation
                })
                
                # Only request revision for MAJOR issues
                if "MAJOR" in recommendation:
                    self._log_action("Major quality issues - requesting revision")
                    self._record_decision(state, "run_drafter", "Address major quality concerns")
                    return "run_drafter"
                
                # For minor issues, only revise if:
                # 1. Score is below threshold (7.5)
                # 2. We have iterations left to improve
                # 3. We haven't already done too many revisions (prevent loops)
                elif overall_score < self.quality_threshold:
                    if state.iteration_count < self.max_iterations - 1:
                        self._log_action("Score below threshold - requesting revision", {
                            "score": overall_score,
                            "threshold": self.quality_threshold
                        })
                        self._record_decision(
                            state, 
                            "run_drafter", 
                            f"Improve quality score from {overall_score} to {self.quality_threshold}+"
                        )
                        return "run_drafter"
                    else:
                        # No iterations left, accept as-is
                        self._log_action("Score below threshold but no iterations left - halting")
                        self._record_decision(
                            state,
                            "halt_for_human",
                            f"Quality score {overall_score} below threshold but max iterations reached"
                        )
                        return "halt_for_human"
                
                # Score meets threshold - ready for human review
                else:
                    self._log_action("Quality score meets threshold - ready for review", {
                        "score": overall_score,
                        "threshold": self.quality_threshold
                    })
        
        # All checks passed - halt for human review
        self._log_action("All validations passed - halting for human review")
        self._record_decision(state, "halt_for_human", "Ready for human review")
        return "halt_for_human"
    
    def _has_recent_safety_check(self, state: Any) -> bool:
        """
        Check if safety validation has been performed for current draft.
        
        Args:
            state: Current protocol state
            
        Returns:
            True if safety check is recent (same iteration)
        """
        safety_checks = state.scratchpad.get("safety_checks", [])
        if not safety_checks:
            return False
        
        # Check if latest safety check is for current iteration
        latest_check = safety_checks[-1] if isinstance(safety_checks[-1], dict) else {}
        return latest_check.get("iteration") == state.iteration_count
    
    def _has_recent_quality_check(self, state: Any) -> bool:
        """
        Check if quality review has been performed for current draft.
        
        Args:
            state: Current protocol state
            
        Returns:
            True if quality check is recent (same iteration)
        """
        critic_feedback = state.scratchpad.get("critic_feedback", [])
        if not critic_feedback:
            return False
        
        # Check if latest review is for current iteration
        latest_review = critic_feedback[-1] if isinstance(critic_feedback[-1], dict) else {}
        return latest_review.get("iteration") == state.iteration_count
    
    def _record_decision(self, state: Any, action: str, reason: str):
        """
        Record supervisor decision in state scratchpad.
        
        This creates an audit trail of all routing decisions.
        
        Args:
            state: Current protocol state
            action: Decision action taken
            reason: Reasoning for the decision
        """
        if "supervisor_decisions" not in state.scratchpad:
            state.scratchpad["supervisor_decisions"] = []
        
        decision_record = {
            "timestamp": datetime.now().isoformat(),  # ISO format for JSON serialization
            "iteration": state.iteration_count,
            "action": action,
            "reason": reason
        }
        
        state.scratchpad["supervisor_decisions"].append(decision_record)
        
        # Also log for monitoring
        self.logger.debug(f"Decision recorded: {action} - {reason}")
    
    def get_workflow_summary(self, state: Any) -> dict:
        """
        Get a summary of the workflow state for debugging/monitoring.
        
        Args:
            state: Current protocol state
            
        Returns:
            Dictionary with workflow summary
        """
        # Get latest quality score if available
        latest_score = None
        critic_feedback = state.scratchpad.get("critic_feedback", [])
        if critic_feedback:
            latest_feedback = [f for f in critic_feedback if isinstance(f, dict)]
            if latest_feedback:
                latest_score = latest_feedback[-1].get("overall_score")
        
        return {
            "current_iteration": state.iteration_count,
            "max_iterations": self.max_iterations,
            "quality_threshold": self.quality_threshold,
            "latest_quality_score": latest_score,
            "has_draft": bool(state.current_draft),
            "draft_word_count": len(state.current_draft.split()) if state.current_draft else 0,
            "total_safety_flags": len(state.scratchpad.get("safety_flags", [])),
            "total_critic_feedback": len(state.scratchpad.get("critic_feedback", [])),
            "total_decisions": len(state.scratchpad.get("supervisor_decisions", [])),
            "is_halted": state.should_halt,
            "is_finalized": state.is_finalized,
            "approval_status": str(state.approval_status)
        }
