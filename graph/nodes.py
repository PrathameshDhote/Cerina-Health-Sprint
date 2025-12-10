"""
LangGraph node implementations.

Each node represents a specific step in the protocol generation workflow,
typically corresponding to an agent action.
"""

from typing import Dict, Any
from datetime import datetime

from state.protocol_state import ProtocolState, AgentRole, SafetySeverity
from agents import (
    CBTDrafterAgent,
    SafetyGuardianAgent,
    ClinicalCriticAgent,
    SupervisorAgent
)
from utils.logger import logger


# Initialize agents (singleton pattern)
_drafter = None
_safety_guardian = None
_clinical_critic = None
_supervisor = None


def get_drafter() -> CBTDrafterAgent:
    """Get or create drafter agent instance."""
    global _drafter
    if _drafter is None:
        _drafter = CBTDrafterAgent()
    return _drafter


def get_safety_guardian() -> SafetyGuardianAgent:
    """Get or create safety guardian agent instance."""
    global _safety_guardian
    if _safety_guardian is None:
        _safety_guardian = SafetyGuardianAgent()
    return _safety_guardian


def get_clinical_critic() -> ClinicalCriticAgent:
    """Get or create clinical critic agent instance."""
    global _clinical_critic
    if _clinical_critic is None:
        _clinical_critic = ClinicalCriticAgent()
    return _clinical_critic


def get_supervisor() -> SupervisorAgent:
    """Get or create supervisor agent instance."""
    global _supervisor
    if _supervisor is None:
        _supervisor = SupervisorAgent()
    return _supervisor


# Node Implementations

def drafter_node(state: ProtocolState) -> ProtocolState:
    """
    Drafter node - generates or revises CBT protocol content.
    
    Args:
        state: Current protocol state
        
    Returns:
        Updated state with new draft
    """
    logger.info(f"[Drafter Node] Starting draft generation (iteration {state.iteration_count})")
    
    try:
        # Get drafter agent
        drafter = get_drafter()
        
        # Process current state
        response = drafter.process(state)
        
        # Update state with new draft
        state.add_draft_version(
            content=response.content,
            agent=AgentRole.DRAFTER,
            changes_summary=response.reasoning
        )
        
        # Add drafter note to scratchpad
        state.add_drafter_note(
            note=response.reasoning,
            word_count=response.metadata.get("word_count", 0),
            has_structure=response.metadata.get("has_structure", False),
            addressed_feedback=response.suggestions
        )
        
        # Increment iteration
        state.increment_iteration()
        
        logger.info(f"[Drafter Node] Draft generation completed (iteration {state.iteration_count})")
        
        return state
        
    except Exception as e:
        logger.error(f"[Drafter Node] Error: {str(e)}")
        state.add_error("drafter_error", str(e), "drafter")
        raise


def safety_guardian_node(state: ProtocolState) -> ProtocolState:
    """
    Safety Guardian node - validates draft for safety concerns.
    
    Args:
        state: Current protocol state
        
    Returns:
        Updated state with safety assessment
    """
    logger.info(f"[Safety Guardian Node] Starting safety validation (iteration {state.iteration_count})")
    
    try:
        # Get safety guardian agent
        safety_guardian = get_safety_guardian()
        
        # Process current state
        response = safety_guardian.process(state)
        
        # Parse and add safety flags
        if response.flags:
            for flag_text in response.flags:
                # Parse severity from flag text
                severity = SafetySeverity.MEDIUM  # Default
                if "HIGH" in flag_text.upper():
                    severity = SafetySeverity.HIGH
                elif "LOW" in flag_text.upper():
                    severity = SafetySeverity.LOW
                
                state.add_safety_flag(
                    severity=severity,
                    issue=flag_text,
                    recommendation=response.suggestions[0] if response.suggestions else "Review and revise",
                    confidence=response.confidence
                )
        
        # Record safety check in scratchpad
        state.scratchpad["safety_checks"].append({
            "timestamp": datetime.now().isoformat(),
            "iteration": state.iteration_count,
            "agent": "safety_guardian",
            "rating": response.metadata.get("safety_rating", "UNKNOWN"),
            "flags_count": len(response.flags),
            "confidence": response.confidence
        })
        
        logger.info(f"[Safety Guardian Node] Safety validation completed - {len(response.flags)} flags found")
        
        return state
        
    except Exception as e:
        logger.error(f"[Safety Guardian Node] Error: {str(e)}")
        state.add_error("safety_error", str(e), "safety_guardian")
        raise


def clinical_critic_node(state: ProtocolState) -> ProtocolState:
    """
    Clinical Critic node - evaluates draft quality and empathy.
    
    Args:
        state: Current protocol state
        
    Returns:
        Updated state with quality assessment
    """
    logger.info(f"[Clinical Critic Node] Starting quality review (iteration {state.iteration_count})")
    
    try:
        # Get clinical critic agent
        critic = get_clinical_critic()
        
        # Process current state
        response = critic.process(state)
        
        # Extract scores from metadata
        metadata = response.metadata
        
        # Add critic feedback to state
        state.add_critic_feedback(
            overall_score=metadata.get("overall_score", 0.0),
            empathy_score=metadata.get("empathy_score", 0.0),
            individual_scores=metadata.get("individual_scores", {}),
            strengths=response.suggestions[:3] if response.suggestions else [],  # First 3 as strengths
            improvements=response.flags if response.flags else [],
            recommendation=metadata.get("recommendation", "REVIEW"),
            feedback=response.content,
            confidence=response.confidence
        )
        
        logger.info(f"[Clinical Critic Node] Quality review completed - Score: {metadata.get('overall_score', 0)}/10")
        
        return state
        
    except Exception as e:
        logger.error(f"[Clinical Critic Node] Error: {str(e)}")
        state.add_error("critic_error", str(e), "clinical_critic")
        raise


def supervisor_node(state: ProtocolState) -> ProtocolState:
    """
    Supervisor node - makes routing decisions.
    
    This node determines the next step in the workflow based on
    current state and agent outputs.
    
    Args:
        state: Current protocol state
        
    Returns:
        Updated state with supervisor decision
    """
    logger.info(f"[Supervisor Node] Evaluating workflow (iteration {state.iteration_count})")
    
    try:
        # Get supervisor agent
        supervisor = get_supervisor()
        
        # Get next action decision
        next_action = supervisor.decide_next_action(state)
        
        logger.info(f"[Supervisor Node] Decision: {next_action}")
        
        # The actual routing happens in the conditional edge
        # This node just records the decision
        return state
        
    except Exception as e:
        logger.error(f"[Supervisor Node] Error: {str(e)}")
        state.add_error("supervisor_error", str(e), "supervisor")
        raise


def halt_node(state: ProtocolState) -> ProtocolState:
    """
    Halt node - prepares state for human review.
    
    This is the critical "human-in-the-loop" interruption point.
    The workflow pauses here and waits for human approval/rejection.
    
    Args:
        state: Current protocol state
        
    Returns:
        Updated state marked for human review
    """
    logger.info(f"[Halt Node] Halting for human review (iteration {state.iteration_count})")
    
    try:
        # Mark state as halted
        state.halt_for_human_review()
        
        logger.info(f"[Halt Node] Workflow halted - awaiting human decision")
        logger.info(f"[Halt Node] Thread ID: {state.thread_id}")
        logger.info(f"[Halt Node] Current draft length: {len(state.current_draft)} chars")
        
        return state
        
    except Exception as e:
        logger.error(f"[Halt Node] Error: {str(e)}")
        state.add_error("halt_error", str(e), "halt")
        raise


def finalize_node(state: ProtocolState) -> ProtocolState:
    """
    Finalize node - completes the workflow after human approval.
    
    Args:
        state: Current protocol state
        
    Returns:
        Finalized state
    """
    logger.info(f"[Finalize Node] Finalizing protocol (iteration {state.iteration_count})")
    
    try:
        # If not already finalized, mark as complete
        if not state.is_finalized:
            # If there's a human-edited draft, use it
            if state.human_edited_draft:
                state.approve(edited_draft=state.human_edited_draft)
            else:
                state.approve()
        
        logger.info(f"[Finalize Node] Protocol finalized successfully")
        logger.info(f"[Finalize Node] Final status: {state.approval_status}")
        logger.info(f"[Finalize Node] Total iterations: {state.iteration_count}")
        
        return state
        
    except Exception as e:
        logger.error(f"[Finalize Node] Error: {str(e)}")
        state.add_error("finalize_error", str(e), "finalize")
        raise


def error_node(state: ProtocolState) -> ProtocolState:
    """
    Error node - handles workflow errors gracefully.
    
    Args:
        state: Current protocol state
        
    Returns:
        State with error information
    """
    logger.error(f"[Error Node] Workflow entered error state")
    logger.error(f"[Error Node] Errors: {state.errors}")
    
    # Mark for human review even in error state
    if not state.should_halt:
        state.halt_for_human_review()
    
    return state


def max_iterations_node(state: ProtocolState) -> ProtocolState:
    """
    Max iterations node - handles reaching iteration limit.
    
    When max iterations is reached, the workflow halts for human review
    even if all validations haven't passed.
    
    Args:
        state: Current protocol state
        
    Returns:
        State marked for human review
    """
    logger.warning(f"[Max Iterations Node] Maximum iterations reached: {state.iteration_count}/{state.max_iterations}")
    
    try:
        # Add a note about reaching max iterations
        state.add_error(
            "max_iterations",
            f"Reached maximum iteration limit ({state.max_iterations}). Current draft may need manual review.",
            "supervisor"
        )
        
        # Halt for human review
        state.halt_for_human_review()
        
        logger.info(f"[Max Iterations Node] Halting for human review due to iteration limit")
        
        return state
        
    except Exception as e:
        logger.error(f"[Max Iterations Node] Error: {str(e)}")
        state.add_error("max_iterations_error", str(e), "max_iterations")
        raise


# Helper node for initialization

def initialize_node(state: ProtocolState) -> ProtocolState:
    """
    Initialize node - sets up the initial state.
    
    Args:
        state: Initial protocol state
        
    Returns:
        Initialized state
    """
    logger.info(f"[Initialize Node] Starting new protocol generation")
    logger.info(f"[Initialize Node] Thread ID: {state.thread_id}")
    logger.info(f"[Initialize Node] User Intent: {state.user_intent}")
    
    # Record initial supervisor decision
    state.add_supervisor_decision(
        action="initialize",
        reason="Starting protocol generation workflow",
        next_agent="drafter"
    )
    
    return state
