"""
Conditional edge logic for the LangGraph workflow.

Edges determine routing between nodes based on state conditions.
"""

from typing import Literal
from state.protocol_state import ProtocolState
from agents import SupervisorAgent
from utils.logger import logger


def supervisor_router(
    state: ProtocolState
) -> Literal["drafter", "safety_guardian", "clinical_critic", "halt", "max_iterations"]:
    """
    Supervisor routing logic - determines which node to execute next.
    
    This is the core routing function that implements the Supervisor pattern.
    It evaluates the current state and decides which agent should run next.
    
    Args:
        state: Current protocol state
        
    Returns:
        Name of the next node to execute
    """
    logger.info(f"[Supervisor Router] Evaluating routing (iteration {state.iteration_count})")
    
    # Check if already halted or finalized
    if state.should_halt or state.is_finalized:
        logger.info("[Supervisor Router] State is halted or finalized, routing to halt")
        return "halt"
    
    # Get supervisor agent
    supervisor = SupervisorAgent()
    
    # Make routing decision
    decision = supervisor.decide_next_action(state)
    
    # Map decision to node name
    routing_map = {
        "run_drafter": "drafter",
        "run_safety": "safety_guardian",
        "run_critic": "clinical_critic",
        "halt_for_human": "halt",
        "max_iterations_reached": "max_iterations"
    }
    
    next_node = routing_map.get(decision, "halt")
    
    logger.info(f"[Supervisor Router] Routing to: {next_node}")
    
    return next_node


def should_continue(
    state: ProtocolState
) -> Literal["continue", "halt"]:
    """
    Determine if workflow should continue or halt.
    
    This is used after agent nodes to check if we should continue
    the workflow or halt for human review.
    
    Args:
        state: Current protocol state
        
    Returns:
        "continue" or "halt"
    """
    # Check if explicitly marked to halt
    if state.should_halt:
        logger.info("[Should Continue] State marked for halt")
        return "halt"
    
    # Check if finalized
    if state.is_finalized:
        logger.info("[Should Continue] State is finalized")
        return "halt"
    
    # Check max iterations
    if state.iteration_count >= state.max_iterations:
        logger.info("[Should Continue] Max iterations reached")
        return "halt"
    
    # Continue workflow
    logger.info("[Should Continue] Continuing workflow")
    return "continue"


def human_decision_router(
    state: ProtocolState
) -> Literal["drafter", "finalize"]:
    """
    Route based on human decision after halt.
    
    If human approved: go to finalize
    If human rejected: go back to drafter for revision
    
    Args:
        state: Current protocol state
        
    Returns:
        Next node based on human decision
    """
    from state.protocol_state import ApprovalStatus
    
    logger.info(f"[Human Decision Router] Human decision: {state.approval_status}")
    
    if state.approval_status in [ApprovalStatus.APPROVED, ApprovalStatus.EDITED]:
        logger.info("[Human Decision Router] Approved - routing to finalize")
        return "finalize"
    elif state.approval_status == ApprovalStatus.REJECTED:
        logger.info("[Human Decision Router] Rejected - routing to drafter for revision")
        # Reset halt flag to allow workflow to continue
        state.should_halt = False
        state.needs_revision = True
        return "drafter"
    else:
        # Still pending - stay halted
        logger.info("[Human Decision Router] Still pending - remaining halted")
        return "finalize"  # This shouldn't be reached in normal flow


def after_drafter_router(
    state: ProtocolState
) -> Literal["supervisor", "error"]:
    """
    Route after drafter node completes.
    
    Args:
        state: Current protocol state
        
    Returns:
        Next node
    """
    # Check for errors
    if state.errors and state.errors[-1].get("agent") == "drafter":
        logger.error("[After Drafter Router] Drafter error detected")
        return "error"
    
    # Normal flow - go to supervisor for decision
    return "supervisor"


def after_safety_router(
    state: ProtocolState
) -> Literal["supervisor", "error"]:
    """
    Route after safety guardian node completes.
    
    Args:
        state: Current protocol state
        
    Returns:
        Next node
    """
    # Check for errors
    if state.errors and state.errors[-1].get("agent") == "safety_guardian":
        logger.error("[After Safety Router] Safety guardian error detected")
        return "error"
    
    # Normal flow - go to supervisor for decision
    return "supervisor"


def after_critic_router(
    state: ProtocolState
) -> Literal["supervisor", "error"]:
    """
    Route after clinical critic node completes.
    
    Args:
        state: Current protocol state
        
    Returns:
        Next node
    """
    # Check for errors
    if state.errors and state.errors[-1].get("agent") == "clinical_critic":
        logger.error("[After Critic Router] Clinical critic error detected")
        return "error"
    
    # Normal flow - go to supervisor for decision
    return "supervisor"
