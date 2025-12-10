"""
Main LangGraph workflow definition.

This module creates and compiles the complete protocol generation workflow graph.
"""

from typing import Optional, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from state.protocol_state import ProtocolState
from database import get_checkpointer
from utils.logger import logger

from .nodes import (
    initialize_node,
    drafter_node,
    safety_guardian_node,
    clinical_critic_node,
    supervisor_node,
    halt_node,
    finalize_node,
    max_iterations_node,
    error_node
)
from .edges import (
    supervisor_router,
    after_drafter_router,
    after_safety_router,
    after_critic_router
)


def create_protocol_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for protocol generation.
    
    Workflow Structure:
    1. Initialize
    2. Enter agent loop (Drafter -> Safety -> Critic)
    3. Supervisor makes routing decisions
    4. Halt for human review
    5. Finalize based on human decision
    
    Returns:
        StateGraph instance (not yet compiled)
    """
    logger.info("Creating protocol generation workflow")
    
    # Create state graph
    workflow = StateGraph(ProtocolState)
    
    # Add nodes
    logger.info("Adding workflow nodes")
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("drafter", drafter_node)
    workflow.add_node("safety_guardian", safety_guardian_node)
    workflow.add_node("clinical_critic", clinical_critic_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("halt", halt_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("max_iterations", max_iterations_node)
    workflow.add_node("error", error_node)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Add edges
    logger.info("Adding workflow edges")
    
    # Initialize -> Supervisor (to start the loop)
    workflow.add_edge("initialize", "supervisor")
    
    # Supervisor -> Conditional routing to agents
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "drafter": "drafter",
            "safety_guardian": "safety_guardian",
            "clinical_critic": "clinical_critic",
            "halt": "halt",
            "max_iterations": "max_iterations"
        }
    )
    
    # After Drafter -> Back to Supervisor
    workflow.add_conditional_edges(
        "drafter",
        after_drafter_router,
        {
            "supervisor": "supervisor",
            "error": "error"
        }
    )
    
    # After Safety Guardian -> Back to Supervisor
    workflow.add_conditional_edges(
        "safety_guardian",
        after_safety_router,
        {
            "supervisor": "supervisor",
            "error": "error"
        }
    )
    
    # After Clinical Critic -> Back to Supervisor
    workflow.add_conditional_edges(
        "clinical_critic",
        after_critic_router,
        {
            "supervisor": "supervisor",
            "error": "error"
        }
    )
    
    # Halt -> END (workflow pauses here for human review)
    workflow.add_edge("halt", END)
    
    # Max Iterations -> Halt -> END
    workflow.add_edge("max_iterations", "halt")
    
    # Finalize -> END
    workflow.add_edge("finalize", END)
    
    # Error -> END
    workflow.add_edge("error", END)
    
    logger.info("Workflow graph created successfully")
    
    return workflow


async def compile_workflow_async(checkpointer: Optional[Any] = None) -> Any:
    """
    Compile the workflow with async checkpointing enabled.
    
    Used for streaming endpoints.
    
    Args:
        checkpointer: Optional custom async checkpointer (uses default if None)
        
    Returns:
        Compiled workflow graph with async support
    """
    logger.info("Compiling workflow with async checkpointing")
    
    # Create workflow
    workflow = create_protocol_workflow()
    
    # Get async checkpointer if not provided
    if checkpointer is None:
        logger.info("Using default async checkpointer from database config")
        from database import get_async_checkpointer
        checkpointer = await get_async_checkpointer()
    
    # Compile with async checkpointer
    compiled = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Async workflow compiled successfully with checkpointing enabled")
    
    return compiled



def create_workflow_diagram() -> str:
    """
    Generate a text representation of the workflow for documentation.
    
    Returns:
        Workflow diagram as string
    """
    diagram = """
    Cerina Protocol Foundry - Workflow Diagram
    ==========================================
    
    START
      ↓
    [Initialize]
      ↓
    [Supervisor] ←─────────────────────┐
      ↓                                │
    (Decision)                         │
      ├─→ [Drafter] ──────────────────┤
      ├─→ [Safety Guardian] ──────────┤
      ├─→ [Clinical Critic] ──────────┤
      ├─→ [Max Iterations] → [Halt]   │
      └─→ [Halt] ─────────────────────┘
           ↓
    (Human Review)
           ↓
    [Finalize]
      ↓
    END
    
    Legend:
    - [ ] = Node (agent action)
    - ( ) = Conditional routing
    - → = Edge (flow direction)
    - ← = Loop back
    """
    return diagram


# Workflow introspection utilities

def get_workflow_stats(state: ProtocolState) -> dict:
    """
    Get statistics about the workflow execution.
    
    Args:
        state: Current protocol state
        
    Returns:
        Dictionary of workflow statistics
    """
    return {
        "thread_id": state.thread_id,
        "iterations_completed": state.iteration_count,
        "max_iterations": state.max_iterations,
        "total_agents_run": (
            len(state.drafter_notes) +
            len(state.safety_flags) +
            len(state.critic_feedbacks)
        ),
        "supervisor_decisions": len(state.supervisor_decisions),
        "draft_versions": len(state.draft_versions),
        "safety_flags": len(state.safety_flags),
        "critic_feedbacks": len(state.critic_feedbacks),
        "errors": len(state.errors),
        "is_halted": state.should_halt,
        "is_finalized": state.is_finalized,
        "approval_status": state.approval_status,
        "metadata_scores": state.metadata.model_dump()
    }


def visualize_workflow_execution(state: ProtocolState) -> str:
    """
    Create a visual representation of the workflow execution path.
    
    Args:
        state: Current protocol state
        
    Returns:
        Visual execution trace
    """
    lines = [
        "Workflow Execution Trace",
        "=" * 50,
        f"Thread ID: {state.thread_id}",
        f"User Intent: {state.user_intent}",
        f"Status: {state.approval_status}",
        "",
        "Execution Path:",
    ]
    
    # Build execution timeline
    all_events = []
    
    # Add drafter events
    for note in state.drafter_notes:
        all_events.append({
            "timestamp": note.timestamp,
            "iteration": note.iteration,
            "agent": "Drafter",
            "action": "Generated draft"
        })
    
    # Add safety events
    for flag in state.safety_flags:
        all_events.append({
            "timestamp": flag.timestamp,
            "iteration": flag.iteration,
            "agent": "Safety Guardian",
            "action": f"Safety check - {flag.severity}"
        })
    
    # Add critic events
    for feedback in state.critic_feedbacks:
        all_events.append({
            "timestamp": feedback.timestamp,
            "iteration": feedback.iteration,
            "agent": "Clinical Critic",
            "action": f"Quality review - {feedback.overall_score}/10"
        })
    
    # Add supervisor events
    for decision in state.supervisor_decisions:
        all_events.append({
            "timestamp": decision.timestamp,
            "iteration": decision.iteration,
            "agent": "Supervisor",
            "action": f"Decision: {decision.action}"
        })
    
    # Sort by timestamp
    all_events.sort(key=lambda x: x["timestamp"])
    
    # Format events
    for event in all_events:
        timestamp_str = event["timestamp"].strftime("%H:%M:%S")
        lines.append(
            f"  [{timestamp_str}] Iteration {event['iteration']}: "
            f"{event['agent']} - {event['action']}"
        )
    
    lines.extend([
        "",
        f"Final Status: {state.approval_status}",
        f"Total Iterations: {state.iteration_count}",
        f"Draft Versions: {len(state.draft_versions)}",
    ])
    
    return "\n".join(lines)
