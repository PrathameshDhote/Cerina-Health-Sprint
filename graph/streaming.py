"""
Streaming utilities for real-time workflow updates.

Provides functions to stream workflow execution events to clients.
"""

from typing import AsyncIterator, Dict, Any
from datetime import datetime
import json

from state.protocol_state import ProtocolState
from state.schemas import StreamEvent
from utils.logger import logger


async def stream_workflow_events(
    graph,
    initial_state: ProtocolState,
    config: Dict[str, Any]
) -> AsyncIterator[str]:
    """
    Stream workflow execution events in real-time.
    
    This is used for the React dashboard to show live agent activity.
    
    Args:
        graph: Compiled workflow graph
        initial_state: Initial state
        config: LangGraph config with thread_id
        
    Yields:
        Server-Sent Events formatted strings
    """
    thread_id = config.get('configurable', {}).get('thread_id', 'unknown')
    logger.info(f"Starting workflow stream for thread: {thread_id}")
    
    try:
        # Send initial event
        start_event = StreamEvent(
            event_type="start",
            iteration=0,
            message=f"Starting workflow for thread {thread_id}",
            agent="system",
            data={"thread_id": thread_id}
        )
        yield f"data: {json.dumps(start_event.model_dump(), default=str)}\n\n"
        
        # Stream graph execution using astream
        async for event in graph.astream(initial_state, config, stream_mode="updates"):
            # event is a dict with node name as key and updated state as value
            for node_name, updated_state in event.items():
                logger.info(f"Stream event from node: {node_name}")
                
                # Handle both dict and ProtocolState responses
                if isinstance(updated_state, dict):
                    # Convert dict to ProtocolState
                    state_obj = ProtocolState(**updated_state)
                else:
                    state_obj = updated_state
                
                # Create stream event
                stream_event = create_stream_event(node_name, state_obj)
                
                # Format as SSE
                sse_data = f"data: {json.dumps(stream_event.model_dump(), default=str)}\n\n"
                
                yield sse_data
                
                # Check if halted
                if state_obj.should_halt or state_obj.is_finalized:
                    logger.info("Workflow halted or finalized - stopping stream")
                    
                    # Send completion event
                    completion_event = StreamEvent(
                        event_type="complete" if state_obj.is_finalized else "halt",
                        iteration=state_obj.iteration_count,
                        message="Workflow completed" if state_obj.is_finalized else "Workflow halted for human review",
                        agent="system",
                        data={
                            "thread_id": thread_id,
                            "approval_status": str(state_obj.approval_status),
                            "iteration_count": state_obj.iteration_count
                        }
                    )
                    yield f"data: {json.dumps(completion_event.model_dump(), default=str)}\n\n"
                    break
        
        logger.info(f"Stream completed for thread: {thread_id}")
        
    except Exception as e:
        logger.error(f"Error during workflow streaming: {str(e)}")
        
        # Send error event
        error_event = StreamEvent(
            event_type="error",
            iteration=0,
            message=f"Workflow error: {str(e)}",
            agent="system",
            data={"error_type": type(e).__name__}
        )
        yield f"data: {json.dumps(error_event.model_dump(), default=str)}\n\n"


def create_stream_event(node_name: str, state: ProtocolState) -> StreamEvent:
    """
    Create a stream event from node execution.
    
    Args:
        node_name: Name of the node that executed
        state: Updated state after node execution
        
    Returns:
        StreamEvent object
    """
    event_type = "agent_start"
    agent = node_name
    message = f"{node_name} is processing"
    data = {}
    
    # Customize based on node
    if node_name == "initialize":
        event_type = "agent_start"
        message = "Initializing workflow"
        data = {
            "thread_id": state.thread_id,
            "user_intent": state.user_intent,
            "max_iterations": state.max_iterations
        }
    
    elif node_name == "drafter":
        event_type = "draft_update"
        message = f"Drafter generated new draft (iteration {state.iteration_count})"
        data = {
            "draft_preview": state.current_draft[:200] + "..." if state.current_draft else "No draft yet",
            "word_count": len(state.current_draft.split()) if state.current_draft else 0,
            "version": len(state.draft_versions)
        }
    
    elif node_name == "safety_guardian":
        event_type = "agent_end"
        latest_flags = state.safety_flags[-3:] if state.safety_flags else []
        message = f"Safety check completed - {len(latest_flags)} issues found"
        data = {
            "flags_count": len(state.safety_flags),
            "latest_flags": [f.issue for f in latest_flags] if latest_flags else []
        }
    
    elif node_name == "clinical_critic":
        event_type = "agent_end"
        latest_feedback = state.get_latest_critic_feedback()
        score = latest_feedback.overall_score if latest_feedback else 0
        message = f"Quality review completed - Score: {score}/10"
        data = {
            "overall_score": score,
            "empathy_score": latest_feedback.empathy_score if latest_feedback else 0,
            "recommendation": latest_feedback.recommendation if latest_feedback else "N/A"
        }
    
    elif node_name == "supervisor":
        event_type = "agent_start"
        message = "Supervisor evaluating next action"
        data = {
            "iteration": state.iteration_count,
            "max_iterations": state.max_iterations
        }
    
    elif node_name == "halt":
        event_type = "halt"
        message = "Workflow halted for human review"
        data = {
            "thread_id": state.thread_id,
            "halted_at_iteration": state.halted_at_iteration
        }
    
    elif node_name == "finalize":
        event_type = "complete"
        message = "Protocol finalized successfully"
        data = {
            "approval_status": str(state.approval_status),
            "total_iterations": state.iteration_count
        }
    
    elif node_name == "error":
        event_type = "error"
        message = "Error occurred in workflow"
        data = {
            "errors": state.errors
        }
    
    return StreamEvent(
        event_type=event_type,
        timestamp=datetime.now(),
        agent=agent,
        iteration=state.iteration_count,
        data=data,
        message=message
    )
