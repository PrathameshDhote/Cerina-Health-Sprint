"""
Main API routes for the Cerina Protocol Foundry.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse

from state.protocol_state import ProtocolState, ApprovalStatus
from state.schemas import (
    GenerationRequest,
    GenerationResponse,
    StateResponse,
    ResumeRequest,
    HealthResponse,
    DetailedStateResponse,
    ErrorResponse
)
from graph.workflow import create_protocol_workflow, get_workflow_stats
from graph.streaming import stream_workflow_events
from database import get_checkpointer, get_async_checkpointer
from utils.logger import logger
from config import settings

# ✅ FIXED: Import only what exists in dependencies
from .dependencies import get_current_state, verify_api_key


router = APIRouter(prefix="/api", tags=["protocol"])


# Health Check

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the service
    """
    try:
        # Test database connection
        checkpointer = get_checkpointer()
        db_connected = checkpointer is not None
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            database_connected=db_connected
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            database_connected=False
        )


# Protocol Generation

@router.post("/generate", response_model=GenerationResponse)
async def generate_protocol(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Initiate a new protocol generation workflow.
    
    This creates a new thread and starts the agent workflow.
    The workflow will run until it reaches the halt point for human review.
    
    Args:
        request: Generation request with user intent
        background_tasks: FastAPI background tasks
        api_key_valid: API key validation result
        
    Returns:
        Generation response with thread ID
    """
    logger.info(f"Received generation request: {request.user_intent}")
    
    try:
        # Generate unique thread ID
        thread_id = str(uuid4())
        
        # Create initial state
        initial_state = ProtocolState(
            thread_id=thread_id,
            user_intent=request.user_intent,
            max_iterations=request.max_iterations or settings.max_agent_iterations
        )
        
        # Configuration for LangGraph with increased recursion limit
        config = {
            "configurable": {
                "thread_id": thread_id
            },
            "recursion_limit": 50
        }
        
        logger.info(f"Starting workflow for thread: {thread_id}")
        
        # ✅ FIXED: Use async checkpointer context manager
        workflow_graph = create_protocol_workflow()
        
        async with get_async_checkpointer() as checkpointer:
            compiled_workflow = workflow_graph.compile(checkpointer=checkpointer)
            result = await compiled_workflow.ainvoke(initial_state, config)
        
        # LangGraph returns a dictionary, not a ProtocolState object
        if isinstance(result, dict):
            final_state = ProtocolState(**result)
        else:
            final_state = result
        
        logger.info(f"Workflow reached halt point for thread: {thread_id}")
        logger.info(f"Final state: {final_state.approval_status}, Iteration: {final_state.iteration_count}")
        
        return GenerationResponse(
            thread_id=thread_id,
            status=str(final_state.approval_status),
            message="Protocol generation initiated successfully. Workflow halted for human review.",
            user_intent=request.user_intent,
            created_at=final_state.created_at
        )
        
    except Exception as e:
        logger.error(f"Error during protocol generation: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Protocol generation failed: {str(e)}"
        )


@router.post("/generate", response_model=GenerationResponse)
async def generate_protocol(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Initiate a new protocol generation workflow.
    
    - For web requests (source="web"): Workflow halts for human review
    - For MCP requests (source="mcp"): Workflow bypasses halt and auto-finalizes
    """
    logger.info(f"Received generation request: {request.user_intent}")
    logger.info(f"Request source: {request.source}")  # ✅ LOG SOURCE
    
    try:
        # Generate unique thread ID
        thread_id = str(uuid4())
        
        # ✅ Determine if we should bypass halt
        bypass_halt = (request.source == "mcp")
        
        logger.info(f"Bypass halt mode: {bypass_halt}")
        
        # Create initial state with bypass flag
        initial_state = ProtocolState(
            thread_id=thread_id,
            user_intent=request.user_intent,
            source= request.source,
            max_iterations=request.max_iterations or settings.max_agent_iterations,
            bypass_halt=bypass_halt  # ✅ Pass bypass flag to state
        )
        
        # Configuration for LangGraph
        config = {
            "configurable": {
                "thread_id": thread_id
            },
            "recursion_limit": 50
        }
        
        logger.info(f"Starting workflow for thread: {thread_id}")
        
        # Run workflow with async checkpointer
        workflow_graph = create_protocol_workflow()
        
        async with get_async_checkpointer() as checkpointer:
            compiled_workflow = workflow_graph.compile(checkpointer=checkpointer)
            result = await compiled_workflow.ainvoke(initial_state, config)
        
        # Convert result to ProtocolState
        if isinstance(result, dict):
            final_state = ProtocolState(**result)
        else:
            final_state = result
        
        logger.info(f"Workflow completed for thread: {thread_id}")
        logger.info(f"Final state: {final_state.approval_status}, Iteration: {final_state.iteration_count}")
        
        # ✅ Different response messages based on mode
        if bypass_halt:
            message = "Protocol generation completed. Workflow finalized automatically (MCP mode)."
        else:
            message = "Protocol generation initiated successfully. Workflow halted for human review."
        
        return GenerationResponse(
            thread_id=thread_id,
            status=str(final_state.approval_status),
            message=message,
            user_intent=request.user_intent,
            created_at=final_state.created_at
        )
        
    except Exception as e:
        logger.error(f"Error during protocol generation: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Protocol generation failed: {str(e)}"
        )




# State Management

@router.get("/state/{thread_id}", response_model=StateResponse)
async def get_state(
    thread_id: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Get the current state of a protocol generation workflow.
    
    Args:
        thread_id: Thread identifier
        api_key_valid: API key validation result
        
    Returns:
        Current state of the workflow
    """
    logger.info(f"Retrieving state for thread: {thread_id}")
    
    try:
        state = get_current_state(thread_id)
        
        return StateResponse(
            thread_id=state.thread_id,
            user_intent=state.user_intent,
            current_draft=state.current_draft,
            final_approved_draft=state.final_approved_draft if state.final_approved_draft else None,
            iteration_count=state.iteration_count,
            max_iterations=state.max_iterations,
            approval_status=state.approval_status,
            metadata=state.metadata,
            safety_flags_count=len(state.safety_flags),
            critic_feedbacks_count=len(state.critic_feedbacks),
            has_blocking_issues=state.has_blocking_safety_issues() or state.has_major_quality_issues(),
            is_finalized=state.is_finalized,
            halted_at_iteration=state.halted_at_iteration,
            created_at=state.created_at,
            last_modified=state.last_modified,
            halted_at=state.halted_at,
            approved_at=state.approved_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving state: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve state: {str(e)}"
        )


@router.get("/state/{thread_id}/detailed", response_model=DetailedStateResponse)
async def get_detailed_state(
    thread_id: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Get detailed state including full agent history.
    
    Args:
        thread_id: Thread identifier
        api_key_valid: API key validation result
        
    Returns:
        Detailed state with full scratchpad and agent outputs
    """
    logger.info(f"Retrieving detailed state for thread: {thread_id}")
    
    try:
        state = get_current_state(thread_id)
        
        return DetailedStateResponse(
            thread_id=state.thread_id,
            user_intent=state.user_intent,
            current_draft=state.current_draft,
            final_approved_draft=state.final_approved_draft if state.final_approved_draft else None,
            iteration_count=state.iteration_count,
            max_iterations=state.max_iterations,
            approval_status=state.approval_status,
            metadata=state.metadata,
            safety_flags_count=len(state.safety_flags),
            critic_feedbacks_count=len(state.critic_feedbacks),
            has_blocking_issues=state.has_blocking_safety_issues() or state.has_major_quality_issues(),
            is_finalized=state.is_finalized,
            halted_at_iteration=state.halted_at_iteration,
            created_at=state.created_at,
            last_modified=state.last_modified,
            halted_at=state.halted_at,
            approved_at=state.approved_at,
            # Additional detailed fields
            draft_versions=[v.model_dump() for v in state.draft_versions],
            safety_flags=[f.model_dump() for f in state.safety_flags],
            critic_feedbacks=[f.model_dump() for f in state.critic_feedbacks],
            supervisor_decisions=[d.model_dump() for d in state.supervisor_decisions],
            drafter_notes=[n.model_dump() for n in state.drafter_notes],
            errors=state.errors,
            scratchpad=state.scratchpad
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving detailed state: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve detailed state: {str(e)}"
        )


# Human-in-Loop

@router.post("/resume/{thread_id}")
async def resume_workflow(
    thread_id: str,
    request: ResumeRequest,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Resume a halted workflow after human review.
    
    This endpoint handles three actions:
    - approve: Accept the protocol as-is
    - edit: Accept with human edits
    - reject: Send back for revision
    
    Args:
        thread_id: Thread identifier
        request: Resume request with action and optional feedback
        api_key_valid: API key validation result
        
    Returns:
        Updated state after resume
    """
    logger.info(f"Resuming workflow for thread: {thread_id} with action: {request.action}")
    
    try:
        # Verify thread_id matches
        if request.thread_id != thread_id:
            raise HTTPException(
                status_code=400,
                detail="Thread ID in URL does not match request body"
            )
        
        # Get current state
        state = get_current_state(thread_id)
        
        # Verify state is halted
        if not state.should_halt and not state.approval_status == ApprovalStatus.PENDING_HUMAN_REVIEW:
            raise HTTPException(
                status_code=400,
                detail=f"Workflow is not halted. Current status: {state.approval_status}"
            )
        
        # Handle action
        if request.action == "approve":
            logger.info(f"Approving protocol for thread: {thread_id}")
            state.approve()
            
        elif request.action == "edit":
            if not request.edited_draft:
                raise HTTPException(
                    status_code=400,
                    detail="edited_draft is required when action is 'edit'"
                )
            logger.info(f"Approving edited protocol for thread: {thread_id}")
            state.approve(edited_draft=request.edited_draft)
            
        elif request.action == "reject":
            if not request.feedback:
                raise HTTPException(
                    status_code=400,
                    detail="feedback is required when action is 'reject'"
                )
            logger.info(f"Rejecting protocol for thread: {thread_id}")
            state.reject(feedback=request.feedback)
            
            # ✅ FIXED: Resume workflow with async checkpointer
            workflow_graph = create_protocol_workflow()
            config = {
                "configurable": {
                    "thread_id": thread_id
                },
                "recursion_limit": 50
            }
            
            async with get_async_checkpointer() as checkpointer:
                compiled_workflow = workflow_graph.compile(checkpointer=checkpointer)
                result = await compiled_workflow.ainvoke(state, config)
            
            # Convert result back to ProtocolState if needed
            if isinstance(result, dict):
                state = ProtocolState(**result)
            else:
                state = result
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {request.action}. Must be 'approve', 'edit', or 'reject'"
            )
        
        # Return updated state
        return StateResponse(
            thread_id=state.thread_id,
            user_intent=state.user_intent,
            current_draft=state.current_draft,
            final_approved_draft=state.final_approved_draft if state.final_approved_draft else None,
            iteration_count=state.iteration_count,
            max_iterations=state.max_iterations,
            approval_status=state.approval_status,
            metadata=state.metadata,
            safety_flags_count=len(state.safety_flags),
            critic_feedbacks_count=len(state.critic_feedbacks),
            has_blocking_issues=state.has_blocking_safety_issues() or state.has_major_quality_issues(),
            is_finalized=state.is_finalized,
            halted_at_iteration=state.halted_at_iteration,
            created_at=state.created_at,
            last_modified=state.last_modified,
            halted_at=state.halted_at,
            approved_at=state.approved_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming workflow: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume workflow: {str(e)}"
        )


# Workflow Management

@router.get("/workflow/stats/{thread_id}")
async def get_workflow_statistics(
    thread_id: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Get detailed statistics about a workflow execution.
    
    Args:
        thread_id: Thread identifier
        api_key_valid: API key validation result
        
    Returns:
        Workflow statistics
    """
    logger.info(f"Retrieving workflow stats for thread: {thread_id}")
    
    try:
        state = get_current_state(thread_id)
        stats = get_workflow_stats(state)
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving workflow stats: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve workflow stats: {str(e)}"
        )


@router.delete("/workflow/{thread_id}")
async def delete_workflow(
    thread_id: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Delete a workflow and its checkpoint data.
    
    Args:
        thread_id: Thread identifier
        api_key_valid: API key validation result
        
    Returns:
        Success message
    """
    logger.info(f"Deleting workflow for thread: {thread_id}")
    
    try:
        # TODO: Implement checkpoint deletion
        # This requires extending the checkpointer with a delete method
        
        return {
            "message": f"Workflow {thread_id} deleted successfully",
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error deleting workflow: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete workflow: {str(e)}"
        )


# History and Analytics

@router.get("/history")
async def get_protocol_history(
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Get history of protocol generations.
    
    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        status: Filter by approval status
        api_key_valid: API key validation result
        
    Returns:
        List of historical protocols
    """
    logger.info(f"Retrieving protocol history (limit={limit}, offset={offset}, status={status})")
    
    try:
        # TODO: Implement history retrieval from database
        # This requires querying all checkpoints and filtering
        
        return {
            "total": 0,
            "limit": limit,
            "offset": offset,
            "protocols": []
        }
        
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


# Draft Management

@router.get("/draft/{thread_id}")
async def get_current_draft(
    thread_id: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Get only the current draft content (for quick preview).
    
    Args:
        thread_id: Thread identifier
        api_key_valid: API key validation result
        
    Returns:
        Current draft content
    """
    logger.info(f"Retrieving current draft for thread: {thread_id}")
    
    try:
        state = get_current_state(thread_id)
        
        return {
            "thread_id": thread_id,
            "current_draft": state.current_draft,
            "word_count": len(state.current_draft.split()) if state.current_draft else 0,
            "iteration": state.iteration_count,
            "last_modified": state.last_modified.isoformat() if state.last_modified else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving draft: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve draft: {str(e)}"
        )


@router.get("/draft/{thread_id}/versions")
async def get_draft_versions(
    thread_id: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Get all versions of the draft for comparison.
    
    Args:
        thread_id: Thread identifier
        api_key_valid: API key validation result
        
    Returns:
        All draft versions
    """
    logger.info(f"Retrieving draft versions for thread: {thread_id}")
    
    try:
        state = get_current_state(thread_id)
        
        versions = [
            {
                "version": v.version_number,
                "content": v.content,
                "word_count": v.word_count,
                "created_at": v.created_at.isoformat(),
                "created_by": v.created_by,
                "changes_summary": v.changes_summary,
                "iteration": v.iteration
            }
            for v in state.draft_versions
        ]
        
        return {
            "thread_id": thread_id,
            "total_versions": len(versions),
            "versions": versions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving draft versions: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve draft versions: {str(e)}"
        )
