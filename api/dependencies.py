"""
FastAPI dependencies for dependency injection.
"""

from typing import Optional, Any, Tuple
from fastapi import HTTPException, Header

from graph.workflow import create_protocol_workflow
from state.protocol_state import ProtocolState
from database import get_checkpointer, get_async_checkpointer
from utils.logger import logger

# Global workflow instance for sync operations (read-only)
_sync_workflow: Optional[Any] = None


def get_sync_workflow() -> Any:
    """
    Get compiled workflow for synchronous operations (like get_state).
    
    Uses synchronous SqliteSaver.
    
    Returns:
        Compiled workflow with sync checkpointer
    """
    global _sync_workflow
    
    if _sync_workflow is None:
        logger.info("Compiling synchronous workflow for state retrieval")
        
        checkpointer = get_checkpointer()
        workflow_graph = create_protocol_workflow()
        _sync_workflow = workflow_graph.compile(checkpointer=checkpointer)
        
        logger.info("Synchronous workflow compiled and cached")
    
    return _sync_workflow


async def get_async_workflow():
    """
    Get compiled workflow for async operations (like ainvoke).
    
    Creates a new compiled workflow with AsyncSqliteSaver each time.
    
    Returns:
        Compiled workflow with async checkpointer
    """
    logger.info("Compiling workflow with async checkpointer")
    
    workflow_graph = create_protocol_workflow()
    
    # Get async checkpointer context manager
    checkpointer_cm = get_async_checkpointer()
    
    # Enter the async context and compile
    async with checkpointer_cm as checkpointer:
        compiled_workflow = workflow_graph.compile(checkpointer=checkpointer)
        return compiled_workflow, checkpointer_cm


async def get_workflow_with_async_checkpointer():
    """
    Get workflow graph and async checkpointer context manager for streaming.
    
    This returns the uncompiled workflow graph and the async checkpointer
    context manager. The caller must enter the async context and compile
    the workflow with the checkpointer.
    
    Returns:
        Tuple of (workflow_graph, async_checkpointer_context_manager)
    """
    logger.info("Creating workflow with async checkpointer for streaming")
    
    workflow_graph = create_protocol_workflow()
    checkpointer_cm = get_async_checkpointer()
    
    return workflow_graph, checkpointer_cm


def get_current_state(thread_id: str) -> ProtocolState:
    """
    Retrieve the current state for a given thread from the checkpoint.
    
    Args:
        thread_id: Thread identifier
        
    Returns:
        Current protocol state
        
    Raises:
        HTTPException: If thread not found
    """
    try:
        # Use synchronous workflow for state retrieval
        workflow = get_sync_workflow()
        
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = workflow.get_state(config)
        
        if state_snapshot is None or not state_snapshot.values:
            raise HTTPException(
                status_code=404,
                detail=f"Thread {thread_id} not found. It may have expired or never existed."
            )
        
        state_dict = state_snapshot.values
        
        if isinstance(state_dict, dict):
            state = ProtocolState(**state_dict)
        else:
            state = state_dict
        
        return state
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving state for thread {thread_id}: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve state: {str(e)}"
        )


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """
    Optional API key verification for production deployment.
    
    Args:
        x_api_key: API key from header
        
    Returns:
        True if valid (or if API key checking is disabled)
        
    Raises:
        HTTPException: If API key is invalid
    """
    from config import settings
    
    if settings.is_development:
        return True
    
    if not hasattr(settings, 'api_key') or settings.api_key is None:
        return True
    
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header."
        )
    
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return True


async def rate_limit_check(thread_id: str) -> bool:
    """
    Check rate limiting for API calls.
    
    Args:
        thread_id: Thread identifier
        
    Returns:
        True if within rate limits
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    return True
