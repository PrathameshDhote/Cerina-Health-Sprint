"""
API request/response schemas for the Cerina Protocol Foundry.

These Pydantic models define the contract between the API and clients.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

from .protocol_state import ApprovalStatus, MetadataScores


# Request Schemas

class GenerationRequest(BaseModel):
    """Request to generate a new CBT protocol."""
    user_intent: str = Field(
        ...,
        description="The user's intent or clinical need for the protocol",
        examples=["Create an exposure hierarchy for agoraphobia"]
    )
    max_iterations: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of agent iterations"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier for tracking"
    )


class ResumeRequest(BaseModel):
    """Request to resume a halted workflow after human review."""
    thread_id: str = Field(..., description="Thread ID of the halted workflow")
    action: Literal["approve", "reject", "edit"] = Field(
        ...,
        description="Action to take: approve, reject, or edit"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Human feedback (required if rejecting)"
    )
    edited_draft: Optional[str] = Field(
        default=None,
        description="Edited version of the draft (required if editing)"
    )


class ApprovalAction(BaseModel):
    """Human approval action on a protocol."""
    action: Literal["approve", "reject", "edit"]
    feedback: Optional[str] = None
    edited_draft: Optional[str] = None


# Response Schemas

class GenerationResponse(BaseModel):
    """Response from protocol generation initiation."""
    thread_id: str = Field(..., description="Unique thread identifier")
    status: str = Field(..., description="Current status of generation")
    message: str = Field(..., description="Human-readable message")
    user_intent: str = Field(..., description="Original user intent")
    created_at: datetime = Field(..., description="Creation timestamp")


class StateResponse(BaseModel):
    """Current state of a protocol generation workflow."""
    thread_id: str
    user_intent: str
    current_draft: str
    final_approved_draft: Optional[str]
    iteration_count: int
    max_iterations: int
    approval_status: ApprovalStatus
    metadata: MetadataScores
    safety_flags_count: int
    critic_feedbacks_count: int
    has_blocking_issues: bool
    is_finalized: bool
    halted_at_iteration: Optional[int]
    created_at: datetime
    last_modified: datetime
    halted_at: Optional[datetime]
    approved_at: Optional[datetime]


class ProtocolHistory(BaseModel):
    """Historical record of a protocol generation."""
    thread_id: str
    user_intent: str
    final_draft: str
    iterations_used: int
    approval_status: ApprovalStatus
    metadata: MetadataScores
    created_at: datetime
    approved_at: Optional[datetime]


class StreamEvent(BaseModel):
    """Event streamed during protocol generation."""
    event_type: Literal["start", "agent_start", "agent_end", "draft_update", "flag", "halt", "error", "complete"]
    timestamp: datetime = Field(default_factory=datetime.now)
    agent: Optional[str] = None
    iteration: int
    data: Dict[str, Any] = Field(default_factory=dict)
    message: str


class ErrorResponse(BaseModel):
    """Error response from API."""
    error: str
    detail: Optional[str] = None
    thread_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    database_connected: bool = True


# Agent-specific schemas (for transparency)

class AgentActivity(BaseModel):
    """Activity record from an agent."""
    agent_name: str
    action: str
    timestamp: datetime
    iteration: int
    details: Dict[str, Any] = Field(default_factory=dict)


class SafetyAssessmentSummary(BaseModel):
    """Summary of safety assessment."""
    overall_safety: str  # SAFE, NEEDS_REVISION, UNSAFE
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    key_concerns: List[str]


class QualityAssessmentSummary(BaseModel):
    """Summary of quality assessment."""
    overall_score: float
    empathy_score: float
    recommendation: str
    key_strengths: List[str]
    key_improvements: List[str]


# Detailed state for debugging/monitoring

class DetailedStateResponse(StateResponse):
    """Extended state response with full agent history."""
    draft_versions: List[Dict[str, Any]]
    safety_flags: List[Dict[str, Any]]
    critic_feedbacks: List[Dict[str, Any]]
    supervisor_decisions: List[Dict[str, Any]]
    drafter_notes: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    scratchpad: Dict[str, List[Any]]


# Batch operation schemas

class BatchGenerationRequest(BaseModel):
    """Request to generate multiple protocols."""
    requests: List[GenerationRequest] = Field(..., max_length=10)


class BatchGenerationResponse(BaseModel):
    """Response from batch generation."""
    results: List[GenerationResponse]
    total: int
    successful: int
    failed: int
