"""
State management for the Cerina Protocol Foundry.

This module defines the shared state structure (the "blackboard") that agents
collaboratively modify throughout the protocol generation workflow.
"""

from .protocol_state import (
    ProtocolState,
    ScratchpadEntry,
    DraftVersion,
    SafetyFlag,
    CriticFeedback,
    SupervisorDecision,
    MetadataScores
)
from .schemas import (
    GenerationRequest,
    GenerationResponse,
    StateResponse,
    ResumeRequest,
    ApprovalAction,
    ProtocolHistory
)

__all__ = [
    # State models
    "ProtocolState",
    "ScratchpadEntry",
    "DraftVersion",
    "SafetyFlag",
    "CriticFeedback",
    "SupervisorDecision",
    "MetadataScores",
    # API schemas
    "GenerationRequest",
    "GenerationResponse",
    "StateResponse",
    "ResumeRequest",
    "ApprovalAction",
    "ProtocolHistory",
]
