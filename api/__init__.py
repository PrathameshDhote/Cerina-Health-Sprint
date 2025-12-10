"""
API layer for the Cerina Protocol Foundry.

This module provides REST endpoints and WebSocket connections for:
- Protocol generation initiation
- State inspection and retrieval
- Human-in-loop approval/rejection
- Real-time workflow streaming
- Health checks and monitoring
"""

from .routes import router
from .websocket import websocket_endpoint
from .dependencies import get_current_state, verify_api_key

__all__ = [
    "router",
    "websocket_endpoint",
    "get_current_state",
    "verify_api_key",
]
