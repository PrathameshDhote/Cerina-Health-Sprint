"""
WebSocket endpoint for real-time bidirectional communication.
"""

import json
from typing import Dict, Any
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from utils.logger import logger
from graph.workflow import compile_workflow_async
from state.protocol_state import ProtocolState
from database import get_checkpointer


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, thread_id: str, websocket: WebSocket):
        """Accept and store a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[thread_id] = websocket
        logger.info(f"WebSocket connected for thread: {thread_id}")
    
    def disconnect(self, thread_id: str):
        """Remove a WebSocket connection."""
        if thread_id in self.active_connections:
            del self.active_connections[thread_id]
            logger.info(f"WebSocket disconnected for thread: {thread_id}")
    
    async def send_message(self, thread_id: str, message: Dict[str, Any]):
        """Send a message to a specific connection."""
        if thread_id in self.active_connections:
            websocket = self.active_connections[thread_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connections."""
        for thread_id, websocket in self.active_connections.items():
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {thread_id}: {str(e)}")


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    """
    WebSocket endpoint for real-time communication.
    
    Supports:
    - Real-time workflow updates
    - Bidirectional messaging
    - Live state synchronization
    
    Args:
        websocket: WebSocket connection
        thread_id: Thread identifier
    """
    await manager.connect(thread_id, websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Listen for messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                logger.info(f"WebSocket message received from {thread_id}: {message.get('type')}")
                
                # Handle different message types
                await handle_websocket_message(thread_id, message, websocket)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected by client: {thread_id}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    finally:
        manager.disconnect(thread_id)


async def handle_websocket_message(
    thread_id: str,
    message: Dict[str, Any],
    websocket: WebSocket
):
    """
    Handle incoming WebSocket messages.
    
    Args:
        thread_id: Thread identifier
        message: Message from client
        websocket: WebSocket connection
    """
    msg_type = message.get("type")
    
    if msg_type == "ping":
        # Heartbeat ping
        await websocket.send_json({
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    elif msg_type == "get_state":
        # Request current state
        try:
            checkpointer = get_checkpointer()
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = checkpointer.get(config)
            
            if checkpoint:
                state_dict = checkpoint["channel_values"]
                state = ProtocolState(**state_dict)
                
                await websocket.send_json({
                    "type": "state_update",
                    "state": state.to_summary_dict(),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Thread {thread_id} not found"
                })
        
        except Exception as e:
            logger.error(f"Error getting state: {str(e)}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    
    elif msg_type == "subscribe_updates":
        # Subscribe to real-time updates
        await websocket.send_json({
            "type": "subscription",
            "status": "subscribed",
            "thread_id": thread_id
        })
    
    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {msg_type}"
        })


async def send_workflow_update(
    thread_id: str,
    event_type: str,
    data: Dict[str, Any]
):
    """
    Send a workflow update to connected clients.
    
    Args:
        thread_id: Thread identifier
        event_type: Type of event
        data: Event data
    """
    message = {
        "type": "workflow_update",
        "event_type": event_type,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    await manager.send_message(thread_id, message)
