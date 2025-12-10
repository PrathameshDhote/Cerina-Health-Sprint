"""
Cerina Protocol Foundry - Main Application Entry Point

A sophisticated multi-agent AI system for autonomous CBT protocol generation
using LangGraph, FastAPI, and the Model Context Protocol (MCP).

Architecture:
- Multi-agent workflow with Supervisor-Worker pattern
- LangGraph for orchestration with database checkpointing
- FastAPI for REST API and WebSocket endpoints
- Real-time streaming with Server-Sent Events
- Human-in-the-loop approval mechanism
- MCP integration for cross-platform tool exposure

Author: Cerina Health - Agentic Architect Sprint
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError

from config import settings
from database import init_database, get_checkpointer
from utils.logger import logger, log_exception
from api.routes import router as api_router
from api.websocket import websocket_endpoint
from api.middleware import LoggingMiddleware
from api.error_handlers import (
    http_exception_handler,
    validation_exception_handler,
    generic_exception_handler
)
from graph.workflow import compile_workflow_async, create_workflow_diagram


# ASCII Art Banner
BANNER = """
╔═════════════════════════════════════════════════════╗
║                                                     ║
║   ██████╗███████╗██████╗ ██╗███╗   ██╗ █████╗       ║
║  ██╔════╝██╔════╝██╔══██╗██║████╗  ██║██╔══██╗      ║
║  ██║     █████╗  ██████╔╝██║██╔██╗ ██║███████║      ║
║  ██║     ██╔══╝  ██╔══██╗██║██║╚██╗██║██╔══██║      ║
║  ╚██████╗███████╗██║  ██║██║██║ ╚████║██║  ██║      ║
║   ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝      ║
║                                                     ║
║      Protocol Foundry - Multi-Agent System          ║
║       Autonomous CBT Protocol Generation            ║
║                                                     ║
╚═════════════════════════════════════════════════════╝
"""


# Application Lifecycle Management

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown operations:
    - Database initialization
    - Workflow compilation
    - Resource cleanup
    """
    # Startup
    logger.info(BANNER)
    logger.info("=" * 70)
    logger.info("Starting Cerina Protocol Foundry")
    logger.info("=" * 70)
    
    try:
        # Initialize database and checkpointer
        logger.info("Initializing database and checkpointing system...")
        init_database()
        checkpointer = get_checkpointer()
        logger.info(f"✓ Database initialized: {settings.database_type}")
        
        # Compile workflow
        logger.info("Compiling LangGraph workflow...")
        workflow = compile_workflow_async(checkpointer)
        app.state.workflow = workflow
        logger.info("✓ Workflow compiled successfully")
        
        # Log configuration
        logger.info("=" * 70)
        logger.info("Configuration:")
        logger.info(f"  Environment: {settings.app_env}")
        logger.info(f"  LLM Provider: {settings.primary_llm_provider}")
        logger.info(f"  OpenAI Model: {settings.openai_model}")
        logger.info(f"  Anthropic Model: {settings.anthropic_model}")
        logger.info(f"  Max Iterations: {settings.max_agent_iterations}")
        logger.info(f"  Database: {settings.database_type}")
        logger.info(f"  CORS Origins: {settings.cors_origins}")
        logger.info("=" * 70)
        
        # Display workflow diagram
        logger.info("\nWorkflow Architecture:")
        logger.info(create_workflow_diagram())
        
        logger.info("=" * 70)
        logger.info("🚀 Cerina Protocol Foundry is ready!")
        logger.info(f"📡 API Server: http://{settings.api_host}:{settings.api_port}")
        logger.info(f"📚 API Docs: http://{settings.api_host}:{settings.api_port}/docs")
        logger.info(f"🔧 MCP Server: Port {settings.mcp_server_port}")
        logger.info("=" * 70)
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        log_exception(e, "Startup failed")
        raise
    
    # Shutdown
    logger.info("=" * 70)
    logger.info("Shutting down Cerina Protocol Foundry...")
    logger.info("=" * 70)
    
    try:
        # Cleanup resources
        logger.info("Cleaning up resources...")
        
        # Close database connections
        # (handled automatically by SQLAlchemy/SQLite)
        
        logger.info("✓ Cleanup completed")
        logger.info("👋 Cerina Protocol Foundry stopped successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        log_exception(e, "Shutdown error")


# Create FastAPI Application

app = FastAPI(
    title="Cerina Protocol Foundry",
    description="""
    **Multi-Agent AI System for Autonomous CBT Protocol Generation**
    
    ## Features
    
    - 🤖 **Multi-Agent Architecture**: Supervisor-Worker pattern with specialized agents
    - 🔄 **LangGraph Workflow**: Sophisticated orchestration with state management
    - 💾 **Database Checkpointing**: Full workflow persistence and recovery
    - 🎯 **Human-in-Loop**: Approval mechanism with halt/resume capability
    - 📊 **Real-Time Streaming**: Server-Sent Events for live updates
    - 🔌 **WebSocket Support**: Bidirectional communication
    - 🛡️ **Safety First**: Dedicated safety validation agent
    - ✨ **Quality Assurance**: Clinical quality critic with empathy scoring
    - 🔧 **MCP Integration**: Model Context Protocol for tool exposure
    
    ## Agent Topology
    
    1. **Supervisor**: Orchestrates workflow and routing decisions
    2. **CBT Drafter**: Generates therapeutic exercise content
    3. **Safety Guardian**: Validates for safety and liability risks
    4. **Clinical Critic**: Evaluates quality and empathy
    
    ## Workflow
    
    ```
    User Request → Initialize → Supervisor Loop:
        ├─> Drafter (generate/revise)
        ├─> Safety Guardian (validate)
        ├─> Clinical Critic (assess quality)
        └─> Decision (continue/halt)
    → Halt for Human Review
    → Human Approval/Rejection
    → Finalize
    ```
    
    ## Quick Start
    
    1. **Generate Protocol**: POST `/api/generate` with user intent
    2. **Check State**: GET `/api/state/{thread_id}`
    3. **Review & Approve**: POST `/api/resume/{thread_id}` with action
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# CORS Middleware Configuration

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Thread-ID", "X-Process-Time"]
)


# Custom Middleware

app.add_middleware(LoggingMiddleware)


# Exception Handlers

app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


# Include API Routes

app.include_router(api_router)


# Root Endpoints

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/info")
async def info():
    """
    Get application information.
    
    Returns basic information about the service, configuration, and capabilities.
    """
    return {
        "name": "Cerina Protocol Foundry",
        "version": "1.0.0",
        "description": "Multi-Agent AI System for Autonomous CBT Protocol Generation",
        "architecture": {
            "pattern": "Supervisor-Worker",
            "agents": [
                {"name": "Supervisor", "role": "Workflow Orchestrator"},
                {"name": "CBT_Drafter", "role": "Clinical Content Generator"},
                {"name": "Safety_Guardian", "role": "Risk Assessment & Safety"},
                {"name": "Clinical_Critic", "role": "Quality Assessment"}
            ],
            "workflow_engine": "LangGraph",
            "state_management": "Pydantic with Database Checkpointing"
        },
        "configuration": {
            "environment": settings.app_env,
            "llm_provider": settings.primary_llm_provider,
            "max_iterations": settings.max_agent_iterations,
            "database_type": settings.database_type
        },
        "capabilities": [
            "Autonomous CBT protocol generation",
            "Multi-agent collaboration",
            "Safety validation",
            "Quality assessment with empathy scoring",
            "Human-in-loop approval",
            "Real-time streaming updates",
            "Database persistence and recovery",
            "MCP integration"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "generate": "/api/generate",
            "generate_stream": "/api/generate/stream",
            "state": "/api/state/{thread_id}",
            "resume": "/api/resume/{thread_id}",
            "websocket": "/ws/{thread_id}"
        }
    }


@app.get("/workflow/diagram")
async def workflow_diagram():
    """
    Get the workflow diagram as text.
    
    Returns a text representation of the agent workflow topology.
    """
    return {
        "diagram": create_workflow_diagram(),
        "format": "text"
    }


# WebSocket Endpoint

@app.websocket("/ws/{thread_id}")
async def websocket_route(websocket: WebSocket, thread_id: str):
    """
    WebSocket endpoint for real-time bidirectional communication.
    
    Args:
        websocket: WebSocket connection
        thread_id: Thread identifier for the workflow
    """
    await websocket_endpoint(websocket, thread_id)


# Development Helpers

if settings.is_development:
    
    @app.get("/dev/test-agents")
    async def test_agents():
        """
        Test agent initialization (development only).
        
        Returns status of all agents.
        """
        from agents import (
            CBTDrafterAgent,
            SafetyGuardianAgent,
            ClinicalCriticAgent,
            SupervisorAgent
        )
        
        try:
            drafter = CBTDrafterAgent()
            safety = SafetyGuardianAgent()
            critic = ClinicalCriticAgent()
            supervisor = SupervisorAgent()
            
            return {
                "status": "success",
                "agents": {
                    "drafter": {
                        "name": drafter.name,
                        "role": drafter.role,
                        "temperature": drafter.temperature
                    },
                    "safety_guardian": {
                        "name": safety.name,
                        "role": safety.role,
                        "temperature": safety.temperature
                    },
                    "clinical_critic": {
                        "name": critic.name,
                        "role": critic.role,
                        "temperature": critic.temperature
                    },
                    "supervisor": {
                        "name": supervisor.name,
                        "role": supervisor.role,
                        "max_iterations": supervisor.max_iterations
                    }
                }
            }
        except Exception as e:
            log_exception(e, "Agent test failed")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": str(e)
                }
            )
    
    @app.get("/dev/test-database")
    async def test_database():
        """
        Test database connection (development only).
        
        Returns database connection status.
        """
        try:
            checkpointer = get_checkpointer()
            
            return {
                "status": "success",
                "database_type": settings.database_type,
                "database_url": settings.database_url,
                "checkpointer": str(type(checkpointer).__name__)
            }
        except Exception as e:
            log_exception(e, "Database test failed")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": str(e)
                }
            )


# CLI Entry Point

def main():
    """
    Main entry point for CLI execution.
    
    Starts the Uvicorn server with configuration from settings.
    """
    import uvicorn
    
    # Log startup message
    print(BANNER)
    print("\nStarting Cerina Protocol Foundry server...")
    print(f"Environment: {settings.app_env}")
    print(f"Host: {settings.api_host}")
    print(f"Port: {settings.api_port}")
    print("\n" + "=" * 70 + "\n")
    
    # Run server
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
        access_log=settings.enable_debug_logging
    )


if __name__ == "__main__":
    main()
