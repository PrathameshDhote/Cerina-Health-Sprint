"""
Database configuration and LangGraph checkpointer setup.
Supports both SQLite (development) and PostgreSQL (production).
Handles both sync and async operations.
"""

import sqlite3
import aiosqlite
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, Union, AsyncGenerator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from config import settings
from utils.logger import logger


# SQLAlchemy setup
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_type == "sqlite" else {},
    echo=settings.enable_debug_logging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()


# Global checkpointer instances
_sync_checkpointer = None


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_checkpointer():
    """
    Get the synchronous LangGraph checkpointer based on database type.
    
    Returns:
        SqliteSaver or PostgresSaver: Configured checkpointer instance
    """
    global _sync_checkpointer
    
    if _sync_checkpointer is not None:
        return _sync_checkpointer
    
    if settings.database_type == "sqlite":
        logger.info(f"Initializing SQLite checkpointer: {settings.database_url}")
        
        # Create connection for checkpointer
        conn = sqlite3.connect(
            settings.database_url.replace("sqlite:///./", ""),
            check_same_thread=False
        )
        
        _sync_checkpointer = SqliteSaver(conn)
        logger.info("SQLite checkpointer initialized successfully")
        return _sync_checkpointer
    
    elif settings.database_type == "postgresql":
        logger.info(f"Initializing PostgreSQL checkpointer: {settings.database_url}")
        
        # PostgreSQL connection string
        _sync_checkpointer = PostgresSaver.from_conn_string(settings.database_url)
        logger.info("PostgreSQL checkpointer initialized successfully")
        return _sync_checkpointer
    
    else:
        raise ValueError(f"Unsupported database type: {settings.database_type}")


@asynccontextmanager
async def get_async_checkpointer():
    """
    Get the asynchronous LangGraph checkpointer as a context manager.
    
    Used for streaming and async operations.
    
    Usage:
        async with get_async_checkpointer() as checkpointer:
            # use checkpointer
    
    Yields:
        AsyncSqliteSaver or AsyncPostgresSaver: Configured async checkpointer instance
    """
    if settings.database_type == "sqlite":
        logger.info(f"Initializing Async SQLite checkpointer: {settings.database_url}")
        
        # Get database path
        db_path = settings.database_url.replace("sqlite:///./", "")
        
        # Create async connection using aiosqlite
        async with aiosqlite.connect(db_path) as conn:
            # Create checkpointer with the connection
            checkpointer = AsyncSqliteSaver(conn)
            
            # Setup tables
            await checkpointer.setup()
            
            logger.info("Async SQLite checkpointer initialized successfully")
            
            try:
                yield checkpointer
            finally:
                # Cleanup happens automatically when context exits
                logger.info("Async SQLite checkpointer closed")
    
    elif settings.database_type == "postgresql":
        logger.info(f"Initializing Async PostgreSQL checkpointer: {settings.database_url}")
        
        # Use the async context manager for PostgreSQL
        async with AsyncPostgresSaver.from_conn_string(
            settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
        ) as checkpointer:
            await checkpointer.setup()
            
            logger.info("Async PostgreSQL checkpointer initialized successfully")
            
            try:
                yield checkpointer
            finally:
                logger.info("Async PostgreSQL checkpointer closed")
    
    else:
        raise ValueError(f"Unsupported database type: {settings.database_type}")


def init_database():
    """Initialize database tables."""
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Initialize checkpointer tables (sync only during startup)
    checkpointer = get_checkpointer()
    checkpointer.setup()
    
    logger.info("Database initialization complete")


@contextmanager
def get_db_context():
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize database on module import
if __name__ != "__main__":
    try:
        init_database()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
