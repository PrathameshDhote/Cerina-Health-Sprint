"""
Configuration management for the Cerina Protocol Foundry backend.
Loads environment variables and provides typed configuration objects.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # LLM Configuration
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    primary_llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = "gpt-4-turbo-preview"
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    
    # Database
    database_type: Literal["sqlite", "postgresql"] = "sqlite"
    database_url: str = "sqlite:///./cerina_protocol.db"
    
    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    max_agent_iterations: int = 5
    enable_debug_logging: bool = True
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Agent Configuration
    drafter_temperature: float = 0.7
    safety_temperature: float = 0.2
    critic_temperature: float = 0.5
    supervisor_temperature: float = 0.3
    
    # MCP Server
    mcp_server_name: str = "cerina-foundry"
    mcp_server_version: str = "1.0.0"
    mcp_server_port: int = 8001
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == "production"


# Global settings instance
settings = Settings()
