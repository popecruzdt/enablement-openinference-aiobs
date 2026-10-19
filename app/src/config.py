"""
Application configuration for AI Travel Advisor
Handles environment configuration and application settings
"""

import os
import logging
from functools import lru_cache
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Basic Application Configuration
    service_name: str = os.getenv("SERVICE_NAME", "ai-travel-advisor")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    app_name: str = "AI Travel Advisor"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Ollama LLM Configuration
    ollama_endpoint: str = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    ai_model: str = os.getenv("AI_MODEL", "orca-mini:3b")
    ai_embedding_model: str = os.getenv("AI_EMBEDDING_MODEL", "nomic-embed-text")
    ai_temperature: float = float(os.getenv("AI_TEMPERATURE", "0.7"))

    # Weaviate Vector Database Configuration
    weaviate_endpoint: str = os.getenv("WEAVIATE_ENDPOINT", "localhost")
    weaviate_port: int = int(os.getenv("WEAVIATE_PORT", "8080"))
    weaviate_scheme: str = os.getenv("WEAVIATE_SCHEME", "http")

    # RAG Configuration
    max_prompt_length: int = int(os.getenv("MAX_PROMPT_LENGTH", "50"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "3"))
    force_reindex: bool = os.getenv("FORCE_REINDEX", "false").lower() == "true"
    min_kb_objects: int = int(os.getenv("MIN_KB_OBJECTS", "450"))

    # Agentic Workflow Configuration
    agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "8"))

    # Observability Configuration
    otel_enabled: bool = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    otel_endpoint: str = os.getenv("OTEL_ENDPOINT", "https://localhost:4317")
    dynatrace_api_url: str = os.getenv("DYNATRACE_API_URL", "")
    dynatrace_api_token: str = os.getenv("DYNATRACE_API_TOKEN", "")
    api_token: str = os.getenv("API_TOKEN", "")

    # Traceloop Configuration
    traceloop_telemetry: bool = os.getenv("TRACELOOP_TELEMETRY", "false").lower() == "true"

    # Application Paths
    destinations_path: str = os.getenv("DESTINATIONS_PATH", "destinations")
    public_path: str = os.getenv("PUBLIC_PATH", "public")
    rag_prompt_path: str = os.getenv("RAG_PROMPT_PATH", "prompts/rag_instructions.txt")
    agentic_prompt_path: str = os.getenv("AGENTIC_PROMPT_PATH", "prompts/agentic_instructions.txt")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached singleton)
    
    Returns:
        Settings: Application configuration instance
    """
    settings = Settings()
    
    # Log configuration summary
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug: {settings.debug}")
    logger.info(f"Ollama endpoint: {settings.ollama_endpoint}")
    logger.info(f"Ollama model: {settings.ai_model}")
    logger.info(f"Ollama temperature: {settings.ai_temperature}")
    logger.info(f"Weaviate: {settings.weaviate_scheme}://{settings.weaviate_endpoint}:{settings.weaviate_port}")
    
    return settings


# Module-level instance for compatibility
settings = get_settings()

