"""Configuration management using Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # AWS Configuration
    aws_region: str = Field(
        default="us-east-1",
        description="Default AWS region for capacity API calls",
    )
    aws_profile: str | None = Field(
        default=None,
        description="AWS profile name (uses default if not set)",
    )
    aws_account_id: str | None = Field(
        default=None,
        description="AWS account ID for filtering resources",
    )

    # Agent Configuration (Bedrock)
    bedrock_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock LLM API calls",
    )
    bedrock_model_id: str = Field(
        default="us.anthropic.claude-sonnet-4-20250514-v1:0",
        description="Bedrock model ID to use for the agent",
    )
    agent_max_retries: int = Field(
        default=2,
        description="Maximum retries for agent tool calls",
    )

    # Chainlit Configuration
    chainlit_host: str = Field(
        default="0.0.0.0",
        description="Host for Chainlit server",
    )
    chainlit_port: int = Field(
        default=8000,
        description="Port for Chainlit server",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )


# Global settings instance
settings = Settings()
