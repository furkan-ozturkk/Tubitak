"""Centralized configuration for the LogRouter benchmark extension.

Holds the Ollama connection settings and model names (14B, 32B, and coder
models), the Level-1 and Level-2 router thresholds, the Postgres connection
settings, and the chunking window size.

The team's shared GPU machine exposes an Ollama endpoint over VPN, as
described in the GPU/LLM usage guide. ``OLLAMA_HOST`` defaults to that
shared endpoint but can be overridden with an environment variable to point
at a local Ollama instance instead.

Reference:
    LogRouter paper, Section III-A and III-E (Hardware and Implementation).
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Runtime configuration for the LogRouter benchmark extension.

    Attributes:
        ollama_host: Base URL of the Ollama server.
        ollama_default_model: Model used for standard chat and generation
            calls.
        ollama_light_model: Smaller model recommended for quick
            experiments, to keep the shared GPU resources free for other
            users.
        ollama_request_timeout: Timeout in seconds for Ollama HTTP
            requests.
        postgres_host: Hostname of the Postgres service.
        postgres_port: Port of the Postgres service.
        postgres_user: Postgres username.
        postgres_password: Postgres password.
        postgres_db: Postgres database name.
        level1_threshold: Default Level-1 router threshold for the SQL and
            keyword signal families.
        level1_event_threshold: Default Level-1 router threshold for the
            event signal family.
        level2_threshold: Default Level-2 router complexity threshold.
        chunk_window_size: Number of log lines per chunk.
        chunk_overlap: Number of overlapping lines between consecutive
            chunks.
    """

    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://10.15.33.66:11435")
    ollama_default_model: str = os.environ.get("OLLAMA_DEFAULT_MODEL", "qwen2.5-coder:14b")
    ollama_light_model: str = os.environ.get("OLLAMA_LIGHT_MODEL", "qwen2.5-coder:7b")
    ollama_request_timeout: int = int(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "600"))

    postgres_host: str = os.environ.get("POSTGRES_HOST", "postgres")
    postgres_port: int = int(os.environ.get("POSTGRES_PORT", "5432"))
    postgres_user: str = os.environ.get("POSTGRES_USER", "logrouter")
    postgres_password: str = os.environ.get("POSTGRES_PASSWORD", "logrouter_pass")
    postgres_db: str = os.environ.get("POSTGRES_DB", "logrouter_db")

    level1_threshold: float = float(os.environ.get("LEVEL1_THRESHOLD", "0.3"))
    level1_event_threshold: float = float(os.environ.get("LEVEL1_EVENT_THRESHOLD", "0.5"))
    level2_threshold: float = float(os.environ.get("LEVEL2_THRESHOLD", "0.5"))

    chunk_window_size: int = int(os.environ.get("CHUNK_WINDOW_SIZE", "25"))
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "3"))

    @classmethod
    def from_env(cls) -> "Config":
        """Builds a Config instance from environment variables.

        Returns:
            Config: A configuration populated from the current process
            environment, falling back to the defaults documented on each
            field.
        """
        return cls()
