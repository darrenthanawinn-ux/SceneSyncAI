"""
SceneSync AI - Configuration Module
=====================================
Centralized configuration for the SceneSync AI pre-production copilot.

Loads all runtime configuration from environment variables (12-factor style),
with sane local-dev defaults so the application boots cleanly on Replit even
before real Google Cloud credentials are attached (MOCK_MODE).

Also implements the Google Cloud Secret Manager *pattern*: if a
`GOOGLE_CLOUD_PROJECT` is configured and the `google-cloud-secret-manager`
library is available, secret values (like API keys) are resolved from
Secret Manager first, falling back to plain environment variables. This lets
the same code path work both locally (.env) and in production (Secret
Manager), without ever hard-coding a credential.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("scenesync.config")


def _resolve_secret(secret_id: str, project_id: Optional[str], fallback_env: Optional[str]) -> Optional[str]:
    """
    Resolve a secret value using the Google Cloud Secret Manager pattern.

    Resolution order:
      1. Google Cloud Secret Manager (if project_id is set and the client
         library is installed and reachable).
      2. Plain environment variable fallback.

    This function NEVER raises: any failure quietly falls back to the
    environment variable so local/mock development is never blocked by
    missing cloud credentials.
    """
    if project_id:
        try:
            from google.cloud import secretmanager  # type: ignore

            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            value = response.payload.data.decode("UTF-8").strip()
            if value:
                logger.info("Resolved secret '%s' from Secret Manager.", secret_id)
                return value
        except Exception as exc:  # noqa: BLE001 - intentional broad catch, secrets are best-effort
            logger.debug("Secret Manager lookup for '%s' failed (%s); using env fallback.", secret_id, exc)

    return os.environ.get(fallback_env) if fallback_env else None


class Settings(BaseSettings):
    """
    Application-wide settings, populated from environment variables.

    All variables can be set in a local `.env` file (see `.env.example`) or
    injected directly as Replit "Secrets" / Cloud Run environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_NAME: str = "SceneSync AI"
    APP_ENV: str = Field(default="development", description="development | production")
    APP_HOST: str = Field(default="0.0.0.0")
    APP_PORT: int = Field(default=8000)
    LOG_LEVEL: str = Field(default="INFO")

    # When true (or when no valid GCP project/credentials are detected),
    # SceneSync AI runs its full multi-agent pipeline using deterministic,
    # locally-computed heuristics and generated placeholder art instead of
    # live Vertex AI calls. This guarantees the demo NEVER errors out even
    # with zero cloud setup, while keeping 100% of the real integration
    # code paths intact and ready for production credentials.
    MOCK_MODE: bool = Field(default=False)

    # ------------------------------------------------------------------
    # Google Cloud / Vertex AI
    # ------------------------------------------------------------------
    GOOGLE_CLOUD_PROJECT: Optional[str] = Field(default=None)
    GOOGLE_CLOUD_LOCATION: str = Field(default="us-central1")
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(default=None)

    GEMINI_MODEL: str = Field(default="gemini-2.0-flash-001")
    GEMINI_REASONING_MODEL: str = Field(default="gemini-2.0-flash-001")
    IMAGEN_MODEL: str = Field(default="imagen-3.0-generate-002")

    # Vertex AI Search (grounding) data store, e.g.
    # projects/<proj>/locations/global/collections/default_collection/dataStores/<id>
    VERTEX_SEARCH_DATASTORE_ID: Optional[str] = Field(default=None)
    VERTEX_SEARCH_LOCATION: str = Field(default="global")

    # ------------------------------------------------------------------
    # Agent Development Kit (ADK)
    # ------------------------------------------------------------------
    ADK_APP_NAME: str = Field(default="scenesync_pipeline")
    AGENT_ENGINE_RESOURCE_NAME: Optional[str] = Field(default=None)

    # ------------------------------------------------------------------
    # Generation tuning
    # ------------------------------------------------------------------
    MAX_SCENES_PER_SCRIPT: int = Field(default=40)
    STORYBOARD_ASPECT_RATIO: str = Field(default="16:9")
    STORYBOARD_IMAGES_PER_SCENE: int = Field(default=1)
    MAX_UPLOAD_MB: int = Field(default=15)

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    ALLOWED_ORIGINS: str = Field(default="*")

    @field_validator("MOCK_MODE", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        return bool(v)

    @property
    def cors_origins(self) -> List[str]:
        if self.ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def effective_mock_mode(self) -> bool:
        """
        MOCK_MODE is forced on automatically whenever the minimum viable
        Google Cloud configuration (a project id) is absent, so the app
        never crashes on missing credentials — a core "zero-error" rule
        for this build.
        """
        if self.MOCK_MODE:
            return True
        if not self.GOOGLE_CLOUD_PROJECT:
            return True
        return False

    # ------------------------------------------------------------------
    # Gemini safety settings (guardrails)
    # ------------------------------------------------------------------
    @property
    def safety_settings(self) -> dict:
        """
        Returns Gemini safety threshold configuration used on every
        generative call. Categories map to Vertex AI's
        HarmCategory / HarmBlockThreshold enums (imported lazily inside
        agent.py to avoid a hard dependency at config-import time).
        """
        return {
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        }

    def resolved_secret(self, secret_id: str, fallback_env: str) -> Optional[str]:
        """Public helper exposing the Secret Manager resolution pattern."""
        return _resolve_secret(secret_id, self.GOOGLE_CLOUD_PROJECT, fallback_env)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — import and call get_settings() everywhere."""
    settings = Settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    if settings.effective_mock_mode:
        logger.warning(
            "SceneSync AI is running in MOCK_MODE (no GOOGLE_CLOUD_PROJECT / credentials "
            "detected, or MOCK_MODE explicitly set). All agent + Imagen calls will use "
            "deterministic local simulations so the app runs with zero errors. Set "
            "GOOGLE_CLOUD_PROJECT and authenticate via `gcloud auth application-default login` "
            "or a service account to enable live Vertex AI + Imagen 3 generation."
        )
    return settings
