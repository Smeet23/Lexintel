"""Worker-specific configuration using shared settings."""

from shared import settings

# Re-export shared settings for backward compatibility
# Workers use the shared configuration directly
__all__ = ["settings"]
