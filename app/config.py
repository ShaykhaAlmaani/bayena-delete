import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_API_KEY_FALLBACK_1: str = os.getenv("GROQ_API_KEY_FALLBACK_1", "")
    GROQ_API_KEY_FALLBACK_2: str = os.getenv("GROQ_API_KEY_FALLBACK_2", "")
    DASHBOARD_AGENT_MODEL: str = os.getenv("DASHBOARD_AGENT_MODEL", "llama-3.3-70b-versatile")
    ANALYST_AGENT_MODEL: str = os.getenv("ANALYST_AGENT_MODEL", "llama-3.1-8b-instant")

    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    DASH_PORT: int = int(os.getenv("DASH_PORT", "8050"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

    @property
    def api_keys(self) -> list:
        """All configured API keys in priority order."""
        return [k for k in [
            self.GROQ_API_KEY,
            self.GROQ_API_KEY_FALLBACK_1,
            self.GROQ_API_KEY_FALLBACK_2,
        ] if k]


settings = Settings()
