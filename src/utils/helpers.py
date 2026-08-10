"""BrandPulse AI — Utility Functions.

Shared helpers for configuration loading, logging setup,
ID generation, and display formatting.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler


# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


# ── Logging Setup ────────────────────────────────────────────────────


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure rich logging for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    return logging.getLogger("brandpulse")


logger = setup_logging()
console = Console()


# ── Environment ──────────────────────────────────────────────────────


def load_env() -> None:
    """Load environment variables from .env file."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded environment from .env")
    else:
        logger.warning(
            "No .env file found. Using system environment variables."
        )


def get_api_key() -> str:
    """Get the Google API key from environment.

    Returns:
        The API key string.

    Raises:
        EnvironmentError: If the key is not configured.
    """
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found. "
            "Set it in your .env file or environment."
        )
    return key


def get_groq_key() -> str:
    """Get the Groq API key from environment.

    Returns:
        The Groq API key string.

    Raises:
        EnvironmentError: If the key is not configured.
    """
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. "
            "Set it in your .env file or environment."
        )
    return key


def get_llm_provider() -> str:
    """Get the configured LLM provider.

    Returns:
        'groq' or 'gemini'
    """
    return os.getenv("LLM_PROVIDER", "groq").lower().strip()


def get_llm(temperature: float = 0.1, max_tokens: int = 500):
    """Create the LLM instance based on configured provider.

    Supports:
    - 'groq': Uses Llama 3.1 8B via Groq (free, fast, 30 RPM)
    - 'gemini': Uses Gemini 2.0 Flash via Google (free tier has limits)

    Args:
        temperature: LLM temperature (0.0 - 1.0).
        max_tokens: Maximum output tokens.

    Returns:
        A LangChain chat model instance.
    """
    provider = get_llm_provider()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=get_groq_key(),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=get_api_key(),
            temperature=temperature,
            max_output_tokens=max_tokens,
        )


# ── Config Loading ───────────────────────────────────────────────────


def load_brands_config() -> dict[str, Any]:
    """Load brand configuration from YAML.

    Returns:
        Parsed brand configuration dictionary.
    """
    config_path = CONFIG_DIR / "brands.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Brand config not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_brand_names() -> list[str]:
    """Get list of all configured brand slugs."""
    config = load_brands_config()
    return list(config.get("brands", {}).keys())


def get_brand_display_name(brand_slug: str) -> str:
    """Get the display name for a brand slug."""
    config = load_brands_config()
    brand = config.get("brands", {}).get(brand_slug, {})
    return brand.get("display_name", brand_slug.title())


def get_brand_products(brand_slug: str) -> list[dict]:
    """Get products for a specific brand."""
    config = load_brands_config()
    brand = config.get("brands", {}).get(brand_slug, {})
    return brand.get("products", [])


def get_settings() -> dict[str, Any]:
    """Get global settings from brand config."""
    config = load_brands_config()
    return config.get("settings", {})


# ── Data Loading ─────────────────────────────────────────────────────


def load_sample_reviews() -> list[dict]:
    """Load sample reviews from JSON data file.

    Returns:
        List of review dictionaries.
    """
    data_path = DATA_DIR / "sample_reviews.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Sample data not found at {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: list[dict], filename: str = "results.json") -> Path:
    """Save processing results to JSON.

    Args:
        results: List of result dictionaries to save.
        filename: Output filename.

    Returns:
        Path to the saved file.
    """
    output_dir = DATA_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")
    return output_path


# ── ID Generation ────────────────────────────────────────────────────


def generate_id(prefix: str = "BP") -> str:
    """Generate a unique ID with a prefix.

    Args:
        prefix: Short string prefix for the ID.

    Returns:
        String like 'BP-a1b2c3d4'.
    """
    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}-{short_uuid}"


def generate_alert_id() -> str:
    """Generate a unique alert ID."""
    return generate_id("ALT")


def generate_insight_id() -> str:
    """Generate a unique insight ID."""
    return generate_id("INS")


# ── Date Helpers ─────────────────────────────────────────────────────


def parse_date(date_str: str) -> datetime:
    """Parse date string in common formats.

    Args:
        date_str: Date string to parse.

    Returns:
        Parsed datetime object.

    Raises:
        ValueError: If format is not recognized.
    """
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {date_str}")


def days_ago(dt: datetime) -> int:
    """Calculate how many days ago a datetime was."""
    return (datetime.now() - dt).days


# ── Display Helpers ──────────────────────────────────────────────────


def severity_color(severity: str) -> str:
    """Map alert severity to rich console color.

    Args:
        severity: Severity level string.

    Returns:
        Rich markup color string.
    """
    colors = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "cyan",
        "info": "dim",
    }
    return colors.get(severity, "white")


def severity_emoji(severity: str) -> str:
    """Map alert severity to an emoji indicator."""
    emojis = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🔵",
        "info": "⚪",
    }
    return emojis.get(severity, "⚪")


def format_sentiment_score(score: float) -> str:
    """Format sentiment score with color indicator.

    Args:
        score: Sentiment score between -1.0 and 1.0.

    Returns:
        Rich-formatted string with directional indicator.
    """
    if score >= 0.3:
        return f"[green]▲ {score:+.2f}[/green]"
    elif score <= -0.3:
        return f"[red]▼ {score:+.2f}[/red]"
    else:
        return f"[yellow]● {score:+.2f}[/yellow]"
