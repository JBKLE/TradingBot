"""Safe .env file reader/writer with backup and structure preservation."""
import shutil
from pathlib import Path

from dotenv import find_dotenv


def _resolve_env_path(env_path: str | None = None) -> Path:
    """Find the .env file using the same logic as python-dotenv."""
    if env_path:
        return Path(env_path)
    found = find_dotenv(usecwd=True)
    if found:
        return Path(found)
    # Fallback: project root (parent of src/)
    return Path(__file__).resolve().parent.parent / ".env"


def read_env_file(env_path: str | None = None) -> dict[str, str]:
    """Read all key=value pairs from .env file."""
    result: dict[str, str] = {}
    path = _resolve_env_path(env_path)
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def update_env_file(updates: dict[str, str], env_path: str | None = None) -> None:
    """Update specific keys in .env while preserving comments and structure.

    Creates a .env.bak backup before writing.
    """
    path = _resolve_env_path(env_path)
    if not path.exists():
        raise FileNotFoundError(f".env file not found: {path}")

    # Backup
    shutil.copy2(path, path.with_suffix(".env.bak"))

    lines = path.read_text(encoding="utf-8").splitlines()
    updated_keys: set[str] = set()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            lines[i] = f"{key}={updates[key]}"
            updated_keys.add(key)

    # Append keys that were not already in the file
    for key, value in updates.items():
        if key not in updated_keys:
            lines.append(f"{key}={value}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
