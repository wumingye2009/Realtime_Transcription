from pathlib import Path


class OutputPathService:
    """Validates local output directories for browser-driven local development."""

    def resolve_output_dir(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.mkdir(parents=True, exist_ok=True)
        return path
