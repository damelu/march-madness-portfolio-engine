from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    ROOT / "configs" / "crawl_profiles",
    ROOT / "configs" / "schemas",
    ROOT / "data" / "landing" / "api",
    ROOT / "data" / "landing" / "crawl",
    ROOT / "data" / "landing" / "manual",
    ROOT / "data" / "raw" / "ncaa",
    ROOT / "data" / "raw" / "athletics_sites",
    ROOT / "data" / "raw" / "ratings",
    ROOT / "data" / "raw" / "injuries",
    ROOT / "data" / "raw" / "recruiting",
    ROOT / "data" / "raw" / "odds",
    ROOT / "data" / "raw" / "news_sentiment",
    ROOT / "data" / "staged" / "teams",
    ROOT / "data" / "staged" / "players",
    ROOT / "data" / "staged" / "games",
    ROOT / "data" / "staged" / "coaches",
    ROOT / "data" / "staged" / "events",
    ROOT / "data" / "features" / "selection_sunday",
    ROOT / "data" / "features" / "pre_tipoff",
    ROOT / "data" / "models" / "training",
    ROOT / "data" / "models" / "inference",
    ROOT / "data" / "models" / "artifacts",
    ROOT / "data" / "reference" / "venues",
    ROOT / "data" / "reference" / "geography",
    ROOT / "data" / "reference" / "officials",
    ROOT / "docs",
    ROOT / "logs",
    ROOT / "outputs",
    ROOT / "scripts",
    ROOT / "sql",
    ROOT / "tmp",
    ROOT / "README.md",
    ROOT / ".env.example",
    ROOT / "package.json",
    ROOT / "pyproject.toml",
]

REQUIRED_COMMANDS = ["curl", "jq", "node", "npm", "npx", "python3"]
OPTIONAL_COMMANDS = ["uv"]


def main() -> int:
    missing_paths = [str(path.relative_to(ROOT)) for path in REQUIRED_PATHS if not path.exists()]
    missing_commands = [cmd for cmd in REQUIRED_COMMANDS if shutil.which(cmd) is None]
    available_optional = [cmd for cmd in OPTIONAL_COMMANDS if shutil.which(cmd) is not None]

    print(f"project_root={ROOT}")
    print(f"missing_paths={missing_paths or 'none'}")
    print(f"missing_commands={missing_commands or 'none'}")
    print(f"optional_commands={available_optional or 'none'}")

    if missing_paths or missing_commands:
        return 1

    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
