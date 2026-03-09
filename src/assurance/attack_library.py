import json
from pathlib import Path
from typing import Dict, List


def load_attack_seeds(file_path: str) -> List[Dict[str, str]]:
    """Load JSONL attack prompts from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Attack seed file not found: {file_path}")

    attacks: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            clean = line.strip()
            if not clean:
                continue
            try:
                item = json.loads(clean)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_no} in {file_path}: {exc}"
                ) from exc

            required = {"id", "category", "prompt"}
            missing = required - set(item.keys())
            if missing:
                raise ValueError(
                    f"Attack line {line_no} is missing required keys: {sorted(missing)}"
                )
            attacks.append(item)

    if not attacks:
        raise ValueError(f"No attack rows found in {file_path}")
    return attacks
