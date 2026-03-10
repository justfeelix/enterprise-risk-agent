import json
import re
from pathlib import Path
from typing import Any, Dict, List


def load_attack_seeds(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL attack prompts from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Attack seed file not found: {file_path}")

    attacks: List[Dict[str, Any]] = []
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

            if "prompt" not in item:
                raise ValueError(
                    f"Attack line {line_no} is missing required key: 'prompt'"
                )

            attack_type = str(
                item.get("attack_type") or item.get("category") or "generic"
            )
            if "category" not in item:
                item["category"] = attack_type
            if "id" not in item:
                prefix = (
                    re.sub(r"[^a-z0-9]+", "_", attack_type.lower()).strip("_")
                    or "attack"
                )
                item["id"] = f"{prefix}_{line_no:03d}"

            attacks.append(item)

    if not attacks:
        raise ValueError(f"No attack rows found in {file_path}")
    return attacks
