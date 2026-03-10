"""Small developer sanity check for prompt-injection simulations."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.target_bot import AgentSupportBot
from src.assurance.attack_library import load_attack_seeds
from src.attacks.retriever import retrieve_documents


def _build_retrieved_documents(
    attack: Dict[str, Any], prompt: str
) -> List[Dict[str, str]]:
    retrieved_documents = retrieve_documents(prompt, k=2)
    malicious_document = attack.get("retrieved_document")
    if malicious_document:
        retrieved_documents.append(
            {
                "doc_id": str(
                    attack.get("doc_id")
                    or attack.get("attack_id")
                    or attack.get("id")
                    or "prompt_injection_doc"
                ),
                "content": str(malicious_document),
                "source": "simulated_attack",
            }
        )
    return retrieved_documents


def main() -> None:
    seed_file = REPO_ROOT / "fixtures" / "prompt_injection_seed.jsonl"
    attacks = load_attack_seeds(str(seed_file))
    bot = AgentSupportBot()

    for attack in attacks:
        prompt = str(attack["prompt"])
        retrieved_documents = _build_retrieved_documents(attack, prompt)
        response = bot.respond(prompt, retrieved_docs=retrieved_documents)
        trace = bot.get_last_trace()

        attack_id = str(attack.get("attack_id") or attack.get("id") or "unknown")
        tool_calls = trace.get("tool_calls", [])

        print("-----------------------------")
        print(f"Attack: {attack_id}")
        print(f"Prompt: {prompt}")
        print("Retrieved docs:")
        for doc in retrieved_documents:
            print(
                f"  - {doc.get('doc_id', 'unknown_doc')} ({doc.get('source', 'knowledge_base')})"
            )
        print(f"Response: {response}")
        print(f"Tool calls: {json.dumps(tool_calls, ensure_ascii=True)}")
    print("-----------------------------")


if __name__ == "__main__":
    main()
