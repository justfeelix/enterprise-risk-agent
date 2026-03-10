"""
Runner
------
Orchestrates a full AI assurance run:

  1. (optional) Ingest EU AI Act PDFs → rebuild FAISS vector index
  2. Load attack seeds from fixtures/
  3. Send each prompt to the selected target bot
  4. Judge each response with the selected judge mode
  5. Write a timestamped JSONL evidence file to outputs/evidence_runs/
  6. Print a run summary to stdout

Usage:
  # Baseline: hardcoded bot + rule-based judge
  python -m src.assurance.runner

  # Rebuild EU AI Act index first
  python -m src.assurance.runner --ingest

  # Real LLM target bot (mirrors an actual enterprise chatbot)
  python -m src.assurance.runner --target llm

  # Full pipeline: real bot + semantic LLM judge + legal citations
  python -m src.assurance.runner --target llm --judge both

  # Generate report after run
  python -m src.assurance.report --latest --open
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from ..agent.target_bot import AgentSupportBot, LLMSupportBot, VulnerableSupportBot
from .attack_library import load_attack_seeds
from .judge import combined_judge
from .legal_oracle import LegalOracle


def _build_index() -> None:
    """Ingest PDFs from data/ and persist a fresh FAISS index to vector_db/."""
    # Import here to keep the module-level imports minimal.
    from ..document_processor import ComplianceDocumentProcessor

    processor = ComplianceDocumentProcessor()
    docs = processor.load_documents()
    if not docs:
        print("⚠️  No PDFs found in data/. Add the EU AI Act PDF and retry.")
        return
    chunks = processor.chunk_text(docs)
    processor.create_embeddings(chunks)


def run_assurance(
    seed_file: str,
    output_dir: str,
    build_index: bool = False,
    target_type: str = "hardcoded",
    judge_mode: str = "rules",
) -> Dict[str, int]:
    """Execute a full assurance run and return a summary dict.

    Parameters
    ----------
    target_type : "hardcoded" | "llm" | "agent"
        hardcoded – VulnerableSupportBot (deterministic, no API cost)
        llm       – LLMSupportBot (real Groq/OpenAI call per prompt)
        agent     – AgentSupportBot (LLM + enterprise tools + trace logging)
    judge_mode  : "rules" | "llm" | "both"
        rules – rule-based regex judge only (fast, deterministic)
        llm   – LLM semantic judge only
        both  – both judges merged (highest coverage)
    """
    use_llm_judge = judge_mode in ("llm", "both")

    # Step 1 – optionally (re)build the EU AI Act vector index.
    if build_index:
        print("\n🔨 Building EU AI Act vector index...")
        _build_index()

    # Step 2 – initialise the legal oracle (lazy-loads the FAISS index).
    oracle = LegalOracle()

    # Step 3 – select target bot.
    attacks = load_attack_seeds(seed_file)
    if target_type == "llm":
        print("🤖 Target: LLMSupportBot (real LLM — Groq/OpenAI)")
        bot = LLMSupportBot()
    elif target_type == "agent":
        print("🤖 Target: AgentSupportBot (LLM + tools runtime)")
        bot = AgentSupportBot()
    else:
        print("🤖 Target: VulnerableSupportBot (hardcoded baseline)")
        bot = VulnerableSupportBot()
    print(f"⚖️  Judge mode: {judge_mode}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"evidence_{run_id}.jsonl"

    verdict_counter: Counter[str] = Counter()
    violation_counter: Counter[str] = Counter()
    citation_count = 0  # how many records received EU AI Act legal citations

    with out_file.open("w", encoding="utf-8") as outfile:
        for attack in attacks:
            prompt = attack["prompt"]

            # Step 4a – get the target bot's response.
            response = bot.respond(prompt)
            trace = bot.get_last_trace() if hasattr(bot, "get_last_trace") else {}

            # Step 4b – judge with selected mode; oracle provides legal citations.
            judgment = combined_judge(
                prompt, response, oracle=oracle, use_llm_judge=use_llm_judge
            )

            verdict_counter[judgment["verdict"]] += 1
            for violation in judgment["violations"]:
                violation_counter[violation] += 1
            if judgment["legal_citations"]:
                citation_count += 1

            # Step 5 – write a complete, self-contained evidence record.
            record = {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "target_type": target_type,
                "judge_mode": judge_mode,
                "attack_id": attack["id"],
                "attack_category": attack["category"],
                "prompt": prompt,
                "response": response,
                "verdict": judgment["verdict"],
                "violations": judgment["violations"],
                "reasons": judgment["reasons"],
                "legal_citations": judgment["legal_citations"],
                "trace": trace,
            }
            outfile.write(json.dumps(record, ensure_ascii=True) + "\n")

    # Step 6 – print summary.
    summary = {
        "run_records": len(attacks),
        "pass_count": verdict_counter.get("pass", 0),
        "fail_count": verdict_counter.get("fail", 0),
        "records_with_legal_citations": citation_count,
    }
    print(f"\n✅ Saved evidence to: {out_file}")
    print("Run summary:")
    print(json.dumps(summary, indent=2))
    print("Violation counts:")
    print(json.dumps(dict(violation_counter), indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AI Assurance Copilot v1 red-team checks."
    )
    parser.add_argument(
        "--seed-file",
        default="fixtures/attacks_seed.jsonl",
        help="Path to attack seeds JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evidence_runs",
        help="Directory for evidence JSONL output.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Rebuild the EU AI Act vector index from data/ before running.",
    )
    parser.add_argument(
        "--target",
        choices=["hardcoded", "llm", "agent"],
        default="hardcoded",
        help=(
            "Target bot to attack. 'hardcoded' = deterministic baseline; "
            "'llm' = real LLM-backed FinVault support bot; "
            "'agent' = LLM-backed support agent with enterprise tool runtime "
            "(requires GROQ_API_KEY)."
        ),
    )
    parser.add_argument(
        "--judge",
        choices=["rules", "llm", "both"],
        default="rules",
        help=(
            "Judge mode. 'rules' = fast regex; 'llm' = semantic LLM judge; "
            "'both' = merged results for maximum coverage."
        ),
    )
    args = parser.parse_args()
    run_assurance(
        seed_file=args.seed_file,
        output_dir=args.output_dir,
        build_index=args.ingest,
        target_type=args.target,
        judge_mode=args.judge,
    )


if __name__ == "__main__":
    main()
