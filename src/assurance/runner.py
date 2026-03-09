"""
Runner
------
Orchestrates a full AI assurance run:

  1. (optional) Ingest EU AI Act PDFs → rebuild FAISS vector index
  2. Load attack seeds from fixtures/
  3. Send each prompt to the VulnerableSupportBot (target)
  4. Judge each response (rule-based + EU AI Act legal citations)
  5. Write a timestamped JSONL evidence file to outputs/evidence_runs/
  6. Print a run summary to stdout

Usage:
  python -m src.assurance.runner                # run with existing index
  python -m src.assurance.runner --ingest       # rebuild index first, then run
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from .attack_library import load_attack_seeds
from .judge import judge_response
from .legal_oracle import LegalOracle
from .target_bot import VulnerableSupportBot


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
) -> Dict[str, int]:
    """Execute a full assurance run and return a summary dict."""

    # Step 1 – optionally (re)build the EU AI Act vector index.
    if build_index:
        print("\n🔨 Building EU AI Act vector index...")
        _build_index()

    # Step 2 – initialise the legal oracle (lazy-loads the FAISS index).
    oracle = LegalOracle()

    # Step 3 – load attack prompts and set up the target bot.
    attacks = load_attack_seeds(seed_file)
    bot = VulnerableSupportBot()

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

            # Step 4b – judge the response; pass oracle for legal grounding.
            judgment = judge_response(prompt, response, oracle=oracle)

            verdict_counter[judgment["verdict"]] += 1
            for violation in judgment["violations"]:
                violation_counter[violation] += 1
            if judgment["legal_citations"]:
                citation_count += 1

            # Step 5 – write a complete, self-contained evidence record.
            record = {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "attack_id": attack["id"],
                "attack_category": attack["category"],
                "prompt": prompt,
                "response": response,
                "verdict": judgment["verdict"],
                "violations": judgment["violations"],
                "reasons": judgment["reasons"],
                # Legal citations link each failure to an EU AI Act article.
                "legal_citations": judgment["legal_citations"],
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
    args = parser.parse_args()
    run_assurance(
        seed_file=args.seed_file,
        output_dir=args.output_dir,
        build_index=args.ingest,
    )


if __name__ == "__main__":
    main()
