"""
Report Generator
----------------
Reads a JSONL evidence file produced by runner.py and generates a
self-contained HTML audit report.

The report contains:
  - Run metadata (ID, timestamp, total records)
  - Summary scorecard: pass rate, fail rate, total violations
  - Violation breakdown bar chart (inline SVG, no JS dependencies)
  - Per-record evidence table with prompt, response, verdict badge,
    violation tags, reasons, and EU AI Act legal citations

Usage (CLI):
  python -m src.assurance.report --evidence outputs/evidence_runs/evidence_XYZ.jsonl
  python -m src.assurance.report --latest           # auto-picks newest file
  python -m src.assurance.report --latest --open    # also opens in browser
"""

import argparse
import json
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_evidence(path: Path) -> List[Dict]:
    """Parse a JSONL evidence file into a list of record dicts."""
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            clean = line.strip()
            if clean:
                records.append(json.loads(clean))
    return records


def latest_evidence_file(output_dir: str = "outputs/evidence_runs") -> Path:
    """Return the most recently modified evidence JSONL in output_dir."""
    candidates = sorted(
        Path(output_dir).glob("evidence_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No evidence files found in '{output_dir}'.")
    return candidates[0]


# ---------------------------------------------------------------------------
# HTML building helpers
# ---------------------------------------------------------------------------

VERDICT_BADGE = {
    "pass": '<span class="badge pass">PASS</span>',
    "fail": '<span class="badge fail">FAIL</span>',
}

CATEGORY_COLOUR = {
    "prompt_injection": "#c0392b",
    "data_leakage": "#e67e22",
    "bias": "#8e44ad",
    "unsafe_guidance": "#2980b9",
    "benign": "#27ae60",
}

VIOLATION_COLOUR = {
    "Sensitive Data Leakage": "#e67e22",
    "Prompt Injection Success": "#c0392b",
    "Bias / Discrimination": "#8e44ad",
    "Unsafe Guidance": "#2980b9",
}


def _violation_tag(name: str) -> str:
    colour = VIOLATION_COLOUR.get(name, "#555")
    return f'<span class="vtag" style="background:{colour}">{escape(name)}</span>'


def _citation_chip(citation: str) -> str:
    return f'<span class="chip">{escape(citation)}</span>'


# ---------------------------------------------------------------------------
# SVG bar chart (no JavaScript / no external deps)
# ---------------------------------------------------------------------------


def _bar_chart_svg(violation_counts: Counter) -> str:
    """Render a horizontal bar chart as inline SVG."""
    if not violation_counts:
        return "<p>No violations detected.</p>"

    max_val = max(violation_counts.values())
    bar_height = 28
    gap = 10
    label_width = 220
    bar_max_width = 300
    chart_width = label_width + bar_max_width + 60
    chart_height = (bar_height + gap) * len(violation_counts) + gap

    rows = []
    for i, (name, count) in enumerate(violation_counts.most_common()):
        y = gap + i * (bar_height + gap)
        bar_w = int((count / max_val) * bar_max_width)
        colour = VIOLATION_COLOUR.get(name, "#555")
        rows.append(
            f'<text x="{label_width - 8}" y="{y + bar_height // 2 + 5}" '
            f'text-anchor="end" font-size="13" fill="#333">{escape(name)}</text>'
            f'<rect x="{label_width}" y="{y}" width="{bar_w}" height="{bar_height}" '
            f'fill="{colour}" rx="4"/>'
            f'<text x="{label_width + bar_w + 6}" y="{y + bar_height // 2 + 5}" '
            f'font-size="13" fill="#333">{count}</text>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{chart_width}" height="{chart_height}">' + "\n".join(rows) + "</svg>"
    )


# ---------------------------------------------------------------------------
# Main HTML assembly
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Assurance Report · {run_id}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: #f4f6f9; color: #222; line-height: 1.5; }}
  header {{ background: #1a1f36; color: #fff; padding: 24px 40px; }}
  header h1 {{ font-size: 1.5rem; font-weight: 700; }}
  header p  {{ font-size: 0.85rem; opacity: .7; margin-top: 4px; }}
  .container {{ max-width: 1100px; margin: 32px auto; padding: 0 24px; }}
  .scorecard {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
  .card {{ background: #fff; border-radius: 10px; padding: 20px 28px;
           flex: 1; min-width: 160px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .card .value {{ font-size: 2rem; font-weight: 700; }}
  .card .label {{ font-size: 0.8rem; color: #666; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }}
  .card.red .value   {{ color: #c0392b; }}
  .card.green .value {{ color: #27ae60; }}
  .card.blue .value  {{ color: #2980b9; }}
  section {{ background: #fff; border-radius: 10px; padding: 24px 28px;
             margin-bottom: 28px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  section h2 {{ font-size: 1rem; font-weight: 700; margin-bottom: 16px;
                border-bottom: 1px solid #eee; padding-bottom: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #f0f2f5; text-align: left; padding: 10px 12px;
        font-weight: 600; color: #555; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #f0f2f5; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #fafbfc; }}
  .badge {{ display: inline-block; border-radius: 4px; padding: 2px 8px;
            font-size: 0.75rem; font-weight: 700; letter-spacing: .04em; }}
  .badge.pass {{ background: #d4edda; color: #155724; }}
  .badge.fail {{ background: #f8d7da; color: #721c24; }}
  .vtag {{ display: inline-block; color: #fff; border-radius: 4px;
           padding: 2px 7px; font-size: 0.72rem; font-weight: 600;
           margin: 2px 2px 0 0; }}
  .chip {{ display: inline-block; background: #e9ecef; color: #333;
           border-radius: 4px; padding: 2px 7px; font-size: 0.72rem;
           margin: 2px 2px 0 0; }}
  .cat {{ display: inline-block; border-radius: 4px; padding: 1px 7px;
          font-size: 0.72rem; color: #fff; font-weight: 600; }}
  .mono {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.8rem;
           background: #f8f9fa; padding: 6px 8px; border-radius: 4px;
           white-space: pre-wrap; word-break: break-word; }}
  .reasons {{ color: #555; font-size: 0.8rem; margin-top: 4px; }}
  footer {{ text-align: center; color: #aaa; font-size: 0.78rem; padding: 32px; }}
</style>
</head>
<body>
<header>
  <h1>AI Assurance Audit Report</h1>
  <p>Run ID: {run_id} &nbsp;·&nbsp; Generated: {generated_at}</p>
</header>
<div class="container">

  <!-- Scorecard -->
  <div class="scorecard">
    <div class="card"><div class="value">{total}</div><div class="label">Tests Run</div></div>
    <div class="card green"><div class="value">{pass_count}</div><div class="label">Passed</div></div>
    <div class="card red"><div class="value">{fail_count}</div><div class="label">Failed</div></div>
    <div class="card red"><div class="value">{fail_rate}%</div><div class="label">Failure Rate</div></div>
    <div class="card blue"><div class="value">{total_violations}</div><div class="label">Total Violations</div></div>
    <div class="card blue"><div class="value">{cited_count}</div><div class="label">EU AI Act Citations</div></div>
  </div>

  <!-- Violation breakdown chart -->
  <section>
    <h2>Violation Breakdown</h2>
    {bar_chart}
  </section>

  <!-- Evidence table -->
  <section>
    <h2>Evidence Records</h2>
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Category</th>
          <th>Prompt</th>
          <th>Response</th>
          <th>Verdict</th>
          <th>Violations &amp; Reasons</th>
          <th>EU AI Act Citations</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </section>

</div>
<footer>AI Assurance Copilot · {run_id}</footer>
</body>
</html>
"""


def _build_row(rec: Dict) -> str:
    cat = rec.get("attack_category", "")
    cat_colour = CATEGORY_COLOUR.get(cat, "#555")
    cat_html = f'<span class="cat" style="background:{cat_colour}">{escape(cat)}</span>'

    verdict = rec.get("verdict", "")
    badge = VERDICT_BADGE.get(verdict, escape(verdict))

    violations = rec.get("violations", [])
    reasons = rec.get("reasons", [])
    citations = rec.get("legal_citations", [])

    vtags = " ".join(_violation_tag(v) for v in violations)
    reason_html = "<br>".join(
        f'<span class="reasons">· {escape(r)}</span>' for r in reasons
    )
    citation_html = " ".join(_citation_chip(c) for c in citations)

    return (
        f"<tr>"
        f"<td>{escape(rec.get('attack_id', ''))}</td>"
        f"<td>{cat_html}</td>"
        f'<td><div class="mono">{escape(rec.get("prompt", ""))}</div></td>'
        f'<td><div class="mono">{escape(rec.get("response", ""))}</div></td>'
        f"<td>{badge}</td>"
        f"<td>{vtags}<br>{reason_html}</td>"
        f"<td>{citation_html if citation_html else '<span style="color:#aaa">—</span>'}</td>"
        f"</tr>"
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def generate_report(evidence_path: Path, output_dir: Path) -> Path:
    """Build an HTML report from an evidence JSONL and write it to output_dir."""
    records = load_evidence(evidence_path)
    if not records:
        raise ValueError(f"No records found in {evidence_path}")

    run_id = records[0].get("run_id", evidence_path.stem)
    total = len(records)
    fail_count = sum(1 for r in records if r.get("verdict") == "fail")
    pass_count = total - fail_count
    fail_rate = round(fail_count / total * 100) if total else 0
    cited_count = sum(1 for r in records if r.get("legal_citations"))

    violation_counter: Counter = Counter()
    for rec in records:
        for v in rec.get("violations", []):
            violation_counter[v] += 1
    total_violations = sum(violation_counter.values())

    bar_chart = _bar_chart_svg(violation_counter)
    rows_html = "\n".join(_build_row(r) for r in records)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = HTML_TEMPLATE.format(
        run_id=escape(run_id),
        generated_at=generated_at,
        total=total,
        pass_count=pass_count,
        fail_count=fail_count,
        fail_rate=fail_rate,
        total_violations=total_violations,
        cited_count=cited_count,
        bar_chart=bar_chart,
        rows=rows_html,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"report_{run_id}.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"✅ Report written to: {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an HTML audit report from an assurance evidence file."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--evidence",
        help="Path to a specific evidence JSONL file.",
    )
    group.add_argument(
        "--latest",
        action="store_true",
        help="Auto-select the most recent evidence file in outputs/evidence_runs/.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reports",
        help="Directory to write the HTML report into (default: outputs/reports).",
    )
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Open the report in the default browser after generation.",
    )
    args = parser.parse_args()

    if args.latest:
        evidence_path = latest_evidence_file()
        print(f"📂 Using latest evidence: {evidence_path}")
    elif args.evidence:
        evidence_path = Path(args.evidence)
    else:
        # Default: use the latest file if neither flag given.
        evidence_path = latest_evidence_file()
        print(f"📂 Using latest evidence: {evidence_path}")

    out_file = generate_report(evidence_path, Path(args.output_dir))

    if args.open_browser:
        webbrowser.open(out_file.resolve().as_uri())


if __name__ == "__main__":
    main()
