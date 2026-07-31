"""Run the golden set through the RAG, score it, save results."""
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ask import answer_question_with_context

from evals.judge import judge
from evals.metrics import (answer_cites_expected_clause, clause_hit_at_k,
                           is_refusal, retrieved_clauses)
from langfuse import propagate_attributes
from tracing import langfuse

# The five judge axes, mapped to the score names they get in Langfuse.
SCORE_AXES = (
    "factual_correctness", "completeness", "relevance",
    "citation_correctness", "grounding",
)

GOLDEN_FILE = Path(__file__).resolve().parent / "golden.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Every row carries every column so the CSV is rectangular even though probe
# rows have no judge scores.
FIELDNAMES = [
    "id", "difficulty", "tags", "question", "expected_clause",
    "system_answer", "refused",
    "factual", "complete", "relevant", "citation", "grounding", "overall",
    "hit_1", "hit_3", "hit_5", "cites_expected", "top_clauses",
    "reasoning", "elapsed_sec",
]


def _blank_row(rec: dict) -> dict:
    return {
        "id": rec["id"],
        "difficulty": rec["difficulty"],
        "tags": ",".join(rec["tags"]),
        "question": rec["question"],
        "expected_clause": rec["expected_clause"] or "",
        "system_answer": "",
        "refused": "",
        "factual": "", "complete": "", "relevant": "",
        "citation": "", "grounding": "", "overall": "",
        "hit_1": "", "hit_3": "", "hit_5": "", "cites_expected": "",
        "top_clauses": "",
        "reasoning": "",
        "elapsed_sec": "",
    }


def run_eval(run_name: str, config_notes: str = "", limit: int | None = None) -> Path:
    """Run the eval, write a CSV, print a summary.

    If `limit` is set, only the first N golden records are run - handy for cheap
    smoke tests that stay under the Voyage free-tier rate limit.
    """
    records = [json.loads(line) for line in GOLDEN_FILE.open(encoding="utf-8")]
    total = len(records)
    if limit is not None:
        records = records[:limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"{timestamp}_{run_name}.csv"
    # One Langfuse session per eval run, so every question in this run shows up
    # grouped under the Sessions view and is comparable run-to-run.
    run_session = f"{run_name}-{timestamp}"
    strategy = os.getenv("RETRIEVAL_STRATEGY", "vanilla")

    rows: list[dict] = []
    # A record that errors is not in the summary's denominator, so a partial run
    # would otherwise read as a clean one. Track the losses and report them.
    dropped: dict[str, list[str]] = {"rag": [], "judge": []}
    scope = f"{len(records)} of {total}" if limit is not None else str(total)
    print(f"Running eval: {run_name} ({scope} records, strategy={strategy})")
    if config_notes:
        print(f"Config: {config_notes}")
    print()

    for i, rec in enumerate(records, 1):
        t0 = time.time()
        is_probe = rec["difficulty"] == "probe"
        # Each golden record is one trace named "eval-item". propagate_attributes
        # stamps the session id and tags onto that trace and everything nested
        # under it (the RAG pipeline + the judge call), so a run is filterable by
        # difficulty and grouped as a session.
        item_tags = ["eval", "corpus:sqf", f"strategy:{strategy}",
                     rec["difficulty"], *rec["tags"]]

        with propagate_attributes(session_id=run_session, tags=item_tags), \
                langfuse.start_as_current_observation(name="eval-item", as_type="span"):
            try:
                system_answer, chunks = answer_question_with_context(rec["question"])
            except Exception as e:
                langfuse.update_current_span(level="ERROR", status_message=f"RAG: {e}")
                print(f"  {i:2d}/{len(records)} {rec['id']} RAG ERROR: {e}")
                dropped["rag"].append(rec["id"])
                continue

            refused = is_refusal(system_answer)
            row = _blank_row(rec)
            row["system_answer"] = system_answer
            row["refused"] = refused
            row["top_clauses"] = ",".join(retrieved_clauses(chunks))
            row["elapsed_sec"] = round(time.time() - t0, 2)

            langfuse.update_current_span(
                input=rec["question"],
                output=system_answer,
                metadata={
                    "id": rec["id"],
                    "difficulty": rec["difficulty"],
                    "expected_clause": rec["expected_clause"],
                    "refused": refused,
                },
            )

            if is_probe:
                # Nothing for the judge to compare against. The only thing that
                # matters is whether the system declined, so score that directly.
                langfuse.score_current_trace(
                    name="appropriate_refusal", value=1 if refused else 0,
                    comment="probe question; expected refusal",
                )
                rows.append(row)
                mark = "REFUSED (pass)" if refused else "ANSWERED (FAIL)"
                print(f"  {i:2d}/{len(records)} {rec['id']} [probe ] {mark}")
                continue

            hits = {k: clause_hit_at_k(chunks, rec["expected_clause"], k) for k in (1, 3, 5)}
            row["hit_1"], row["hit_3"], row["hit_5"] = hits[1], hits[3], hits[5]
            # Recorded alongside hit@k because the two disagree in a way that
            # matters: hit@k reads chunk metadata, this reads the answer, so it
            # stays comparable across configurations that chunk differently.
            row["cites_expected"] = answer_cites_expected_clause(
                system_answer, rec["expected_clause"])

            try:
                scores = judge(rec["question"], rec["reference_answer"],
                               system_answer, chunks)
            except Exception as e:
                langfuse.update_current_span(level="ERROR", status_message=f"JUDGE: {e}")
                print(f"  {i:2d}/{len(records)} {rec['id']} JUDGE ERROR: {e}")
                dropped["judge"].append(rec["id"])
                rows.append(row)
                continue

            overall = sum(scores[a] for a in SCORE_AXES) / len(SCORE_AXES)
            row.update({
                "factual": scores["factual_correctness"],
                "complete": scores["completeness"],
                "relevant": scores["relevance"],
                "citation": scores["citation_correctness"],
                "grounding": scores["grounding"],
                "overall": round(overall, 3),
                "reasoning": scores["reasoning"],
            })

            # Attach the judge's verdict plus the deterministic metrics as
            # Langfuse scores - this is what turns the eval into a quality
            # signal you can filter and chart in the UI.
            for axis in SCORE_AXES:
                langfuse.score_current_trace(
                    name=axis, value=scores[axis], comment=scores["reasoning"]
                )
            langfuse.score_current_trace(name="overall", value=round(overall, 3),
                                         comment=scores["reasoning"])
            langfuse.score_current_trace(name="clause_hit_3", value=1 if hits[3] else 0,
                                         comment=f"expected {rec['expected_clause']}")

            rows.append(row)
            print(f"  {i:2d}/{len(records)} {rec['id']} [{rec['difficulty']:6s}] "
                  f"F={row['factual']} C={row['complete']} R={row['relevant']} "
                  f"Cite={row['citation']} G={row['grounding']} "
                  f"avg={overall:.2f} hit@3={'Y' if hits[3] else 'n'}")

    if not rows:
        print("No rows produced; nothing written.")
        return out_file

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written: {out_file}\n")
    print_summary(rows, dropped)

    # Force-send buffered traces and scores before the script exits.
    langfuse.flush()
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        print(f"\nTraces sent to Langfuse under session '{run_session}'.")
    return out_file


def print_summary(rows: list[dict], dropped: dict[str, list[str]] | None = None) -> None:
    """Print the compliance metrics first, then the judged axes."""
    scored = [r for r in rows if r["difficulty"] != "probe" and r["overall"] != ""]
    probes = [r for r in rows if r["difficulty"] == "probe"]

    dropped = dropped or {}
    if any(dropped.values()):
        print("=== Dropped (excluded from every figure below) ===")
        for stage, ids in dropped.items():
            if ids:
                print(f"  {stage} errors: {len(ids)}  ({', '.join(ids)})")
        print()

    def avg(field, subset):
        return sum(float(r[field]) for r in subset) / len(subset)

    def pct(field, subset):
        return 100.0 * sum(1 for r in subset if r[field] is True) / len(subset)

    if probes:
        refused = sum(1 for r in probes if r["refused"] is True)
        print(f"=== Refusal probes (n={len(probes)}) ===")
        print(f"  Correctly refused:  {refused}/{len(probes)}  ({100.0*refused/len(probes):.0f}%)")

    if not scored:
        return

    false_refusals = sum(1 for r in scored if r["refused"] is True)
    print(f"\n=== Retrieval (n={len(scored)}) ===")
    print(f"  clause hit@1:       {pct('hit_1', scored):.0f}%")
    print(f"  clause hit@3:       {pct('hit_3', scored):.0f}%")
    print(f"  clause hit@5:       {pct('hit_5', scored):.0f}%")
    print(f"  cites expected:     {pct('cites_expected', scored):.0f}%")
    print(f"  False refusals:     {false_refusals}/{len(scored)}")

    print(f"\n=== Judged axes (n={len(scored)}) ===")
    print(f"  Citation:   {avg('citation', scored):.2f}")
    print(f"  Grounding:  {avg('grounding', scored):.2f}")
    print(f"  Factual:    {avg('factual', scored):.2f}")
    print(f"  Complete:   {avg('complete', scored):.2f}")
    print(f"  Relevant:   {avg('relevant', scored):.2f}")
    print(f"  Overall:    {avg('overall', scored):.2f}")

    for diff in ("easy", "medium", "hard"):
        subset = [r for r in scored if r["difficulty"] == diff]
        if subset:
            print(f"\n=== {diff.capitalize()} (n={len(subset)}) ===")
            print(f"  Overall: {avg('overall', subset):.2f}   "
                  f"hit@3: {pct('hit_3', subset):.0f}%")

    print("\n=== Lowest 5 ===")
    for r in sorted(scored, key=lambda r: float(r["overall"]))[:5]:
        print(f"  {r['id']} [{r['difficulty']:6s}] {float(r['overall']):.2f}  "
              f"{r['question'][:70]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the SQF golden set through the RAG and score it.")
    parser.add_argument("name", nargs="?", default="baseline",
                        help="Run name, used in the results filename.")
    parser.add_argument("notes", nargs="?", default="",
                        help="Free-text config notes recorded with the run.")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Only run the first N golden records (cheap smoke test).")
    args = parser.parse_args()
    run_eval(args.name, args.notes, limit=args.limit)