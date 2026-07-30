"""Analysis of SQF eval result CSVs.

Usage:
  uv run python evals/analyze.py                              # summarize latest
  uv run python evals/analyze.py RESULTS.csv                  # summarize one
  uv run python evals/analyze.py BASE.csv VARIANT.csv         # compare two
  uv run python evals/analyze.py compare_three V.csv H.csv R.csv
"""
import sys
from pathlib import Path

import pandas as pd

AXES = ["citation", "grounding", "factual", "complete", "relevant", "overall"]
HITS = ["hit_1", "hit_3", "hit_5"]

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _latest_results() -> Path:
    """Newest results CSV, or exit with a usable message if there are none."""
    csvs = sorted(RESULTS_DIR.glob("*.csv"))
    if not csvs:
        sys.exit(f"No result CSVs in {RESULTS_DIR}. "
                 f"Run 'uv run python evals/run_eval.py' first, "
                 f"or pass a CSV path explicitly.")
    return csvs[-1]


def _split(csv_path: str):
    """Return (scored, probes). Probe rows have no judge scores."""
    df = pd.read_csv(csv_path)
    for col in HITS + ["refused"]:
        if col in df.columns:
            df[col] = df[col].astype("object")
    probes = df[df["difficulty"] == "probe"]
    scored = df[(df["difficulty"] != "probe") & df["overall"].notna()]
    return scored, probes


def _hit_rate(df: pd.DataFrame, col: str) -> float:
    vals = df[col].map(lambda v: str(v).lower() == "true")
    return 100.0 * vals.mean() if len(df) else float("nan")


def _refusal_rate(df: pd.DataFrame) -> float:
    vals = df["refused"].map(lambda v: str(v).lower() == "true")
    return 100.0 * vals.mean() if len(df) else float("nan")


def summarize(csv_path: str) -> None:
    scored, probes = _split(csv_path)

    print(f"\n=== {csv_path} ===")
    print(f"Scored: {len(scored)}   Probes: {len(probes)}\n")

    if len(probes):
        print(f"Probe refusal rate:  {_refusal_rate(probes):.0f}%  "
              f"(higher is better)")
    if not len(scored):
        return
    print(f"False refusal rate:  {_refusal_rate(scored):.0f}%  (lower is better)")

    print("\nClause hit rate:")
    for col in HITS:
        print(f"  {col}: {_hit_rate(scored, col):5.1f}%")

    print("\nAverage by axis:")
    print(scored[AXES].mean().round(2).to_string())

    print("\nAverage by difficulty:")
    print(scored.groupby("difficulty")[AXES].mean().round(2).to_string())

    # Tag analysis: which requirement areas does the system handle worst?
    tag_rows = []
    for _, row in scored.iterrows():
        for tag in str(row["tags"]).split(","):
            if tag.strip():
                tag_rows.append({"tag": tag.strip(), "overall": row["overall"],
                                 "citation": row["citation"]})
    if tag_rows:
        tag_df = pd.DataFrame(tag_rows)
        print("\nWorst 10 tags by overall:")
        print(tag_df.groupby("tag")[["overall", "citation"]]
              .agg(["mean", "count"]).round(2)
              .sort_values(("overall", "mean")).head(10).to_string())


def compare(baseline_csv: str, variant_csv: str) -> None:
    a, a_probes = _split(baseline_csv)
    b, b_probes = _split(variant_csv)

    print(f"Baseline: {baseline_csv}")
    print(f"Variant:  {variant_csv}\n")

    print("Compliance metrics:")
    print(f"  probe refusal   baseline={_refusal_rate(a_probes):5.1f}%  "
          f"variant={_refusal_rate(b_probes):5.1f}%")
    print(f"  false refusal   baseline={_refusal_rate(a):5.1f}%  "
          f"variant={_refusal_rate(b):5.1f}%")
    for col in HITS:
        av, bv = _hit_rate(a, col), _hit_rate(b, col)
        print(f"  {col:13s} baseline={av:5.1f}%  variant={bv:5.1f}%  "
              f"delta={bv-av:+.1f}")

    print("\nJudged axes (variant - baseline):")
    for col in AXES:
        delta = b[col].mean() - a[col].mean()
        print(f"  {col:10s} baseline={a[col].mean():.2f}  "
              f"variant={b[col].mean():.2f}  delta={delta:+.2f}")

    merged = a.merge(b, on="id", suffixes=("_a", "_b"))
    merged["delta"] = merged["overall_b"] - merged["overall_a"]
    print("\nBiggest improvements:")
    for _, row in merged.nlargest(3, "delta").iterrows():
        print(f"  {row['id']} {row['overall_a']:.2f} -> {row['overall_b']:.2f}  "
              f"{str(row['question_a'])[:65]}")
    print("\nBiggest regressions:")
    for _, row in merged.nsmallest(3, "delta").iterrows():
        print(f"  {row['id']} {row['overall_a']:.2f} -> {row['overall_b']:.2f}  "
              f"{str(row['question_a'])[:65]}")


def compare_three(vanilla_csv: str, hybrid_csv: str, rerank_csv: str) -> None:
    v, vp = _split(vanilla_csv)
    h, hp = _split(hybrid_csv)
    r, rp = _split(rerank_csv)

    print(f"{'Metric':16s}  {'Vanilla':>9s}  {'Hybrid':>9s}  {'Rerank':>9s}  "
          f"{'H-V':>7s}  {'R-V':>7s}")
    print("-" * 68)

    # Compliance metrics first - these are the headline for this corpus.
    for label, frames in (("probe refusal %", (vp, hp, rp)),):
        vm, hm, rm = (_refusal_rate(f) for f in frames)
        print(f"{label:16s}  {vm:>9.1f}  {hm:>9.1f}  {rm:>9.1f}  "
              f"{hm-vm:>+7.1f}  {rm-vm:>+7.1f}")
    vm, hm, rm = (_refusal_rate(f) for f in (v, h, r))
    print(f"{'false refusal %':16s}  {vm:>9.1f}  {hm:>9.1f}  {rm:>9.1f}  "
          f"{hm-vm:>+7.1f}  {rm-vm:>+7.1f}")
    for col in HITS:
        vm, hm, rm = (_hit_rate(f, col) for f in (v, h, r))
        print(f"{col + ' %':16s}  {vm:>9.1f}  {hm:>9.1f}  {rm:>9.1f}  "
              f"{hm-vm:>+7.1f}  {rm-vm:>+7.1f}")

    print()
    for col in AXES:
        vm, hm, rm = v[col].mean(), h[col].mean(), r[col].mean()
        print(f"{col:16s}  {vm:>9.2f}  {hm:>9.2f}  {rm:>9.2f}  "
              f"{hm-vm:>+7.2f}  {rm-vm:>+7.2f}")

    for diff in ("easy", "medium", "hard"):
        vs = v[v["difficulty"] == diff]["overall"].mean()
        hs = h[h["difficulty"] == diff]["overall"].mean()
        rs = r[r["difficulty"] == diff]["overall"].mean()
        print(f"\n{diff:8s}  vanilla={vs:.2f}  hybrid={hs:.2f}  rerank={rs:.2f}")

    merged = v.merge(r, on="id", suffixes=("_v", "_r"))
    merged["delta"] = merged["overall_r"] - merged["overall_v"]
    print("\nBiggest improvements (rerank vs vanilla):")
    for _, row in merged.nlargest(5, "delta").iterrows():
        print(f"  {row['id']} {row['overall_v']:.2f} -> {row['overall_r']:.2f}  "
              f"{str(row['question_v'])[:65]}")
    print("\nBiggest regressions (rerank vs vanilla):")
    for _, row in merged.nsmallest(5, "delta").iterrows():
        print(f"  {row['id']} {row['overall_v']:.2f} -> {row['overall_r']:.2f}  "
              f"{str(row['question_v'])[:65]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "compare_three":
        if len(args) != 4:
            sys.exit("usage: analyze.py compare_three VANILLA.csv HYBRID.csv RERANK.csv")
        compare_three(args[1], args[2], args[3])
    elif len(args) >= 2:
        compare(args[0], args[1])
    elif len(args) == 1:
        summarize(args[0])
    else:
        summarize(str(_latest_results()))