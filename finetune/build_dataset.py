"""Day 9: split the brand-voice instruction dataset into train and test.

The walkthrough's version of this file was a placeholder list to hand-populate
toward 150 examples. That step is already done: finetune/brand_voice.jsonl is a
real export of 454 published Spices, Inc. product descriptions, so this script
validates and splits rather than builds. Every `output` here is human-written
catalog copy - Day 9's "Option A", which is the whole reason the fine-tune has
a chance of learning a voice instead of imitating one.

Two things this script exists to protect:

  - The split must be reproducible. Day 11 trains on train.jsonl and Day 12
    scores test.jsonl, and if the seed drifts between those two days the model
    is evaluated on records it was trained on and every number is inflated.
    Hence a fixed SEED and a sorted-then-shuffled order rather than relying on
    the file's own ordering.

  - max_seq_length has to be chosen from the data, not guessed. These outputs
    run ~410 tokens at the median against the ~50 of the walkthrough's toy
    example, and SFTTrainer truncates silently past max_seq_length. A truncated
    target is a target with its EOS cut off, which teaches the model never to
    stop - so this prints the truncation cost at each candidate setting and the
    Colab notebook gets set from that table.
"""
import json
import random
from pathlib import Path

IN_FILE = Path("finetune/brand_voice.jsonl")
TRAIN_FILE = Path("finetune/train.jsonl")
TEST_FILE = Path("finetune/test.jsonl")

# 10% held out. The walkthrough says 15 records, sized for a 150-example set;
# at 454 that would be 3%, too few for Day 12 to distinguish a real voice gain
# from noise across five judged product categories.
TEST_SIZE = 45

SEED = 20260903  # never change: Day 12's eval is only honest against Day 11's split

# The Llama-3 tokenizer averages a shade under 4 chars/token on this kind of
# English prose. Good enough to pick max_seq_length; the exact count comes from
# the tokenizer in Colab.
CHARS_PER_TOKEN = 4

# The settings worth checking before training. 512 is the Unsloth notebook's
# default in several forks and is the trap here.
CANDIDATE_LENGTHS = (512, 1024, 2048, 4096)

REQUIRED_KEYS = ("instruction", "input", "output")


def load(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.open(encoding="utf-8")]

    for i, rec in enumerate(records, 1):
        missing = [k for k in REQUIRED_KEYS if k not in rec]
        if missing:
            raise ValueError(f"{path}:{i} missing {missing}")
        if not rec["output"].strip():
            raise ValueError(f"{path}:{i} has an empty output")

    # A duplicated output is a duplicated gradient: the blend appears twice in
    # the catalog under different names and the model over-weights that voice.
    outputs = [r["output"] for r in records]
    if len(set(outputs)) != len(outputs):
        raise ValueError(f"{path}: {len(outputs) - len(set(outputs))} duplicate outputs")

    return records


def seq_tokens(rec: dict) -> int:
    """Estimated tokens for the full training sequence, prompt and target."""
    return (len(rec["instruction"]) + len(rec["input"]) + len(rec["output"])) // CHARS_PER_TOKEN


def report_lengths(records: list[dict]) -> None:
    lengths = sorted(seq_tokens(r) for r in records)

    def pct(p: float) -> int:
        return lengths[min(int(len(lengths) * p), len(lengths) - 1)]

    print(f"\nSequence length (est. tokens, prompt + target), n={len(lengths)}")
    print(f"  p50={pct(.50)}  p75={pct(.75)}  p90={pct(.90)}  "
          f"p95={pct(.95)}  p99={pct(.99)}  max={lengths[-1]}")
    print(f"  total training tokens ~{sum(lengths):,}")

    print("\nTruncation cost by max_seq_length:")
    for limit in CANDIDATE_LENGTHS:
        over = sum(1 for n in lengths if n > limit)
        note = "  <- silently cuts the EOS off these" if over else ""
        print(f"  {limit:>5}: {over:>3} records ({over / len(lengths):5.1%}){note}")


def report_coverage(records: list[dict]) -> None:
    """Which input fields are actually present.

    Sparse fields are a feature, not a defect: 'Ingredients' is absent from over
    half the catalog, so the model has to learn to write without it rather than
    expecting every field on every product.
    """
    counts: dict[str, int] = {}
    for rec in records:
        for line in rec["input"].splitlines():
            if ":" in line:
                counts[line.split(":", 1)[0].strip()] = counts.get(line.split(":", 1)[0].strip(), 0) + 1

    print("\nInput field coverage:")
    for field, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {field:<16} {n:>3}/{len(records)} ({n / len(records):4.0%})")


def write(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records):>3} records to {path}")


def main() -> None:
    records = load(IN_FILE)
    print(f"Loaded {len(records)} brand-voice examples from {IN_FILE}")

    report_lengths(records)
    report_coverage(records)

    # Sort before shuffling so the split does not depend on the export's own
    # ordering - re-exporting the catalog in a different order would otherwise
    # silently reshuffle train and test.
    ordered = sorted(records, key=lambda r: r["input"])
    random.Random(SEED).shuffle(ordered)

    write(TEST_FILE, ordered[:TEST_SIZE])
    write(TRAIN_FILE, ordered[TEST_SIZE:])

    train_tokens = sum(seq_tokens(r) for r in ordered[TEST_SIZE:])
    print(f"\nTrain set is ~{train_tokens:,} tokens. "
          f"See finetune/FINE_TUNING_NOTES.md for the Day 11 config these numbers imply.")


if __name__ == "__main__":
    main()
