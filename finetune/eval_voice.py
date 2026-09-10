"""Day 12: score brand-voice adherence across base, few-shot and fine-tuned.

Runs locally. The three models ran in Colab on the 15 held-out products and
wrote finetune/outputs.json; nothing here touches a GPU. Claude is the judge,
the same LLM-as-judge pattern as evals/run_eval.py, scoring voice instead of
factual correctness.

Three arms, because two is not enough to answer the question that matters:

  base     - stock Llama 3.1 8B, bare prompt
  fewshot  - stock Llama 3.1 8B, brand rules + 3 exemplars in the prompt
  ft       - the QLoRA adapter

A fine-tune only earns its keep if it beats a good prompt. Comparing it to a
bare prompt flatters it and answers nothing.

The walkthrough's version of this file shipped a placeholder rubric, and it is
worth knowing why it was replaced: measured against the 454 real descriptions,
it forbids "gourmet" (used 43 times), forbids em dashes (57 of them), prefers
five phrases that appear zero times, and prefers "building" over "bold" for
heat when the corpus uses bold 67 times and building 3. Judging the fine-tune
against those rules would have scored the human-written catalog as off-brand.
Every rule below is a measurement instead.
"""
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic

# Repo root on the path so `env` resolves by top-level name, same as the
# evals/ scripts do.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import env  # noqa: E402,F401  - loads .env before ANTHROPIC_API_KEY is read

# Not imported from ask.py on purpose: ask.py picks a retrieval strategy at
# import time and opens Chroma, which this has no use for.
JUDGE_MODEL = "claude-sonnet-4-6"

_HERE = Path(__file__).resolve().parent
OUTPUTS = _HERE / "outputs.json"
CORPUS = _HERE / "brand_voice.jsonl"
RESULTS_DIR = _HERE / "results"

ARMS = ("base", "fewshot", "ft")

# Measured over all 454 published descriptions. Percentages are how many
# contain the feature, so the judge scores against the house norm rather than
# against a generic idea of good copy.
BRAND_RULES = """Spices, Inc. brand voice, measured from 454 published product descriptions.

Structure (a strong description has most of these):
- Opens with the product name as the subject of the first sentence (82% do)
- "Flavor Profile" section (96%)
- "How To Use" section (96%) - concrete dishes and techniques, not moods
- An "also known as" / "sometimes called" line of real alternate names (37%)
- An "is popular with" line naming customer types, semicolon-separated (35%)
- Often a short origin or history paragraph before the sections

Register:
- Warm and expert, never patronizing. Addresses the reader directly but
  sparingly (about 1-2 uses of you/your per description)
- Almost never exclaims (0.06 exclamation marks per description)
- Concrete before evocative: origin, ingredients, technique and use before mood
- Around 290 words

Vocabulary, as actually used - do NOT penalize these:
- "gourmet" (43 uses), "bold" (67), "pungent" (91), "mild" (118), "hot" (224)
- Em dashes appear but are rare (57 across the whole corpus)

What is genuinely off-voice:
- Contentless filler: "perfect for any occasion", "will have your guests asking
  for the recipe", "high quality spices sourced from around the world"
- Praise of the product in place of information about it
- Alternate names that are just rearrangements of the product name
"""

JUDGE_TOOL = {
    "name": "score_voice",
    "description": "Score a product description against the Spices, Inc. house voice.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "Two sentences, evidence first."},
            "voice_adherence": {
                "type": "integer", "minimum": 1, "maximum": 5,
                "description": "5 = indistinguishable from published copy; 1 = generic AI marketing.",
            },
            "structure_adherence": {
                "type": "integer", "minimum": 1, "maximum": 5,
                "description": "How completely it follows the house section layout.",
            },
            "filler_phrases": {
                "type": "array", "items": {"type": "string"},
                "description": "Verbatim contentless marketing phrases, empty if none.",
            },
        },
        "required": ["reasoning", "voice_adherence", "structure_adherence", "filler_phrases"],
    },
}

# Section headings and template words that are Title Case but are not names.
_NOT_A_NAME = {
    "Flavor Profile", "How To Use", "How to Use", "Also Known", "Spices Inc",
    "Spices, Inc", "Instruction", "Input", "Response", "Category", "Name",
    "Ingredients", "Characteristics", "Cuisine",
}
_TITLE_PHRASE = re.compile(r"\b[A-Z][a-z’'-]+(?:\s+[A-Z][a-z’'-]+){1,4}\b")

# Cross-sell constructions. A name asserted here is a claim that the company
# sells it, so it is checked against the SKU list exactly - "Thai Seasoning" is
# a fabrication even though "Spicy Thai Seasoning" exists.
_CROSS_SELL = re.compile(
    r"(?:we also (?:have|offer|carry|sell)|try our|check out our|also available)\s*:?\s*(.{0,140})",
    re.I,
)


def load_corpus() -> tuple[set[str], str]:
    """SKU names and the full text of every real description."""
    recs = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    names = {r["input"].splitlines()[0].replace("Name:", "").strip() for r in recs}
    return names, "\n".join(r["output"] for r in recs)


def fabrications(text: str, names: set[str], corpus_text: str) -> list[str]:
    """Title-case phrases the model asserts that appear nowhere real.

    A heuristic, and it is triage for a human rather than a score. Any
    multi-word proper noun the model produced that appears in neither the SKU
    list nor any published description is a claim nothing backs. It catches
    invented product cross-sells and invented recipe titles - the failure mode
    a style fine-tune produces, and the reason facts belong in retrieval.

    Measured on the Satay output and the full corpus: catches 4 of the 5 known
    fabrications, and fires on 11% of real descriptions. It misses "Grilled
    Satay Chicken" because all three words appear in real SKU names, which is
    the price of getting the false-positive rate down from 20%. Read the
    `fabricated` column, do not just sum it.
    """
    found = []

    # Product claims: must match a real SKU exactly.
    for span in _CROSS_SELL.finditer(text):
        for part in re.split(r",|\band\b", span.group(1)):
            for phrase in _TITLE_PHRASE.findall(part):
                phrase = phrase.strip(" ,.")
                if phrase in _NOT_A_NAME or phrase in names:
                    continue
                found.append(phrase)

    # Everything else: a proper noun attested nowhere in the real copy. A
    # phrase built entirely from words that occur in real SKU names is let
    # through ("Ground Aleppo Pepper" is a fair way to refer to a product we
    # sell); one that introduces a new noun is not ("Satay Chicken Skewers"
    # needs "Skewers", which no product uses).
    sku_vocab = {w.lower() for n in names for w in re.findall(r"[A-Za-z’'-]+", n)}
    for phrase in _TITLE_PHRASE.findall(text):
        phrase = phrase.strip(" ,.")
        if phrase in _NOT_A_NAME or any(phrase.startswith(h) for h in _NOT_A_NAME):
            continue
        if phrase in names or re.search(rf"\b{re.escape(phrase)}\b", corpus_text):
            continue
        if all(w.lower() in sku_vocab for w in phrase.split()):
            continue
        found.append(phrase)

    # Preserve order, drop repeats within one description.
    return list(dict.fromkeys(found))


def _product_names(path: Path) -> set[str]:
    return {
        json.loads(line)["input"].splitlines()[0].replace("Name:", "").strip()
        for line in path.open(encoding="utf-8")
    }


def assert_held_out(products: list[str]) -> None:
    """Refuse to score products the model was trained on.

    This exists because it already happened. The split was regenerated locally
    after train.jsonl and test.jsonl had been uploaded to Colab, so the eval
    ran against a superseded test set whose 15 products were all in the current
    training data. Nothing caught it: a 439-record train file and a 15-record
    test file look correct whichever split produced them, and the resulting
    scores looked plausible.

    A contaminated eval is worse than no eval, because it reports a number.
    """
    train, test = _product_names(_HERE / "train.jsonl"), _product_names(_HERE / "test.jsonl")
    seen = sorted(set(products) & train)
    missing = sorted(set(products) - test)

    if seen:
        raise SystemExit(
            f"TRAIN/TEST LEAK: {len(seen)} of {len(products)} scored products are in "
            f"train.jsonl.\n  {', '.join(seen[:6])}{' ...' if len(seen) > 6 else ''}\n\n"
            "outputs.json was generated from a different split than the one on disk.\n"
            "Upload the current finetune/test.jsonl to Drive, regenerate, and retry."
        )
    if missing:
        raise SystemExit(
            f"{len(missing)} scored products are not in test.jsonl: "
            f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}\n"
            "outputs.json and the local split disagree."
        )


def score(client: Anthropic, description: str) -> dict:
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        system=f"You score product descriptions for house-voice adherence.\n\n{BRAND_RULES}",
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "score_voice"},
        messages=[{"role": "user", "content": f"Score this description:\n\n{description}"}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("judge returned no tool_use block")


def main() -> None:
    if not OUTPUTS.exists():
        raise SystemExit(
            f"{OUTPUTS} not found. Generate it in Colab (see finetune/NOTEBOOK_CHANGES.md),\n"
            "then download it from Drive into finetune/."
        )

    data = json.loads(OUTPUTS.read_text(encoding="utf-8"))
    names, corpus_text = load_corpus()
    products = data["names"]
    assert_held_out(products)
    client = Anthropic()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = RESULTS_DIR / f"voice_{stamp}.csv"

    rows = []
    for arm in ARMS:
        for product, text in zip(products, data[arm]):
            judged = score(client, text)
            invented = fabrications(text, names, corpus_text)
            rows.append({
                "arm": arm,
                "product": product,
                "voice": judged["voice_adherence"],
                "structure": judged["structure_adherence"],
                "n_filler": len(judged["filler_phrases"]),
                "n_fabricated": len(invented),
                "words": len(text.split()),
                "fabricated": "; ".join(invented),
                "filler": "; ".join(judged["filler_phrases"]),
                "reasoning": judged["reasoning"],
            })
            print(f"{arm:<8} {product[:34]:<34} voice={judged['voice_adherence']} "
                  f"struct={judged['structure_adherence']} fabricated={len(invented)}")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_csv}\n")
    print(f"{'arm':<9}{'voice':>7}{'struct':>8}{'filler':>8}{'fabricated':>12}{'words':>7}")
    for arm in ARMS:
        r = [x for x in rows if x["arm"] == arm]
        k = len(r)
        print(f"{arm:<9}{sum(x['voice'] for x in r)/k:>7.2f}"
              f"{sum(x['structure'] for x in r)/k:>8.2f}"
              f"{sum(x['n_filler'] for x in r)/k:>8.2f}"
              f"{sum(x['n_fabricated'] for x in r)/k:>12.2f}"
              f"{sum(x['words'] for x in r)//k:>7}")

    ft_voice = sum(x["voice"] for x in rows if x["arm"] == "ft") / len(products)
    fs_voice = sum(x["voice"] for x in rows if x["arm"] == "fewshot") / len(products)
    print(f"\nfine-tune minus few-shot prompt: {ft_voice - fs_voice:+.2f}")
    print("A delta near zero means prompting already got you there - which is a")
    print("finding, not a failure. Read finetune/results/*.csv before concluding.")

    top = Counter(p for x in rows if x["arm"] == "ft" for p in x["fabricated"].split("; ") if p)
    if top:
        print("\nMost-invented names from the fine-tuned arm:")
        for phrase, n in top.most_common(8):
            print(f"  {n}x  {phrase}")


if __name__ == "__main__":
    main()
