# Fine-Tuning Notes: Brand-Voice LoRA

Week 6 of the Weeks 5-6 walkthrough. Sections for Days 11-13 get filled in as
those days happen; what follows is settled through Day 9.

## Why the fine-tune target is brand voice and not the SQF corpus

The obvious question for a repo that is otherwise entirely about compliance
documents: why train on spice descriptions?

Because Day 8's decision tree answers it before the week starts. Fine-tuning
teaches behavior, not facts, and the SQF corpus is a facts problem — the model
needs clause text it has never seen, which is the first branch of the tree and
routes to RAG. That system already exists and is measured in
`evals/EVAL_REPORT.md`. Training a model on clause text would be building the
exact anti-pattern the week teaches you to talk a customer out of.

Brand voice is the one asset here that lands on the *second* branch: a
consistent style and format, with real human-written ground truth to learn it
from. Note the split of concerns, because it is easy to conflate them:

- **On-prem inference comes from choosing open weights**, not from fine-tuning.
  Swapping Sonnet for a local model would keep documents in the building with
  no training at all.
- **Fine-tuning buys quality at small scale** — dragging a small model up to a
  format contract a large one already satisfies.

The two are independent axes. QLoRA is a training technique (LoRA over a 4-bit
base, so the step fits in 16GB); it says nothing about where the result runs.

## Day 9: the dataset

`finetune/brand_voice.jsonl` — **454 published Spices, Inc. product
descriptions**, exported from the Drupal catalog in the
`{instruction, input, output}` schema the walkthrough specifies.

This is Day 9's "Option A" (real catalog copy) rather than Option B (Claude
bootstrapping the voice from a brand guide), at 3x the 150-example target. It
matters more than the volume does: every `output` is copy a person wrote, so
the fine-tune learns the voice rather than imitating an imitation of it.

Verified clean by `build_dataset.py` — uniform keys across all 454, no empty
outputs, no duplicate outputs.

| | Walkthrough assumes | This dataset |
|---|---|---|
| Examples | 150 | 454 (409 train / 45 test) |
| Output length | ~50 tokens | ~410 median, 2,900 max |
| Full sequence | — | p50 484, p95 678, p99 1,669 |
| Total train tokens | — | ~214,000 |

Input fields are deliberately sparse: `Name` 100%, `Characteristics` 99%,
`Category` 98%, but `Cuisine` 47% and `Ingredients` 46%. That variation is
worth keeping — it forces the model to write without an ingredient list rather
than expecting every field on every product.

### Split

`uv run python finetune/build_dataset.py` writes `train.jsonl` (409) and
`test.jsonl` (45) from a fixed seed. 45 rather than the walkthrough's 15: at
454 records, 15 is 3% and too few for Day 12 to separate a real voice gain from
noise. **Do not change `SEED`** — Day 12 is only honest if it scores the split
Day 11 trained against.

## Day 11: training config, corrected for this data

The walkthrough's numbers were sized for 150 examples of ~50-token outputs.
Both assumptions are wrong here, and two settings change as a result.

```python
max_seq_length = 2048       # NOT 512

per_device_train_batch_size = 2
gradient_accumulation_steps = 4
num_train_epochs = 2        # replaces max_steps = 60
max_steps = -1
warmup_steps = 5
learning_rate = 2e-4
logging_steps = 1
```

**`max_seq_length = 2048`.** Several Unsloth notebook forks default to 512,
which truncates 150 of these 454 records — 33% of the corpus. SFTTrainer
truncates silently, and a truncated target is a target with its EOS cut off, so
the model would be trained to never stop generating. 2048 costs 2 records their
tails; 1024 costs 11 and is the fallback if the T4 runs out of memory. Keep
Unsloth's gradient checkpointing on either way.

**`num_train_epochs = 2` instead of `max_steps = 60`.** At `bs=2 x ga=4` that's
8 examples per optimizer step, so 409 records is 51 steps per epoch. The
walkthrough's 60 steps meant ~3 epochs over 150 records; against 454 the same
number is 1.2 epochs, a different run than intended. Two epochs is ~102 steps,
roughly 25-40 minutes on a T4 at this sequence length — longer than the
walkthrough's 10-20 minute estimate, which assumed short targets.

LoRA config is unchanged (`r=16`, `alpha=16`, `dropout=0`, all seven projection
modules). The walkthrough's overfitting caution at `r=16` was aimed at a
150-example set; 409 has more room.

## Day 12: what to watch for

One failure is predictable from reading the data, so look for it deliberately
rather than discovering it as a surprise.

The descriptions have real internal structure — an "also known as" line, an
etymology or sourcing paragraph, then `Flavor Profile` and `How to Use`
sections. That structure is a legitimate LoRA target, and is the part most
likely to transfer.

But `How to Use` is dense with **specific recipe names and links**: "Roasted
Whole Chicken with 18 Spice Chicken Rub," "Turkey and Pumpkin Chili." The model
will learn that the section contains confident recipe titles and will invent
them fluently for held-out products. Expect fabricated recipe names in the
Day 12 outputs.

That is not a defect to fix. It is the week's own thesis reproduced inside the
brand-voice task: the fine-tune learned the *shape* of the section and none of
the *facts* in it. Which is the same reason the SQF corpus is a RAG problem.

The honest comparison Day 12 exists for: how much of the gain would a good
system prompt plus three few-shot examples have gotten on its own? Run that arm
before concluding the LoRA was necessary.

## Day 13: conclusions

_To fill in after the training run._

- [ ] Fine-tuned vs base, scored on the 45 held-out records
- [ ] Fine-tuned vs a strong system prompt on the base model
- [ ] The fine-tune-vs-prompt recommendation, with numbers behind it

## Housekeeping

`finetune/` is deliberately **not** in `.gitignore`, unlike `data/` (ignored
wholesale) and the two Chroma indexes. The brand-voice dataset is committed:
~976KB across `brand_voice.jsonl`, `train.jsonl` and `test.jsonl`.

That is a decision, not an oversight. The copy is already public on the Spices,
Inc. site, so there is nothing to leak, and unlike `data/chunks.jsonl` this is
not regenerable from anything in the repo — the source is a Drupal export. A
committed dataset also makes the training run reproducible by anyone reading
the model card, which a gitignored one would not.

The Hugging Face model card can therefore point at the split directly rather
than describing it second-hand.
