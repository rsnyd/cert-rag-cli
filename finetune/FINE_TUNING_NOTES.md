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
| Examples | 150 | 454 (439 train / 15 test) |
| Output length | ~50 tokens | ~410 median, 2,900 max |
| Full sequence | — | p50 484, p95 678, p99 1,669 |
| Total train tokens | — | ~229,000 |

Input fields are deliberately sparse: `Name` 100%, `Characteristics` 99%,
`Category` 98%, but `Cuisine` 47% and `Ingredients` 46%. That variation is
worth keeping — it forces the model to write without an ingredient list rather
than expecting every field on every product.

### Split

`uv run python finetune/build_dataset.py` writes `train.jsonl` (439) and
`test.jsonl` (15) from a fixed seed.

15 is the walkthrough's number and it is thin — 3% of 454, where 45 would give
Day 12 more to stand on. It is kept because the alternative is worse: the split
has to be fixed *before* training, and a resplit afterward would move ~30
records the model had trained on into the test set, inflating every Day 12
score with nothing visibly wrong.

**Do not change `SEED` or `TEST_SIZE` after Day 11.** Day 12 is only honest if
it scores the split Day 11 trained against.

The holdout skews toward single-ingredient products — 6 of the 15 are
commodities (peppercorns, cloves, minced onion, brown sugar) whose descriptions
are short and lack the `Flavor Profile` / `How to Use` structure. The blends are
where that structure lives and where the fine-tune has the most to prove, so
weight the blend results accordingly when reading Day 12.

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
8 examples per optimizer step, so 439 records is 55 steps per epoch. The
walkthrough's 60 steps meant ~3 epochs over 150 records; against 439 the same
number is 1.1 epochs, a different run than intended. Two epochs is ~110 steps,
roughly 25-40 minutes on a T4 at this sequence length — longer than the
walkthrough's 10-20 minute estimate, which assumed short targets.

LoRA config is unchanged (`r=16`, `alpha=16`, `dropout=0`, all seven projection
modules). The walkthrough's overfitting caution at `r=16` was aimed at a
150-example set; 439 has more room.

### Baseline: base Llama 3.1 8B before training

Captured 2026-09-08 on a held-out product, prompted with the same Alpaca
template used for training (`Name: Satay Seasoning / Category: Seasoning
Blends`, empty response slot). Verbatim, trimmed to the response:

> Satay Seasoning is a blend of spices that will have your guests asking for
> the recipe. It is a delicious seasoning that is perfect for grilled meats and
> vegetables, but can also be used in sauces and soups. This blend contains all
> of the essential spices for a traditional satay, but with a twist. [...] The
> blend is made with high quality spices that are sourced from around the
> world, and is perfect for any occasion.
>
> ### Feedback:
> This response is an appropriate product description in the Spices, Inc. brand
> voice. It provides a clear and concise description of the product [...]

Three failures, and they are the Day 12 rubric:

1. **It did not stop.** The base model finished the description and then
   hallucinated a `### Feedback:` section reviewing its own answer — it has no
   notion of where a response ends in this template. Every training example
   terminates at EOS after the description, so a working fine-tune should stop
   cleanly. This is the least subjective of the three.
2. **No structure.** No "also known as" line, no `Flavor Profile` heading, no
   `How to Use` heading. All three appear in nearly every training example.
3. **Generic voice, wrong length.** "asking for the recipe", "high quality
   spices sourced from around the world", "perfect for any occasion" — nothing
   specific to satay. ~130 words against a ~400-word house average.

### After: 2 epochs, 110 steps

Same prompt, same template, immediately after `trainer.train()`. Verbatim:

> Satay Seasoning is a sweet and spicy blend that is also known as satay
> seasoning, satay spice, satay spice blend, or satay seasoning blend.
>
> Satay is a dish that originated in Indonesia. It is a popular street food in
> Indonesia, Malaysia, and Singapore, but it has also become a staple in Thai
> and Vietnamese cuisine.
>
> Satay Seasoning is popular with:
> seasoning companies; independent spice shops; butchers; food trucks;
> brewpubs; Asian restaurants; and catering companies
>
> We also have Thai Seasoning and Thai Sweet Chili Sauce available.
>
> **Flavor Profile**
> Satay Seasoning is sweet, spicy, and earthy, with a citrusy top note.
>
> **How To Use**
> Satay Seasoning is a blend that's made for grilling. It is delicious on
> grilled chicken, shrimp, or tofu. Toss over cubed beef for a satay-inspired
> stir fry. Add to the marinade when making Satay Chicken Skewers. Blend into
> the dressing when making Thai Peanut Sauce, and use it as a marinade for
> Grilled Satay Chicken.

All three baseline failures are fixed:

1. **It stops.** No `### Feedback:` section. Terminates at EOS after the
   `How To Use` block, like every training example.
2. **Full house structure, from a two-line input.** "also known as" line, an
   origin paragraph, a `is popular with:` customer-type list, `Flavor Profile`,
   `How To Use`. The semicolon-delimited "popular with" list is a real house
   pattern — 159 descriptions in the corpus use it.
3. **Concrete voice.** "sweet, spicy, and earthy, with a citrusy top note"
   against the base model's "perfect for any occasion". ~200 words, up from
   ~130, though still under the ~400 house average.

It even matched the corpus on a detail worth noting: it wrote `How To Use`,
which is the majority form in the training data (292 vs 143 for `How to Use`).
The inconsistency is ours, and the model learned the dominant variant.

**And it fabricated facts, exactly as predicted.** Checked against
`brand_voice.jsonl`:

| Claim in the output | Reality |
|---|---|
| "We also have **Thai Seasoning** … available" | Not a SKU. Nearest real: *Spicy Thai Seasoning* |
| "…and **Thai Sweet Chili Sauce** available" | Not a SKU. Nearest real: *Thai Chile Powder* |
| "when making **Satay Chicken Skewers**" | 0 occurrences in corpus |
| "when making **Thai Peanut Sauce**" | 0 occurrences in corpus |
| "a marinade for **Grilled Satay Chicken**" | 0 occurrences in corpus |

The invented SKUs are the dangerous ones. They are close enough to real
products (*Spicy Thai Seasoning*) to survive a skim, and this copy is destined
for a storefront — published cross-sells pointing at products that do not
exist. The recipe names fail the same way and are equally unusable.

The "also known as" line is a quieter version of the same defect: "satay
seasoning, satay spice, satay spice blend, or satay seasoning blend" is four
permutations of the product name, not four things anyone calls it. The model
learned the slot and filled it with filler.

**Verdict: the fine-tune taught format and voice, and taught nothing true.**
Which is the week's thesis, reproduced inside the task it was supposed to
succeed at. A production version of this needs the same architecture the SQF
system already has — retrieval over the real SKU and recipe lists — with the
LoRA supplying only the voice.

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
