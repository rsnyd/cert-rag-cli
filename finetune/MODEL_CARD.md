---
base_model: unsloth/Meta-Llama-3.1-8B-Instruct
library_name: peft
tags:
  - lora
  - qlora
  - unsloth
  - text-generation
  - brand-voice
license: llama3.1
---

# spice-voice-lora

A LoRA adapter that writes product descriptions in the Spices, Inc. house
voice, given only a product name, category and ingredient list.

Trained as a deliberate test of a question rather than as a product: **how much
of a house style can a small model learn, and what does it fail to learn?** The
answer to the second half turned out to be the useful one.

## What it does

Given bare facts:

```
Name: Satay Seasoning
Category: Seasoning Blends
```

it produces the full house format — an "also known as" line, an origin
paragraph, a customer-type list, a `Flavor Profile` section and a `How To Use`
section — at roughly the length and register of the published catalog.

## What it gets wrong, and why that matters

**It invents products and recipes that do not exist.** Measured over 15
products, it fabricated 1.87 proper names per description: cross-sells to SKUs
the company does not sell, and recipe titles appearing nowhere in the catalog.

This is not a training defect to be fixed with more steps. LoRA transfers
behavior, not facts. The adapter learned that a `How To Use` section contains
confident recipe names and produced confident recipe names.

Fabrication rose *with* fluency across all three arms tested:

| arm | voice (1-5) | structure (1-5) | fabricated names |
|---|---|---|---|
| base model, bare prompt | 1.53 | 1.27 | 0.00 |
| base model, 3 few-shot | 1.87 | 2.33 | 1.00 |
| this adapter | 3.07 | 4.20 | 1.87 |

The base model scores zero fabrications not because it is careful but because
it writes nothing specific enough to be wrong — 2.87 contentless marketing
phrases per description ("perfect for any occasion") and no names at all.

**Do not use this to generate publishable copy unsupervised.** Any real
deployment needs retrieval over the actual SKU and recipe lists, with this
adapter supplying only the voice.

## Evaluation

Three arms on 15 products, judged by Claude Sonnet against a rubric measured
from the 454-description corpus rather than written by hand — 96% carry a
`Flavor Profile` section, 96% a `How To Use` section, 82% open with the product
name, ~293 words average. A hand-written rubric would have scored the company's
own copywriters as off-brand.

**Caveat:** the 15 evaluation products came from a superseded train/test split
and appear in the training data, so this is not a clean held-out evaluation.
Checked for memorization before reporting: output-to-published similarity is
0.075 mean, 0.157 max, against 0.026 for the base model — the adapter learned
the template rather than the text. The scores are reported with that
qualification.

## Training

| | |
|---|---|
| Base | `unsloth/Meta-Llama-3.1-8B-Instruct`, 4-bit |
| Method | QLoRA, `r=16`, `alpha=16`, dropout 0 |
| Target modules | q, k, v, o, gate, up, down |
| Data | 439 published product descriptions |
| Epochs | 2 (110 optimizer steps) |
| Batch | 2 x 4 gradient accumulation |
| LR | 2e-4, linear, 5 warmup steps |
| Max seq length | 2048 |
| Hardware | one free Colab T4, ~35 minutes |

Data is the company's own published catalog copy, already public on the
Spices, Inc. website. No customer data, no internal documents.

## Usage

Prompts must use the **Alpaca format it was trained on**. Using a chat template
instead produces incoherent output.

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "rsnyd/spice-voice-lora", max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

inputs = tokenizer([alpaca_prompt.format(
    "Write a product description in the Spices, Inc. brand voice for the following blend.",
    "Name: Satay Seasoning\nCategory: Seasoning Blends",
    "",
)], return_tensors="pt").to("cuda")

print(tokenizer.batch_decode(model.generate(**inputs, max_new_tokens=600))[0])
```

## Full writeup

Dataset construction, the notebook changes, the measured rubric, the
fabrication detector and the fine-tune-vs-prompt-vs-RAG conclusions:
[github.com/rsnyd/cert-rag-cli/tree/main/finetune](https://github.com/rsnyd/cert-rag-cli/tree/main/finetune)
