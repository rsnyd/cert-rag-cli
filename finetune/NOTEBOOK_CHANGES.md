# Colab Notebook Changes

Every edit made to Unsloth's stock notebook to train the brand-voice adapter.
Recorded here because the notebook lives in Colab, outside this repo, and a
Colab session is not a durable artifact — the edits below are what make the run
reproducible from the stock notebook.

**Base notebook:** [Llama3.1_(8B)-Alpaca.ipynb](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_%288B%29-Alpaca.ipynb)
**Runtime:** T4 GPU (Runtime -> Change runtime type)

The stock notebook uses Alpaca's `{instruction, input, output}` schema, which is
identical to `brand_voice.jsonl`. That is why `alpaca_prompt`, `EOS_TOKEN` and
`formatting_prompts_func` are used unchanged — the only reason this notebook is
a near drop-in.

## 1. Mount Drive

New cell, before Data Prep. `train.jsonl` and `test.jsonl` are uploaded to
`MyDrive/spice-voice/` by hand. Drive rather than `files.upload()` because free
Colab drops the runtime often and uploaded files do not survive it.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Point the dataset cell at our data

In the Data Prep cell, replace only the `load_dataset` call. The `split="train"`
form matters: the dict form returns a `DatasetDict`, which does not drop into
`SFTTrainer(train_dataset=...)` below.

```python
dataset = load_dataset(
    "json",
    data_files = "/content/drive/MyDrive/spice-voice/train.jsonl",
    split = "train",
)
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

Map output must read **439/439**. If it reads 51760, the cell did not re-run and
training would silently use Alpaca.

## 3. Capture the base model's output before training

New cell above "Train the model". Once the adapter is trained the base
behavior is gone, and Day 11's before/after cannot be reconstructed without
reloading the model.

```python
FastLanguageModel.for_inference(model)
inputs = tokenizer([
    alpaca_prompt.format(
        "Write a product description in the Spices, Inc. brand voice for the following blend.",
        "Name: Satay Seasoning\nCategory: Seasoning Blends",
        "",
    )
], return_tensors = "pt").to("cuda")

out = model.generate(**inputs, max_new_tokens = 400, use_cache = True)
print(tokenizer.batch_decode(out)[0])

FastLanguageModel.for_training(model);   # required, or training after this misbehaves
```

The trailing `;` suppresses Jupyter echoing the whole `PeftModelForCausalLM`
repr, which is several hundred lines.

## 4. `SFTConfig`: epochs instead of a step count

In the `SFTTrainer(...)` cell — `SFTConfig`, not `TrainingArguments`; TRL
renamed it.

```python
        num_train_epochs = 2,   # was: commented out
      # max_steps = 60,         # was: active
```

`max_steps = 60` was sized for the walkthrough's 150-example set (~3 epochs).
Against 439 records at `bs=2 x ga=4` it is 1.1 epochs — a different run than
intended. Two epochs is ~110 steps. Commenting the line out is cleaner than
`max_steps = None`; the underlying default is `-1`.

`max_seq_length = 2048` in the model-loading cell is the stock value and is
correct here — see FINE_TUNING_NOTES.md for why 512 would be destructive.

## 5. Pre-flight checks

New cell between `SFTTrainer(...)` and `trainer.train()`. Five seconds, and it
catches every way this run can be silently wrong.

```python
import math, torch

print("records:", len(dataset))
assert len(dataset) == 439, f"got {len(dataset)} - the dataset cell did not re-run"

print("\n" + "=" * 70)
print(dataset[0]["text"])          # read this in full, do not skim
print("=" * 70)

eos = tokenizer.eos_token
missing = sum(1 for t in dataset["text"] if not t.endswith(eos))
print(f"\nEOS token: {eos!r}   missing from: {missing} records  (must be 0)")

lens = [len(tokenizer(t)["input_ids"]) for t in dataset["text"]]
over = sum(1 for n in lens if n > max_seq_length)
print(f"max_seq_length = {max_seq_length}")
print(f"tokens: max={max(lens)}  mean={sum(lens) // len(lens)}")
print(f"would truncate: {over} records  (2 is expected, 100+ means 512)")

a = trainer.args
eff = a.per_device_train_batch_size * a.gradient_accumulation_steps
per_epoch = math.ceil(len(dataset) / eff)
total = a.max_steps if a.max_steps and a.max_steps > 0 else per_epoch * a.num_train_epochs
print(f"\nepochs={a.num_train_epochs}  max_steps={a.max_steps}  batch x accum={eff}")
print(f"-> {total} optimizer steps  (expect ~110, NOT 60)")

print(f"\n{torch.cuda.get_device_name(0)}")
free, tot = torch.cuda.mem_get_info()
print(f"GPU memory free: {free/1e9:.1f} / {tot/1e9:.1f} GB")
```

Values from the actual run: 439 records, 0 missing EOS, max 2577 tokens,
2 truncated, 110 steps, Tesla T4, 8.8/15.6 GB free.

## 6. Push the adapter

Uncomment at the bottom, filling in a real HF write token:

```python
model.push_to_hub("rsnyd/spice-voice-lora", token = "hf_...")
tokenizer.push_to_hub("rsnyd/spice-voice-lora", token = "hf_...")
```

## Cells left alone

Everything below the save section — GGUF, float16-for-vLLM, Ollama, the other
quantized exports — stays commented out. Those are 20+ minutes for formats this
project does not use.

Also unchanged: `alpaca_prompt`, `EOS_TOKEN`, `formatting_prompts_func`, the
whole `get_peft_model` LoRA config (`r=16`, `alpha=16`, seven target modules),
and every `SFTConfig` field except the two above.

## The one trap

Training uses the **Alpaca prompt format**, not a chat template. Inference must
use the identical format — `alpaca_prompt.format(instruction, input, "")` —
including on Day 12 and anywhere the adapter is used later. Training in Alpaca
format and inferring with `tokenizer.apply_chat_template` produces garbage that
looks exactly like a failed fine-tune. The walkthrough's Day 11 snippet suggests
the chat template; it is a different convention and mixing them breaks the run.
