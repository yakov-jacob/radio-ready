import gc
import json
import random
from pathlib import Path

import torch
from radio_ready.train import format_prompt, load_model

data_dir = Path("data/processed/sft")
model_dir = Path("outputs/sft_models")
out_dir = Path("outputs/evaluation/expanded_test")
out_dir.mkdir(parents=True, exist_ok=True)

def load_rows(name):
    path = data_dir / f"{name}.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]

plain = {
    r["id"]: r for r in load_rows("sft_plain")
    if r["split"] == "test"
}
controlled = {
    r["id"]: r for r in load_rows("sft_edit_syl_rhyme")
    if r["split"] == "test"
}

# Select identical, held-out test examples.
shared = sorted(set(plain) & set(controlled))
selected = []

for source in ("genius", "kidzbop"):
    pool = [sid for sid in shared if plain[sid]["source"] == source]
    random.Random(42).shuffle(pool)
    selected.extend(pool[:20])

print(f"Selected {len(selected)} held-out lyric pairs")

results = {
    sid: {
        "id": sid,
        "source": plain[sid]["source"],
        "original": plain[sid]["prompt"].split("\n", 1)[1].strip(),
        "reference": plain[sid]["completion"],
        "predictions": {}
    }
    for sid in selected
}

for model_name, dataset in [
    ("plain_stage2", plain),
    ("controlled_stage2", controlled),
]:
    print(f"\nGenerating with {model_name}...", flush=True)

    adapter = model_dir / model_name
    cfg = json.loads((adapter / "config.json").read_text())
    tok, model, device = load_model(cfg, adapter, train=False)

    for i, sid in enumerate(selected, 1):
        prompt = format_prompt(tok, cfg, dataset[sid]["prompt"])  # as trained
        encoded = tok(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).to(device)

        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=192,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id
            )

        new_tokens = output[0, encoded["input_ids"].shape[1]:]
        prediction = tok.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        results[sid]["predictions"][model_name] = prediction
        print(f"  {i}/{len(selected)}", flush=True)

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

output_path = out_dir / "predictions.jsonl"
with output_path.open("w", encoding="utf-8") as f:
    for record in results.values():
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\nPRELIMINARY RESULTS")

for model_name in ("plain_stage2", "controlled_stage2"):
    valid = 0
    exact = 0

    for record in results.values():
        prediction = record["predictions"][model_name]
        predicted_lines = [x.strip() for x in prediction.splitlines()]
        original_lines = record["original"].splitlines()
        reference_lines = [x.strip() for x in record["reference"].splitlines()]

        valid += (
            len(predicted_lines) == len(original_lines)
            and all(predicted_lines)
        )
        exact += predicted_lines == reference_lines

    print(f"{model_name}:")
    print(f"  Valid line count: {valid}/{len(results)}")
    print(f"  Exact reference matches: {exact}/{len(results)}")

print(f"\nSaved predictions to: {output_path}")

# Show one example for initial inspection.
if results:
    example = next(iter(results.values()))
    print("\nEXAMPLE COMPARISON")
    print("Original:\n", example["original"])
    print("\nReference clean version:\n", example["reference"])
    print("\nBaseline:\n", example["predictions"]["plain_stage2"])
    print("\nConditioned:\n", example["predictions"]["controlled_stage2"])
