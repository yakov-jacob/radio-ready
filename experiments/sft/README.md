# RadioReady SFT Experiments

This directory contains the milestone evaluation scripts, duplicate-audit script, and saved training configurations for the plain and conditioned Qwen2.5-1.5B-Instruct models.

## Files

- `quick_eval.py`: initial SFT evaluation.
- `expanded_eval.py`: expanded held-out evaluation.
- `leakage_check.py`: training/test duplicate audit.
- `configs/`: configurations saved from all four training runs.

## Core Project Modules

- `src/radio_ready/sft.py`: builds plain or structurally conditioned training examples.
- `src/radio_ready/train.py`: custom LoRA training and prediction implementation.
- `src/radio_ready/eval_set.py`: builds held-out evaluation sections.
- `src/radio_ready/references.py`: provides human-approved reference handling.

The training implementation and supporting modules were recovered from Durga's project branch.

## Training Pipeline

1. Prepare ParaDeHate and vetted lyric pairs.
2. Build plain SFT examples using `python -m radio_ready.sft`.
3. Build conditioned examples using `python -m radio_ready.sft --controls edit syl rhyme`.
4. Train one Stage 1 model on all of ParaDeHate for one full epoch (shared by both variants):
   `python -m radio_ready.train train --data sft_plain --sources paradehate --name stage1 --batch-size 1 --grad-accum 2 --max-len 512`
5. Continue it twice with Stage 2 lyric training (3 epochs each):
   `--data sft_plain ... --resume stage1 --name plain_stage2` and
   `--data sft_edit_syl_rhyme ... --resume stage1 --name controlled_stage2`.
6. Select the best validation checkpoints.
7. Generate held-out predictions using greedy decoding.
8. Evaluate line-count accuracy, cleanliness, syllable preservation, and end-rhyme preservation.

## Corrections since the milestone runs

The milestone runs (`configs/`) had three problems, fixed in `sft.py` and `train.py`:

- **Control tokens on ParaDeHate.** Stage 1 gave every ParaDeHate example `<syl:N>` and `<edit>`,
  but only 12% of its rewritten lines stay within one syllable of `<syl:N>` and 46% of them have
  no `<edit>` (no detected explicit word), against 92% and all for the lyric pairs. Stage 1 so
  taught the model that the tokens do not constrain anything. ParaDeHate now has no control
  tokens; they are learned in Stage 2 only, and Stage 1 is the same for both variants.
- **Stage 1 length.** 300 steps at effective batch 2 covered 600 of 6,094 ParaDeHate pairs
  (about 10% of an epoch). The default is now one full epoch (no `--max-steps`).
- **Token initialization.** Every control token started as the same vector (the mean
  embedding), so `<syl:8>` and `<syl:9>` were indistinguishable at first. Each now starts from
  the embeddings of the words it stands for (`<syl:21>`: "twenty-one" and "syllables").

New runs also use the base model's chat template for the prompt (`--chat`, the default); an
adapter keeps the format it was trained with, and `predict` and the evaluation scripts read it
from its `config.json`. The milestone results below come from the old setup.

## Experiment Settings

See the four JSON files in `configs/` for the original saved configurations.

The recorded Stage 1 runs used 300 optimization steps per model.

Stage 2 trained for three epochs and completed 489 steps for the plain model and 480 for the conditioned model.

All runs used a learning rate of 0.0002, LoRA rank 16, LoRA alpha 32, dropout 0.05, effective batch size 2, and maximum sequence length 512.

## Results

See:
- `../../findings/sft_milestone.md`
- `../../findings/sft_results.csv`
- `../../findings/sft_leakage_audit.md`

## Checkpoints and Data

Model checkpoint files and generated lyric predictions are intentionally not stored in this public repository.

Original checkpoint directories:

- `outputs/sft_models/plain_stage1/`
- `outputs/sft_models/controlled_stage1/`
- `outputs/sft_models/plain_stage2/`
- `outputs/sft_models/controlled_stage2/`

The model files and evaluation outputs are preserved separately in the milestone backup.

## Reproduction Notes

Exact evaluation commands and checkpoint retrieval instructions still need verification against the original scripts and saved environment.

Saved configurations document the completed experiments, but successful end-to-end reproduction has not yet been independently verified.
