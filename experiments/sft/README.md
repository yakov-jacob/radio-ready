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
4. Train plain and conditioned Stage 1 models on ParaDeHate.
5. Continue each model with Stage 2 lyric training.
6. Select the best validation checkpoints.
7. Generate held-out predictions using greedy decoding.
8. Evaluate line-count accuracy, cleanliness, syllable preservation, and end-rhyme preservation.

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
