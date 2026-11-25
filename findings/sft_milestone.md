# RadioReady: SFT Milestone Experiments

## Overview

We completed a two-stage LoRA fine-tuning experiment comparing two Qwen2.5-1.5B-Instruct models:

1. Plain SFT: standard clean-lyric rewriting instructions.
2. Conditioned SFT: the same task with `<edit>`, `<syl:N>`, and `<rhyme:X>` control tokens.

The objective was to determine whether structural conditioning improves explicit-content removal, syllable preservation, rhyme preservation, and line-count consistency.

## Training Data

We prepared 8,216 paired rewriting records:

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| ParaDeHate | 6,094 | 782 | 761 |
| Lyric pairs | 329 | 138 | 112 |
| Total | 6,423 | 920 | 873 |

After sequence-length filtering, 326 lyric training records remained for plain SFT and 320 for conditioned SFT.

## Training Configuration

- Base model: Qwen/Qwen2.5-1.5B-Instruct
- GPU: NVIDIA L4, 24 GB VRAM
- Precision: bfloat16
- Optimizer: AdamW
- Learning rate: 0.0002
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Micro-batch size: 1
- Gradient accumulation: 2
- Effective batch size: 2
- Maximum sequence length: 512
- Validation interval: 50 optimization steps
- Checkpoint selection: lowest validation loss

## Training Results

| Run | Steps | Best validation loss | Duration |
|---|---:|---:|---:|
| Plain Stage 1 | 300 | 0.9673 | 5.0 min |
| Conditioned Stage 1 | 300 | 0.9913 | 5.1 min |
| Plain Stage 2 | 489 | 0.2171 | 8.0 min |
| Conditioned Stage 2 | 480 | 0.2168 | 7.9 min |

Stage 1 used ParaDeHate. Stage 2 continued from the corresponding Stage 1 checkpoint and trained for three epochs on lyric pairs.

The four original configurations are saved under `experiments/sft/configs/`.

## Held-Out Evaluation

Both models were evaluated on 40 lyric sections: 20 Genius and 20 Kidz Bop.

Line-count accuracy uses all 40 sections. The remaining generation metrics use the 35 sections with valid outputs from both models. Syllable and rhyme metrics are calculated over changed lines.

| Metric | Plain | Conditioned |
|---|---:|---:|
| Correct line count | 90.0% | 97.5% |
| Profanity-free sections | 94.3% | 94.3% |
| Exact syllable preservation | 74.4% | 75.0% |
| Mean absolute syllable deviation | 0.326 | 0.409 |
| End-rhyme preservation | 83.7% | 84.1% |

The machine-readable results are in `findings/sft_results.csv`.

## Main Findings

Structural conditioning improved line-count accuracy substantially.

However, the conditioned model did not consistently improve syllable or rhyme preservation. Its mean absolute syllable deviation was higher than that of the plain baseline.

Both models removed detected profanity from most commonly valid outputs.

## Observed Limitations

- Occasional retained profanity.
- Semantic drift from the original lyric meaning.
- Awkward or unnatural substitutions.
- Preserved rhymes sometimes resulted from retaining the original end word.
- Preliminary evaluation covers a relatively small held-out sample.

## Future Work

1. Expand evaluation, including semantic similarity and human judgments.
2. Implement syllable- and rhyme-aware decoding or reranking.
3. Compare teacher models using matched structural instructions.
4. Prepare human preference pairs for DPO.
5. Run additional ablations, including training without ParaDeHate.
