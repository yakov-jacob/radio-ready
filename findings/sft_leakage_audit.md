# SFT Training/Test Leakage Audit

## Purpose

Check for exact and near-duplicate lyric sections between SFT training data and held-out evaluation data.

## Comparison

- Lyric training sections: 329
- Held-out evaluation sections: 40
- Near-duplicate character similarity threshold: 0.92

## Results

- No exact or near-duplicate training/test matches at or above the similarity threshold.
- Two near-duplicate pairs were identified within the held-out test sample.

## Interpretation

The audit found no training/test overlap meeting the selected threshold.

This does not establish the absence of semantic duplicates, related lyric variants, or overlaps below the threshold.

## Code

See `experiments/sft/leakage_check.py` for the original audit implementation.

The reported results should be reproduced with the same training and evaluation inputs before changing the threshold or preprocessing.
