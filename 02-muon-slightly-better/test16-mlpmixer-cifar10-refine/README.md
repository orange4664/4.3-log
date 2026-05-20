# test16-mlpmixer-cifar10-refine

Targeted follow-up benchmark for the near-miss `test12-mlpmixer-cifar10`.

## Why this exists

`test12` turned out to be very close under the formal run:

- best `adamw`: `0.7766`
- best `muon_ns`: `0.7741`

This folder keeps the same real-data, matrix-heavy MLP-Mixer model family, but
focuses the search budget around the promising region instead of spending full
budget on clearly weak settings.

## Acceptance gate

This benchmark is only accepted if original `muon_ns` beats `adamw`.
