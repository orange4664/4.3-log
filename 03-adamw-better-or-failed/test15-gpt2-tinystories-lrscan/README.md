# test15-gpt2-tinystories-lrscan

Non-HamGNN language-model benchmark that revisits the small GPT-2 TinyStories
direction with an explicit learning-rate screen for `adamw` and original
`muon_ns`.

## Why this exists

`test14` showed that a single fixed learning rate can strongly disadvantage one
optimizer family. This folder is a fairer follow-up:

- same real TinyStories local text
- same GPT-2 style from-scratch decoder
- small enough for server screening
- explicit per-optimizer LR scan before accepting or rejecting the benchmark

## Acceptance gate

This benchmark is only accepted if the best original `muon_ns` run beats the
best `adamw` run on validation loss.
