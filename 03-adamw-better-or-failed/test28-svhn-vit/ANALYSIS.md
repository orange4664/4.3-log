# test28-svhn-vit analysis

## Run record

First smoke submission:

- server: `nmcc-n46h1`
- work tree: `~/run/4.3-log/test28-svhn-vit`
- smoke job id: `73423`
- state: failed before training due dataset download timeout

Failure cause from the smoke log:

- `torchvision.datasets.SVHN(..., download=True)` timed out while trying to
  fetch the dataset inside the job

Follow-up action:

- pre-download SVHN onto the server under
  `/data/run01/scwb923/4.3-log/_datasets/svhn`
- rerun smoke using the local cached dataset instead of relying on runtime
  network download

Rerun:

- smoke rerun job id: `73424`
- rerun completed enough to analyze smoke outputs
- landed `adamw` smoke summary:
  - `best_test_acc = 0.61171875`
- landed `muon_ns` smoke summary:
  - `best_test_acc = 0.5779296875`

## Acceptance rule

This benchmark is only accepted if:

- `muon_ns` beats `adamw`

## Result

This benchmark is **not accepted** under the user's rule.

On the completed rerun smoke configuration:

- `adamw best_test_acc = 0.61171875`
- `muon_ns best_test_acc = 0.5779296875`

So original `muon_ns` is clearly below `adamw` on this real SVHN ViT smoke
benchmark, and the benchmark should not be promoted to formal.

## Notes

This benchmark reuses the accepted CIFAR ViT code path but swaps the dataset to
SVHN so the optimizer is tested on a different real-data visual regime.
