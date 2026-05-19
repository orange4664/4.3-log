# test1-hamgnn-si

This folder contains the corrected HamGNN `v4.3` adapter code and the completed silicon benchmark artifacts for the packed sweep job `73239`.

## Scope

- Task type: HamGNN silicon Hamiltonian prediction benchmark
- Compared methods:
  - `adamw`
  - `muon_ns`
  - `rt_v43_stream_a`
  - `rt_v43_stream_b`
  - `rt_v43_stream_c`
  - `rt_v43_ns_a`
  - `rt_v43_ns_b`
  - `rt_v43_ns_c`
- Main metric:
  - `test/total_loss`
- Secondary metric:
  - `test/L1Loss_hamiltonian`

## Main result

Best method in this sweep:

- `rt_v43_ns_b`
- `test/total_loss = 0.0004526643315330148`
- `test/L1Loss_hamiltonian = 0.00042148586362600327`

This means the best result in the completed silicon sweep is a `v4.3*` variant, and specifically an `rt_v43_ns_*` variant.

## Ranking by test/total_loss

| Rank | Method | test/total_loss | test/L1Loss_hamiltonian |
| --- | --- | ---: | ---: |
| 1 | `rt_v43_ns_b` | 0.0004526643315330148 | 0.00042148586362600327 |
| 2 | `rt_v43_ns_a` | 0.00045320147182792425 | 0.0004219861002638936 |
| 3 | `adamw` | 0.0004533186147455126 | 0.00042209512321278453 |
| 4 | `rt_v43_ns_c` | 0.00045364003744907677 | 0.0004223943396937102 |
| 5 | `rt_v43_stream_a` | 0.00045369635336101055 | 0.00042244690121151507 |
| 6 | `rt_v43_stream_c` | 0.0004541226080618799 | 0.0004228437028359622 |
| 7 | `rt_v43_stream_b` | 0.00045423145638778806 | 0.000422945071477443 |
| 8 | `muon_ns` | 0.00045428014709614217 | 0.0004229903861414641 |

## Contents

- `code/`
  - corrected local adapter used to repair the remote runtime path
  - includes `hamgnn_rt/optimizers.py` and `hamgnn_rt/run_hamgnn_rt.py`
- `benchmark/configs/`
  - packed sweep config set and manifest
- `benchmark/submit_hamgnn_silicon_v43_sweep_packed_a800.sbatch`
  - submission script used by the sweep
- `benchmark/logs/hgnn_v43_sw_73239.out`
  - packed launcher log showing method launch and completion order
- `benchmark/diag/v43_*_73239/`
  - per-method `train.log`
  - per-method `gpu_monitor.csv`
  - optimizer diagnostics JSONL for non-AdamW methods

## Notes

- This is not a PINN or PDE benchmark.
- This is a HamGNN silicon benchmark on Hamiltonian prediction.
- The `v4.3` logic used here corresponds to the corrected selector-style implementation path that was validated before running job `73239`.
