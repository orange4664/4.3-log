# v4.3 repair chain, why only 73239 counts, and how test2-pinn was corrected

This note is the fuller audit trail for three separate claims:

1. what `v4.3` actually is;
2. why the accepted HamGNN silicon result is still only the packed sweep job `73239`;
3. how `test2-pinn` was corrected from a selector/toy placeholder into a real PDE/PINN benchmark track.

## 1. What `v4.3` is, and what it is not

The source package that settled the interpretation was:

- `rt_muon_v4_3_selectorbench.zip`

The important conclusion is narrow:

- `v4.3` is a selector family;
- it is not a new standalone penalty formula;
- at small learning rate it takes the hard-quench branch;
- at larger learning rate it takes the soft-quench branch;
- in the selector benchmark package the default switch is `0.05`.

That means the correct conceptual picture is:

- low-LR regime -> `v4.1`-style hard quench
- higher-LR regime -> `v4.2`-style soft quench
- `v4.3` chooses between them

## 2. Why earlier HamGNN-side "v43" runs were not accepted

Earlier remote runs were not rejected because the numbers were slightly worse.
They were rejected because the active code path was not yet the repaired one we
intended to test.

The gaps were:

- the HamGNN wrapper did not export the full selector-era runtime knobs;
- the runtime path was missing fields such as:
  - `rt_lambda_consistency`
  - `rt_lambda_diff`
  - `rt_lambda_uncert`
  - `rt_beta_grid`
- the optimizer branch being exercised on the HamGNN side was still a simplified
  path, not the repaired selector logic we wanted to evaluate.

That is why older "v43" numbers were treated as provisional and not used as
evidence.

## 3. What was repaired before the accepted run

The accepted local repair landed in:

- `test1-hamgnn-si/code/hamgnn_rt/optimizers.py`
- `test1-hamgnn-si/code/hamgnn_rt/run_hamgnn_rt.py`

The key repair points were:

- the runtime wrapper exports the selector-relevant environment knobs;
- the optimizer config now includes:
  - `lr_switch`
  - `hard_delta`
  - `soft_delta`
  - `lambda_noise`
  - `lambda_consistency`
  - `lambda_diff`
  - `lambda_uncert`
  - `lambda_tube`
  - `phi_threshold_hard`
  - `phi_threshold_soft`
  - `soft_tau`
- the Muon-source branch and stream branch both route into the repaired
  `_rt_weights(...)` selector logic;
- the HamGNN modes explicitly include:
  - `rt_v43_stream`
  - `rt_v43_ns`

The decisive point is that the accepted HamGNN code path and the later PINN
benchmark code path now point back to the same repaired torch implementation.

## 4. Why only job 73239 is the accepted HamGNN silicon result

Only job `73239` satisfies both requirements at once:

### 4.1 Correct repaired code path was active

The live optimizer diagnostics produced by the run showed selector-era fields,
including:

- `beta_candidate`
- `soft_shrink`
- `phi`
- `raw_cv_gain`
- `diffusion_cost`
- `uncertainty_cost`
- `tube_penalty`

Those fields matter because they are only emitted by the repaired selector path,
not by the earlier simplified path.

### 4.2 The sweep was complete

Job `73239` completed the full packed method set:

- `adamw`
- `muon_ns`
- `rt_v43_stream_a`
- `rt_v43_stream_b`
- `rt_v43_stream_c`
- `rt_v43_ns_a`
- `rt_v43_ns_b`
- `rt_v43_ns_c`

The preserved launcher log is:

- `test1-hamgnn-si/benchmark/logs/hgnn_v43_sw_73239.out`

The per-method diagnostics are under:

- `test1-hamgnn-si/benchmark/diag/`

### 4.3 The accepted ranking from 73239

By `test/total_loss`:

1. `rt_v43_ns_b` - `0.0004526643315330148`
2. `rt_v43_ns_a` - `0.00045320147182792425`
3. `adamw` - `0.0004533186147455126`
4. `rt_v43_ns_c` - `0.00045364003744907677`
5. `rt_v43_stream_a` - `0.00045369635336101055`
6. `rt_v43_stream_c` - `0.0004541226080618799`
7. `rt_v43_stream_b` - `0.00045423145638778806`
8. `muon_ns` - `0.00045428014709614217`

So the accepted HamGNN silicon claim stays narrow:

- the corrected `v4.3` path was run in a complete sweep;
- the best method in that accepted sweep was `rt_v43_ns_b`;
- earlier "v43" HamGNN runs are not used because the repair was not complete.

## 5. Why test2-pinn initially needed correction

The repository already had a `test2-pinn` folder, but its contents were still a
selector/toy benchmark bundle:

- `test2-pinn/code/experiments/run_sq_rt_v43_selector_bench.py`
- `test2-pinn/results/full_budget_matched_20260519/`

That selector bench is still useful as optimizer-family reference material, but
it is not a PINN or PDE result. It does not justify calling the folder a PDE
benchmark by itself.

That mismatch is why `test2-pinn` had to be corrected.

## 6. What was added to make test2-pinn a real PDE/PINN track

The real PDE/PINN addition is:

- `test2-pinn/code/pinn_heat/run_heat_pinn_benchmark.py`
- `test2-pinn/code/pinn_heat/README.md`
- `test2-pinn/scripts/submit_heat1d_pinn_a800.sbatch`

This benchmark is a genuine PINN setup:

- PDE: 1D heat equation
- model: MLP PINN
- losses:
  - PDE residual
  - initial condition
  - boundary condition
- evaluation:
  - grid MSE
  - grid relative L2
  - max absolute grid error

The benchmark compares:

- `adamw`
- `muon_ns_adamw`
- `rt_v43_stream_adamw`
- `rt_v43_ns_adamw`

For the three Muon-family entries, the matrix parameters use the repaired
`HamGNNMuonFDTOptimizer` implementation from `test1-hamgnn-si/code/hamgnn_rt/optimizers.py`,
while non-matrix parameters use its AdamW fallback. That is exactly the hybrid
split we accepted conceptually on the HamGNN side.

## 6.1 What actually ran on the server

The accepted PDE/PINN run was not a local proxy run. It was completed on the
server through Slurm:

- login host used for submission: `nmcc-n46h1`
- actual login node shown by `hostname`: `ln02`
- work directory: `~/run/4.3-log/test2-pinn`
- submit script:
  - `test2-pinn/scripts/submit_heat1d_pinn_a800.sbatch`
- accepted completed job id:
  - `73302`
- queue:
  - `gpu_a800`
- python environment chosen by the script:
  - `/data/home/scwb923/run/envs/hamgnn/bin/python`

The preserved server-side evidence is:

- `test2-pinn/logs/test2_pinn_v43_73302.out`
- `test2-pinn/logs/test2_pinn_v43_73302.err`
- `test2-pinn/results/heat1d_pinn_20260519/run_args.json`

The `run_args.json` file records the server-side resolved paths:

- script:
  - `/data/run01/scwb923/4.3-log/test2-pinn/code/pinn_heat/run_heat_pinn_benchmark.py`
- repaired optimizer source:
  - `/data/run01/scwb923/4.3-log/test1-hamgnn-si/code/hamgnn_rt/optimizers.py`

## 7. Why this PDE benchmark is acceptable even though it is not Burgers

The user asked for a PINN/PDE benchmark, not specifically Burgers-only.

The heat-equation PINN is acceptable because:

- it is a real PDE residual benchmark, not a toy matrix surrogate;
- it has exact boundary and initial data;
- it has a closed-form exact solution, which gives direct error metrics;
- it lets us reuse the repaired torch v4.3 implementation instead of inventing a
  second, unverified PDE-side implementation.

So this benchmark extends the evidence in the right direction:

- `test1-hamgnn-si` remains the accepted Hamiltonian result;
- `test2-pinn` now becomes an actual PDE/PINN follow-up track;
- and the accepted result is specifically the completed server run from job
  `73302`, not an unfinished local smoke run.

## 8. What still remains out of scope

Even after this correction, the claims should stay scoped correctly.

The accepted HamGNN result is still:

- a silicon Hamiltonian benchmark

The `test2-pinn` result is:

- a PDE/PINN follow-up benchmark

Neither one should be misdescribed as:

- a large PDE foundation-model benchmark,
- a Burgers DeepONet benchmark,
- a materials benchmark,
- or a proof that every `v4.3` variant dominates every baseline everywhere.

## 9. Practical summary

The evidence chain that is defensible now is:

1. `v4.3` was interpreted correctly as a selector family.
2. The HamGNN adapter and runtime path were repaired accordingly.
3. Only packed sweep `73239` is accepted as the HamGNN silicon result because it
   uses the repaired path and completes the full method set.
4. `test2-pinn` was corrected from a selector/toy placeholder into a real
   PDE/PINN benchmark track that reuses the same repaired torch implementation.
