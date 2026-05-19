# v4.3 HamGNN repair and why only job 73239 counts

This note records the repair trail for the HamGNN `v4.3` path and explains why the accepted silicon result is taken from job `73239`, not from earlier runs.

## 1. The key source correction

The important source package for `v4.3` was the selector-bench package:

- `rt_muon_v4_3_selectorbench.zip`

The important conclusion from that package was:

- `v4.3` is not a new penalty term;
- `v4.3` is a selector family;
- the rule is:
  - hard-quench path for small learning rate;
  - soft-quench path for larger learning rate;
  - switch threshold `0.05`.

So the accepted interpretation is:

- small-LR regime: `v4.1`-style hard quench
- larger-LR regime: `v4.2`-style soft quench
- `v4.3` chooses between them

## 2. Why older HamGNN-side results were not trusted

Before repair, the active remote HamGNN adapter was incomplete.

The main problems were:

- the wrapper did not export the expected environment knobs for the repaired `v4.3` logic;
- important fields such as:
  - `rt_lambda_consistency`
  - `rt_lambda_diff`
  - `rt_lambda_uncert`
  - `rt_beta_grid`
  were missing from the effective runtime path;
- the optimizer-side implementation used by those old runs was a simplified path and not aligned with the fuller selector-style logic we wanted to test.

Because of that, earlier "v43" HamGNN runs were not accepted as proof of correct `v4.3` behavior.

## 3. What was repaired

The repaired local adapter used for the accepted run was:

- `test1-hamgnn-si/code/hamgnn_rt/optimizers.py`
- `test1-hamgnn-si/code/hamgnn_rt/run_hamgnn_rt.py`

The repair established the following:

- the optimizer config contains the selector-related knobs:
  - `lr_switch`
  - `phi_threshold_hard`
  - `phi_threshold_soft`
  - `soft_tau`
  - `lambda_consistency`
  - `lambda_diff`
  - `lambda_uncert`
  - `lambda_tube`
- the runtime wrapper exports the matching environment variables;
- the HamGNN modes include:
  - `rt_v43_stream`
  - `rt_v43_ns`
- the `rt_v43_ns` path explicitly uses the Muon-style source branch;
- the selector behavior is implemented in the repaired `_rt_weights(...)` path.

## 4. Why job 73239 is the trusted benchmark

Job `73239` is accepted because it satisfies both the code-path requirement and the result-completeness requirement.

### Code-path requirement

During the run, live optimizer diagnostics from the `v4.3` methods showed the expected selector-era fields, including:

- `beta_candidate`
- `soft_shrink`
- `phi`
- `raw_cv_gain`
- `diffusion_cost`
- `uncertainty_cost`
- `tube_penalty`

That matters because it proves the repaired path was really active during execution.

### Result-completeness requirement

Job `73239` completed the whole packed sweep:

- `adamw`
- `muon_ns`
- `rt_v43_stream_a`
- `rt_v43_stream_b`
- `rt_v43_stream_c`
- `rt_v43_ns_a`
- `rt_v43_ns_b`
- `rt_v43_ns_c`

The launcher log is preserved at:

- `test1-hamgnn-si/benchmark/logs/hgnn_v43_sw_73239.out`

The per-method logs and diagnostics are preserved under:

- `test1-hamgnn-si/benchmark/diag/`

## 5. Final accepted result from 73239

The accepted ranking by `test/total_loss` is:

1. `rt_v43_ns_b` - `0.0004526643315330148`
2. `rt_v43_ns_a` - `0.00045320147182792425`
3. `adamw` - `0.0004533186147455126`
4. `rt_v43_ns_c` - `0.00045364003744907677`
5. `rt_v43_stream_a` - `0.00045369635336101055`
6. `rt_v43_stream_c` - `0.0004541226080618799`
7. `rt_v43_stream_b` - `0.00045423145638778806`
8. `muon_ns` - `0.00045428014709614217`

The best run is therefore:

- `rt_v43_ns_b`

and it beats both:

- `adamw`
- `muon_ns`

## 6. Why we do not collapse this into a PINN claim

The completed accepted benchmark is a HamGNN silicon Hamiltonian benchmark.

It is **not**:

- a Burgers PINN result,
- a DeepONet result,
- a PDE benchmark result.

That is why `test2-pinn` is kept separate.

## 7. Practical takeaway

The accepted claim is narrow and defensible:

- after repair, the corrected HamGNN `v4.3` path was run in a complete silicon packed sweep;
- in that accepted sweep, the best method was `rt_v43_ns_b`;
- earlier "v43" HamGNN results are not used as evidence because the runtime path was not yet correctly wired.
