# test2-pinn status

This folder is now the PDE/PINN-side follow-up track for the repaired `v4.3`
line.

It remains intentionally separate from `test1-hamgnn-si`:

- `test1-hamgnn-si` is the accepted HamGNN silicon Hamiltonian benchmark.
- `test2-pinn` is the PDE/PINN follow-up track.

## What is included now

### Real PDE/PINN benchmark

- `code/pinn_heat/`
  - real 1D heat-equation PINN benchmark
  - compares:
    - `adamw`
    - `muon_ns_adamw`
    - `rt_v43_stream_adamw`
    - `rt_v43_ns_adamw`
- `results/heat1d_pinn_20260519/`
  - completed real PDE/PINN benchmark outputs
- `logs/test2_pinn_v43_73302.out`
- `logs/test2_pinn_v43_73302.err`
  - server-side Slurm evidence for the completed run

The accepted completed PDE run in this folder is:

- server: `nmcc-n46h1` (`ln02`)
- work tree: `~/run/4.3-log/test2-pinn`
- submit method: `sbatch`
- job id: `73302`
- queue: `gpu_a800`
- python environment: `~/run/envs/hamgnn`

### Legacy selector reference material

- `code/README.md`
  - upstream selector-bench README
- `code/requirements.txt`
  - selector-bench package requirements
- `code/experiments/`
  - `run_sq_rt_v41_bench.py`
  - `run_sq_rt_v42_bench.py`
  - `run_sq_rt_v43_selector_bench.py`
- `code/rtmuon/`
  - selector/toy benchmark implementation
- `code/scripts/`
  - selector-bench helper scripts
- `results/smoke_v43/`
  - selector-bench smoke outputs
- `results/full_budget_matched_20260519/`
  - completed selector/toy full benchmark

## What is claimed here

Two different claims are kept separate:

1. `test1-hamgnn-si` shows the accepted repaired HamGNN silicon result.
2. `test2-pinn` now includes a real PDE/PINN benchmark that reuses the repaired
   torch `v4.3` implementation and was completed on the server through Slurm job
   `73302`.

## What is not claimed here

This folder still does **not** claim any of the following:

- a Burgers PINN result unless the run explicitly says so,
- a DeepONet result,
- a PDEBench or PDEArena result,
- or that the selector/toy benchmark itself is a PDE benchmark.

## Why this folder exists

The work sequence is now:

1. repair the HamGNN `v4.3` path;
2. accept only the completed repaired HamGNN sweep `73239`;
3. keep the selector/toy benchmark as optimizer-family reference;
4. add a real PDE/PINN benchmark in a separate follow-up track.
