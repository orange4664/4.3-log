# SQ-RT-Muon v4.2 Note

v4.1 reached a useful state: AUC was consistently better than Muon and positive-p HTMuon-like baselines, while final loss was strictly better in fullrank/spiked-noise and tied in lowrank/mixed.  The lowrank/mixed final-loss tie was not a true nonzero improvement: the selected SQ-RT configuration fully quenched back to Muon (`beta=0`).

v4.2 tests a minimal mechanism for strict improvement:

\[
\beta_{\rm eff}=\beta\,\sigma\left(\frac{\Phi(\beta)-\theta}{\tau}\right),
\]

where \(\Phi\) is the same cross-replica lower-bound free energy used by SQ-RT.  If \(\Phi\) is strongly positive, the update behaves like RT; if \(\Phi\) is marginal, the update becomes a tiny near-Muon perturbation.  This is intended to avoid the hard switch between exact Muon and finite spectral symmetry breaking.

The diffusion cost is also rescaled:

\[
D(\beta)
=\eta\,{\sum_i \nu_i (w_i^2-1)^2 \over \sum_i \nu_i+\epsilon},
\]

instead of the earlier \(\eta^2\operatorname{mean}_i[\nu_i(w_i^2-1)^2]\), which was usually too small to affect selection.

The v4.2 experiment compares two SQ families:

- `sq_hard_v41`: hard-quench v4.1-style reference.
- `sq_soft_v42`: soft-quench plus normalized diffusion.

The target result is not merely better AUC.  The target is strict or near-strict final-loss improvement over Muon in lowrank/mixed, while retaining the large-LR AUC/robustness advantage from v1-v4.1.
