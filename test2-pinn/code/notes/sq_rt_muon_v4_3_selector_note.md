# SQ-RT-Muon v4.3: hard/soft thermostat selector

## Motivation

v4.1 hard-quench SQ-RT is the current safe main version.  It wins Muon on AUC, wins or ties Muon on exact final loss, and fully restores the Muon state when nonzero spectral symmetry breaking is not justified.

v4.2 soft-quench SQ-RT is faster.  It improves AUC across all toy settings but leaves small residual spectral breaking in lowrank/mixed final-loss regimes, which slightly hurts exact final loss.

The observation from v1--v4.2 is therefore not that one thermostat dominates the other, but that the useful regime differs:

\[
\text{small lr / final-quality regime} \Rightarrow \text{hard quench},
\]

\[
\text{medium-large lr / speed regime} \Rightarrow \text{soft quench}.
\]

v4.3 turns this into an explicit selector family rather than adding another penalty term.

## Algorithm

Let `lr_switch` be a scalar threshold.  For each matrix block and training step:

\[
Q_{v4.3} =
\begin{cases}
Q_{\rm hard}, & \eta < \eta_{\rm switch},\\
Q_{\rm soft}, & \eta \ge \eta_{\rm switch}.
\end{cases}
\]

Here \(Q_{\rm hard}\) is v4.1 hard-quench SQ-RT and \(Q_{\rm soft}\) is v4.2 soft-quench SQ-RT.  Both remain in the Muon spectral basis and both preserve Muon Frobenius scale.

The default is:

\[
\eta_{\rm switch}=0.05.
\]

This threshold is not a claim of universality.  It is a toy-level test of the learning-rate regime phenomenon.  If it works, the next version should replace this scalar threshold with a statistic such as normalized cross-replica free-energy gain or expected diffusion ratio.

## Why this is not a new theory leap

v4.3 is intentionally conservative.  Its purpose is to test a clear hypothesis before changing theory again:

\[
\text{Can hard-quench and soft-quench be combined to dominate Muon on AUC, final loss, and LR robustness?}
\]

If v4.3 succeeds, the physics interpretation is:

- hard quench = symmetry restoration / final-quality phase;
- soft quench = weakly broken symmetry / fast exploration phase;
- selector = a crude thermostat phase rule.

If it fails, we should not keep adding knobs; we should return to the self-quench free energy and derive a better continuous order parameter.

## Scoring

The benchmark reports:

- best-over-LR AUC and final loss;
- positive-p HTMuon-like comparison;
- Muon-quality constrained AUC at epsilon = 0, 0.5%, 1%, 2%;
- per-LR regime scorecard;
- selector mode diagnostics.

The desired outcome is not only large-LR acceleration.  The stronger target is:

\[
\operatorname{AUC}(\text{selector}) < \operatorname{AUC}(\text{Muon})
\]

and

\[
L_T(\text{selector}) \le L_T(\text{Muon})
\]

in all four toy settings under matched tuning budget.
