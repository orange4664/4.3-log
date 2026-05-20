"""Exact-SVD matrix update rules for Muon-family toy experiments.

The new v4 method is SQ-RT-Muon: Self-Quenched Replica-Thermostatted Muon.
It keeps the Muon spectral basis U,V and Frobenius scale, but selects the
spectral-symmetry-breaking temperature beta by a cross-replica lower-bound
free energy:

    phi(beta) = crossfit_gain
                - lambda_diff * diffusion_cost
                - lambda_uncert * replica_uncertainty
                - lambda_tube * tube_penalty.

Because beta=0 is always included, the method restores exact Muon whenever
non-uniform spectral weights are not justified by replica-stable gain.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .linalg_utils import (
    EPS,
    compact_svd,
    fro_norm,
    rms_align,
    robust_center_scale,
    spectral_entropy_from_weights,
    stable_softmax,
)


@dataclass
class UpdateInfo:
    method: str
    entropy: float
    pmax: float
    update_fro: float
    beta_star: float = 0.0        # effective beta used to make the final update
    beta_candidate: float = 0.0   # candidate before soft-shrink
    soft_shrink: float = 0.0      # shrink factor in [0,1]
    cv_gain: float = 0.0          # for rt_muon: raw gain; for sq_rt: selected phi
    raw_cv_gain: float = 0.0      # selected raw cross-fit gain before penalties
    diffusion_cost: float = 0.0
    uncertainty_cost: float = 0.0
    tube_penalty: float = 0.0
    phi: float = 0.0
    selected: str = ""
    mean_abs_w_minus_1: float = 0.0
    min_w: float = 1.0
    max_w: float = 1.0

    def asdict(self) -> Dict[str, float | str]:
        return asdict(self)


def _info(method: str, w: np.ndarray, q: np.ndarray, **kwargs) -> UpdateInfo:
    w = np.asarray(w, dtype=float)
    w2 = w * w
    p = w2 / (np.sum(w2) + EPS)
    return UpdateInfo(
        method=method,
        entropy=spectral_entropy_from_weights(w),
        pmax=float(np.max(p)) if p.size else 0.0,
        update_fro=fro_norm(q),
        mean_abs_w_minus_1=float(np.mean(np.abs(w - 1.0))) if w.size else 0.0,
        min_w=float(np.min(w)) if w.size else 1.0,
        max_w=float(np.max(w)) if w.size else 1.0,
        **kwargs,
    )


def _diag_projections(u: np.ndarray, vt: np.ndarray, g: np.ndarray) -> np.ndarray:
    """diag(U^T G V), where V = Vt.T."""
    return np.diag(u.T @ g @ vt.T)


def normalized_momentum_update(m: np.ndarray, target_fro: float) -> Tuple[np.ndarray, UpdateInfo]:
    q = rms_align(m, target_fro)
    _, s, _ = compact_svd(q)
    return q, _info("sgdm", s, q)


def muon_update(m: np.ndarray, target_fro: float) -> Tuple[np.ndarray, UpdateInfo]:
    u, _, vt = compact_svd(m)
    r = min(m.shape)
    w = np.ones(r)
    q = rms_align(u @ vt, target_fro)
    return q, _info("muon", w, q)


def ht_power_update(m: np.ndarray, target_fro: float, power: float = 0.125) -> Tuple[np.ndarray, UpdateInfo]:
    """Exact-SVD HTMuon-like baseline: U diag((sigma+eps)^p) V^T, RMS-aligned."""
    u, s, vt = compact_svd(m)
    w0 = (s + EPS) ** float(power)
    q0 = u @ np.diag(w0) @ vt
    q = rms_align(q0, target_fro)
    scale = target_fro / (fro_norm(q0) + EPS) if fro_norm(q0) > EPS else 0.0
    return q, _info(f"htmuon_p{power:g}", w0 * scale, q)


def mclip_update(m: np.ndarray, target_fro: float, tau_ratio: float = 0.25) -> Tuple[np.ndarray, UpdateInfo]:
    u, s, vt = compact_svd(m)
    tau = max(float(tau_ratio) * float(np.max(s)), EPS)
    w0 = np.minimum(s / tau, 1.0)
    q0 = u @ np.diag(w0) @ vt
    q = rms_align(q0, target_fro)
    scale = target_fro / (fro_norm(q0) + EPS) if fro_norm(q0) > EPS else 0.0
    return q, _info(f"mclip_{tau_ratio:g}", w0 * scale, q)


def sfem_eps4_update(
    m: np.ndarray,
    g_a: np.ndarray,
    g_b: np.ndarray,
    target_fro: float,
    kappa_diag: Optional[np.ndarray] = None,
    temp: float = 0.6,
    lambda_unc: float = 4.0,
    lambda_flip: float = 0.5,
    lambda_util: float = 1.5,
    ht_prior_power: float = 0.125,
) -> Tuple[np.ndarray, UpdateInfo]:
    """Previous Gibbs-energy baseline, retained only as an old baseline."""
    u, s, vt = compact_svd(m)
    r = len(s)
    g1 = _diag_projections(u, vt, g_a)
    g2 = _diag_projections(u, vt, g_b)
    gbar = 0.5 * (g1 + g2)
    nu = 0.25 * (g1 - g2) ** 2
    c = 2.0 * g1 * g2 / (g1 * g1 + g2 * g2 + EPS)
    kappa = np.ones(r) if kappa_diag is None else np.asarray(kappa_diag[:r], dtype=float)
    R = (gbar * gbar + EPS) / (gbar * gbar + lambda_unc * nu + EPS)
    energy = -np.log(R + EPS) + lambda_flip * (1.0 - c) - lambda_util * np.log1p((gbar * gbar) / (kappa + EPS))
    prior = (s * s + EPS) ** float(ht_prior_power)
    logits = np.log(prior + EPS) - energy / max(temp, EPS)
    p = stable_softmax(logits, 1.0)
    w = np.sqrt(r * p)
    q0 = u @ np.diag(w) @ vt
    q = rms_align(q0, target_fro)
    scale = target_fro / (fro_norm(q0) + EPS) if fro_norm(q0) > EPS else 0.0
    return q, _info("sfem_eps4_ht", w * scale, q)


def _tube_project(w: np.ndarray, delta: float, target_l2: float, n_iter: int = 4) -> np.ndarray:
    """Approximate projection into a Muon tube and L2 sphere."""
    w = np.asarray(w, dtype=float).copy()
    if delta is not None and np.isfinite(delta):
        lo, hi = max(0.0, 1.0 - float(delta)), 1.0 + float(delta)
    else:
        lo, hi = 0.0, np.inf
    for _ in range(max(1, n_iter)):
        w = np.clip(w, lo, hi)
        nrm = float(np.linalg.norm(w))
        if nrm < EPS:
            w = np.ones_like(w)
            nrm = float(np.linalg.norm(w))
        w = w * (target_l2 / (nrm + EPS))
    w = np.clip(w, lo, hi)
    nrm = float(np.linalg.norm(w))
    if nrm > EPS:
        w = w * (target_l2 / (nrm + EPS))
    return w


def _weights_from_advantage(a: np.ndarray, beta: float, delta: float) -> np.ndarray:
    r = len(a)
    if r == 0 or beta <= 0.0:
        return np.ones(r)
    z = robust_center_scale(a)
    p = stable_softmax(z, temperature_inverse=float(beta))
    w = np.sqrt(r * p)
    return _tube_project(w, delta=delta, target_l2=np.sqrt(r))


def _surrogate(g: np.ndarray, B: np.ndarray, w: np.ndarray, eta: float) -> float:
    return float(eta * np.sum(g * w) - 0.5 * eta * eta * np.sum(B * w * w))


def _rt_basis_terms(
    m: np.ndarray,
    g_a: np.ndarray,
    g_b: np.ndarray,
    eta: float,
    lambda_nu: float,
    lambda_c: float,
    kappa_diag: Optional[np.ndarray],
    no_noise: bool = False,
    no_consistency: bool = False,
):
    u, s, vt = compact_svd(m)
    r = len(s)
    g1 = _diag_projections(u, vt, g_a)
    g2 = _diag_projections(u, vt, g_b)
    nu = 0.25 * (g1 - g2) ** 2
    c = 2.0 * g1 * g2 / (g1 * g1 + g2 * g2 + EPS)
    kappa = np.ones(r) if kappa_diag is None else np.asarray(kappa_diag[:r], dtype=float)
    B = kappa.copy()
    if not no_noise:
        B = B + float(lambda_nu) * nu
    if not no_consistency:
        B = B + float(lambda_c) * (1.0 - c)
    B = np.maximum(B, EPS)
    a1 = eta * g1 - eta * eta * B
    a2 = eta * g2 - eta * eta * B
    return u, s, vt, g1, g2, nu, c, B, a1, a2


def rt_muon_update(
    m: np.ndarray,
    g_a: np.ndarray,
    g_b: np.ndarray,
    target_fro: float,
    eta: float,
    beta_grid: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
    delta: float = 0.35,
    lambda_nu: float = 2.0,
    lambda_c: float = 0.25,
    kappa_diag: Optional[np.ndarray] = None,
    cross_fit: bool = True,
    fixed_beta: Optional[float] = None,
    no_noise: bool = False,
    no_consistency: bool = False,
    no_tube: bool = False,
    method_name: str = "rt_muon",
) -> Tuple[np.ndarray, UpdateInfo]:
    """v3 Replica-Thermostatted Muon: select beta by positive cross-fit gain."""
    u, _, vt, g1, g2, _, _, B, a1, a2 = _rt_basis_terms(
        m, g_a, g_b, eta, lambda_nu, lambda_c, kappa_diag, no_noise, no_consistency
    )
    r = len(g1)
    if r == 0:
        q = np.zeros_like(m)
        return q, _info(method_name, np.zeros(0), q)
    ones = np.ones(r)
    tube_delta = 10.0 if no_tube else float(delta)

    candidates = [float(b) for b in beta_grid]
    if fixed_beta is not None:
        candidates = [float(fixed_beta)]

    best_beta = 0.0
    best_gain = 0.0
    if fixed_beta is not None:
        best_beta = float(fixed_beta)
        best_gain = float("nan")
    else:
        for beta in candidates:
            if beta == 0.0:
                gain = 0.0
            else:
                w1 = _weights_from_advantage(a1, beta, tube_delta)
                w2 = _weights_from_advantage(a2, beta, tube_delta)
                if cross_fit:
                    gain_2 = _surrogate(g2, B, w1, eta) - _surrogate(g2, B, ones, eta)
                    gain_1 = _surrogate(g1, B, w2, eta) - _surrogate(g1, B, ones, eta)
                    gain = 0.5 * (gain_1 + gain_2)
                else:
                    gbar = 0.5 * (g1 + g2)
                    gain = _surrogate(gbar, B, w1, eta) - _surrogate(gbar, B, ones, eta)
            if gain > best_gain + 1e-15:
                best_gain = float(gain)
                best_beta = float(beta)

    a = 0.5 * (a1 + a2)
    w = _weights_from_advantage(a, best_beta, tube_delta)
    q0 = u @ np.diag(w) @ vt
    q = rms_align(q0, target_fro)
    return q, _info(
        method_name, w, q, beta_star=float(best_beta), cv_gain=float(best_gain) if np.isfinite(best_gain) else float("nan"),
        raw_cv_gain=float(best_gain) if np.isfinite(best_gain) else float("nan"), phi=float(best_gain) if np.isfinite(best_gain) else float("nan"),
        selected=f"beta={best_beta:g}",
    )



def _sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0:
        z = np.exp(-min(x, 60.0))
        return float(1.0 / (1.0 + z))
    z = np.exp(min(x, 60.0))
    return float(z / (1.0 + z))


def sq_rt_muon_update(
    m: np.ndarray,
    g_a: np.ndarray,
    g_b: np.ndarray,
    target_fro: float,
    eta: float,
    beta_grid: Sequence[float] = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
    delta: float = 0.35,
    lambda_nu: float = 2.0,
    lambda_c: float = 0.25,
    lambda_diff: float = 0.5,
    lambda_uncert: float = 0.5,
    lambda_tube: float = 0.05,
    phi_threshold: float = 0.0,
    soft_tau: float = 0.001,
    quench_mode: str = "hard",
    diffusion_mode: str = "mean",
    kappa_diag: Optional[np.ndarray] = None,
    no_diffusion_penalty: bool = False,
    no_uncertainty_penalty: bool = False,
    no_tube_penalty: bool = False,
    no_quench: bool = False,
    no_noise: bool = False,
    no_consistency: bool = False,
    no_tube: bool = False,
    method_name: str = "sq_rt_muon",
) -> Tuple[np.ndarray, UpdateInfo]:
    """Self-Quenched RT-Muon v4/v4.2.

    v4.2 adds soft-quench and normalized diffusion.  Soft-quench replaces the
    hard jump beta -> 0 by beta_eff = beta * sigmoid((phi-threshold)/tau).
    Normalized diffusion uses eta * sum(nu * excess) / sum(nu), instead of the
    older eta^2 * mean(nu * excess), whose scale was often negligible.
    """
    u, _, vt, g1, g2, nu, _, B, a1, a2 = _rt_basis_terms(
        m, g_a, g_b, eta, lambda_nu, lambda_c, kappa_diag, no_noise, no_consistency
    )
    r = len(g1)
    if r == 0:
        q = np.zeros_like(m)
        return q, _info(method_name, np.zeros(0), q)
    ones = np.ones(r)
    tube_delta = 10.0 if no_tube else float(delta)
    qmode = str(quench_mode).lower().strip()
    dmode = str(diffusion_mode).lower().strip()
    if qmode not in {"hard", "soft"}:
        raise ValueError(f"Unknown quench_mode={quench_mode!r}")
    if dmode not in {"mean", "normalized"}:
        raise ValueError(f"Unknown diffusion_mode={diffusion_mode!r}")

    def metrics(beta: float):
        beta = float(beta)
        if beta <= 0.0:
            return {"phi":0.0,"gain":0.0,"diff":0.0,"uncert":0.0,"tube":0.0}
        w1 = _weights_from_advantage(a1, beta, tube_delta)
        w2 = _weights_from_advantage(a2, beta, tube_delta)
        gain_2 = _surrogate(g2, B, w1, eta) - _surrogate(g2, B, ones, eta)
        gain_1 = _surrogate(g1, B, w2, eta) - _surrogate(g1, B, ones, eta)
        gain = 0.5 * (gain_1 + gain_2)
        ex1 = (w1*w1 - 1.0)**2
        ex2 = (w2*w2 - 1.0)**2
        if dmode == "normalized":
            diff_raw = 0.5*(float(np.sum(nu*ex1)/(np.sum(nu)+EPS)) + float(np.sum(nu*ex2)/(np.sum(nu)+EPS)))
            diff = eta * diff_raw
        else:
            diff_raw = 0.5*(float(np.mean(nu*ex1)) + float(np.mean(nu*ex2)))
            diff = eta*eta*diff_raw
        if no_diffusion_penalty:
            diff = 0.0
        uncert = abs(float(gain_1 - gain_2))
        if no_uncertainty_penalty:
            uncert = 0.0
        tube_raw = 0.5*(float(np.mean((w1-1.0)**2)) + float(np.mean((w2-1.0)**2)))
        tube = eta * tube_raw
        if no_tube_penalty:
            tube = 0.0
        phi = gain - float(lambda_diff)*diff - float(lambda_uncert)*uncert - float(lambda_tube)*tube
        return {"phi":float(phi),"gain":float(gain),"diff":float(diff),"uncert":float(uncert),"tube":float(tube)}

    best = {"beta":0.0,"candidate":0.0,"shrink":0.0,"phi":0.0,"gain":0.0,"diff":0.0,"uncert":0.0,"tube":0.0,"raw_phi":0.0}
    for b0 in [float(b) for b in beta_grid]:
        raw = metrics(b0)
        if qmode == "soft" and b0 > 0.0 and not no_quench:
            if raw["phi"] <= 0.0:
                beff, shrink, met = 0.0, 0.0, metrics(0.0)
            else:
                shrink = _sigmoid((raw["phi"] - float(phi_threshold)) / (float(soft_tau) + EPS))
                beff = b0 * shrink
                met = metrics(beff)
        else:
            beff, shrink, met = b0, (1.0 if b0 > 0 else 0.0), raw
        if met["phi"] > best["phi"] + 1e-15:
            best.update(beta=beff,candidate=b0,shrink=shrink,phi=met["phi"],gain=met["gain"],diff=met["diff"],uncert=met["uncert"],tube=met["tube"],raw_phi=raw["phi"])

    if not no_quench:
        if qmode == "hard" and best["phi"] <= float(phi_threshold):
            best.update(beta=0.0,candidate=0.0,shrink=0.0,phi=0.0,gain=0.0,diff=0.0,uncert=0.0,tube=0.0,raw_phi=0.0)
        elif qmode == "soft" and best["phi"] <= 0.0:
            best.update(beta=0.0,candidate=0.0,shrink=0.0,phi=0.0,gain=0.0,diff=0.0,uncert=0.0,tube=0.0,raw_phi=0.0)

    a = 0.5*(a1+a2)
    w = _weights_from_advantage(a, best["beta"], tube_delta)
    q0 = u @ np.diag(w) @ vt
    q = rms_align(q0, target_fro)
    return q, _info(
        method_name, w, q,
        beta_star=float(best["beta"]), beta_candidate=float(best["candidate"]), soft_shrink=float(best["shrink"]),
        cv_gain=float(best["phi"]), raw_cv_gain=float(best["gain"]), diffusion_cost=float(best["diff"]),
        uncertainty_cost=float(best["uncert"]), tube_penalty=float(best["tube"]), phi=float(best["phi"]),
        selected=(f"mode={qmode};diff={dmode};beta={best['beta']:g};cand={best['candidate']:g};"
                  f"shrink={best['shrink']:.3f};phi={best['phi']:.3e};rawphi={best['raw_phi']:.3e}")
    )


def make_update(
    method: str,
    m: np.ndarray,
    g_a: np.ndarray,
    g_b: np.ndarray,
    target_fro: float,
    eta: float,
    kappa_diag: Optional[np.ndarray] = None,
    beta_grid: Sequence[float] = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
    rt_delta: float = 0.35,
    rt_lambda_nu: float = 2.0,
    rt_lambda_c: float = 0.25,
    sq_lambda_diff: float = 0.5,
    sq_lambda_uncert: float = 0.5,
    sq_lambda_tube: float = 0.05,
    sq_phi_threshold: float = 0.0,
    sq_soft_tau: float = 0.001,
    sq_quench_mode: str = "hard",
    sq_diffusion_mode: str = "mean",
) -> Tuple[np.ndarray, UpdateInfo]:
    if method == "sgdm":
        return normalized_momentum_update(m, target_fro)
    if method == "muon":
        return muon_update(m, target_fro)
    if method.startswith("ht_power") or method.startswith("htmuon"):
        power = 0.125
        token = method.split("_")[-1]
        if token.startswith("p"):
            token = token[1:]
        try:
            power = float(token)
        except ValueError:
            pass
        return ht_power_update(m, target_fro, power=power)
    if method.startswith("mclip"):
        ratio = 0.25
        try:
            ratio = float(method.split("_")[-1])
        except Exception:
            pass
        return mclip_update(m, target_fro, tau_ratio=ratio)
    if method == "sfem_eps4_ht":
        return sfem_eps4_update(m, g_a, g_b, target_fro, kappa_diag=kappa_diag)
    if method == "rt_muon":
        return rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, kappa_diag, method_name="rt_muon")
    if method == "rt_no_cv":
        return rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, kappa_diag, cross_fit=False, method_name="rt_no_cv")
    if method == "rt_no_noise":
        return rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, kappa_diag, no_noise=True, method_name="rt_no_noise")
    if method == "rt_no_consistency":
        return rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, kappa_diag, no_consistency=True, method_name="rt_no_consistency")
    if method == "rt_fixed_beta1":
        return rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, kappa_diag, fixed_beta=1.0, method_name="rt_fixed_beta1")
    if method == "rt_no_tube":
        return rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, kappa_diag, no_tube=True, method_name="rt_no_tube")
    if method in {"sq_rt_muon", "sq_hard_quench", "sq_rt_soft"}:
        qmode = "soft" if method == "sq_rt_soft" else sq_quench_mode
        if method == "sq_hard_quench":
            qmode = "hard"
        return sq_rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, sq_lambda_diff, sq_lambda_uncert, sq_lambda_tube, sq_phi_threshold, sq_soft_tau, qmode, sq_diffusion_mode, kappa_diag, method_name=method)
    if method == "sq_no_diffusion":
        return sq_rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, sq_lambda_diff, sq_lambda_uncert, sq_lambda_tube, sq_phi_threshold, sq_soft_tau, sq_quench_mode, sq_diffusion_mode, kappa_diag, no_diffusion_penalty=True, method_name="sq_no_diffusion")
    if method == "sq_no_uncertainty":
        return sq_rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, sq_lambda_diff, sq_lambda_uncert, sq_lambda_tube, sq_phi_threshold, sq_soft_tau, sq_quench_mode, sq_diffusion_mode, kappa_diag, no_uncertainty_penalty=True, method_name="sq_no_uncertainty")
    if method == "sq_no_tube_penalty":
        return sq_rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, sq_lambda_diff, sq_lambda_uncert, sq_lambda_tube, sq_phi_threshold, sq_soft_tau, sq_quench_mode, sq_diffusion_mode, kappa_diag, no_tube_penalty=True, method_name="sq_no_tube_penalty")
    if method == "sq_no_quench":
        return sq_rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, sq_lambda_diff, sq_lambda_uncert, sq_lambda_tube, sq_phi_threshold, sq_soft_tau, sq_quench_mode, sq_diffusion_mode, kappa_diag, no_quench=True, method_name="sq_no_quench")
    if method == "sq_no_noise":
        return sq_rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, sq_lambda_diff, sq_lambda_uncert, sq_lambda_tube, sq_phi_threshold, sq_soft_tau, sq_quench_mode, sq_diffusion_mode, kappa_diag, no_noise=True, method_name="sq_no_noise")
    if method == "sq_no_consistency":
        return sq_rt_muon_update(m, g_a, g_b, target_fro, eta, beta_grid, rt_delta, rt_lambda_nu, rt_lambda_c, sq_lambda_diff, sq_lambda_uncert, sq_lambda_tube, sq_phi_threshold, sq_soft_tau, sq_quench_mode, sq_diffusion_mode, kappa_diag, no_consistency=True, method_name="sq_no_consistency")
    raise ValueError(f"Unknown method: {method}")
