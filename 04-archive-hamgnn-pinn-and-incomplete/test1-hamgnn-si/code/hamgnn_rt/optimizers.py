"""RT/FDT-Metric optimizers for HamGNN.

This module is intentionally optimizer-only: it works with PyTorch Lightning's
automatic optimization and does not require modifying HamGNN's training step.
For HamGNN's common batch_size=1 setting, replica/FDT quantities are estimated
from temporal gradient fluctuations rather than two explicit microbatch passes.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer


def _as_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _as_int(x: Any, default: int) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def _rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x.float().pow(2).mean().sqrt().to(dtype=x.dtype).clamp_min(eps)


def _safe_col_norm(y: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    s = torch.linalg.vector_norm(y, dim=0).clamp_min(eps)
    return y / s.unsqueeze(0), s


def _center_scale(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float()
    med = x.mean()
    xc = x - med
    scale = xc.pow(2).mean().sqrt().clamp_min(eps)
    return (xc / scale).to(dtype=x.dtype)


def _geomean_normalize_temperature(t: torch.Tensor, tmin: float, tmax: float, eps: float) -> torch.Tensor:
    t = torch.nan_to_num(t.float(), nan=1.0, posinf=tmax, neginf=tmin).clamp_min(eps)
    log_g = t.log().mean()
    t = (t / log_g.exp()).clamp(tmin, tmax)
    # clipping changes geomean; recenter once more, then clip again
    t = (t / t.log().mean().exp()).clamp(tmin, tmax)
    return t


def _parse_float_tuple(value: Any, default: Tuple[float, ...]) -> Tuple[float, ...]:
    if value is None:
        return default
    if isinstance(value, (tuple, list)):
        try:
            return tuple(float(x) for x in value)
        except Exception:
            return default
    try:
        out = tuple(float(x.strip()) for x in str(value).split(",") if x.strip())
        return out or default
    except Exception:
        return default


def _tube_project(w: torch.Tensor, delta: float, target_l2: float, eps: float, n_iter: int = 4) -> torch.Tensor:
    lo = max(0.0, 1.0 - float(delta)) if math.isfinite(float(delta)) else 0.0
    hi = 1.0 + float(delta) if math.isfinite(float(delta)) else float("inf")
    w = w.float()
    for _ in range(max(1, n_iter)):
        w = w.clamp(lo, hi)
        w = w * (float(target_l2) / w.norm().clamp_min(eps))
    w = w.clamp(lo, hi)
    return w * (float(target_l2) / w.norm().clamp_min(eps))


def _weights_from_advantage(a: torch.Tensor, beta: float, delta: float, eps: float) -> torch.Tensor:
    r = a.numel()
    if r == 0 or beta <= 0.0:
        return torch.ones_like(a)
    z = _center_scale(a, eps)
    prob = torch.softmax(float(beta) * z, dim=0)
    w = torch.sqrt(prob * float(r))
    return _tube_project(w, delta=delta, target_l2=math.sqrt(float(r)), eps=eps).to(dtype=a.dtype)


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the polar factor used by Muon with quintic Newton-Schulz steps."""
    orig_dtype = g.dtype
    x = g.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.t()
    x = x / x.norm().clamp_min(eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(1, steps)):
        xx_t = x @ x.t()
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    if transposed:
        x = x.t()
    return x.to(dtype=orig_dtype)


@dataclass
class RTConfig:
    mode: str = "adamw"
    beta: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    stream_k: int = 0
    power_iters: int = 1
    beta_grid: Tuple[float, ...] = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    rt_period: int = 1
    lr_switch: float = 0.05
    hard_delta: float = 0.35
    soft_delta: float = 0.35
    lambda_noise: float = 2.0
    lambda_consistency: float = 0.25
    lambda_diff: float = 0.5
    lambda_uncert: float = 0.5
    lambda_tube: float = 0.05
    phi_threshold_hard: float = 0.0
    phi_threshold_soft: float = 0.0
    soft_tau: float = 0.01
    metric_alpha_row: float = 0.25
    metric_alpha_col: float = 0.25
    metric_lambda_noise: float = 1.0
    metric_tmin: float = 0.5
    metric_tmax: float = 2.0
    metric_ema_beta: float = 0.95
    ht_power: float = 0.125
    mclip_delta: float = 0.25
    ns_steps: int = 5
    nesterov: bool = True
    fallback: str = "adamw"  # adamw or sgd for non-matrix params
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    log_interval: int = 50
    diag_dir: str = ""
    name: str = "hamgnn_rt"


class HamGNNMuonFDTOptimizer(Optimizer):
    """Streaming-power Muon/RT/FDT-Metric optimizer.

    Supported modes:
      - adamw: AdamW fallback, useful as control.
      - muon_ns: Muon with Newton-Schulz polar orthogonalization.
      - muon_stream: ScienceSpace-style streaming-power polar update.
      - muon_stream_qqq: streaming-power Muon with target-RMS Q normalization.
      - ht_stream: streaming HT-like U Sigma^p V^T update.
      - mclip_stream: streaming power + clipped spectral weights.
      - rt_v43_stream: temporal-FDT RT spectral bath in Euclidean metric.
      - rt_v43_ns: v4.3 selector on a Muon-style Nesterov momentum source.
      - muon_stream_fdt_metric: FDT-metric streaming Muon.
      - rt_v6_fdt_metric: FDT-metric streaming RT.

    The optimizer only uses gradients exposed by Lightning automatic
    optimization. For HamGNN's typical batch_size=1, stochastic information is
    estimated from temporal gradient innovations.
    """

    def __init__(self, params: Iterable[Any], lr: float = 1e-3, **kwargs: Any) -> None:
        cfg = RTConfig(**{k: v for k, v in kwargs.items() if hasattr(RTConfig, k)})
        cfg.mode = str(cfg.mode).lower().replace("-", "_")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.cfg = cfg
        self.global_step = 0
        self._diag_accum: Dict[str, float] = {}
        self._diag_count = 0
        if cfg.diag_dir:
            os.makedirs(cfg.diag_dir, exist_ok=True)
            self._diag_path = os.path.join(cfg.diag_dir, f"{cfg.name}_optimizer_diag.jsonl")
        else:
            self._diag_path = ""

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.global_step += 1
        cfg = self.cfg
        if cfg.mode == "adamw":
            self._adamw_step_only()
            return loss
        for group in self.param_groups:
            lr = float(group.get("lr", 0.0))
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("HamGNNMuonFDTOptimizer does not support sparse gradients.")
                if not torch.is_floating_point(g):
                    continue
                if p.ndim < 2 or min(p.shape[0], int(p.numel() // p.shape[0])) < 2:
                    self._adamw_param_step(p, g, lr)
                    continue
                if cfg.mode == "muon_ns":
                    q, diag = self._muon_ns_update(p, g)
                else:
                    q, diag = self._matrix_update(p, g, lr)
                if cfg.weight_decay:
                    p.mul_(1.0 - lr * cfg.weight_decay)
                p.add_(q.reshape_as(p), alpha=-lr)
                self._accumulate_diag(diag)
        if self._diag_path and cfg.log_interval > 0 and self.global_step % cfg.log_interval == 0:
            self._flush_diag()
        return loss

    def _muon_ns_update(self, p: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        st = self.state[p]
        n = int(p.shape[0])
        m = int(p.numel() // n)
        gmat = g.reshape(n, m)
        if "momentum" not in st or st["momentum"].shape != gmat.shape:
            st["momentum"] = torch.zeros_like(gmat)
        M = st["momentum"]
        M.mul_(cfg.beta).add_(gmat, alpha=1.0 - cfg.beta)
        update_source = gmat.lerp(M, cfg.beta) if cfg.nesterov else M
        q = _zeropower_via_newtonschulz5(update_source, steps=cfg.ns_steps, eps=cfg.eps)
        q = q * max(1.0, math.sqrt(float(n) / float(max(m, 1))))
        diag = {
            "beta": 0.0,
            "accept": 0.0,
            "abs_w_minus_1": 0.0,
            "phi": 0.0,
            "mode_is_metric": 0.0,
            "momentum_beta": float(cfg.beta),
            "nesterov": 1.0 if cfg.nesterov else 0.0,
            "update_rms": float(_rms(q, cfg.eps).item()),
            "matrix_rows": float(n),
            "matrix_cols": float(m),
            "ns_steps": float(cfg.ns_steps),
        }
        return q.reshape_as(p), diag

    def _adamw_step_only(self) -> None:
        for group in self.param_groups:
            lr = float(group.get("lr", 0.0))
            for p in group["params"]:
                if p.grad is None:
                    continue
                self._adamw_param_step(p, p.grad.detach(), lr)

    @torch.no_grad()
    def _adamw_param_step(self, p: torch.Tensor, g: torch.Tensor, lr: float) -> None:
        cfg = self.cfg
        st = self.state[p]
        if "adam_exp_avg" not in st:
            st["adam_exp_avg"] = torch.zeros_like(p)
            st["adam_exp_avg_sq"] = torch.zeros_like(p)
            st["adam_step"] = 0
        exp_avg = st["adam_exp_avg"]
        exp_avg_sq = st["adam_exp_avg_sq"]
        st["adam_step"] += 1
        beta1, beta2 = cfg.adam_beta1, cfg.adam_beta2
        exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        bias_c1 = 1.0 - beta1 ** st["adam_step"]
        bias_c2 = 1.0 - beta2 ** st["adam_step"]
        step_size = lr / bias_c1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(cfg.eps)
        if cfg.weight_decay:
            p.mul_(1.0 - lr * cfg.weight_decay)
        p.addcdiv_(exp_avg, denom, value=-step_size)

    def _matrix_update(self, p: torch.Tensor, g: torch.Tensor, lr: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        st = self.state[p]
        orig_shape = p.shape
        n = int(orig_shape[0])
        m = int(p.numel() // n)
        gmat = g.reshape(n, m)
        if "momentum" not in st or st["momentum"].shape != gmat.shape:
            st["momentum"] = torch.zeros_like(gmat)
            st["grad_ema"] = gmat.detach().clone()
            prev_grad = gmat.detach().clone()
        else:
            prev_grad = st["grad_ema"].detach().clone()
        M = st["momentum"]
        M.mul_(cfg.beta).add_(gmat, alpha=1.0 - cfg.beta)
        muon_source = gmat.lerp(M, cfg.beta) if cfg.nesterov else M

        # The previous gradient EMA is the best cheap replica proxy available
        # under Lightning automatic optimization.
        grad_ema_old = st["grad_ema"]
        innovation = gmat - prev_grad
        grad_ema_old.mul_(cfg.metric_ema_beta).add_(gmat, alpha=1.0 - cfg.metric_ema_beta)

        mode = cfg.mode
        use_metric = mode in {"muon_stream_fdt_metric", "rt_v6_fdt_metric"}
        use_muon_source = mode in {"muon_stream", "muon_stream_qqq", "muon_stream_fdt_metric", "rt_v43_ns"}
        base_matrix = muon_source if use_muon_source else M
        if use_metric:
            sqrt_tr, sqrt_tc, metric_diag = self._get_metric_temperatures(st, gmat, innovation, cfg)
            M_work = sqrt_tr[:, None] * base_matrix * sqrt_tc[None, :]
            G_work = sqrt_tr[:, None] * gmat * sqrt_tc[None, :]
            R_work = sqrt_tr[:, None] * prev_grad * sqrt_tc[None, :]
        else:
            sqrt_tr = sqrt_tc = None
            M_work = base_matrix
            G_work = gmat
            R_work = prev_grad
            metric_diag = {}

        U, sigma, V, transposed = self._streaming_basis(st, M_work, cfg)
        r = sigma.numel()
        active_rt = (cfg.rt_period <= 1) or (self.global_step % cfg.rt_period == 0)
        if mode in {"muon_stream", "muon_stream_qqq"} or (mode == "muon_stream_fdt_metric") or (not active_rt and mode.startswith("rt_")):
            w = torch.ones_like(sigma)
            rt_diag = {"beta": 0.0, "accept": 0.0, "abs_w_minus_1": 0.0, "phi": 0.0}
        elif mode == "ht_stream":
            raw = sigma.clamp_min(cfg.eps).pow(cfg.ht_power)
            w = raw * (math.sqrt(r) / raw.norm().clamp_min(cfg.eps))
            rt_diag = {"beta": -1.0, "accept": 1.0, "abs_w_minus_1": float((w - 1).abs().mean().item()), "phi": 0.0}
        elif mode == "mclip_stream":
            raw = sigma.clamp_min(cfg.eps)
            raw = raw / raw.mean().clamp_min(cfg.eps)
            w = torch.clamp(raw, 1.0 - cfg.mclip_delta, 1.0 + cfg.mclip_delta)
            w = w * (math.sqrt(r) / w.norm().clamp_min(cfg.eps))
            rt_diag = {"beta": -2.0, "accept": 1.0, "abs_w_minus_1": float((w - 1).abs().mean().item()), "phi": 0.0}
        elif mode in {"rt_v43_stream", "rt_v43_ns", "rt_v6_fdt_metric"}:
            w, rt_diag = self._rt_weights(U, V, G_work, R_work, lr, cfg)
        else:
            # Unknown modes fall back to streaming Muon rather than crashing a long HamGNN run.
            w = torch.ones_like(sigma)
            rt_diag = {"beta": 0.0, "accept": 0.0, "abs_w_minus_1": 0.0, "phi": 0.0}

        X = self._compose_from_basis(U, V, w, transposed)
        if use_metric and sqrt_tr is not None and sqrt_tc is not None:
            Q = sqrt_tr[:, None] * X * sqrt_tc[None, :]
        else:
            Q = X
        uses_target_rms = mode != "muon_stream"
        if uses_target_rms:
            target_rms = math.sqrt(float(min(n, m)) / float(n * m))
            Q = Q * (target_rms / float(_rms(Q, cfg.eps).item()))
        else:
            shape_scale = max(1.0, float(n) / float(max(m, 1))) ** 0.5
            Q = Q * shape_scale
        diag = {
            "mode_is_metric": float(use_metric),
            "mode_uses_muon_source": float(use_muon_source),
            "mode_uses_target_rms": 1.0 if uses_target_rms else 0.0,
            "momentum_beta": float(cfg.beta),
            "nesterov": 1.0 if cfg.nesterov else 0.0,
            "update_rms": float(_rms(Q, cfg.eps).item()),
            "matrix_rows": float(n),
            "matrix_cols": float(m),
            **rt_diag,
            **metric_diag,
        }
        return Q.reshape(orig_shape), diag

    def _get_metric_temperatures(self, st: Dict[str, Any], g: torch.Tensor, innovation: torch.Tensor, cfg: RTConfig):
        n, m = g.shape
        device = g.device
        if "row_signal" not in st or st["row_signal"].numel() != n:
            st["row_signal"] = torch.zeros(n, device=device, dtype=torch.float32)
            st["row_noise"] = torch.zeros(n, device=device, dtype=torch.float32)
        if "col_signal" not in st or st["col_signal"].numel() != m:
            st["col_signal"] = torch.zeros(m, device=device, dtype=torch.float32)
            st["col_noise"] = torch.zeros(m, device=device, dtype=torch.float32)
        row_signal_now = g.float().pow(2).sum(dim=1)
        col_signal_now = g.float().pow(2).sum(dim=0)
        row_noise_now = innovation.float().pow(2).sum(dim=1)
        col_noise_now = innovation.float().pow(2).sum(dim=0)
        b = cfg.metric_ema_beta
        st["row_signal"].mul_(b).add_(row_signal_now, alpha=1-b)
        st["col_signal"].mul_(b).add_(col_signal_now, alpha=1-b)
        st["row_noise"].mul_(b).add_(row_noise_now, alpha=1-b)
        st["col_noise"].mul_(b).add_(col_noise_now, alpha=1-b)
        row_rel = (st["row_signal"] + cfg.eps) / (st["row_signal"] + cfg.metric_lambda_noise * st["row_noise"] + cfg.eps)
        col_rel = (st["col_signal"] + cfg.eps) / (st["col_signal"] + cfg.metric_lambda_noise * st["col_noise"] + cfg.eps)
        tr = _geomean_normalize_temperature((row_rel + cfg.eps).pow(cfg.metric_alpha_row), cfg.metric_tmin, cfg.metric_tmax, cfg.eps)
        tc = _geomean_normalize_temperature((col_rel + cfg.eps).pow(cfg.metric_alpha_col), cfg.metric_tmin, cfg.metric_tmax, cfg.eps)
        sqrt_tr = tr.sqrt().to(dtype=g.dtype)
        sqrt_tc = tc.sqrt().to(dtype=g.dtype)
        diag = {
            "metric_row_temp_cv": float(tr.std(unbiased=False).div(tr.mean().clamp_min(cfg.eps)).item()),
            "metric_col_temp_cv": float(tc.std(unbiased=False).div(tc.mean().clamp_min(cfg.eps)).item()),
            "metric_row_rel_mean": float(row_rel.mean().item()),
            "metric_col_rel_mean": float(col_rel.mean().item()),
        }
        return sqrt_tr, sqrt_tc, diag

    def _streaming_basis(self, st: Dict[str, Any], A: torch.Tensor, cfg: RTConfig):
        n, m = A.shape
        if n >= m:
            return self._streaming_basis_n_ge_m(st, A, cfg, transposed=False)
        sub = st.setdefault("transpose_state", {})
        U_t, s_t, V_t, _ = self._streaming_basis_n_ge_m(sub, A.t().contiguous(), cfg, transposed=True)
        # For A.T = U_t S V_t.T, A = V_t S U_t.T. Return U=V_t, V=U_t and flag.
        return V_t, s_t, U_t, False

    def _streaming_basis_n_ge_m(self, st: Dict[str, Any], A: torch.Tensor, cfg: RTConfig, transposed: bool = False):
        n, m = A.shape
        r = m if cfg.stream_k <= 0 else min(cfg.stream_k, m, n)
        key = "V_basis"
        V = st.get(key)
        if V is None or tuple(V.shape) != (m, r) or V.device != A.device or V.dtype != A.dtype:
            if r == m:
                V = torch.eye(m, r, device=A.device, dtype=A.dtype)
            else:
                V = torch.randn(m, r, device=A.device, dtype=A.dtype) / math.sqrt(max(m, 1))
                V, _ = torch.linalg.qr(V, mode="reduced")
            st[key] = V
        V = st[key]
        for _ in range(max(1, cfg.power_iters)):
            Z = A.t().matmul(A.matmul(V))
            V, _ = torch.linalg.qr(Z, mode="reduced")
        st[key] = V
        Y = A.matmul(V)
        U, sigma = _safe_col_norm(Y, cfg.eps)
        return U, sigma, V, transposed

    @staticmethod
    def _compose_from_basis(U: torch.Tensor, V: torch.Tensor, w: torch.Tensor, transposed: bool = False) -> torch.Tensor:
        return (U * w.unsqueeze(0)).matmul(V.t())

    def _rt_weights(self, U: torch.Tensor, V: torch.Tensor, G: torch.Tensor, G_replica: torch.Tensor, lr: float, cfg: RTConfig):
        # v4.3 selector proxy: project current and temporal-replica gradients into
        # the Muon basis, then apply the SQ-RT hard/soft cross-fit selector.
        GV = G.matmul(V)
        RV = G_replica.matmul(V)
        g1 = (U * GV).sum(dim=0).float()
        g2 = (U * RV).sum(dim=0).float()
        r = g1.numel()
        if r == 0:
            return torch.ones(0, device=G.device, dtype=G.dtype), {"beta": 0.0, "accept": 0.0, "phi": 0.0}

        nu = 0.25 * (g1 - g2).pow(2)
        consistency = 2.0 * g1 * g2 / (g1.pow(2) + g2.pow(2) + cfg.eps)
        B = (1.0 + cfg.lambda_noise * nu + cfg.lambda_consistency * (1.0 - consistency)).clamp_min(cfg.eps)
        a1 = float(lr) * g1 - (float(lr) ** 2) * B
        a2 = float(lr) * g2 - (float(lr) ** 2) * B
        is_soft = lr >= cfg.lr_switch
        delta = cfg.soft_delta if is_soft else cfg.hard_delta
        threshold = cfg.phi_threshold_soft if is_soft else cfg.phi_threshold_hard
        ones = torch.ones(r, device=G.device, dtype=torch.float32)

        def surrogate(gproj: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return float(lr) * (gproj * w).sum() - 0.5 * (float(lr) ** 2) * (B * w.pow(2)).sum()

        def metrics(beta: float):
            if beta <= 0.0:
                zero = torch.tensor(0.0, device=G.device)
                return zero, zero, zero, zero, zero
            w1 = _weights_from_advantage(a1, beta, delta, cfg.eps)
            w2 = _weights_from_advantage(a2, beta, delta, cfg.eps)
            gain_2 = surrogate(g2, w1) - surrogate(g2, ones)
            gain_1 = surrogate(g1, w2) - surrogate(g1, ones)
            gain = 0.5 * (gain_1 + gain_2)
            ex1 = (w1.pow(2) - 1.0).pow(2)
            ex2 = (w2.pow(2) - 1.0).pow(2)
            diff_raw = 0.5 * ((nu * ex1).sum() / nu.sum().clamp_min(cfg.eps) + (nu * ex2).sum() / nu.sum().clamp_min(cfg.eps))
            diff = float(lr) * diff_raw
            uncert = (gain_1 - gain_2).abs()
            tube = float(lr) * 0.5 * ((w1 - 1.0).pow(2).mean() + (w2 - 1.0).pow(2).mean())
            phi = gain - cfg.lambda_diff * diff - cfg.lambda_uncert * uncert - cfg.lambda_tube * tube
            return phi, gain, diff, uncert, tube

        best_phi = torch.tensor(0.0, device=G.device)
        best_gain = torch.tensor(0.0, device=G.device)
        best_diff = torch.tensor(0.0, device=G.device)
        best_uncert = torch.tensor(0.0, device=G.device)
        best_tube = torch.tensor(0.0, device=G.device)
        best_beta = 0.0
        best_candidate = 0.0
        best_shrink = 0.0
        for beta in cfg.beta_grid:
            phi, gain, diff, uncert, tube = metrics(float(beta))
            beff = float(beta)
            shrink = 1.0 if beta > 0.0 else 0.0
            if is_soft and beta > 0.0:
                if phi <= 0.0:
                    phi = gain = diff = uncert = tube = torch.tensor(0.0, device=G.device)
                    beff = 0.0
                    shrink = 0.0
                else:
                    shrink_t = torch.sigmoid((phi - threshold) / max(cfg.soft_tau, cfg.eps))
                    beff = float(beta) * float(shrink_t.item())
                    shrink = float(shrink_t.item())
                    phi, gain, diff, uncert, tube = metrics(beff)
            elif (not is_soft) and phi <= threshold:
                phi = gain = diff = uncert = tube = torch.tensor(0.0, device=G.device)
                beff = 0.0
                shrink = 0.0
            if float(phi.item()) > float(best_phi.item()):
                best_phi = phi
                best_gain = gain
                best_diff = diff
                best_uncert = uncert
                best_tube = tube
                best_beta = beff
                best_candidate = float(beta)
                best_shrink = shrink

        a = 0.5 * (a1 + a2)
        best_w = _weights_from_advantage(a, best_beta, delta, cfg.eps).to(dtype=G.dtype)
        accept = 1.0 if best_beta > 0 and float(best_phi.item()) > 0.0 else 0.0
        diag = {
            "beta": best_beta,
            "beta_candidate": best_candidate,
            "soft_shrink": best_shrink,
            "accept": accept,
            "abs_w_minus_1": float((best_w.float() - 1.0).abs().mean().item()),
            "phi": float(best_phi.item()),
            "raw_cv_gain": float(best_gain.item()),
            "diffusion_cost": float(best_diff.item()),
            "uncertainty_cost": float(best_uncert.item()),
            "tube_penalty": float(best_tube.item()),
            "nu_mean": float(nu.mean().item()),
            "consistency_mean": float(consistency.mean().item()),
            "B_mean": float(B.mean().item()),
        }
        return best_w, diag

    def _accumulate_diag(self, diag: Dict[str, float]) -> None:
        self._diag_count += 1
        for k, v in diag.items():
            if isinstance(v, (float, int)) and math.isfinite(float(v)):
                self._diag_accum[k] = self._diag_accum.get(k, 0.0) + float(v)

    def _flush_diag(self) -> None:
        if not self._diag_path or self._diag_count == 0:
            return
        rec = {"time": time.time(), "step": self.global_step, "count": self._diag_count, "mode": self.cfg.mode}
        for k, v in self._diag_accum.items():
            rec[k] = v / max(1, self._diag_count)
        with open(self._diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        self._diag_accum.clear()
        self._diag_count = 0


def build_optimizer_from_env(named_parameters, lr: float, eps: float = 1e-8, beta1: float = 0.9, beta2: float = 0.999, amsgrad: bool = True):
    """Build optimizer from HAMGNN_RT_* environment variables."""
    mode = os.environ.get("HAMGNN_RT_OPTIMIZER", os.environ.get("HAMGNN_OPTIMIZER", "adamw")).lower().replace("-", "_")
    wd = _as_float(os.environ.get("HAMGNN_RT_WEIGHT_DECAY"), 0.0)
    if mode == "adamw":
        return torch.optim.AdamW([p for _, p in named_parameters], lr=lr, eps=eps, betas=(beta1, beta2), weight_decay=wd, amsgrad=amsgrad)
    groups = []
    for name, p in named_parameters:
        if p.requires_grad:
            groups.append({"params": [p], "name": name})
    cfg = dict(
        mode=mode,
        beta=_as_float(os.environ.get("HAMGNN_RT_MOMENTUM"), 0.95),
        eps=eps,
        weight_decay=wd,
        stream_k=_as_int(os.environ.get("HAMGNN_RT_STREAM_K"), 0),
        power_iters=_as_int(os.environ.get("HAMGNN_RT_POWER_ITERS"), 1),
        rt_period=_as_int(os.environ.get("HAMGNN_RT_PERIOD"), 1),
        lr_switch=_as_float(os.environ.get("HAMGNN_RT_LR_SWITCH"), 0.05),
        hard_delta=_as_float(os.environ.get("HAMGNN_RT_HARD_DELTA"), 0.35),
        soft_delta=_as_float(os.environ.get("HAMGNN_RT_SOFT_DELTA"), 0.35),
        lambda_noise=_as_float(os.environ.get("HAMGNN_RT_LAMBDA_NOISE"), 2.0),
        lambda_consistency=_as_float(os.environ.get("HAMGNN_RT_LAMBDA_CONSISTENCY"), _as_float(os.environ.get("HAMGNN_RT_LAMBDA_C"), 0.25)),
        lambda_diff=_as_float(os.environ.get("HAMGNN_RT_LAMBDA_DIFF"), 0.5),
        lambda_uncert=_as_float(os.environ.get("HAMGNN_RT_LAMBDA_UNCERT"), 0.5),
        lambda_tube=_as_float(os.environ.get("HAMGNN_RT_LAMBDA_TUBE"), 0.05),
        phi_threshold_hard=_as_float(os.environ.get("HAMGNN_RT_HARD_PHI_THRESHOLD"), 0.0),
        phi_threshold_soft=_as_float(os.environ.get("HAMGNN_RT_SOFT_PHI_THRESHOLD"), 0.0),
        soft_tau=_as_float(os.environ.get("HAMGNN_RT_SOFT_TAU"), 0.01),
        metric_alpha_row=_as_float(os.environ.get("HAMGNN_RT_METRIC_ALPHA_ROW"), 0.25),
        metric_alpha_col=_as_float(os.environ.get("HAMGNN_RT_METRIC_ALPHA_COL"), 0.25),
        metric_lambda_noise=_as_float(os.environ.get("HAMGNN_RT_METRIC_LAMBDA_NOISE"), 1.0),
        metric_tmin=_as_float(os.environ.get("HAMGNN_RT_METRIC_TMIN"), 0.5),
        metric_tmax=_as_float(os.environ.get("HAMGNN_RT_METRIC_TMAX"), 2.0),
        metric_ema_beta=_as_float(os.environ.get("HAMGNN_RT_METRIC_EMA_BETA"), 0.95),
        ht_power=_as_float(os.environ.get("HAMGNN_RT_HT_POWER"), 0.125),
        mclip_delta=_as_float(os.environ.get("HAMGNN_RT_MCLIP_DELTA"), 0.25),
        ns_steps=_as_int(os.environ.get("HAMGNN_MUON_NS_STEPS"), _as_int(os.environ.get("HAMGNN_RT_NS_STEPS"), 5)),
        nesterov=_as_bool(os.environ.get("HAMGNN_MUON_NESTEROV"), True),
        adam_beta1=beta1,
        adam_beta2=beta2,
        log_interval=_as_int(os.environ.get("HAMGNN_RT_LOG_INTERVAL"), 50),
        diag_dir=os.environ.get("HAMGNN_RT_DIAG_DIR", ""),
        name=os.environ.get("HAMGNN_RT_RUN_NAME", mode),
    )
    cfg["beta_grid"] = _parse_float_tuple(os.environ.get("HAMGNN_RT_BETA_GRID"), RTConfig.beta_grid)
    return HamGNNMuonFDTOptimizer(groups, lr=lr, **cfg)
