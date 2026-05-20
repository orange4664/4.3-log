"""Small linear algebra helpers for exact-SVD Muon toy benchmarks."""
from __future__ import annotations
import numpy as np

EPS = 1e-12

def compact_svd(x: np.ndarray):
    return np.linalg.svd(x, full_matrices=False)

def fro_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord='fro'))

def rms_align(x: np.ndarray, target_fro: float) -> np.ndarray:
    n = fro_norm(x)
    if n < EPS:
        return np.zeros_like(x)
    return x * (float(target_fro) / (n + EPS))

def spectral_entropy_from_weights(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    if w.size == 0:
        return 0.0
    p = w*w / (np.sum(w*w) + EPS)
    return float(-np.sum(p * np.log(p + EPS)))

def stable_softmax(logits: np.ndarray, temperature_inverse: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=float) * float(temperature_inverse)
    if z.size == 0:
        return z
    z = z - np.max(z)
    e = np.exp(np.clip(z, -80, 80))
    s = np.sum(e)
    if s < EPS:
        return np.ones_like(e) / max(1, e.size)
    return e / s

def robust_center_scale(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    med = np.median(a)
    centered = a - med
    mad = np.median(np.abs(centered))
    if mad < EPS:
        std = np.std(centered)
        scale = std if std > EPS else 1.0
    else:
        scale = 1.4826 * mad
    return centered / (scale + EPS)
