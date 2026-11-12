"""MeanFlow policy helpers: interval sampling, JVP, iMF / naive regression, diagnostics.

Convention (identical to FPO++ / RFPO): t=1 is the noise end, t=0 the action end,

    z_t = t * eps + (1 - t) * a,        v_c = dz_t/dt = eps - a.

``u(o, z_t, r, t)`` is the average velocity over [r, t] with 0 <= r <= t <= 1, so
``z_r = z_t - (t - r) * u`` and the one-step action is ``a = eps - u(o, eps, 0, 1)``.
The boundary ``u(o, z_t, t, t)`` is the instantaneous velocity ``v_hat``.

iMF regression (ours):   V = u + (t - r) * sg(d),  d = JVP[u; (v_hat, 0, 1)],  loss = w * ||V - v_c||^2
naive MeanFlow:          u_tgt = sg(v_c - (t - r) * JVP[u; (v_c, 0, 1)]),       loss = w * ||u - u_tgt||^2
"""

from __future__ import annotations

from typing import Callable

import torch

MEANFLOW_INTERVAL_BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0 + 1e-6)


def sample_meanflow_r(
    t: torch.Tensor,
    interval_mode: str,
    boundary_prob: float,
) -> torch.Tensor:
    """Sample r with 0 <= r <= t.

    - ``full``:   r = 0 (whole interval [0, t]).
    - ``random``: r ~ U(0, t).
    A ``boundary_prob`` fraction of samples uses r = t, which is plain instantaneous
    flow matching on u(z, t, t) and keeps ``v_hat`` (the JVP tangent) supervised.
    """
    if interval_mode == "full":
        r = torch.zeros_like(t)
    elif interval_mode == "random":
        r = torch.rand_like(t) * t
    else:
        raise ValueError(f"Unknown meanflow_interval_mode: {interval_mode}")
    if boundary_prob > 0.0:
        r = torch.where(torch.rand_like(t) < boundary_prob, t, r)
    return r


def meanflow_interval_weight(r: torch.Tensor, t: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-sample weight w(r, t) with shape ``t.shape[:-1]``."""
    if mode == "uniform":
        return torch.ones_like(t[..., 0])
    raise ValueError(f"Unknown meanflow_loss_weighting: {mode}")


def compute_meanflow_jvp(
    u_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    z_t: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    tangent_z: torch.Tensor,
) -> torch.Tensor:
    """d = (du/dz) tangent_z + du/dt, i.e. JVP of u(z, r, t) along (tangent_z, 0, 1).

    Forward-mode AD under ``no_grad``: the result carries no autograd graph, so the
    loss never backpropagates through the JVP (no second-order terms).
    """
    tangent_z = tangent_z.detach()
    with torch.no_grad():
        _, d = torch.func.jvp(
            lambda z, tt: u_fn(z, r, tt),
            (z_t.detach(), t.detach()),
            (tangent_z, torch.ones_like(t)),
        )
    return d


class MeanFlowDiagnostics:
    """Accumulates per-sample MeanFlow diagnostics over the mini-batches of one update."""

    def __init__(self, quantile_subsample: int = 4096):
        self.quantile_subsample = int(quantile_subsample)
        self._sums: dict[str, float] = {}
        self._count = 0
        self._d_norm_samples: list[torch.Tensor] = []
        self._gap_bin_sums = torch.zeros(len(MEANFLOW_INTERVAL_BIN_EDGES) - 1, dtype=torch.float64)
        self._gap_bin_counts = torch.zeros_like(self._gap_bin_sums)

    @torch.no_grad()
    def add(self, aux: dict[str, torch.Tensor]) -> None:
        if not aux:
            return
        d_norm = aux["jvp_norm"].reshape(-1).float()
        n = d_norm.numel()
        for key in ("jvp_norm", "u_vhat_gap", "V_err", "u_err", "interval"):
            self._sums[key] = self._sums.get(key, 0.0) + aux[key].float().sum().item()
        self._sums["jvp_norm_sq"] = self._sums.get("jvp_norm_sq", 0.0) + d_norm.pow(2).sum().item()
        self._count += n

        if self.quantile_subsample > 0:
            k = min(self.quantile_subsample, n)
            idx = torch.randperm(n, device=d_norm.device)[:k]
            self._d_norm_samples.append(d_norm[idx].cpu())

        interval = aux["interval"].reshape(-1).float()
        gap = aux["u_vhat_gap"].reshape(-1).float()
        for i in range(len(MEANFLOW_INTERVAL_BIN_EDGES) - 1):
            lo, hi = MEANFLOW_INTERVAL_BIN_EDGES[i], MEANFLOW_INTERVAL_BIN_EDGES[i + 1]
            mask = (interval >= lo) & (interval < hi)
            self._gap_bin_sums[i] += gap[mask].sum().item()
            self._gap_bin_counts[i] += mask.sum().item()

    def summary(self, prefix: str = "meanflow/") -> dict[str, float]:
        if self._count == 0:
            return {}
        n = float(self._count)
        mean_d = self._sums["jvp_norm"] / n
        var_d = max(self._sums["jvp_norm_sq"] / n - mean_d**2, 0.0)
        out = {
            f"{prefix}jvp_norm_mean": mean_d,
            f"{prefix}jvp_norm_std": var_d**0.5,
            f"{prefix}u_vhat_gap_mean": self._sums["u_vhat_gap"] / n,
            f"{prefix}V_minus_vc_sq_mean": self._sums["V_err"] / n,
            f"{prefix}u_minus_vc_sq_mean": self._sums["u_err"] / n,
            f"{prefix}interval_mean": self._sums["interval"] / n,
        }
        if self._d_norm_samples:
            samples = torch.cat(self._d_norm_samples)
            q = torch.quantile(samples, torch.tensor([0.9, 0.99]))
            out[f"{prefix}jvp_norm_p90"] = q[0].item()
            out[f"{prefix}jvp_norm_p99"] = q[1].item()
        for i in range(len(MEANFLOW_INTERVAL_BIN_EDGES) - 1):
            if self._gap_bin_counts[i] > 0:
                lo, hi = MEANFLOW_INTERVAL_BIN_EDGES[i], min(MEANFLOW_INTERVAL_BIN_EDGES[i + 1], 1.0)
                out[f"{prefix}u_vhat_gap_interval_{lo:.2f}_{hi:.2f}"] = (
                    self._gap_bin_sums[i] / self._gap_bin_counts[i]
                ).item()
        return out
