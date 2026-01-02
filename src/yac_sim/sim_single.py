from __future__ import annotations

import numpy as np

from .config import SimConfig
from .models import double_integrator_2d
from .utils import dlqr, norm2
from .channels import GilbertElliottChannel
from .quantization import uniform_quantize


def _cost_matrices(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    Q = np.diag([cfg.Q_px, cfg.Q_vx, cfg.Q_py, cfg.Q_vy]).astype(float)
    R = (cfg.R_u * np.eye(2)).astype(float)
    return Q, R


def _bounded_vec(rng: np.random.Generator, bound: float, size: int) -> np.ndarray:
    """Sample a vector with component-wise bound in [-bound, bound].

    Paper assumption is ||w_k||_2 <= \bar w and ||v_k||_2 <= \bar v.
    We sample uniformly in an L_infty box; this implies an L2 bound:
      ||z||_2 <= sqrt(size) * bound.
    """
    b = float(bound)
    if b <= 0.0:
        return np.zeros((size,), dtype=float)
    return rng.uniform(-b, b, size=(size,)).astype(float)


def _measurement_matrices(cfg: SimConfig, n: int) -> tuple[np.ndarray, int]:
    """Return (C, p). Default is full-state measurement (C = I)."""
    if getattr(cfg, "C_full_state", True):
        C = np.eye(n, dtype=float)
        p = n
        return C, p

    # If you ever want partial measurement, customize here.
    # For now, we keep it simple and explicit: measure positions only.
    # State is [p_x, v_x, p_y, v_y]^T -> measure [p_x, p_y]
    C = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]], dtype=float)
    p = C.shape[0]
    return C, p


def _observer_gain(cfg: SimConfig, n: int, p: int) -> np.ndarray:
    """Fixed observer gain L used in the paper update: xhat = xhat^- + L(y - C xhat^-)."""
    # Default: full-state, L = I (equivalent to reset-to-measurement, but written in the paper's form)
    gain = float(getattr(cfg, "L_gain", 1.0))
    if getattr(cfg, "C_full_state", True) and p == n:
        return gain * np.eye(n, dtype=float)

    # For partial measurement, use a simple gain mapping innovation (p) into state (n)
    # Here we inject the position innovations into the corresponding state components.
    L = np.zeros((n, p), dtype=float)
    if n == 4 and p == 2:
        L[0, 0] = gain
        L[2, 1] = gain
    else:
        # Fallback: scaled identity on min(n, p)
        for i in range(min(n, p)):
            L[i, i] = gain
    return L


def simulate(cfg: SimConfig, policy: str = "ET", rng: np.random.Generator | None = None) -> dict:
    """Simulate one rollout over a finite horizon.

    Policies:
      - ET: innovation-triggered (paper eq. (trigger))
      - PER: periodic with period_M
      - RAND: Bernoulli random with probability random_p
    """
    if rng is None:
        rng = np.random.default_rng(int(cfg.seed) & 0xFFFFFFFF)

    A, B = double_integrator_2d(cfg.Ts)
    n = A.shape[0]
    C, p = _measurement_matrices(cfg, n)
    L = _observer_gain(cfg, n, p)

    Q, R = _cost_matrices(cfg)
    K = dlqr(A, B, Q, R)  # control law uses estimate: u = -K x_hat

    # Predictor may have mismatch (robust experiment)
    A_hat = A + float(cfg.mismatch_eps) * np.eye(n, dtype=float)

    # Channel used only in robust mode
    ch = GilbertElliottChannel(cfg.p_good_to_bad, cfg.p_bad_to_good, cfg.loss_good, cfg.loss_bad, rng)

    # initial state: random position/velocity
    x = np.array([rng.normal(0, 5.0), rng.normal(0, 1.0), rng.normal(0, 5.0), rng.normal(0, 1.0)], dtype=float)
    x_hat = x.copy()
    u_prev = np.zeros((2,), dtype=float)

    # logs
    x_norm = np.zeros(cfg.T_steps)
    tilde_x_norm = np.zeros(cfg.T_steps)          # ||x - x_hat||
    innovation_norm = np.zeros(cfg.T_steps)       # ||y - C x_hat^-||
    tx_attempt = np.zeros(cfg.T_steps, dtype=int)
    tx_deliv = np.zeros(cfg.T_steps, dtype=int)

    J = 0.0

    for k in range(cfg.T_steps):
        # --- plant update (paper: bounded disturbance) ---
        if cfg.mode == "theory":
            w = _bounded_vec(rng, cfg.w_bar, n)
        else:
            # robust mode can still use bounded disturbances; keep it bounded for paper consistency
            w = _bounded_vec(rng, max(cfg.w_bar, 0.0), n)
        x = A @ x + B @ u_prev + w

        # --- measurement (paper: y = Cx + v, bounded noise) ---
        if cfg.mode == "theory":
            v = _bounded_vec(rng, cfg.v_bar, p)
        else:
            v = _bounded_vec(rng, max(cfg.v_bar, 0.0), p)
        y = (C @ x) + v

        # --- remote prediction at controller ---
        x_hat_pred = A_hat @ x_hat + B @ u_prev

        # innovation used for triggering (paper eq. (trigger))
        innovation = y - (C @ x_hat_pred)
        innovation_norm[k] = norm2(innovation)

        # decide whether to transmit
        if policy == "ET":
            do_tx = innovation_norm[k] > float(cfg.delta)
        elif policy == "PER":
            do_tx = (k % max(int(cfg.period_M), 1) == 0)
        elif policy == "RAND":
            do_tx = (float(rng.random()) < float(cfg.random_p))
        else:
            raise ValueError(f"Unknown policy: {policy}")

        if do_tx:
            tx_attempt[k] = 1
            delivered = True if cfg.mode == "theory" else ch.deliver()

            if delivered:
                tx_deliv[k] = 1

                # quantization (robust only)
                if cfg.mode == "robust" and int(cfg.bits_per_value) < 32:
                    y_rx = uniform_quantize(y, cfg.bits_per_value, cfg.q_min, cfg.q_max)
                else:
                    y_rx = y

                innovation_rx = y_rx - (C @ x_hat_pred)
                x_hat = x_hat_pred + (L @ innovation_rx)
            else:
                x_hat = x_hat_pred
        else:
            x_hat = x_hat_pred

        # control uses remote estimate only (paper eq. (control))
        u = -(K @ x_hat).reshape(-1)

        # instantaneous LQR cost (paper simulation setup)
        J += float(x.T @ Q @ x + u.T @ R @ u)

        # prediction error (paper eq. (tilde))
        tilde_x = x - x_hat
        tilde_x_norm[k] = norm2(tilde_x)

        x_norm[k] = norm2(x)
        u_prev = u

    # packet counts are primary in the paper; bits are auxiliary (for legacy plots)
    N_attempt = int(tx_attempt.sum())
    N_deliv = int(tx_deliv.sum())
    bits_attempt = N_attempt * n * int(cfg.bits_per_value)
    bits_deliv = N_deliv * n * int(cfg.bits_per_value)

    return dict(
        J=J,
        x_norm=x_norm,
        tilde_x_norm=tilde_x_norm,
        innovation_norm=innovation_norm,
        tx_attempt=tx_attempt,
        tx_deliv=tx_deliv,
        N_attempt=N_attempt,
        N_deliv=N_deliv,
        bits_attempt=bits_attempt,
        bits_deliv=bits_deliv,
    )


def monte_carlo(
    cfg: SimConfig,
    policy: str,
    deltas: list[float] | None = None,
    periods: list[int] | None = None,
    random_ps: list[float] | None = None,
) -> list[dict]:
    """Run Monte Carlo for a sweep of a single policy knob.

    Returns a list of dicts, one per sweep point, matching the plotting helpers in experiments.py.
    Each dict contains:
      - param: sweep value (delta / period / p)
      - J: array of costs over MC runs
      - N_deliv: array of delivered packet counts over MC runs
      - bits_deliv: array of delivered bits over MC runs (auxiliary)
      - N_attempt / bits_attempt also included for completeness
    """
    rng_master = np.random.default_rng(int(cfg.seed) & 0xFFFFFFFF)
    seeds = rng_master.integers(0, 2**31 - 1, size=int(cfg.mc_runs), dtype=np.int64).tolist()

    results: list[dict] = []

    if policy == "ET":
        assert deltas is not None
        for d in deltas:
            Js, Nd, Bd, Na, Ba = [], [], [], [], []
            for s in seeds:
                rng = np.random.default_rng(int(s))
                cfg2 = SimConfig(**cfg.__dict__)
                cfg2.delta = float(d)
                out = simulate(cfg2, "ET", rng=rng)
                Js.append(out["J"])
                Nd.append(out["N_deliv"])
                Bd.append(out["bits_deliv"])
                Na.append(out["N_attempt"])
                Ba.append(out["bits_attempt"])
            results.append(
                dict(
                    param=float(d),
                    J=np.asarray(Js, dtype=float),
                    N_deliv=np.asarray(Nd, dtype=float),
                    bits_deliv=np.asarray(Bd, dtype=float),
                    N_attempt=np.asarray(Na, dtype=float),
                    bits_attempt=np.asarray(Ba, dtype=float),
                )
            )

    elif policy == "PER":
        assert periods is not None
        for M in periods:
            Js, Nd, Bd, Na, Ba = [], [], [], [], []
            for s in seeds:
                rng = np.random.default_rng(int(s))
                cfg2 = SimConfig(**cfg.__dict__)
                cfg2.period_M = int(M)
                out = simulate(cfg2, "PER", rng=rng)
                Js.append(out["J"])
                Nd.append(out["N_deliv"])
                Bd.append(out["bits_deliv"])
                Na.append(out["N_attempt"])
                Ba.append(out["bits_attempt"])
            results.append(
                dict(
                    param=int(M),
                    J=np.asarray(Js, dtype=float),
                    N_deliv=np.asarray(Nd, dtype=float),
                    bits_deliv=np.asarray(Bd, dtype=float),
                    N_attempt=np.asarray(Na, dtype=float),
                    bits_attempt=np.asarray(Ba, dtype=float),
                )
            )

    elif policy == "RAND":
        assert random_ps is not None
        for p in random_ps:
            Js, Nd, Bd, Na, Ba = [], [], [], [], []
            for s in seeds:
                rng = np.random.default_rng(int(s))
                cfg2 = SimConfig(**cfg.__dict__)
                cfg2.random_p = float(p)
                out = simulate(cfg2, "RAND", rng=rng)
                Js.append(out["J"])
                Nd.append(out["N_deliv"])
                Bd.append(out["bits_deliv"])
                Na.append(out["N_attempt"])
                Ba.append(out["bits_attempt"])
            results.append(
                dict(
                    param=float(p),
                    J=np.asarray(Js, dtype=float),
                    N_deliv=np.asarray(Nd, dtype=float),
                    bits_deliv=np.asarray(Bd, dtype=float),
                    N_attempt=np.asarray(Na, dtype=float),
                    bits_attempt=np.asarray(Ba, dtype=float),
                )
            )
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return results
