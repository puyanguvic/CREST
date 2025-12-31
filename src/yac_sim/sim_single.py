
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

def simulate(cfg: SimConfig, policy: str = "ET", rng: np.random.Generator | None = None) -> dict:
    """Simulate one rollout. policy in {ET, PER, RAND}."""
    if rng is None:
        rng = np.random.default_rng(0)

    A, B = double_integrator_2d(cfg.Ts)
    n = A.shape[0]
    Q, R = _cost_matrices(cfg)
    K = dlqr(A, B, Q, R)  # u = -K x_hat

    A_hat = A + cfg.mismatch_eps * np.eye(n)

    ch = GilbertElliottChannel(cfg.p_good_to_bad, cfg.p_bad_to_good, cfg.loss_good, cfg.loss_bad, rng)

    # initial state: random position/velocity
    x = np.array([rng.normal(0, 5.0), rng.normal(0, 1.0), rng.normal(0, 5.0), rng.normal(0, 1.0)], dtype=float)
    x_hat = x.copy()  # controller-side estimate
    u_prev = np.zeros((2,), dtype=float)

    # logs
    x_norm = np.zeros(cfg.T_steps)
    tilde_pred_norm = np.zeros(cfg.T_steps)
    tx_attempt = np.zeros(cfg.T_steps, dtype=int)
    tx_deliv = np.zeros(cfg.T_steps, dtype=int)

    J = 0.0

    for k in range(cfg.T_steps):
        # plant evolves with process noise
        w = rng.normal(0.0, cfg.sigma_w, size=(n,))
        x = A @ x + B @ u_prev + w

        # measurement
        v = rng.normal(0.0, cfg.sigma_v, size=(n,))
        y = x + v

        # predictor at controller
        x_hat_pred = A_hat @ x_hat + B @ u_prev
        tilde_pred = y - x_hat_pred  # innovation / pred error proxy
        tilde_pred_norm[k] = norm2(tilde_pred)

        # decide whether to transmit
        do_tx = False
        if policy == "ET":
            do_tx = tilde_pred_norm[k] > cfg.delta
        elif policy == "PER":
            do_tx = (k % max(int(cfg.period_M), 1) == 0)
        elif policy == "RAND":
            do_tx = (float(rng.random()) < cfg.random_p)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        if do_tx:
            tx_attempt[k] = 1
            delivered = True if cfg.mode == "theory" else ch.deliver()
            if delivered:
                tx_deliv[k] = 1
                if cfg.mode == "theory":
                    x_hat = x.copy()  # exact reset
                else:
                    yq = uniform_quantize(y, cfg.bits_per_value, cfg.q_min, cfg.q_max)
                    x_hat = yq.copy()
            else:
                x_hat = x_hat_pred
        else:
            x_hat = x_hat_pred

        # control
        u = -(K @ x_hat).reshape(-1)
        # cost (use true x and u)
        J += float(x.T @ Q @ x + u.T @ R @ u)

        # log
        x_norm[k] = norm2(x)
        u_prev = u

    bits_attempt = int(tx_attempt.sum()) * n * int(cfg.bits_per_value)
    bits_deliv = int(tx_deliv.sum()) * n * int(cfg.bits_per_value)

    return dict(
        J=J,
        x_norm=x_norm,
        tilde_pred_norm=tilde_pred_norm,
        tx_attempt=tx_attempt,
        tx_deliv=tx_deliv,
        N_attempt=int(tx_attempt.sum()),
        N_deliv=int(tx_deliv.sum()),
        bits_attempt=bits_attempt,
        bits_deliv=bits_deliv,
    )

def monte_carlo(cfg: SimConfig, policy: str, deltas: list[float] | None = None, periods: list[int] | None = None, ps: list[float] | None = None) -> list[dict]:
    """Run MC for a sweep of a single policy knob."""
    rng_master = np.random.default_rng(cfg.seed)
    seeds = rng_master.integers(0, 2**31-1, size=cfg.mc_runs, dtype=np.int64).tolist()

    results = []
    if policy == "ET":
        assert deltas is not None
        for d in deltas:
            Js, bitsD, bitsA = [], [], []
            for s in seeds:
                rng = np.random.default_rng(int(s))
                cfg2 = SimConfig(**cfg.__dict__)
                cfg2.delta = float(d)
                out = simulate(cfg2, "ET", rng=rng)
                Js.append(out["J"]); bitsD.append(out["bits_deliv"]); bitsA.append(out["bits_attempt"])
            results.append(dict(param=float(d), J=np.array(Js), bits_deliv=np.array(bitsD), bits_attempt=np.array(bitsA)))
    elif policy == "PER":
        assert periods is not None
        for M in periods:
            Js, bitsD, bitsA = [], [], []
            for s in seeds:
                rng = np.random.default_rng(int(s))
                cfg2 = SimConfig(**cfg.__dict__)
                cfg2.period_M = int(M)
                out = simulate(cfg2, "PER", rng=rng)
                Js.append(out["J"]); bitsD.append(out["bits_deliv"]); bitsA.append(out["bits_attempt"])
            results.append(dict(param=int(M), J=np.array(Js), bits_deliv=np.array(bitsD), bits_attempt=np.array(bitsA)))
    elif policy == "RAND":
        assert ps is not None
        for p in ps:
            Js, bitsD, bitsA = [], [], []
            for s in seeds:
                rng = np.random.default_rng(int(s))
                cfg2 = SimConfig(**cfg.__dict__)
                cfg2.random_p = float(p)
                out = simulate(cfg2, "RAND", rng=rng)
                Js.append(out["J"]); bitsD.append(out["bits_deliv"]); bitsA.append(out["bits_attempt"])
            results.append(dict(param=float(p), J=np.array(Js), bits_deliv=np.array(bitsD), bits_attempt=np.array(bitsA)))
    else:
        raise ValueError(policy)

    return results
