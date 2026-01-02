
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SimConfig:
    # timing
    Ts: float = 0.1
    T_steps: int = 400
    mc_runs: int = 30
    seed: int = 7

    # modes: "theory" (paper baseline) or "robust" (noise/quant/loss/mismatch)
    mode: str = "theory"

    # trigger / periodic
    delta: float = 0.4
    period_M: int = 10
    random_p: float = 0.1  # for random baseline

    # disturbances / noise (paper: bounded)
    w_bar: float = 0.06   # ||w_k||_inf bound used for sampling; paper assumes ||w_k||_2 <= w_bar (see sim_single)
    v_bar: float = 0.00   # ||v_k||_inf bound
    # legacy Gaussian stds (kept for backwards compatibility; ignored when mode='theory' and bounds are used)
    sigma_w: float = 0.03
    sigma_v: float = 0.00
    bits_per_value: int = 32

    # measurement model (paper uses y = Cx + v; default is full-state)
    C_full_state: bool = True

    # observer gain (paper uses fixed L; default L = I for full-state)
    L_gain: float = 1.0

    # channel (Gilbert–Elliott)
    p_good_to_bad: float =  0.03
    p_bad_to_good: float =  0.08
    loss_good: float =  0.02
    loss_bad: float =  0.45

    # model mismatch (predictor uses A_hat = A + eps*I)
    mismatch_eps: float = 0.0

    # quantizer range
    q_min: float = -20.0
    q_max: float = 20.0

    # cost matrices (as in paper)
    Q_px: float = 10.0
    Q_vx: float = 1.0
    Q_py: float = 10.0
    Q_vy: float = 1.0
    R_u: float = 0.1

    def force_theory(self) -> None:
        """Configure idealized 'theory' mode consistent with the paper assumptions."""
        self.mode = "theory"
        self.v_bar = 0.0
        self.bits_per_value = 32
        self.loss_good = 0.0
        self.loss_bad = 0.0
        self.mismatch_eps = 0.0

