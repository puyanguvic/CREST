from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SimConfig
from .sim_single import simulate_single_uav


def apply_plot_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "grid.alpha": 0.3,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def monte_carlo_single(cfg: SimConfig, policy: str, runs: int = 30, **kwargs):
    stats = []
    for i in range(runs):
        c = SimConfig(**{**cfg.__dict__, "seed": cfg.seed + i})
        _, s = simulate_single_uav(c, policy, **kwargs)
        stats.append(s)
    df = pd.DataFrame(stats)
    return df, {
        "rms_mean": df["rms_err"].mean(),
        "rms_std": df["rms_err"].std(),
        "J_mean": df["J_cost"].mean(),
        "J_std": df["J_cost"].std(),
        "N_tx_mean": df["N_tx"].mean(),
        "N_tx_std": df["N_tx"].std(),
        "N_tx_attempt_mean": df["N_tx_attempt"].mean(),
        "tx_rate_mean": df["tx_rate"].mean(),
        "tx_attempt_rate_mean": df["tx_attempt_rate"].mean(),
        "bits_mean": df["bits_used"].mean(),
        "bits_std": df["bits_used"].std(),
        "energy_mean": df["energy"].mean(),
        "fail_rate": df["failed"].mean(),
        "avg_qerr": df["avg_qerr"].mean(),
    }


def run_experiments(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    base = SimConfig()

    delta_list = np.logspace(-2, 0.5, num=9)
    pareto_rows = []
    for delta in delta_list:
        cfg = SimConfig(**{**base.__dict__, "delta": delta})
        _, summ = monte_carlo_single(cfg, "event", runs=30)
        pareto_rows.append(
            {
                "delta": delta,
                "J_mean": summ["J_mean"],
                "J_std": summ["J_std"],
                "N_tx_mean": summ["N_tx_mean"],
                "N_tx_std": summ["N_tx_std"],
                "tx_rate_mean": summ["tx_rate_mean"],
                "bits_mean": summ["bits_mean"],
            }
        )
    df_pareto = pd.DataFrame(pareto_rows)
    df_pareto.to_csv(output_dir / "exp_pareto_tradeoff.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        df_pareto["N_tx_mean"],
        df_pareto["J_mean"],
        yerr=df_pareto["J_std"],
        fmt="o-",
        capsize=3,
        label="Event-triggered",
    )
    ax.set_xlabel("Average delivered updates")
    ax.set_ylabel("Quadratic cost J")
    ax.set_title("Performance--communication trade-off")
    ax.legend()
    ax.grid(True)
    save_figure(fig, output_dir / "fig_pareto_tradeoff.png")

    time_delta_list = [
        float(delta_list[0]),
        float(delta_list[len(delta_list) // 2]),
        float(delta_list[-1]),
    ]
    time_rows = []
    fig, axes = plt.subplots(len(time_delta_list), 1, figsize=(8, 8), sharex=True)
    if len(time_delta_list) == 1:
        axes = [axes]
    for ax, delta in zip(axes, time_delta_list):
        cfg = SimConfig(**{**base.__dict__, "delta": delta})
        df, _ = simulate_single_uav(cfg, "event")
        df = df.copy()
        df["delta"] = delta
        time_rows.append(df[["k", "delta", "tilde_norm", "x_norm", "tx"]])

        ax.plot(df["k"], df["tilde_norm"], label="||tilde_x||", color="tab:blue")
        ax.plot(df["k"], df["x_norm"], label="||x||", linestyle="--", color="tab:orange")
        ax.axhline(delta, color="k", linestyle=":", linewidth=1.0, label="delta")
        ax.scatter(
            df.loc[df["tx"] == 1, "k"],
            [0.0] * int(df["tx"].sum()),
            s=18,
            alpha=0.6,
            marker="|",
            color="tab:green",
            label="tx",
        )
        ax.set_ylabel("norm")
        ax.set_title(f"time response (delta={delta:.3f})")
        ax.grid(True)
    axes[-1].set_xlabel("time step k")
    axes[0].legend(loc="upper right", ncol=2)
    save_figure(fig, output_dir / "fig_time_response.png")

    if time_rows:
        df_time = pd.concat(time_rows, ignore_index=True)
        df_time.to_csv(output_dir / "exp_time_response.csv", index=False)

    bits_list = [4, 6, 8, 10, 12]
    rows = []
    for b in bits_list:
        cfg = SimConfig(**{**base.__dict__, "bits_per_value": b, "delta": 0.5})
        _, summ = monte_carlo_single(cfg, "event", runs=30)
        rows.append({"bits_per_value": b, **summ})
    df_q = pd.DataFrame(rows)
    df_q.to_csv(output_dir / "exp_quant_tradeoff.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        df_q["bits_mean"],
        df_q["rms_mean"],
        yerr=df_q["rms_std"],
        fmt="o-",
        capsize=3,
        color="tab:purple",
    )
    ax.set_xlabel("Average bits used (total)")
    ax.set_ylabel("RMS tracking error (m)")
    ax.set_title("Rate--distortion--control trade-off (quantization)")
    ax.grid(True)
    save_figure(fig, output_dir / "fig_quant_tradeoff.png")

    budget_list = [200_000, 500_000, 1_000_000, 2_000_000]
    budget_rows = []
    for budget in budget_list:
        cfg = SimConfig(**{**base.__dict__, "bit_budget_total": budget, "delta": 0.5})
        _, summ = monte_carlo_single(cfg, "event", runs=30)
        budget_rows.append(
            {
                "bit_budget_total": budget,
                "J_mean": summ["J_mean"],
                "J_std": summ["J_std"],
                "bits_mean": summ["bits_mean"],
                "rms_mean": summ["rms_mean"],
                "N_tx_mean": summ["N_tx_mean"],
            }
        )
    df_budget = pd.DataFrame(budget_rows)
    df_budget.to_csv(output_dir / "exp_budget_tradeoff.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        df_budget["bits_mean"],
        df_budget["J_mean"],
        yerr=df_budget["J_std"],
        fmt="o-",
        capsize=3,
        color="tab:red",
    )
    ax.set_xlabel("Average bits used (total)")
    ax.set_ylabel("Quadratic cost J")
    ax.set_title("Budget robustness (event-triggered)")
    ax.grid(True)
    save_figure(fig, output_dir / "fig_budget_tradeoff.png")

    p_bad_list = [0.3, 0.5, 0.7]
    burst_list = [(0.01, 0.2), (0.02, 0.1), (0.05, 0.05)]
    rob_rows = []
    for p_bad in p_bad_list:
        for (p_g2b, p_b2g) in burst_list:
            cfg = SimConfig(
                **{
                    **base.__dict__,
                    "p_loss_bad": p_bad,
                    "p_g2b": p_g2b,
                    "p_b2g": p_b2g,
                    "delta": 0.5,
                    "bits_per_value": 8,
                }
            )
            _, summ = monte_carlo_single(cfg, "event", runs=30)
            rob_rows.append({"p_loss_bad": p_bad, "p_g2b": p_g2b, "p_b2g": p_b2g, **summ})
    df_rob = pd.DataFrame(rob_rows)
    df_rob.to_csv(output_dir / "exp_markov_robustness.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = df_rob[df_rob["p_loss_bad"] == 0.5].copy()
    sub["burst_len_proxy"] = 1.0 / (sub["p_b2g"] + 1e-12)
    for (p_g2b, p_b2g) in burst_list:
        s2 = sub[(sub["p_g2b"] == p_g2b) & (sub["p_b2g"] == p_b2g)]
        if len(s2) == 0:
            continue
        ax.scatter(
            s2["burst_len_proxy"],
            s2["rms_mean"],
            label=f"g2b={p_g2b}, b2g={p_b2g}",
            s=50,
        )
    ax.set_xlabel("Bad-state burst length proxy (1/p_b2g)")
    ax.set_ylabel("RMS tracking error (m)")
    ax.set_title("Robustness under bursty Markov losses (p_bad=0.5)")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir / "fig_markov_robustness.png")

    compare_rows = []
    compare_delta_list = [0.1, 0.5, 1.0]
    for delta in compare_delta_list:
        cfg_event = SimConfig(**{**base.__dict__, "delta": delta})
        _, summ_event = monte_carlo_single(cfg_event, "event", runs=30)

        target_updates = max(1.0, summ_event["N_tx_mean"])
        period_updates = max(1, int(round(cfg_event.T_steps / target_updates)))
        _, summ_period_updates = monte_carlo_single(
            cfg_event, "periodic", runs=30, periodic_M=period_updates
        )
        compare_rows.append(
            {
                "delta": delta,
                "match": "updates",
                "event_J": summ_event["J_mean"],
                "event_J_std": summ_event["J_std"],
                "event_bits": summ_event["bits_mean"],
                "event_bits_std": summ_event["bits_std"],
                "event_N_tx": summ_event["N_tx_mean"],
                "periodic_M": period_updates,
                "periodic_J": summ_period_updates["J_mean"],
                "periodic_J_std": summ_period_updates["J_std"],
                "periodic_bits": summ_period_updates["bits_mean"],
                "periodic_bits_std": summ_period_updates["bits_std"],
                "periodic_N_tx": summ_period_updates["N_tx_mean"],
            }
        )

        bits_per_packet = cfg_event.bits_per_packet_overhead + 4 * cfg_event.bits_per_value
        target_bits = max(bits_per_packet, summ_event["bits_mean"])
        target_attempts = target_bits / bits_per_packet
        period_bits = max(1, int(round(cfg_event.T_steps / max(1.0, target_attempts))))
        _, summ_period_bits = monte_carlo_single(
            cfg_event, "periodic", runs=30, periodic_M=period_bits
        )
        compare_rows.append(
            {
                "delta": delta,
                "match": "bits",
                "event_J": summ_event["J_mean"],
                "event_J_std": summ_event["J_std"],
                "event_bits": summ_event["bits_mean"],
                "event_bits_std": summ_event["bits_std"],
                "event_N_tx": summ_event["N_tx_mean"],
                "periodic_M": period_bits,
                "periodic_J": summ_period_bits["J_mean"],
                "periodic_J_std": summ_period_bits["J_std"],
                "periodic_bits": summ_period_bits["bits_mean"],
                "periodic_bits_std": summ_period_bits["bits_std"],
                "periodic_N_tx": summ_period_bits["N_tx_mean"],
            }
        )

    df_compare = pd.DataFrame(compare_rows)
    df_compare.to_csv(output_dir / "exp_periodic_comparison.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)
    for match, linestyle in [("updates", "-"), ("bits", "--")]:
        sub = df_compare[df_compare["match"] == match].sort_values("delta")
        axes[0].errorbar(
            sub["delta"],
            sub["event_J"],
            yerr=sub["event_J_std"],
            marker="o",
            linestyle=linestyle,
            capsize=3,
            label=f"Event ({match})",
        )
        axes[0].errorbar(
            sub["delta"],
            sub["periodic_J"],
            yerr=sub["periodic_J_std"],
            marker="s",
            linestyle=linestyle,
            capsize=3,
            label=f"Periodic ({match})",
        )
        axes[1].errorbar(
            sub["delta"],
            sub["event_bits"],
            yerr=sub["event_bits_std"],
            marker="o",
            linestyle=linestyle,
            capsize=3,
            label=f"Event ({match})",
        )
        axes[1].errorbar(
            sub["delta"],
            sub["periodic_bits"],
            yerr=sub["periodic_bits_std"],
            marker="s",
            linestyle=linestyle,
            capsize=3,
            label=f"Periodic ({match})",
        )
    axes[0].set_ylabel("Quadratic cost J")
    axes[1].set_ylabel("Average bits used (total)")
    axes[1].set_xlabel("Event-trigger threshold delta")
    axes[0].set_title("Event vs periodic baselines")
    axes[0].legend(ncol=2, fontsize=9)
    save_figure(fig, output_dir / "fig_periodic_comparison.png")

    baseline_delta = 0.5
    cfg_event = SimConfig(**{**base.__dict__, "delta": baseline_delta})
    _, summ_event = monte_carlo_single(cfg_event, "event", runs=30)
    target_updates = max(1.0, summ_event["N_tx_mean"])
    periodic_M = max(1, int(round(cfg_event.T_steps / target_updates)))
    random_q = min(1.0, max(0.01, target_updates / cfg_event.T_steps))

    base_rows = []
    for policy, kwargs in [
        ("event", {}),
        ("periodic", {"periodic_M": periodic_M}),
        ("random", {"random_q": random_q}),
    ]:
        _, s = monte_carlo_single(cfg_event, policy, runs=30, **kwargs)
        base_rows.append(
            {
                "policy": policy,
                "delta": baseline_delta,
                "periodic_M": periodic_M if policy == "periodic" else np.nan,
                "random_q": random_q if policy == "random" else np.nan,
                **s,
            }
        )
    df_base = pd.DataFrame(base_rows)
    df_base.to_csv(output_dir / "exp_single_uav_baselines.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    labels = df_base["policy"].str.capitalize()
    x = np.arange(len(df_base))
    axes[0].bar(x, df_base["J_mean"], yerr=df_base["J_std"], capsize=3, color="tab:blue")
    axes[0].set_ylabel("Quadratic cost J")
    axes[0].set_title("Baseline J (matched updates)")
    axes[1].bar(x, df_base["rms_mean"], yerr=df_base["rms_std"], capsize=3, color="tab:green")
    axes[1].set_ylabel("RMS tracking error (m)")
    axes[1].set_title("Baseline RMS (matched updates)")
    axes[2].bar(x, df_base["bits_mean"], yerr=df_base["bits_std"], capsize=3, color="tab:orange")
    axes[2].set_ylabel("Average bits used (total)")
    axes[2].set_title("Baseline bits (matched updates)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y")
    save_figure(fig, output_dir / "fig_single_uav_baselines.png")

    metrics = [
        ("J_mean", "J (lower better)"),
        ("rms_mean", "RMS (lower better)"),
        ("bits_mean", "Bits (lower better)"),
        ("tx_rate_mean", "Tx rate (lower better)"),
        ("fail_rate", "Fail rate (lower better)"),
    ]
    vals = np.array([df_base[m[0]].values for m in metrics])
    min_v = vals.min(axis=1, keepdims=True)
    max_v = vals.max(axis=1, keepdims=True)
    norm = (vals - min_v) / (max_v - min_v + 1e-12)
    radar = 1.0 - norm
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.r_[angles, angles[0]]
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, polar=True)
    for idx, row in enumerate(df_base.itertuples(index=False)):
        vals_r = np.r_[radar[:, idx], radar[0, idx]]
        ax.plot(angles, vals_r, label=str(row.policy).capitalize())
        ax.fill(angles, vals_r, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_yticklabels([])
    ax.set_title("Single-UAV multi-metric comparison (normalized)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    save_figure(fig, output_dir / "fig_single_uav_radar.png")

    heat_delta = np.logspace(-2, 0.5, num=6)
    heat_pbad = [0.1, 0.3, 0.5, 0.7, 0.9]
    heat_rows = []
    heat_grid = np.zeros((len(heat_pbad), len(heat_delta)))
    for i, p_bad in enumerate(heat_pbad):
        for j, delta in enumerate(heat_delta):
            cfg = SimConfig(
                **{
                    **base.__dict__,
                    "delta": float(delta),
                    "p_loss_bad": float(p_bad),
                }
            )
            _, summ = monte_carlo_single(cfg, "event", runs=20)
            heat_grid[i, j] = summ["rms_mean"]
            heat_rows.append(
                {
                    "p_loss_bad": float(p_bad),
                    "delta": float(delta),
                    "rms_mean": summ["rms_mean"],
                    "rms_std": summ["rms_std"],
                    "J_mean": summ["J_mean"],
                    "J_std": summ["J_std"],
                    "bits_mean": summ["bits_mean"],
                    "bits_std": summ["bits_std"],
                }
            )
    df_heat = pd.DataFrame(heat_rows)
    df_heat.to_csv(output_dir / "exp_sensitivity_heatmap.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(heat_grid, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(heat_delta)))
    ax.set_xticklabels([f"{d:.2f}" for d in heat_delta])
    ax.set_yticks(np.arange(len(heat_pbad)))
    ax.set_yticklabels([f"{p:.1f}" for p in heat_pbad])
    ax.set_xlabel("Event-trigger threshold delta")
    ax.set_ylabel("p_loss_bad")
    ax.set_title("Sensitivity: RMS vs delta and p_loss_bad")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RMS tracking error (m)")
    save_figure(fig, output_dir / "fig_sensitivity_heatmap.png")

    print(
        "Saved figures & CSVs:",
        "fig_pareto_tradeoff.png, fig_time_response.png, fig_quant_tradeoff.png, fig_budget_tradeoff.png, fig_markov_robustness.png, fig_periodic_comparison.png, fig_single_uav_baselines.png, fig_single_uav_radar.png, fig_sensitivity_heatmap.png",
        "exp_pareto_tradeoff.csv, exp_time_response.csv, exp_quant_tradeoff.csv, exp_budget_tradeoff.csv, exp_markov_robustness.csv, exp_periodic_comparison.csv, exp_single_uav_baselines.csv, exp_sensitivity_heatmap.csv",
    )
