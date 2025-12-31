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
            "lines.markersize": 7,
        }
    )


def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def pareto_front(x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
    order = np.argsort(x_vals)
    x_sorted = x_vals[order]
    y_sorted = y_vals[order]
    keep = []
    best_y = np.inf
    for x, y in zip(x_sorted, y_sorted):
        if y < best_y:
            keep.append((x, y))
            best_y = y
    if not keep:
        return np.empty((0, 2))
    return np.array(keep)


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
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()
    apply_plot_style()
    base = SimConfig()

    delta_list = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0])
    pareto_rows = []
    pareto_runs = []
    for delta in delta_list:
        cfg = SimConfig(**{**base.__dict__, "delta": delta})
        df_runs, summ = monte_carlo_single(cfg, "event", runs=30)
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
        df_runs = df_runs.copy()
        df_runs["delta"] = delta
        pareto_runs.append(df_runs)
    df_pareto = pd.DataFrame(pareto_rows)
    df_pareto.to_csv(output_dir / "exp_pareto_tradeoff.csv", index=False)
    df_pareto_runs = pd.concat(pareto_runs, ignore_index=True)
    df_pareto_runs.to_csv(output_dir / "exp_pareto_tradeoff_runs.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    df_plot = df_pareto.sort_values("N_tx_mean")
    x_vals = df_plot["N_tx_mean"].to_numpy()
    y_vals = df_plot["J_mean"].to_numpy()
    color = "tab:blue"
    ax.plot(
        x_vals,
        y_vals,
        linestyle="-",
        marker="o",
        color=color,
        markerfacecolor=color,
        markeredgecolor=color,
    )
    ax.set_xlabel("Communication usage: Number of updates N_tx")
    ax.set_ylabel("Control performance: Quadratic cost J")
    if y_vals.max() / max(y_vals.min(), 1e-9) > 10.0:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.6, color="0.8")
    mid_idx = len(x_vals) // 2
    ax.annotate(
        "Balanced operating point",
        xy=(x_vals[mid_idx], y_vals[mid_idx]),
        xytext=(10, 12),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8),
    )
    save_figure(fig, output_dir / "fig_pareto_tradeoff.png")

    grouped = df_pareto_runs.groupby("delta", as_index=False)
    q = grouped.agg(
        J_q25=("J_cost", lambda s: np.quantile(s, 0.25)),
        J_med=("J_cost", "median"),
        J_q75=("J_cost", lambda s: np.quantile(s, 0.75)),
        bits_q25=("bits_used", lambda s: np.quantile(s, 0.25)),
        bits_med=("bits_used", "median"),
        bits_q75=("bits_used", lambda s: np.quantile(s, 0.75)),
    )
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    axes[0].fill_between(q["delta"], q["J_q25"], q["J_q75"], alpha=0.2)
    axes[0].plot(q["delta"], q["J_med"], marker="o")
    axes[0].set_ylabel("Quadratic cost J")
    axes[0].set_title("Event-triggered quantile bands (IQR)")
    axes[0].grid(True)
    axes[1].fill_between(q["delta"], q["bits_q25"], q["bits_q75"], alpha=0.2, color="tab:orange")
    axes[1].plot(q["delta"], q["bits_med"], marker="o", color="tab:orange")
    axes[1].set_xlabel("Event-trigger threshold delta")
    axes[1].set_ylabel("Bits used (total)")
    axes[1].grid(True)
    axes[1].set_xscale("log")
    save_figure(fig, output_dir / "fig_quantile_band.png")

    delta_mid = float(delta_list[len(delta_list) // 2])
    cfg = SimConfig(**{**base.__dict__, "delta": delta_mid})
    df, _ = simulate_single_uav(cfg, "event")
    df = df.copy()
    df["delta"] = delta_mid

    time_rows = [df[["k", "delta", "tilde_norm", "x_norm", "tx"]]]

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    ax_err, ax_state = axes
    ax_err.plot(df["k"], df["tilde_norm"], linestyle="-")
    ax_err.axhline(delta_mid, linestyle="--", linewidth=1.0)
    trigger_steps = df.loc[df["tx"] == 1, "k"].to_numpy()
    for k in trigger_steps:
        ax_err.axvline(k, linestyle="--", linewidth=0.7, color="0.7")
    ax_err.set_ylabel(r"$\|\tilde{x}_k\|_2$")
    ax_err.set_title("(a) Prediction error (grow-and-reset)")
    ax_err.grid(True, linestyle="--", linewidth=0.6, color="0.8")

    ax_state.plot(df["k"], df["x_norm"], linestyle="-")
    ax_state.set_xlabel(r"Time step $k$")
    ax_state.set_ylabel(r"$\|x_k\|_2$")
    ax_state.set_title("(b) Closed-loop state response")
    ax_state.grid(True, linestyle="--", linewidth=0.6, color="0.8")

    save_figure(fig, output_dir / "fig_time_response.pdf")

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

    robust_delta_list = delta_list[:8]
    robust_M_list = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]
    robust_rows = []
    for delta in robust_delta_list:
        cfg_event = SimConfig(
            **{
                **base.__dict__,
                "delta": float(delta),
                "sigma_v": 0.2,
                "bits_per_value": 8,
                "p_loss_good": 0.05,
                "p_loss_bad": 0.5,
                "p_g2b": 0.02,
                "p_b2g": 0.1,
            }
        )
        _, summ_event = monte_carlo_single(cfg_event, "event", runs=30)
        robust_rows.append(
            {
                "scheme": "ET",
                "param": float(delta),
                "bits_mean": summ_event["bits_mean"],
                "J_mean": summ_event["J_mean"],
            }
        )
    for M in robust_M_list:
        cfg_per = SimConfig(
            **{
                **base.__dict__,
                "sigma_v": 0.2,
                "bits_per_value": 8,
                "p_loss_good": 0.05,
                "p_loss_bad": 0.5,
                "p_g2b": 0.02,
                "p_b2g": 0.1,
            }
        )
        _, summ_periodic = monte_carlo_single(cfg_per, "periodic", runs=30, periodic_M=M)
        robust_rows.append(
            {
                "scheme": "PER",
                "param": int(M),
                "bits_mean": summ_periodic["bits_mean"],
                "J_mean": summ_periodic["J_mean"],
            }
        )
    df_robust = pd.DataFrame(robust_rows)
    df_robust.to_csv(output_dir / "exp_robustness_summary.csv", index=False)
    df_et = df_robust[df_robust["scheme"] == "ET"].sort_values("bits_mean")
    df_per = df_robust[df_robust["scheme"] == "PER"].sort_values("bits_mean")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        df_et["bits_mean"],
        df_et["J_mean"],
        linestyle="-",
        marker="o",
        label="Event-triggered (ET)",
    )
    ax.plot(
        df_per["bits_mean"],
        df_per["J_mean"],
        linestyle="--",
        marker="o",
        markerfacecolor="none",
        label="Periodic (PER)",
    )
    ax.set_xlabel(r"Average bits used $B(0{:}T)$")
    ax.set_ylabel(r"Quadratic cost $J$")
    ax.grid(True, linestyle="--", linewidth=0.6, color="0.8")
    ax.legend()
    save_figure(fig, output_dir / "fig_robustness_summary.pdf")

    candidate_M = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50]
    per_rows = []
    for M in candidate_M:
        _, summ_periodic = monte_carlo_single(base, "periodic", runs=30, periodic_M=M)
        per_rows.append(
            {
                "M": M,
                "J_mean": summ_periodic["J_mean"],
                "bits_mean": summ_periodic["bits_mean"],
                "updates_mean": summ_periodic["N_tx_mean"],
            }
        )
    df_per = pd.DataFrame(per_rows)

    et_rows = []
    matched_rows = []
    compare_delta_list = delta_list
    for delta in compare_delta_list:
        cfg_event = SimConfig(**{**base.__dict__, "delta": float(delta)})
        _, summ_event = monte_carlo_single(cfg_event, "event", runs=30)
        et_rows.append(
            {
                "delta": float(delta),
                "J_mean": summ_event["J_mean"],
                "bits_mean": summ_event["bits_mean"],
                "updates_mean": summ_event["N_tx_mean"],
            }
        )
        diff = (df_per["bits_mean"] - summ_event["bits_mean"]).abs()
        best_idx = int(diff.idxmin())
        per_match = df_per.loc[best_idx]
        matched_rows.append(
            {
                "delta": float(delta),
                "M_star": int(per_match["M"]),
                "bits_et": float(summ_event["bits_mean"]),
                "bits_per": float(per_match["bits_mean"]),
                "J_et": float(summ_event["J_mean"]),
                "J_per": float(per_match["J_mean"]),
                "updates_et": float(summ_event["N_tx_mean"]),
                "updates_per": float(per_match["updates_mean"]),
            }
        )

    df_et = pd.DataFrame(et_rows)
    df_et.to_csv(output_dir / "exp_periodic_et.csv", index=False)
    df_per.to_csv(output_dir / "exp_periodic_per.csv", index=False)
    df_matched = pd.DataFrame(matched_rows)
    df_matched.to_csv(output_dir / "exp_periodic_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    df_plot = df_matched.sort_values("bits_et")
    ax.plot(
        df_plot["bits_et"],
        df_plot["J_et"],
        linestyle="-",
        marker="o",
        color="tab:blue",
        markerfacecolor="tab:blue",
        markeredgecolor="tab:blue",
        label="Event-triggered (ET)",
    )
    ax.plot(
        df_plot["bits_per"],
        df_plot["J_per"],
        linestyle="--",
        marker="o",
        color="tab:orange",
        markerfacecolor="white",
        markeredgecolor="tab:orange",
        label="Periodic (PER)",
    )
    ax.set_xlabel(r"Average bits used $B(0{:}T)$")
    ax.set_ylabel(r"Quadratic cost $J$")
    ax.grid(True, linestyle="--", linewidth=0.6, color="0.8")
    ax.legend()
    first = df_plot.iloc[0]
    ax.annotate(
        "ET dominates PER under matched budget",
        xy=(first["bits_et"], first["J_et"]),
        xytext=(12, 12),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8),
    )
    save_figure(fig, output_dir / "fig_periodic_comparison.pdf")

    baseline_delta = 0.5
    cfg_event = SimConfig(**{**base.__dict__, "delta": baseline_delta})
    _, summ_event = monte_carlo_single(cfg_event, "event", runs=30)
    target_updates = max(1.0, summ_event["N_tx_mean"])
    periodic_M = max(1, int(round(cfg_event.T_steps / target_updates)))
    random_q = min(1.0, max(0.01, target_updates / cfg_event.T_steps))

    base_rows = []
    base_runs = []
    for policy, kwargs in [
        ("event", {}),
        ("periodic", {"periodic_M": periodic_M}),
        ("random", {"random_q": random_q}),
    ]:
        df_runs, s = monte_carlo_single(cfg_event, policy, runs=30, **kwargs)
        df_runs = df_runs.copy()
        df_runs["policy"] = policy
        base_runs.append(df_runs)
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
    df_runs = pd.concat(base_runs, ignore_index=True)
    df_runs.to_csv(output_dir / "exp_single_uav_baseline_runs.csv", index=False)

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

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    policies = ["event", "periodic", "random"]
    labels = [p.capitalize() for p in policies]
    data_J = [df_runs.loc[df_runs["policy"] == p, "J_cost"] for p in policies]
    data_rms = [df_runs.loc[df_runs["policy"] == p, "rms_err"] for p in policies]
    data_bits = [df_runs.loc[df_runs["policy"] == p, "bits_used"] for p in policies]
    axes[0].boxplot(data_J, labels=labels, patch_artist=True)
    axes[0].set_ylabel("Quadratic cost J")
    axes[0].set_title("Distribution of J")
    axes[1].boxplot(data_rms, labels=labels, patch_artist=True)
    axes[1].set_ylabel("RMS tracking error (m)")
    axes[1].set_title("Distribution of RMS")
    axes[2].boxplot(data_bits, labels=labels, patch_artist=True)
    axes[2].set_ylabel("Bits used (total)")
    axes[2].set_title("Distribution of bits")
    for ax in axes:
        ax.grid(axis="y")
    save_figure(fig, output_dir / "fig_single_uav_boxplot.png")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for policy, marker in [("event", "o"), ("periodic", "s"), ("random", "^")]:
        sub = df_runs[df_runs["policy"] == policy]
        ax.scatter(
            sub["bits_used"],
            sub["J_cost"],
            alpha=0.6,
            label=policy.capitalize(),
            marker=marker,
        )
        front = pareto_front(sub["bits_used"].values, sub["J_cost"].values)
        if len(front) > 0:
            ax.plot(front[:, 0], front[:, 1], linewidth=2)
    ax.set_xlabel("Bits used (total)")
    ax.set_ylabel("Quadratic cost J")
    ax.set_title("Cost vs communication (per-run scatter)")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir / "fig_single_uav_scatter.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for policy in ["event", "periodic", "random"]:
        sub = df_runs[df_runs["policy"] == policy]
        for ax, col, title in [
            (axes[0], "J_cost", "CDF of J"),
            (axes[1], "rms_err", "CDF of RMS"),
        ]:
            vals = np.sort(sub[col].values)
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, label=policy.capitalize())
            ax.set_title(title)
            ax.grid(True)
    axes[0].set_xlabel("Quadratic cost J")
    axes[1].set_xlabel("RMS tracking error (m)")
    axes[0].set_ylabel("CDF")
    axes[0].legend()
    save_figure(fig, output_dir / "fig_single_uav_cdf.png")

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
        "fig_pareto_tradeoff.png, fig_quantile_band.png, fig_time_response.pdf, fig_quant_tradeoff.png, fig_budget_tradeoff.png, fig_markov_robustness.png, fig_robustness_summary.pdf, fig_periodic_comparison.pdf, fig_single_uav_baselines.png, fig_single_uav_boxplot.png, fig_single_uav_scatter.png, fig_single_uav_cdf.png, fig_single_uav_radar.png, fig_sensitivity_heatmap.png",
        "exp_pareto_tradeoff.csv, exp_pareto_tradeoff_runs.csv, exp_time_response.csv, exp_quant_tradeoff.csv, exp_budget_tradeoff.csv, exp_markov_robustness.csv, exp_periodic_comparison.csv, exp_single_uav_baselines.csv, exp_single_uav_baseline_runs.csv, exp_sensitivity_heatmap.csv",
    )
