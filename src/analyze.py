"""
Analysis: Phase transition detection in alignment resistance scaling.
Loads experiment results and performs:
1. Descriptive statistics per model scale
2. Curve fitting: power-law vs. sigmoid (phase transition)
3. Discontinuity detection via first derivatives
4. Schaeffer et al. analysis: continuous vs. discrete metrics
5. Visualization
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import bootstrap, pearsonr
from pathlib import Path

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.figsize': (10, 6), 'figure.dpi': 150,
})


# ── Curve Models ─────────────────────────────────────────────────────────────
def power_law(x, a, b, c):
    """Power law: y = a * x^b + c"""
    return a * np.power(x, b) + c

def sigmoid(x, L, k, x0, b):
    """Sigmoid: y = L / (1 + exp(-k*(x - x0))) + b"""
    return L / (1.0 + np.exp(-k * (x - x0))) + b

def linear(x, a, b):
    return a * x + b


# ── Analysis Functions ───────────────────────────────────────────────────────
def compute_model_stats(results_data):
    """Compute per-model aggregate statistics from experiment results."""
    stats = []
    for model_result in results_data:
        scores = model_result.get("scores", [])
        if not scores:
            continue
        safety_scores = [s["safety_score"] for s in scores]
        refused = [s["refused"] for s in scores]

        stats.append({
            "model": model_result["model"],
            "label": model_result["label"],
            "params": model_result["params_approx"],
            "log_params": np.log10(model_result["params_approx"]),
            "mean_safety": np.mean(safety_scores),
            "std_safety": np.std(safety_scores),
            "median_safety": np.median(safety_scores),
            "refusal_rate": np.mean(refused),
            "n_samples": len(scores),
        })
    return stats


def compute_erosion_stats(results_data):
    """Compute per-model, per-n_shots statistics for erosion experiment."""
    stats = []
    for model_result in results_data:
        for n_shots_str, scores in model_result.get("scores_by_shots", {}).items():
            safety_scores = [s["safety_score"] for s in scores]
            refused = [s["refused"] for s in scores]
            stats.append({
                "model": model_result["model"],
                "label": model_result["label"],
                "params": model_result["params_approx"],
                "log_params": np.log10(model_result["params_approx"]),
                "n_shots": int(n_shots_str),
                "mean_safety": np.mean(safety_scores),
                "std_safety": np.std(safety_scores),
                "refusal_rate": np.mean(refused),
            })
    return stats


def fit_scaling_curves(log_params, metric_values):
    """Fit power-law and sigmoid to data, return R² for each."""
    x = np.array(log_params)
    y = np.array(metric_values)

    results = {}

    # Power law fit (in log space, so effectively linear)
    try:
        popt_lin, _ = curve_fit(linear, x, y, maxfev=5000)
        y_pred_lin = linear(x, *popt_lin)
        ss_res = np.sum((y - y_pred_lin)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results["linear"] = {"params": popt_lin.tolist(), "r2": r2_lin, "y_pred": y_pred_lin}
    except Exception as e:
        results["linear"] = {"error": str(e), "r2": -1}

    # Sigmoid fit
    try:
        # Initial guess: L=range of y, k=1, x0=midpoint of x, b=min(y)
        p0 = [np.ptp(y), 2.0, np.mean(x), np.min(y)]
        bounds = ([0, 0.01, x.min()-2, -20], [20, 50, x.max()+2, 20])
        popt_sig, _ = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds, maxfev=10000)
        y_pred_sig = sigmoid(x, *popt_sig)
        ss_res = np.sum((y - y_pred_sig)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_sig = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results["sigmoid"] = {"params": popt_sig.tolist(), "r2": r2_sig, "y_pred": y_pred_sig}
    except Exception as e:
        results["sigmoid"] = {"error": str(e), "r2": -1}

    return results


def compute_derivatives(log_params, values):
    """Compute numerical first derivative to detect discontinuities."""
    x = np.array(log_params)
    y = np.array(values)
    if len(x) < 2:
        return [], []
    dx = np.diff(x)
    dy = np.diff(y)
    derivatives = dy / dx
    midpoints = (x[:-1] + x[1:]) / 2
    return midpoints.tolist(), derivatives.tolist()


# ── Plotting Functions ───────────────────────────────────────────────────────
def plot_jailbreak_scaling(stats, fits):
    """Plot jailbreak resistance vs. model scale with curve fits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    log_params = [s["log_params"] for s in stats]
    labels = [s["label"] for s in stats]

    # Plot 1: Continuous metric (mean safety score)
    ax = axes[0]
    mean_safety = [s["mean_safety"] for s in stats]
    std_safety = [s["std_safety"] for s in stats]
    ax.errorbar(log_params, mean_safety, yerr=std_safety, fmt='o-', capsize=5,
                markersize=8, linewidth=2, color='steelblue', label='Data')

    # Add fit curves
    if "linear" in fits and "r2" in fits["linear"] and fits["linear"]["r2"] > -1:
        x_fine = np.linspace(min(log_params)-0.2, max(log_params)+0.2, 100)
        y_lin = linear(x_fine, *fits["linear"]["params"])
        ax.plot(x_fine, y_lin, '--', color='orange', alpha=0.7,
                label=f'Linear (R²={fits["linear"]["r2"]:.3f})')
    if "sigmoid" in fits and "r2" in fits["sigmoid"] and fits["sigmoid"]["r2"] > -1:
        x_fine = np.linspace(min(log_params)-0.2, max(log_params)+0.2, 100)
        y_sig = sigmoid(x_fine, *fits["sigmoid"]["params"])
        ax.plot(x_fine, y_sig, '-', color='red', alpha=0.7,
                label=f'Sigmoid (R²={fits["sigmoid"]["r2"]:.3f})')

    ax.set_xlabel("log₁₀(Parameters)")
    ax.set_ylabel("Mean Safety Score (0-10)")
    ax.set_title("Continuous Metric: Safety Score vs. Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(log_params)
    ax.set_xticklabels([l.split('(')[0].strip() for l in labels], rotation=30, ha='right', fontsize=9)

    # Plot 2: Discrete metric (refusal rate)
    ax = axes[1]
    refusal_rates = [s["refusal_rate"] for s in stats]
    ax.bar(range(len(stats)), refusal_rates, color='coral', alpha=0.8)
    ax.plot(range(len(stats)), refusal_rates, 'o-', color='darkred', linewidth=2, markersize=8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Refusal Rate")
    ax.set_title("Discrete Metric: Refusal Rate vs. Scale")
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([l.split('(')[0].strip() for l in labels], rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp1_jailbreak_scaling.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'exp1_jailbreak_scaling.png'}")


def plot_jailbreak_by_template(stats_by_template):
    """Plot jailbreak resistance broken down by template type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    templates = list(stats_by_template.keys())
    colors = {'direct': 'steelblue', 'roleplay': 'coral', 'hypothetical': 'seagreen'}

    for template in templates:
        tdata = stats_by_template[template]
        ax.plot([d["log_params"] for d in tdata],
                [d["mean_safety"] for d in tdata],
                'o-', label=template, color=colors.get(template, 'gray'),
                linewidth=2, markersize=8)

    ax.set_xlabel("log₁₀(Parameters)")
    ax.set_ylabel("Mean Safety Score")
    ax.set_title("Jailbreak Resistance by Template Type")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp1_by_template.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'exp1_by_template.png'}")


def plot_erosion(erosion_stats):
    """Plot alignment erosion: safety score vs. n_shots for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = sorted(set(s["model"] for s in erosion_stats),
                   key=lambda m: next(s["params"] for s in erosion_stats if s["model"]==m))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(models)))

    for idx, model in enumerate(models):
        mdata = sorted([s for s in erosion_stats if s["model"]==model],
                      key=lambda s: s["n_shots"])
        n_shots = [d["n_shots"] for d in mdata]
        safety = [d["mean_safety"] for d in mdata]
        label_short = mdata[0]["label"].split("(")[0].strip()
        ax.plot(n_shots, safety, 'o-', label=label_short, color=colors[idx],
                linewidth=2, markersize=8)

    ax.set_xlabel("Number of Misaligned Few-Shot Examples")
    ax.set_ylabel("Mean Safety Score")
    ax.set_title("Few-Shot Alignment Erosion Across Model Scales")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 1, 3, 5])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp2_erosion.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'exp2_erosion.png'}")


def plot_erosion_rate(erosion_stats):
    """Plot erosion RATE (slope) vs. model scale to detect phase transitions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = sorted(set(s["model"] for s in erosion_stats),
                   key=lambda m: next(s["params"] for s in erosion_stats if s["model"]==m))

    erosion_rates = []
    log_params = []

    for model in models:
        mdata = sorted([s for s in erosion_stats if s["model"]==model],
                      key=lambda s: s["n_shots"])
        if len(mdata) >= 2:
            # Erosion rate = change in safety per few-shot example
            safety_0 = next((d["mean_safety"] for d in mdata if d["n_shots"]==0), None)
            safety_5 = next((d["mean_safety"] for d in mdata if d["n_shots"]==5), None)
            if safety_0 is not None and safety_5 is not None:
                rate = (safety_0 - safety_5) / 5.0  # safety drop per example
                erosion_rates.append(rate)
                log_params.append(mdata[0]["log_params"])

    ax.plot(log_params, erosion_rates, 'o-', color='crimson', linewidth=2, markersize=10)
    ax.set_xlabel("log₁₀(Parameters)")
    ax.set_ylabel("Erosion Rate (safety drop per few-shot example)")
    ax.set_title("Alignment Erosion Rate vs. Model Scale")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp2_erosion_rate.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'exp2_erosion_rate.png'}")
    return log_params, erosion_rates


def plot_conflict(conflict_stats):
    """Plot instruction conflict resolution across scales."""
    fig, ax = plt.subplots(figsize=(10, 6))

    log_params = [s["log_params"] for s in conflict_stats]
    mean_safety = [s["mean_safety"] for s in conflict_stats]
    std_safety = [s["std_safety"] for s in conflict_stats]
    labels = [s["label"].split("(")[0].strip() for s in conflict_stats]

    ax.errorbar(log_params, mean_safety, yerr=std_safety, fmt='s-', capsize=5,
                markersize=10, linewidth=2, color='purple')
    ax.set_xlabel("log₁₀(Parameters)")
    ax.set_ylabel("Mean Safety Score Under Conflict")
    ax.set_title("Instruction Conflict Resolution vs. Model Scale")
    ax.set_xticks(log_params)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp3_conflict.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'exp3_conflict.png'}")


def plot_derivative(log_params, values, metric_name, filename):
    """Plot first derivative to detect discontinuities."""
    midpoints, derivs = compute_derivatives(log_params, values)
    if not midpoints:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Original values
    ax = axes[0]
    ax.plot(log_params, values, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.set_xlabel("log₁₀(Parameters)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs. Scale")
    ax.grid(True, alpha=0.3)

    # First derivative
    ax = axes[1]
    ax.bar(midpoints, derivs, width=0.3, color='coral', alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel("log₁₀(Parameters)")
    ax.set_ylabel(f"d({metric_name})/d(log₁₀(params))")
    ax.set_title("First Derivative (Phase Transition Indicator)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / filename}")


def plot_schaeffer_comparison(stats):
    """Schaeffer et al. analysis: compare continuous vs. discrete metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    log_params = [s["log_params"] for s in stats]
    mean_safety = [s["mean_safety"] for s in stats]
    refusal_rate = [s["refusal_rate"] for s in stats]

    # Normalize both to 0-1 for comparison
    safety_norm = [(s - min(mean_safety)) / (max(mean_safety) - min(mean_safety) + 1e-8)
                   for s in mean_safety]

    ax.plot(log_params, safety_norm, 'o-', color='steelblue', linewidth=2, markersize=8,
            label='Continuous: Safety Score (normalized)')
    ax.plot(log_params, refusal_rate, 's-', color='coral', linewidth=2, markersize=8,
            label='Discrete: Refusal Rate')

    ax.set_xlabel("log₁₀(Parameters)")
    ax.set_ylabel("Normalized Metric Value")
    ax.set_title("Schaeffer Analysis: Continuous vs. Discrete Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "schaeffer_comparison.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'schaeffer_comparison.png'}")


def plot_comprehensive_summary(jailbreak_stats, erosion_stats, conflict_stats):
    """Create a comprehensive 2x2 summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Jailbreak safety score
    ax = axes[0, 0]
    lp = [s["log_params"] for s in jailbreak_stats]
    ms = [s["mean_safety"] for s in jailbreak_stats]
    ss = [s["std_safety"] for s in jailbreak_stats]
    ax.errorbar(lp, ms, yerr=ss, fmt='o-', capsize=5, markersize=8, linewidth=2, color='steelblue')
    ax.set_xlabel("log₁₀(Parameters)")
    ax.set_ylabel("Safety Score")
    ax.set_title("A) Jailbreak Resistance")
    ax.grid(True, alpha=0.3)

    # Panel 2: Refusal rate
    ax = axes[0, 1]
    rr = [s["refusal_rate"] for s in jailbreak_stats]
    ax.plot(lp, rr, 's-', color='coral', linewidth=2, markersize=8)
    ax.set_xlabel("log₁₀(Parameters)")
    ax.set_ylabel("Refusal Rate")
    ax.set_title("B) Refusal Rate (Discrete)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 3: Erosion curves
    ax = axes[1, 0]
    models_sorted = sorted(set(s["model"] for s in erosion_stats),
                          key=lambda m: next(s["params"] for s in erosion_stats if s["model"]==m))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(models_sorted)))
    for idx, model in enumerate(models_sorted):
        mdata = sorted([s for s in erosion_stats if s["model"]==model], key=lambda s: s["n_shots"])
        ax.plot([d["n_shots"] for d in mdata], [d["mean_safety"] for d in mdata],
                'o-', color=colors[idx], linewidth=2, markersize=6,
                label=mdata[0]["label"].split("(")[0].strip())
    ax.set_xlabel("Few-Shot Misaligned Examples")
    ax.set_ylabel("Safety Score")
    ax.set_title("C) Alignment Erosion")
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # Panel 4: Conflict resolution
    ax = axes[1, 1]
    if conflict_stats:
        clp = [s["log_params"] for s in conflict_stats]
        cms = [s["mean_safety"] for s in conflict_stats]
        css = [s["std_safety"] for s in conflict_stats]
        ax.errorbar(clp, cms, yerr=css, fmt='D-', capsize=5, markersize=8, linewidth=2, color='purple')
    ax.set_xlabel("log₁₀(Parameters)")
    ax.set_ylabel("Safety Score")
    ax.set_title("D) Instruction Conflict Resolution")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Alignment Resistance Across Model Scales", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comprehensive_summary.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'comprehensive_summary.png'}")


# ── Main Analysis ────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("ANALYSIS: Phase Transition Detection")
    print("="*70)

    # Load experiment results
    with open(RESULTS_DIR / "exp1_jailbreak.json") as f:
        exp1_data = json.load(f)
    with open(RESULTS_DIR / "exp2_erosion.json") as f:
        exp2_data = json.load(f)
    with open(RESULTS_DIR / "exp3_conflict.json") as f:
        exp3_data = json.load(f)

    # ── Experiment 1 Analysis ────────────────────────────────────────────
    print("\n--- Experiment 1: Jailbreak Resistance ---")
    jailbreak_stats = compute_model_stats(exp1_data)

    # Per-template breakdown
    stats_by_template = {}
    for model_result in exp1_data:
        for template in ["direct", "roleplay", "hypothetical"]:
            scores = [s for s in model_result["scores"] if s["template"] == template]
            if template not in stats_by_template:
                stats_by_template[template] = []
            stats_by_template[template].append({
                "model": model_result["model"],
                "label": model_result["label"],
                "log_params": np.log10(model_result["params_approx"]),
                "mean_safety": np.mean([s["safety_score"] for s in scores]),
                "refusal_rate": np.mean([s["refused"] for s in scores]),
            })

    for s in jailbreak_stats:
        print(f"  {s['label']}: safety={s['mean_safety']:.2f}±{s['std_safety']:.2f}, "
              f"refusal={s['refusal_rate']:.2%}")

    # Curve fitting
    log_params = [s["log_params"] for s in jailbreak_stats]
    mean_safety = [s["mean_safety"] for s in jailbreak_stats]
    fits = fit_scaling_curves(log_params, mean_safety)

    print(f"\n  Curve Fitting Results:")
    for fit_name, fit_result in fits.items():
        if "r2" in fit_result:
            print(f"    {fit_name}: R² = {fit_result['r2']:.4f}")
        else:
            print(f"    {fit_name}: {fit_result.get('error', 'unknown error')}")

    # Phase transition test: sigmoid vs linear
    if fits.get("sigmoid", {}).get("r2", -1) > fits.get("linear", {}).get("r2", -1):
        print("  → Sigmoid fits BETTER than linear → supports phase transition")
    else:
        print("  → Linear fits better or equal → supports gradual scaling")

    # Plots
    plot_jailbreak_scaling(jailbreak_stats, fits)
    plot_jailbreak_by_template(stats_by_template)
    plot_derivative(log_params, mean_safety, "Mean Safety Score", "exp1_derivative.png")
    plot_schaeffer_comparison(jailbreak_stats)

    # ── Experiment 2 Analysis ────────────────────────────────────────────
    print("\n--- Experiment 2: Alignment Erosion ---")
    erosion_stats = compute_erosion_stats(exp2_data)

    for s in erosion_stats:
        print(f"  {s['label']} (shots={s['n_shots']}): safety={s['mean_safety']:.2f}")

    plot_erosion(erosion_stats)
    erosion_lp, erosion_rates = plot_erosion_rate(erosion_stats)
    if erosion_lp:
        plot_derivative(erosion_lp, erosion_rates, "Erosion Rate", "exp2_erosion_derivative.png")

    # ── Experiment 3 Analysis ────────────────────────────────────────────
    print("\n--- Experiment 3: Conflict Resolution ---")
    conflict_stats = compute_model_stats(exp3_data)

    for s in conflict_stats:
        print(f"  {s['label']}: safety={s['mean_safety']:.2f}±{s['std_safety']:.2f}")

    plot_conflict(conflict_stats)

    conflict_log_params = [s["log_params"] for s in conflict_stats]
    conflict_safety = [s["mean_safety"] for s in conflict_stats]
    conflict_fits = fit_scaling_curves(conflict_log_params, conflict_safety)
    print(f"\n  Conflict Curve Fitting:")
    for fit_name, fit_result in conflict_fits.items():
        if "r2" in fit_result:
            print(f"    {fit_name}: R² = {fit_result['r2']:.4f}")

    # ── Comprehensive Summary Plot ───────────────────────────────────────
    plot_comprehensive_summary(jailbreak_stats, erosion_stats, conflict_stats)

    # ── Save Analysis Results ────────────────────────────────────────────
    analysis = {
        "jailbreak": {
            "stats": jailbreak_stats,
            "curve_fits": {k: {kk: vv for kk, vv in v.items() if kk != "y_pred"}
                          for k, v in fits.items()},
            "stats_by_template": {k: v for k, v in stats_by_template.items()},
        },
        "erosion": {
            "stats": erosion_stats,
            "erosion_rates": {"log_params": erosion_lp, "rates": erosion_rates} if erosion_lp else {},
        },
        "conflict": {
            "stats": conflict_stats,
            "curve_fits": {k: {kk: vv for kk, vv in v.items() if kk != "y_pred"}
                          for k, v in conflict_fits.items()},
        },
    }

    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    print(f"\nAnalysis saved to {RESULTS_DIR / 'analysis.json'}")
    print(f"Plots saved to {PLOTS_DIR}/")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
