"""
analysis/evaluate.py

Reads all results CSVs and produces:
1. A summary accuracy table (printed + saved as CSV)
2. Accuracy bar charts by dataset and model
3. Consistency rate analysis for Self-Consistency runs

Run this after all 16 experiments are complete.

Usage:
    python analysis/evaluate.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


RESULTS_DIR = "results"
MODELS = ["gemma-2b"]
DATASETS = ["gsm8k", "csqa"]
STRATEGIES = ["standard", "fewshot", "cot", "self_consistency"]
STRATEGY_LABELS = {
    "standard": "Zero-shot Standard",
    "fewshot": "Few-shot Standard",
    "cot": "Zero-shot CoT",
    "self_consistency": "Self-Consistency CoT",
}


def load_all_results() -> pd.DataFrame:
    """
    Loads all 16 results CSVs into a single DataFrame.
    Prints a warning for any files not yet generated.
    """
    frames = []
    for dataset in DATASETS:
        for model in MODELS:
            for strategy in STRATEGIES:
                path = os.path.join(RESULTS_DIR, f"{dataset}_{model}_{strategy}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    frames.append(df)
                else:
                    print(f"[MISSING] {path}")
    if not frames:
        raise FileNotFoundError("No results found in results/. Run experiments first.")
    return pd.concat(frames, ignore_index=True)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the main results table used in the paper.
    Rows: strategy × model. Columns: dataset accuracy (+ consistency rate).
    """
    rows = []
    for strategy in STRATEGIES:
        for model in MODELS:
            row: dict = {
                "Strategy": STRATEGY_LABELS[strategy],
                "Model": model,
            }
            for dataset in DATASETS:
                subset = df[
                    (df["strategy"] == strategy)
                    & (df["model"] == model)
                    & (df["dataset"] == dataset)
                ]
                if len(subset) > 0:
                    acc = subset["correct"].mean() * 100
                    row[f"{dataset.upper()} Accuracy (%)"] = round(acc, 2)
                else:
                    row[f"{dataset.upper()} Accuracy (%)"] = None

                # Consistency rate — Self-Consistency only
                if (
                    strategy == "self_consistency"
                    and "consistency_rate" in subset.columns
                    and len(subset) > 0
                ):
                    cr = subset["consistency_rate"].mean() * 100
                    row[f"{dataset.upper()} Consistency (%)"] = round(cr, 2)

            rows.append(row)

    return pd.DataFrame(rows)


def plot_accuracy_by_dataset(df: pd.DataFrame):
    """
    Grouped bar chart: accuracy per strategy, grouped by model.
    One chart per dataset. Saved to results/.
    """
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(11, 5))

        x = np.arange(len(STRATEGIES))
        width = 0.35
        colors = ["#4C72B0", "#DD8452"]

        for i, model in enumerate(MODELS):
            accuracies = []
            for strategy in STRATEGIES:
                subset = df[
                    (df["strategy"] == strategy)
                    & (df["model"] == model)
                    & (df["dataset"] == dataset)
                ]
                acc = subset["correct"].mean() * 100 if len(subset) > 0 else 0
                accuracies.append(acc)

            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, accuracies, width, label=model, color=colors[i])

            for bar, val in zip(bars, accuracies):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xlabel("Prompting Strategy", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(
            f"Accuracy by Prompting Strategy and Model Scale\nDataset: {dataset.upper()}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [STRATEGY_LABELS[s] for s in STRATEGIES], rotation=10, ha="right"
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, 105)
        ax.legend(title="Model")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = f"results/plot_{dataset}_accuracy.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")
        plt.close()


## Added by Zaina
def plot_consistency_rates(df: pd.DataFrame):
    """
    Plot the consistency rates across datasets for different models using self-consistency strategy.
    This function filters the input dataframe for rows with 'self_consistency' strategy,
    calculates the mean consistency rate for each model-dataset combination, and generates
    a grouped bar chart showing the agreement rates across 5 samples.
    Args:
        df (pd.DataFrame): Input dataframe containing columns 'strategy', 'model', 'dataset',
                          and 'consistency_rate'. The dataframe should include results from
                          self-consistency experiments.
    Returns:
        None: Generates and saves a plot to 'results/plot_consistency_rates.png'.
              Prints a message upon successful save. If 'consistency_rate' column is missing,
              prints a skip message and returns early.
    Side Effects:
        - Creates a bar plot with dataset names on x-axis and consistency rates (%) on y-axis
        - Saves the plot as PNG file at 'results/plot_consistency_rates.png' with 150 dpi
        - Prints status messages to stdout
        - Closes the matplotlib figure to free memory
    Note:
        Requires global variables DATASETS and MODELS to be defined.
        Uses matplotlib styling with grid lines and legend for model identification.
    """

    sc_df = df[df["strategy"] == "self_consistency"].copy()
    if "consistency_rate" not in sc_df.columns:
        print("No consistency_rate column found. Skipping consistency plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(DATASETS))
    width = 0.35
    colors = ["#4C72B0", "#DD8452"]

    for i, model in enumerate(MODELS):
        rates = []
        for dataset in DATASETS:
            subset = sc_df[(sc_df["model"] == model) & (sc_df["dataset"] == dataset)]
            rate = subset["consistency_rate"].mean() * 100 if len(subset) > 0 else 0
            rates.append(rate)
        offset = (i - 0.5) * width
        ax.bar(x + offset, rates, width, label=model, color=colors[i])

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Avg Consistency Rate (%)")
    ax.set_title("Self-Consistency: Agreement Rate Across 5 Samples", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in DATASETS])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 100)
    ax.legend(title="Model")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("results/plot_consistency_rates.png", dpi=150)
    print("Saved plot: results/plot_consistency_rates.png")
    plt.close()

if __name__ == "__main__":
    print("Loading results...\n")
    df = load_all_results()

    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    summary = build_summary_table(df)
    print(summary.to_string(index=False))
    summary.to_csv("results/summary_table.csv", index=False)
    print("\nSaved: results/summary_table.csv\n")

    print("Generating plots...")
    plot_accuracy_by_dataset(df)
    plot_consistency_rates(df)
    print("\nDone. All outputs saved to results/")
