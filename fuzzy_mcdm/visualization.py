"""Matplotlib figures for paper-style reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_cluster_weights(df: pd.DataFrame, path: str | Path) -> None:
    plot_df = df.sort_values("weight", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(plot_df["label"], plot_df["weight"])
    ax.set_ylabel("Priority weight")
    ax.set_xlabel("Barrier cluster")
    ax.set_title("Fuzzy AHP priority weights of barrier clusters")
    ax.set_ylim(0, max(plot_df["weight"].max() * 1.2, 0.1))
    for idx, value in enumerate(plot_df["weight"]):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_subbarrier_ranking(df: pd.DataFrame, path: str | Path, top_n: int | None = None) -> None:
    plot_df = df.sort_values("global_weight", ascending=True)
    if top_n is not None:
        plot_df = df.sort_values("global_weight", ascending=False).head(top_n).sort_values("global_weight", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["code"] + " - " + plot_df["barrier"], plot_df["global_weight"])
    ax.set_xlabel("Global priority weight")
    ax.set_title("Global ranking of renovation sub-barriers")
    for y, value in enumerate(plot_df["global_weight"]):
        ax.text(value, y, f" {value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_dematel_map(df: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(0, linewidth=1)
    ax.scatter(df["prominence_D_plus_R"], df["relation_D_minus_R"], s=90)
    for _, row in df.iterrows():
        ax.annotate(row["barrier"], (row["prominence_D_plus_R"], row["relation_D_minus_R"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Prominence (D + R)")
    ax.set_ylabel("Relation (D - R)")
    ax.set_title("DEMATEL cause-effect map")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_policy_scenarios(df: pd.DataFrame, path: str | Path) -> None:
    plot_df = df.sort_values("total_leverage_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(plot_df["scenario"], plot_df["total_leverage_score"])
    ax.set_xlabel("Total leverage score")
    ax.set_title("Policy scenario leverage based on DEMATEL relations")
    for y, value in enumerate(plot_df["total_leverage_score"]):
        ax.text(value, y, f" {value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
