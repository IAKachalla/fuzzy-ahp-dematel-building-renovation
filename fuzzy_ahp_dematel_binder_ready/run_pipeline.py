"""Run the full Fuzzy AHP-DEMATEL analysis pipeline.

Usage:
    python run_pipeline.py
    python run_pipeline.py --data-dir data --output-dir outputs --sensitivity-iterations 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fuzzy_mcdm.fuzzy_ahp import long_pairwise_to_matrices, run_fuzzy_ahp_from_pairwise
from fuzzy_mcdm.fuzzy_dematel import run_fuzzy_dematel_from_influence
from fuzzy_mcdm.scenario_modeling import evaluate_policy_scenarios
from fuzzy_mcdm.sensitivity import rank_stability
from fuzzy_mcdm.visualization import (
    plot_cluster_weights,
    plot_dematel_map,
    plot_policy_scenarios,
    plot_subbarrier_ranking,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuzzy AHP-DEMATEL pipeline for sustainable renovation barrier analysis.")
    parser.add_argument("--data-dir", default="data", help="Input data directory.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    parser.add_argument("--sensitivity-iterations", type=int, default=500, help="Monte Carlo iterations for sensitivity analysis.")
    parser.add_argument("--dematel-threshold", type=float, default=None, help="Optional fixed DEMATEL threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    barriers = pd.read_csv(data_dir / "barriers.csv")
    cluster_meta = (
        barriers[["cluster_code", "cluster"]]
        .drop_duplicates()
        .rename(columns={"cluster_code": "item", "cluster": "label"})
    )
    cluster_order = cluster_meta["item"].tolist()
    cluster_labels = dict(zip(cluster_meta["item"], cluster_meta["label"]))

    # 1) Fuzzy AHP for barrier clusters.
    cluster_result = run_fuzzy_ahp_from_pairwise(data_dir / "sample_cluster_pairwise.csv", "clusters", cluster_order)
    cluster_df = cluster_result.as_dataframe().merge(cluster_meta, on="item", how="left")
    cluster_df = cluster_df.rename(columns={"item": "cluster_code"})
    cluster_df.to_csv(out_dir / "cluster_priority.csv", index=False)
    cluster_result.expert_consistency.to_csv(out_dir / "cluster_expert_consistency.csv", index=False)

    # 2) Fuzzy AHP for sub-barriers within each cluster.
    sub_rows = []
    sub_consistency = []
    sub_pairwise = pd.read_csv(data_dir / "sample_subbarrier_pairwise.csv")

    for cluster_code in cluster_order:
        context = cluster_code
        item_order = barriers.loc[barriers["cluster_code"] == cluster_code, "code"].tolist()
        result = run_fuzzy_ahp_from_pairwise(data_dir / "sample_subbarrier_pairwise.csv", context, item_order)
        local_df = result.as_dataframe().rename(columns={"item": "code", "weight": "local_weight"})
        local_df["cluster_code"] = cluster_code
        local_df["cluster"] = cluster_labels[cluster_code]
        local_df = local_df.merge(barriers[["code", "barrier"]], on="code", how="left")
        cluster_weight = float(cluster_df.loc[cluster_df["cluster_code"] == cluster_code, "weight"].iloc[0])
        local_df["cluster_weight"] = cluster_weight
        local_df["global_weight"] = local_df["local_weight"] * cluster_weight
        sub_rows.append(local_df)
        tmp = result.expert_consistency.copy()
        tmp["cluster_code"] = cluster_code
        tmp["cluster"] = cluster_labels[cluster_code]
        sub_consistency.append(tmp)

    sub_df = pd.concat(sub_rows, ignore_index=True)
    sub_df["global_rank"] = sub_df["global_weight"].rank(ascending=False, method="min").astype(int)
    sub_df = sub_df.sort_values("global_rank")
    sub_df.to_csv(out_dir / "subbarrier_priority.csv", index=False)
    pd.concat(sub_consistency, ignore_index=True).to_csv(out_dir / "subbarrier_expert_consistency.csv", index=False)

    # 3) Fuzzy DEMATEL causal mapping for critical barriers.
    dematel_items = pd.read_csv(data_dir / "sample_dematel_influence.csv")["source"].drop_duplicates().tolist()
    dematel_result = run_fuzzy_dematel_from_influence(
        data_dir / "sample_dematel_influence.csv",
        dematel_items,
        threshold=args.dematel_threshold,
    )
    dematel_df = dematel_result.results.rename(columns={"barrier": "code"}).merge(barriers[["code", "barrier"]], on="code", how="left")
    dematel_df.to_csv(out_dir / "dematel_results.csv", index=False)
    pd.DataFrame(dematel_result.total_relation_matrix, index=dematel_result.items, columns=dematel_result.items).to_csv(out_dir / "dematel_total_relation_matrix.csv")
    pd.DataFrame(dematel_result.direct_matrix_crisp, index=dematel_result.items, columns=dematel_result.items).to_csv(out_dir / "dematel_direct_matrix_crisp.csv")

    # 4) Sensitivity analysis for global sub-barrier ranking.
    #    To keep the method transparent, sensitivity is performed within each local cluster matrix.
    stability_frames = []
    for cluster_code in cluster_order:
        item_order = barriers.loc[barriers["cluster_code"] == cluster_code, "code"].tolist()
        context_df = sub_pairwise[sub_pairwise["context"] == cluster_code].copy()
        matrices = long_pairwise_to_matrices(context_df, item_order)
        stab = rank_stability(matrices, item_order, iterations=args.sensitivity_iterations, pct=0.10, seed=42)
        stab["cluster_code"] = cluster_code
        stab["cluster"] = cluster_labels[cluster_code]
        stability_frames.append(stab)
    stability_df = pd.concat(stability_frames, ignore_index=True).merge(barriers[["code", "barrier"]], left_on="item", right_on="code", how="left")
    stability_df.to_csv(out_dir / "sensitivity_rank_stability.csv", index=False)

    # 5) Scenario modelling.
    scenario_df = evaluate_policy_scenarios(dematel_result.total_relation_matrix, dematel_result.items)
    scenario_df.to_csv(out_dir / "policy_scenarios.csv", index=False)

    # 6) Figures.
    plot_cluster_weights(cluster_df, out_dir / "cluster_weights.png")
    plot_subbarrier_ranking(sub_df, out_dir / "subbarrier_global_ranking.png", top_n=14)
    plot_dematel_map(dematel_df, out_dir / "dematel_cause_effect_map.png")
    plot_policy_scenarios(scenario_df, out_dir / "policy_scenario_impacts.png")

    print("Analysis complete.")
    print(f"Results saved to: {out_dir.resolve()}")
    print("\nTop five global sub-barriers:")
    print(sub_df[["global_rank", "code", "barrier", "global_weight"]].head(5).to_string(index=False))
    print("\nDEMATEL cause/effect summary:")
    print(dematel_df[["code", "barrier", "prominence_D_plus_R", "relation_D_minus_R", "type"]].to_string(index=False))


if __name__ == "__main__":
    main()
