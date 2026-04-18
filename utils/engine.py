"""
Cross-Sell Recommendation Engine
Implements hierarchical gap analysis + collaborative filtering
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple


def load_and_validate(transactions_df, products_df, customers_df):
    """Merge datasets into unified analysis frame."""
    df = transactions_df.merge(products_df, on="ProductID", how="left")
    df = df.merge(customers_df, on="CustomerID", how="left")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_customer_product_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Binary matrix: customers × products (1 if purchased)."""
    matrix = df.pivot_table(
        index="CustomerID", columns="ProductID", values="Revenue",
        aggfunc="sum", fill_value=0
    )
    return (matrix > 0).astype(int)


def compute_similarity_matrix(cp_matrix: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between customers for collaborative filtering."""
    sim = cosine_similarity(cp_matrix.values)
    return pd.DataFrame(sim, index=cp_matrix.index, columns=cp_matrix.index)


def get_avg_price(df: pd.DataFrame, product_id: str) -> float:
    """Historical average unit price for a product."""
    prices = df[df["ProductID"] == product_id]["UnitPrice"]
    return float(prices.mean()) if len(prices) > 0 else 0.0


def get_avg_quantity(df: pd.DataFrame, product_id: str) -> float:
    """Historical average quantity per transaction."""
    qtys = df[df["ProductID"] == product_id]["Quantity"]
    return float(qtys.mean()) if len(qtys) > 0 else 1.0


def hierarchical_gap_analysis(
    customer_id: str,
    df: pd.DataFrame,
    products_df: pd.DataFrame,
    cp_matrix: pd.DataFrame,
) -> List[Dict]:
    """
    Level 1: Missing products in same Product Line  (weight 1.0)
    Level 2: Missing products in same Product Group (weight 0.65)
    Level 3: Missing products in same Product Class (weight 0.35)
    """
    if customer_id not in cp_matrix.index:
        return []

    owned = cp_matrix.loc[customer_id]
    owned_products = owned[owned == 1].index.tolist()

    if not owned_products:
        return []

    owned_meta = products_df[products_df["ProductID"].isin(owned_products)]
    all_meta = products_df.copy()

    owned_lines = set(owned_meta["ProductLine"].dropna())
    owned_groups = set(owned_meta["ProductGroup"].dropna())
    owned_classes = set(owned_meta["ProductClass"].dropna())

    missing = products_df[~products_df["ProductID"].isin(owned_products)]
    recommendations = []

    for _, row in missing.iterrows():
        pid = row["ProductID"]
        level, reason, priority_weight = None, None, 0.0

        if row.get("ProductLine") in owned_lines:
            level = 1
            priority_weight = 1.0
            reason = f"Same Product Line: {row['ProductLine']}"
        elif row.get("ProductGroup") in owned_groups:
            level = 2
            priority_weight = 0.65
            reason = f"Same Product Group: {row['ProductGroup']}"
        elif row.get("ProductClass") in owned_classes:
            level = 3
            priority_weight = 0.35
            reason = f"Same Product Class: {row['ProductClass']}"
        else:
            continue

        avg_price = get_avg_price(df, pid)
        if avg_price == 0:
            avg_price = float(row.get("StandardPrice", 100))
        avg_qty = get_avg_quantity(df, pid)

        # Propensity: how many existing customers bought this product
        if pid in cp_matrix.columns:
            adoption_rate = cp_matrix[pid].mean()
        else:
            adoption_rate = 0.05

        propensity = min(0.95, adoption_rate * priority_weight * 1.5 + 0.05)
        predicted_revenue = avg_price * avg_qty * propensity

        recommendations.append({
            "CustomerID": customer_id,
            "ProductID": pid,
            "ProductName": row.get("ProductName", pid),
            "ProductLine": row.get("ProductLine", ""),
            "ProductGroup": row.get("ProductGroup", ""),
            "ProductClass": row.get("ProductClass", ""),
            "Level": level,
            "Reason": reason,
            "PropensityScore": round(propensity * 100, 1),
            "AvgPrice": round(avg_price, 2),
            "AvgQty": round(avg_qty, 1),
            "PredictedRevenue": round(predicted_revenue, 2),
            "Type": "Gap Analysis",
        })

    return recommendations


def collaborative_filter_new_ideas(
    customer_id: str,
    df: pd.DataFrame,
    products_df: pd.DataFrame,
    cp_matrix: pd.DataFrame,
    sim_matrix: pd.DataFrame,
    top_n_similar: int = 5,
    top_n_products: int = 5,
) -> List[Dict]:
    """
    Discovery Logic: suggest Product Lines customer hasn't engaged with,
    based on similar customers' purchasing patterns.
    """
    if customer_id not in cp_matrix.index:
        return []

    owned = cp_matrix.loc[customer_id]
    owned_products = set(owned[owned == 1].index.tolist())

    # Get top-N similar customers
    if customer_id not in sim_matrix.index:
        return []
    similarities = sim_matrix.loc[customer_id].drop(customer_id, errors="ignore")
    top_similar = similarities.nlargest(top_n_similar).index.tolist()

    if not top_similar:
        return []

    # Score products by weighted purchase frequency among similar customers
    product_scores: Dict[str, float] = {}
    for sim_cust in top_similar:
        if sim_cust not in cp_matrix.index:
            continue
        sim_score = float(sim_matrix.loc[customer_id, sim_cust])
        sim_owned = cp_matrix.loc[sim_cust]
        for pid in sim_owned[sim_owned == 1].index:
            if pid not in owned_products:
                product_scores[pid] = product_scores.get(pid, 0) + sim_score

    # Only keep products from "new" product lines
    owned_meta = products_df[products_df["ProductID"].isin(owned_products)]
    owned_lines = set(owned_meta["ProductLine"].dropna())

    new_idea_scores = {
        pid: score for pid, score in product_scores.items()
        if products_df[products_df["ProductID"] == pid]["ProductLine"].isin(
            products_df[~products_df["ProductLine"].isin(owned_lines)]["ProductLine"]
        ).any()
    }

    if not new_idea_scores:
        return []

    top_ideas = sorted(new_idea_scores.items(), key=lambda x: x[1], reverse=True)[:top_n_products]
    results = []

    for pid, score in top_ideas:
        meta = products_df[products_df["ProductID"] == pid]
        if meta.empty:
            continue
        row = meta.iloc[0]
        avg_price = get_avg_price(df, pid)
        if avg_price == 0:
            avg_price = float(row.get("StandardPrice", 100))
        avg_qty = get_avg_quantity(df, pid)

        max_score = max(new_idea_scores.values()) if new_idea_scores else 1
        max_score = max_score if max_score != 0 else 1
        propensity = min(0.85, (score / max_score) * 0.5 + 0.1)
        predicted_revenue = avg_price * avg_qty * propensity

        results.append({
            "CustomerID": customer_id,
            "ProductID": pid,
            "ProductName": row.get("ProductName", pid),
            "ProductLine": row.get("ProductLine", ""),
            "ProductGroup": row.get("ProductGroup", ""),
            "ProductClass": row.get("ProductClass", ""),
            "Level": 4,
            "Reason": f"New Idea – Similar customers buy this ({row.get('ProductLine', '')})",
            "PropensityScore": round(propensity * 100, 1),
            "AvgPrice": round(avg_price, 2),
            "AvgQty": round(avg_qty, 1),
            "PredictedRevenue": round(predicted_revenue, 2),
            "Type": "New Idea (CF)",
        })

    return results


def run_full_analysis(df: pd.DataFrame, products_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    """Run complete cross-sell analysis for all customers."""
    cp_matrix = build_customer_product_matrix(df)
    sim_matrix = compute_similarity_matrix(cp_matrix)

    all_recs = []
    for cid in customers_df["CustomerID"].unique():
        gap_recs = hierarchical_gap_analysis(cid, df, products_df, cp_matrix)
        cf_recs = collaborative_filter_new_ideas(cid, df, products_df, cp_matrix, sim_matrix)
        all_recs.extend(gap_recs)
        all_recs.extend(cf_recs)

    if not all_recs:
        return pd.DataFrame()

    recs_df = pd.DataFrame(all_recs)
    recs_df = recs_df.merge(
        customers_df[["CustomerID", "CustomerName", "Segment", "Region", "Industry"]],
        on="CustomerID", how="left"
    )

    # Customer-level rollup: total predicted revenue opportunity
    customer_totals = (
        recs_df.groupby("CustomerID")["PredictedRevenue"]
        .sum()
        .rename("TotalOpportunity")
        .reset_index()
    )
    recs_df = recs_df.merge(customer_totals, on="CustomerID", how="left")
    recs_df = recs_df.sort_values(["TotalOpportunity", "PredictedRevenue"], ascending=False)

    return recs_df
