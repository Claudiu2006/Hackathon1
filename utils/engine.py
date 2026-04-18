"""
Cross-Sell Recommendation Engine
Mapped to the real schema:
  Sales:     key_source_soldto, key_source_material_pl, date, Value, QTY
  Products:  key_source_material_pl, material, division, key_pl,
             keytext_productgroup, keytext_productclass
  Customers: key_source_soldto, soldto, customer_type_2, danfoss_region_2,
             customer_group_1..3, text_country, sf_primary_segment,
             sf_primary_application
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional

# Column aliases
C_CUST_KEY   = "key_source_soldto"
C_PROD_KEY   = "key_source_material_pl"
C_VALUE      = "Value"
C_QTY        = "QTY"
C_DATE       = "date"
C_MATERIAL   = "material"
C_DIVISION   = "division"
C_PL         = "key_pl"
C_PG         = "keytext_productgroup"
C_PC         = "keytext_productclass"
C_CUST_NAME  = "soldto"
C_CUST_TYPE  = "customer_type_2"
C_REGION     = "danfoss_region_2"
C_COUNTRY    = "text_country"
C_SEGMENT    = "sf_primary_segment"
C_APPLICATION= "sf_primary_application"

LEVEL_LABELS = {
    1: "L1 – Product Line",
    2: "L2 – Product Group",
    3: "L3 – Product Class",
    4: "New Idea (CF)",
}


def load_excel(uploaded_file) -> pd.DataFrame:
    if hasattr(uploaded_file, "name"):
        fname = uploaded_file.name.lower()
    else:
        fname = str(uploaded_file).lower()
    if fname.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file, engine="openpyxl")


def load_and_validate(sales_df, products_df, customers_df):
    sales_df = sales_df.copy()
    sales_df[C_VALUE] = pd.to_numeric(sales_df[C_VALUE], errors="coerce").fillna(0)
    sales_df[C_QTY]   = pd.to_numeric(sales_df[C_QTY],   errors="coerce").fillna(0)
    df = sales_df.merge(products_df,  on=C_PROD_KEY, how="left")
    df = df.merge(customers_df,       on=C_CUST_KEY, how="left")
    return df


def build_customer_pl_matrix(df):
    matrix = df.pivot_table(
        index=C_CUST_KEY, columns=C_PL,
        values=C_VALUE, aggfunc="sum", fill_value=0
    )
    return (matrix > 0).astype(np.int8)


def build_customer_product_matrix(df):
    matrix = df.pivot_table(
        index=C_CUST_KEY, columns=C_PROD_KEY,
        values=C_VALUE, aggfunc="sum", fill_value=0
    )
    return (matrix > 0).astype(np.int8)


def compute_similarity_matrix(cp_matrix):
    sim = cosine_similarity(cp_matrix.values)
    return pd.DataFrame(sim, index=cp_matrix.index, columns=cp_matrix.index)


def build_pl_adoption_rates(cp_matrix):
    return cp_matrix.mean()


def _pl_avg_revenue(df, pl):
    v = df.loc[df[C_PL] == pl, C_VALUE]
    return float(v[v > 0].mean()) if len(v[v > 0]) > 0 else 0.0


def hierarchical_gap_analysis(customer_key, df, products_df, cp_pl_matrix, pl_adoption):
    if customer_key not in cp_pl_matrix.index:
        return []

    owned_row = cp_pl_matrix.loc[customer_key]
    owned_pls = set(owned_row[owned_row == 1].index.tolist())
    if not owned_pls:
        return []

    cust_df = df[df[C_CUST_KEY] == customer_key]
    owned_divisions = set(cust_df[C_DIVISION].dropna().unique())

    all_pls = products_df[C_PL].dropna().unique()
    recs = []

    for pl in all_pls:
        if pl in owned_pls:
            continue
        pl_divisions = set(products_df.loc[products_df[C_PL] == pl, C_DIVISION].dropna().unique())
        if not pl_divisions.intersection(owned_divisions):
            continue

        adoption = float(pl_adoption.get(pl, 0))
        propensity = min(0.92, adoption * 1.4 + 0.05)
        avg_rev = _pl_avg_revenue(df, pl)
        shared_divs = pl_divisions & owned_divisions

        recs.append({
            "CustomerKey":      customer_key,
            "Level":            1,
            "Reason":           f"Buys in Division {', '.join(sorted(shared_divs))} but missing {pl}",
            "ProductLine":      pl,
            "ProductGroup":     "",
            "ProductClass":     "",
            "PropensityScore":  round(propensity * 100, 1),
            "AvgRevPerLine":    round(avg_rev, 2),
            "PredictedRevenue": round(avg_rev * propensity, 2),
            "Type":             "Gap Analysis",
        })

    return recs


def collaborative_filter_new_ideas(customer_key, df, products_df, cp_pl_matrix, sim_matrix,
                                    top_n_similar=10, top_n_pls=5):
    if customer_key not in cp_pl_matrix.index:
        return []

    owned_row = cp_pl_matrix.loc[customer_key]
    owned_pls = set(owned_row[owned_row == 1].index.tolist())

    if customer_key not in sim_matrix.index:
        return []

    similarities = sim_matrix.loc[customer_key].drop(customer_key, errors="ignore")
    top_similar  = similarities.nlargest(top_n_similar).index.tolist()

    pl_scores = {}
    for sim_cust in top_similar:
        if sim_cust not in cp_pl_matrix.index:
            continue
        sim_score = float(sim_matrix.loc[customer_key, sim_cust])
        sim_owned = cp_pl_matrix.loc[sim_cust]
        for pl in sim_owned[sim_owned == 1].index:
            if pl not in owned_pls:
                pl_scores[pl] = pl_scores.get(pl, 0.0) + sim_score

    if not pl_scores:
        return []

    max_score = max(pl_scores.values())
    if max_score == 0:
        return []

    top_pls = sorted(pl_scores.items(), key=lambda x: x[1], reverse=True)[:top_n_pls]
    results = []

    for pl, score in top_pls:
        propensity = min(0.75, (score / max_score) * 0.55 + 0.08)
        avg_rev    = _pl_avg_revenue(df, pl)
        results.append({
            "CustomerKey":      customer_key,
            "Level":            4,
            "Reason":           f"Similar customers also buy {pl} — New Idea",
            "ProductLine":      pl,
            "ProductGroup":     "",
            "ProductClass":     "",
            "PropensityScore":  round(propensity * 100, 1),
            "AvgRevPerLine":    round(avg_rev, 2),
            "PredictedRevenue": round(avg_rev * propensity, 2),
            "Type":             "New Idea (CF)",
        })

    return results


def run_full_analysis(df, products_df, customers_df, max_customers=None):
    cp_pl_matrix = build_customer_pl_matrix(df)
    sim_matrix   = compute_similarity_matrix(cp_pl_matrix)
    pl_adoption  = build_pl_adoption_rates(cp_pl_matrix)
    products_pl  = products_df[[C_PROD_KEY, C_PL, C_PG, C_PC, C_DIVISION]].drop_duplicates()

    all_customers = customers_df[C_CUST_KEY].unique()
    if max_customers:
        rev_by_cust   = df.groupby(C_CUST_KEY)[C_VALUE].sum().sort_values(ascending=False)
        all_customers = rev_by_cust.head(max_customers).index.tolist()

    all_recs = []
    for cid in all_customers:
        gap_recs = [r for r in hierarchical_gap_analysis(
            cid, df, products_pl, cp_pl_matrix, pl_adoption) if r["Level"] == 1]
        cf_recs  = collaborative_filter_new_ideas(
            cid, df, products_pl, cp_pl_matrix, sim_matrix)
        all_recs.extend(gap_recs)
        all_recs.extend(cf_recs)

    if not all_recs:
        return pd.DataFrame()

    recs_df = pd.DataFrame(all_recs)
    cust_meta = customers_df[[
        C_CUST_KEY, C_CUST_NAME, C_CUST_TYPE, C_REGION,
        C_COUNTRY, C_SEGMENT, C_APPLICATION
    ]].drop_duplicates(subset=[C_CUST_KEY])

    cust_meta = cust_meta.rename(columns={C_CUST_KEY: "CustomerKey"})
    recs_df = recs_df.merge(cust_meta, on="CustomerKey", how="left")
    totals   = recs_df.groupby("CustomerKey")["PredictedRevenue"].sum().rename("TotalOpportunity").reset_index()
    recs_df  = recs_df.merge(totals, on="CustomerKey", how="left")
    recs_df  = recs_df.sort_values(["TotalOpportunity", "PredictedRevenue"], ascending=False).reset_index(drop=True)
    return recs_df
