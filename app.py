"""
Cross-Sell Intelligence Dashboard
Works with Sales.xlsx / Products.xlsx / Customers.xlsx
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.engine import (
    load_and_validate, build_customer_pl_matrix,
    compute_similarity_matrix, run_full_analysis,
    LEVEL_LABELS,
    C_CUST_KEY, C_CUST_NAME, C_REGION, C_COUNTRY,
    C_CUST_TYPE, C_SEGMENT, C_APPLICATION, C_PL, C_VALUE,
)
from utils.charts import (
    chart_revenue_by_region, chart_revenue_by_segment,
    chart_revenue_by_customer_type, chart_revenue_by_country,
    chart_propensity_distribution, chart_scatter_propensity_vs_revenue,
    chart_top_customers, chart_opportunity_by_level,
    chart_pl_gap_heatmap, LEVEL_COLORS, PALETTE,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cross-Sell Intelligence",
    page_icon="🎯", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0D1117; color: #E6EDF3;
}
h1, h2, h3 { font-family: 'Space Mono', monospace; }
.block-container { padding-top: 1.5rem; max-width: 1500px; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161B22 0%, #1C2128 100%);
    border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
    padding: 1.2rem 1.4rem;
}
[data-testid="metric-container"] label {
    color: #8B949E !important; font-size: 0.78rem !important;
    letter-spacing: 0.08em; text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E8630A !important; font-family: 'Space Mono', monospace;
    font-size: 1.5rem !important;
}
[data-testid="stSidebar"] {
    background-color: #0D1117;
    border-right: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #161B22; border-radius: 10px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #8B949E; font-size: 0.85rem; font-weight: 500;
}
.stTabs [aria-selected="true"] { background-color: #E8630A !important; color: #fff !important; }
.section-header {
    font-family: 'Space Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: #E8630A;
    margin: 1.5rem 0 0.5rem; padding-bottom: 6px;
    border-bottom: 1px solid rgba(232,99,10,0.3);
}
.info-box {
    background: #161B22; border-left: 3px solid #E8630A;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.9rem;
}
.hero-banner {
    background: linear-gradient(135deg, #0F4C81 0%, #0D1117 60%);
    border: 1px solid rgba(15,76,129,0.4); border-radius: 16px;
    padding: 1.8rem 2rem; margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k in ["recs_df", "merged_df", "products_df", "customers_df",
          "cp_pl_matrix", "analysis_done", "filters"]:
    if k not in st.session_state:
        st.session_state[k] = None
if not st.session_state["analysis_done"]:
    st.session_state["analysis_done"] = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Cross-Sell Intel")
    st.markdown("---")
    st.markdown("### Upload Data Files")
    st.caption("Required: Sales · Products · Customers (Excel or CSV)")

    s_file = st.file_uploader("Sales file",     type=["xlsx","xls","csv"], key="sf")
    p_file = st.file_uploader("Products file",  type=["xlsx","xls","csv"], key="pf")
    c_file = st.file_uploader("Customers file", type=["xlsx","xls","csv"], key="cf")

    st.markdown("---")
    max_cust = st.number_input(
        "Max customers to analyse",
        min_value=50, max_value=10000, value=500, step=50,
        help="Top customers by revenue. Increase for fuller results (slower)."
    )

    run_btn = st.button("▶  Run Analysis", use_container_width=True, type="primary",
                        disabled=not all([s_file, p_file, c_file]))

    if run_btn and all([s_file, p_file, c_file]):
        with st.spinner("Loading & joining data…"):
            try:
                t_df = pd.read_excel(s_file, engine="openpyxl") if s_file.name.endswith(("xlsx","xls")) else pd.read_csv(s_file)
                p_df = pd.read_excel(p_file, engine="openpyxl") if p_file.name.endswith(("xlsx","xls")) else pd.read_csv(p_file)
                c_df = pd.read_excel(c_file, engine="openpyxl") if c_file.name.endswith(("xlsx","xls")) else pd.read_csv(c_file)
            except Exception as e:
                st.error(f"File read error: {e}")
                st.stop()

        with st.spinner(f"Running analysis on top {max_cust} customers…"):
            try:
                merged = load_and_validate(t_df, p_df, c_df)
                cp_pl  = build_customer_pl_matrix(merged)
                recs   = run_full_analysis(merged, p_df, c_df, max_customers=int(max_cust))
                st.session_state.update({
                    "recs_df": recs, "merged_df": merged,
                    "products_df": p_df, "customers_df": c_df,
                    "cp_pl_matrix": cp_pl, "analysis_done": True,
                })
                st.success(f"✅ Done — {len(recs):,} recommendations generated")
            except Exception as e:
                st.error(f"Analysis error: {e}")
                raise

    st.markdown("---")
    st.markdown("### Filters")
    if st.session_state["analysis_done"] and st.session_state["recs_df"] is not None:
        recs_df = st.session_state["recs_df"]

        regions  = ["All"] + sorted(recs_df[C_REGION].dropna().unique().tolist())   if C_REGION  in recs_df.columns else ["All"]
        segments = ["All"] + sorted(recs_df[C_SEGMENT].dropna().unique().tolist())  if C_SEGMENT in recs_df.columns else ["All"]
        countries= ["All"] + sorted(recs_df[C_COUNTRY].dropna().unique().tolist())  if C_COUNTRY in recs_df.columns else ["All"]

        sel_region   = st.selectbox("Region",   regions)
        sel_segment  = st.selectbox("Segment",  segments)
        sel_country  = st.selectbox("Country",  countries)
        sel_levels   = st.multiselect(
            "Rec. Level", options=[1, 4],
            format_func=lambda x: LEVEL_LABELS.get(x, str(x)),
            default=[1, 4],
        )
        min_prop = st.slider("Min Propensity (%)", 0, 100, 0)

        st.session_state["filters"] = {
            "region": sel_region, "segment": sel_segment,
            "country": sel_country, "levels": sel_levels,
            "min_propensity": min_prop,
        }
    else:
        st.caption("Upload files and run analysis to enable filters.")

    st.markdown("---")
    st.caption("Cross-Sell Intelligence v2.0")


# ── Landing page ──────────────────────────────────────────────────────────────
if not st.session_state["analysis_done"]:
    st.markdown("""
    <div class="hero-banner">
        <h1 style="margin:0;font-size:1.8rem">🎯 Cross-Sell Intelligence</h1>
        <p style="color:#8B949E;margin-top:0.5rem;font-size:1rem">
        Revenue gap analysis &amp; collaborative filtering across your customer base.
        </p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="info-box">
        <strong>📥 Upload 3 files</strong><br>
        Sales · Products · Customers (Excel or CSV)</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-box">
        <strong>🧠 L1 Gap Analysis</strong><br>
        Missing Product Lines within the same Division hierarchy</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="info-box">
        <strong>🤝 Collaborative Filtering</strong><br>
        New Product Lines suggested from similar-customer behaviour</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Expected column names")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Sales**\n- `key_source_soldto`\n- `key_source_material_pl`\n- `Value`\n- `QTY`\n- `date` *(optional)*")
    with col_b:
        st.markdown("**Products**\n- `key_source_material_pl`\n- `key_pl`\n- `keytext_productgroup`\n- `keytext_productclass`\n- `division`")
    with col_c:
        st.markdown("**Customers**\n- `key_source_soldto`\n- `soldto`\n- `danfoss_region_2`\n- `text_country`\n- `sf_primary_segment`")
    st.stop()


# ── Apply filters ─────────────────────────────────────────────────────────────
recs_df      = st.session_state["recs_df"].copy()
merged_df    = st.session_state["merged_df"]
products_df  = st.session_state["products_df"]
customers_df = st.session_state["customers_df"]
filters      = st.session_state.get("filters") or {}

if filters.get("region") not in (None, "All"):
    recs_df = recs_df[recs_df[C_REGION] == filters["region"]]
if filters.get("segment") not in (None, "All"):
    recs_df = recs_df[recs_df[C_SEGMENT] == filters["segment"]]
if filters.get("country") not in (None, "All"):
    recs_df = recs_df[recs_df[C_COUNTRY] == filters["country"]]
if filters.get("levels"):
    recs_df = recs_df[recs_df["Level"].isin(filters["levels"])]
if filters.get("min_propensity", 0) > 0:
    recs_df = recs_df[recs_df["PropensityScore"] >= filters["min_propensity"]]


# ── Executive Summary ─────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Executive Summary</p>', unsafe_allow_html=True)

total_opp    = recs_df["PredictedRevenue"].sum()
n_customers  = recs_df["CustomerKey"].nunique()
n_recs       = len(recs_df)
avg_prop     = recs_df["PropensityScore"].mean()
total_actual = merged_df[C_VALUE].sum()
uplift_pct   = (total_opp / total_actual * 100) if total_actual > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue Opportunity",  f"€{total_opp:,.0f}")
c2.metric("Actual YTD Revenue",         f"€{total_actual:,.0f}")
c3.metric("Opportunity Uplift",         f"{uplift_pct:.1f}%")
c4.metric("Customers with Gaps",        f"{n_customers:,}")
c5.metric("Avg Propensity Score",       f"{avg_prop:.1f}%")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Opportunity List", "📈 Analytics", "🔍 Customer Detail", "🗃️ Raw Data"])


# ── TAB 1: Ranked Opportunity Table ──────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Ranked Leads — Sorted by Predicted Revenue Opportunity</p>',
                unsafe_allow_html=True)

    name_col = C_CUST_NAME if C_CUST_NAME in recs_df.columns else "CustomerKey"
    summary = (
        recs_df.groupby(["CustomerKey", name_col,
                          C_CUST_TYPE if C_CUST_TYPE in recs_df.columns else "CustomerKey",
                          C_REGION    if C_REGION    in recs_df.columns else "CustomerKey",
                          C_COUNTRY   if C_COUNTRY   in recs_df.columns else "CustomerKey"])
        .agg(
            TotalOpportunity  =("PredictedRevenue", "sum"),
            NumRecommendations=("ProductLine", "count"),
            AvgPropensity     =("PropensityScore", "mean"),
            TopProductLine    =("ProductLine", "first"),
        )
        .reset_index()
        .sort_values("TotalOpportunity", ascending=False)
        .reset_index(drop=True)
    )
    summary.index += 1
    disp = summary.copy()
    disp["TotalOpportunity"] = disp["TotalOpportunity"].apply(lambda v: f"€{v:,.0f}")
    disp["AvgPropensity"]    = disp["AvgPropensity"].apply(lambda v: f"{v:.1f}%")
    col_rename = {
        name_col:            "Customer",
        C_CUST_TYPE:         "Type",
        C_REGION:            "Region",
        C_COUNTRY:           "Country",
        "TotalOpportunity":  "Opportunity",
        "NumRecommendations":"# Recs",
        "AvgPropensity":     "Avg Propensity",
        "TopProductLine":    "Top PL",
    }
    st.dataframe(disp.rename(columns=col_rename), use_container_width=True, height=440)

    st.markdown('<p class="section-header">All Recommendations</p>', unsafe_allow_html=True)

    show_cols = [name_col, "ProductLine", "Level", "Reason",
                 "PropensityScore", "AvgRevPerLine", "PredictedRevenue", "Type"]
    show_cols = [c for c in show_cols if c in recs_df.columns]
    disp2 = recs_df[show_cols].copy()
    disp2["Level"]            = disp2["Level"].map(LEVEL_LABELS)
    disp2["PropensityScore"]  = disp2["PropensityScore"].apply(lambda v: f"{v}%")
    disp2["AvgRevPerLine"]    = disp2["AvgRevPerLine"].apply(lambda v: f"€{v:,.2f}")
    disp2["PredictedRevenue"] = disp2["PredictedRevenue"].apply(lambda v: f"€{v:,.2f}")
    st.dataframe(disp2.rename(columns={
        name_col: "Customer", "ProductLine": "Product Line",
        "PropensityScore": "Propensity", "AvgRevPerLine": "Avg Rev/Line",
        "PredictedRevenue": "Predicted Revenue",
    }), use_container_width=True, height=500)

    csv = recs_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Export full recommendations CSV",
                       csv, "cross_sell_recommendations.csv", "text/csv")


# ── TAB 2: Analytics ──────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Revenue Analytics</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_revenue_by_region(recs_df),       use_container_width=True)
    with col_b:
        st.plotly_chart(chart_revenue_by_segment(recs_df),      use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(chart_opportunity_by_level(recs_df),    use_container_width=True)
    with col_d:
        st.plotly_chart(chart_propensity_distribution(recs_df), use_container_width=True)

    st.plotly_chart(chart_scatter_propensity_vs_revenue(recs_df), use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        st.plotly_chart(chart_revenue_by_customer_type(recs_df), use_container_width=True)
    with col_f:
        st.plotly_chart(chart_revenue_by_country(recs_df),       use_container_width=True)

    st.plotly_chart(chart_top_customers(recs_df),  use_container_width=True)
    st.plotly_chart(chart_pl_gap_heatmap(recs_df), use_container_width=True)


# ── TAB 3: Customer Detail ────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Customer Detail View</p>', unsafe_allow_html=True)

    name_col = C_CUST_NAME if C_CUST_NAME in recs_df.columns else "CustomerKey"
    cust_options = sorted(recs_df[name_col].dropna().unique().tolist())
    sel_cust = st.selectbox("Select Customer", cust_options)

    if sel_cust:
        cust_recs = recs_df[recs_df[name_col] == sel_cust].copy()
        cust_key  = cust_recs["CustomerKey"].iloc[0]

        # Metadata row
        meta = customers_df[customers_df[C_CUST_KEY] == cust_key]
        if not meta.empty:
            m = meta.iloc[0]
            cols = st.columns(5)
            for i, (label, field) in enumerate([
                ("Type",    C_CUST_TYPE),
                ("Region",  C_REGION),
                ("Country", C_COUNTRY),
                ("Segment", C_SEGMENT),
                ("Application", C_APPLICATION),
            ]):
                cols[i].metric(label, str(m.get(field, "—"))[:30])

        # Revenue stats
        actual_rev = float(merged_df[merged_df[C_CUST_KEY] == cust_key][C_VALUE].sum())
        opp_rev    = float(cust_recs["PredictedRevenue"].sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Actual YTD Revenue",     f"€{actual_rev:,.0f}")
        m2.metric("Predicted Opportunity",  f"€{opp_rev:,.0f}")
        m3.metric("Recommendations",        str(len(cust_recs)))

        # Current Product Lines
        owned_pls = merged_df[merged_df[C_CUST_KEY] == cust_key][C_PL].dropna().unique().tolist()
        if owned_pls:
            st.markdown("**Currently buys:** " + "  ·  ".join([f"`{pl}`" for pl in sorted(owned_pls)]))

        st.markdown("---")
        st.markdown("#### Recommendations")

        for _, row in cust_recs.sort_values("PredictedRevenue", ascending=False).iterrows():
            label = LEVEL_LABELS.get(row["Level"], str(row["Level"]))
            with st.expander(f"**{row['ProductLine']}** — {label} — €{row['PredictedRevenue']:,.2f}"):
                c1, c2 = st.columns(2)
                c1.metric("Propensity Score",    f"{row['PropensityScore']}%")
                c2.metric("Avg Rev / Line",      f"€{row['AvgRevPerLine']:,.2f}")
                st.markdown(f"**Reason:** {row['Reason']}")
                st.markdown(f"**Type:** `{row['Type']}`")


# ── TAB 4: Raw Data ───────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">Raw Data Explorer</p>', unsafe_allow_html=True)

    sub1, sub2, sub3, sub4 = st.tabs(["Sales (sample)", "Products", "Customers", "Merged (sample)"])
    with sub1:
        st.dataframe(merged_df[["key_source_soldto","key_source_material_pl",
                                  "Value","QTY"]].head(500), use_container_width=True)
    with sub2:
        st.dataframe(products_df.head(500), use_container_width=True)
    with sub3:
        st.dataframe(customers_df.head(500), use_container_width=True)
    with sub4:
        st.dataframe(merged_df.head(500), use_container_width=True)
