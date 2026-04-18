"""
Cross-Sell Intelligence Dashboard
Streamlit entry point
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.engine import (
    load_and_validate,
    build_customer_product_matrix,
    compute_similarity_matrix,
    run_full_analysis,
)
from utils.data_loader import load_sample_data
from utils.charts import (
    chart_revenue_by_region,
    chart_revenue_by_segment,
    chart_propensity_distribution,
    chart_scatter_propensity_vs_revenue,
    chart_top_customers,
    chart_opportunity_by_level,
    chart_heatmap_customer_product,
    LEVEL_LABELS,
    PALETTE,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cross-Sell Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0D1117;
    color: #E6EDF3;
}

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.block-container { padding-top: 1.5rem; max-width: 1400px; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161B22 0%, #1C2128 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
}
[data-testid="metric-container"] label {
    color: #8B949E !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E8630A !important;
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0D1117;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #161B22; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #8B949E;
    font-size: 0.85rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background-color: #E8630A !important;
    color: #fff !important;
}

/* Section header */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #E8630A;
    margin: 1.5rem 0 0.5rem;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(232,99,10,0.3);
}

/* Level badge */
.badge-l1 { background:#2563EB22; color:#60A5FA; border:1px solid #2563EB44; border-radius:6px; padding:2px 8px; font-size:0.75rem; }
.badge-l2 { background:#7C3AED22; color:#C4B5FD; border:1px solid #7C3AED44; border-radius:6px; padding:2px 8px; font-size:0.75rem; }
.badge-l3 { background:#0891B222; color:#67E8F9; border:1px solid #0891B244; border-radius:6px; padding:2px 8px; font-size:0.75rem; }
.badge-l4 { background:#D9770622; color:#FCD34D; border:1px solid #D9770644; border-radius:6px; padding:2px 8px; font-size:0.75rem; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0F4C81 0%, #0D1117 60%);
    border: 1px solid rgba(15,76,129,0.4);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
}

/* Info box */
.info-box {
    background: #161B22;
    border-left: 3px solid #E8630A;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
for key in ["recs_df", "merged_df", "products_df", "customers_df", "cp_matrix", "analysis_done"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Cross-Sell Intel")
    st.markdown("---")

    mode = st.radio("Data Source", ["📂 Upload Files", "🧪 Use Sample Data"], index=1)

    if mode == "📂 Upload Files":
        st.markdown('<p class="section-header">Upload Datasets</p>', unsafe_allow_html=True)
        t_file = st.file_uploader("Transactions (CSV/Excel)", type=["csv", "xlsx", "xls"], key="t")
        p_file = st.file_uploader("Products (CSV/Excel)", type=["csv", "xlsx", "xls"], key="p")
        c_file = st.file_uploader("Customers (CSV/Excel)", type=["csv", "xlsx", "xls"], key="c")

        if st.button("▶  Run Analysis", use_container_width=True, type="primary"):
            if not all([t_file, p_file, c_file]):
                st.error("Please upload all three files.")
            else:
                with st.spinner("Processing…"):
                    try:
                        t_df = pd.read_csv(t_file) if t_file.name.endswith(".csv") else pd.read_excel(t_file)
                        p_df = pd.read_csv(p_file) if p_file.name.endswith(".csv") else pd.read_excel(p_file)
                        c_df = pd.read_csv(c_file) if c_file.name.endswith(".csv") else pd.read_excel(c_file)
                        merged = load_and_validate(t_df, p_df, c_df)
                        cp_matrix = build_customer_product_matrix(merged)
                        recs = run_full_analysis(merged, p_df, c_df)
                        st.session_state.update({
                            "recs_df": recs, "merged_df": merged,
                            "products_df": p_df, "customers_df": c_df,
                            "cp_matrix": cp_matrix, "analysis_done": True,
                        })
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        if st.button("▶  Load Sample Data", use_container_width=True, type="primary"):
            with st.spinner("Loading sample data…"):
                t_df, p_df, c_df = load_sample_data()
                merged = load_and_validate(t_df, p_df, c_df)
                cp_matrix = build_customer_product_matrix(merged)
                recs = run_full_analysis(merged, p_df, c_df)
                st.session_state.update({
                    "recs_df": recs, "merged_df": merged,
                    "products_df": p_df, "customers_df": c_df,
                    "cp_matrix": cp_matrix, "analysis_done": True,
                })
                st.success("✅ Sample data loaded!")

    st.markdown("---")
    st.markdown("### Filters")
    if st.session_state["analysis_done"] and st.session_state["recs_df"] is not None:
        recs_df = st.session_state["recs_df"]
        segments = ["All"] + sorted(recs_df["Segment"].dropna().unique().tolist())
        regions = ["All"] + sorted(recs_df["Region"].dropna().unique().tolist())
        sel_segment = st.selectbox("Segment", segments)
        sel_region = st.selectbox("Region", regions)
        sel_levels = st.multiselect(
            "Rec. Level", options=[1, 2, 3, 4],
            format_func=lambda x: LEVEL_LABELS.get(x, str(x)),
            default=[1, 2, 3, 4],
        )
        min_propensity = st.slider("Min Propensity (%)", 0, 100, 0)
        st.session_state["filters"] = {
            "segment": sel_segment, "region": sel_region,
            "levels": sel_levels, "min_propensity": min_propensity,
        }
    else:
        st.caption("Run analysis to enable filters.")

    st.markdown("---")
    st.caption("Cross-Sell Intelligence v1.0")


# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state["analysis_done"]:
    st.markdown("""
    <div class="hero-banner">
        <h1 style="margin:0;font-size:1.8rem">🎯 Cross-Sell Intelligence</h1>
        <p style="color:#8B949E;margin-top:0.5rem;font-size:1rem">
        Identify revenue opportunities through hierarchical gap analysis &amp; collaborative filtering.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>📊 Gap Analysis</strong><br>
        Detects missing products across Product Line → Group → Class hierarchy.
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
        <strong>🤝 Collaborative Filtering</strong><br>
        Suggests new Product Lines based on similar customers' behaviour.
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-box">
        <strong>💰 Revenue Prediction</strong><br>
        Propensity-weighted predicted revenue for each recommendation.
        </div>""", unsafe_allow_html=True)

    st.info("👈 Select **Use Sample Data** in the sidebar and click **Load Sample Data** to begin.")
    st.stop()


# ── Apply filters ──────────────────────────────────────────────────────────────
recs_df = st.session_state["recs_df"].copy()
merged_df = st.session_state["merged_df"]
products_df = st.session_state["products_df"]
customers_df = st.session_state["customers_df"]
cp_matrix = st.session_state["cp_matrix"]
filters = st.session_state.get("filters", {})

if filters.get("segment") and filters["segment"] != "All":
    recs_df = recs_df[recs_df["Segment"] == filters["segment"]]
if filters.get("region") and filters["region"] != "All":
    recs_df = recs_df[recs_df["Region"] == filters["region"]]
if filters.get("levels"):
    recs_df = recs_df[recs_df["Level"].isin(filters["levels"])]
if filters.get("min_propensity", 0) > 0:
    recs_df = recs_df[recs_df["PropensityScore"] >= filters["min_propensity"]]


# ── Executive Summary ──────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Executive Summary</p>', unsafe_allow_html=True)

total_opp = recs_df["PredictedRevenue"].sum()
num_customers = recs_df["CustomerID"].nunique()
num_recs = len(recs_df)
avg_propensity = recs_df["PropensityScore"].mean()
top_opp = recs_df.groupby("CustomerID")["PredictedRevenue"].sum().max()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue Opportunity", f"€{total_opp:,.0f}")
c2.metric("Customers with Gaps", f"{num_customers}")
c3.metric("Total Recommendations", f"{num_recs}")
c4.metric("Avg. Propensity Score", f"{avg_propensity:.1f}%")
c5.metric("Top Customer Opp.", f"€{top_opp:,.0f}")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📋 Opportunities", "📈 Analytics", "🗺️ Customer Detail", "🔍 Data Explorer"])

# ── TAB 1: Ranked Opportunity Table ───────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Ranked Lead Table – Sorted by Revenue Opportunity</p>', unsafe_allow_html=True)

    # Customer-level summary
    summary = (
        recs_df.groupby(["CustomerID", "CustomerName", "Segment", "Region", "Industry"])
        .agg(
            TotalOpportunity=("PredictedRevenue", "sum"),
            NumRecommendations=("ProductID", "count"),
            AvgPropensity=("PropensityScore", "mean"),
            TopProduct=("ProductName", "first"),
        )
        .reset_index()
        .sort_values("TotalOpportunity", ascending=False)
        .reset_index(drop=True)
    )
    summary.index += 1
    summary["TotalOpportunity"] = summary["TotalOpportunity"].apply(lambda v: f"€{v:,.0f}")
    summary["AvgPropensity"] = summary["AvgPropensity"].apply(lambda v: f"{v:.1f}%")

    st.dataframe(
        summary.rename(columns={
            "CustomerName": "Customer", "TotalOpportunity": "Opportunity",
            "NumRecommendations": "# Recs", "AvgPropensity": "Avg Propensity",
            "TopProduct": "Top Product",
        }),
        use_container_width=True,
        height=420,
    )

    st.markdown('<p class="section-header">All Recommendations</p>', unsafe_allow_html=True)
    display_cols = ["CustomerName", "ProductName", "Level", "Reason", "PropensityScore", "AvgPrice", "PredictedRevenue", "Type"]
    display = recs_df[display_cols].copy()
    display["Level"] = display["Level"].map(LEVEL_LABELS)
    display["PredictedRevenue"] = display["PredictedRevenue"].apply(lambda v: f"€{v:,.2f}")
    display["AvgPrice"] = display["AvgPrice"].apply(lambda v: f"€{v:,.2f}")
    display["PropensityScore"] = display["PropensityScore"].apply(lambda v: f"{v}%")

    st.dataframe(
        display.rename(columns={
            "CustomerName": "Customer", "ProductName": "Product",
            "PropensityScore": "Propensity", "AvgPrice": "Avg Price",
            "PredictedRevenue": "Predicted Revenue",
        }),
        use_container_width=True,
        height=500,
    )

    csv = recs_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Export Recommendations CSV", csv, "cross_sell_recommendations.csv", "text/csv")


# ── TAB 2: Analytics Suite ────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Revenue Analytics</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_revenue_by_region(recs_df), use_container_width=True)
    with col_b:
        st.plotly_chart(chart_revenue_by_segment(recs_df), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(chart_opportunity_by_level(recs_df), use_container_width=True)
    with col_d:
        st.plotly_chart(chart_propensity_distribution(recs_df), use_container_width=True)

    st.markdown('<p class="section-header">Conversion Probability vs. Deal Size</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_scatter_propensity_vs_revenue(recs_df), use_container_width=True)

    st.markdown('<p class="section-header">Top Customers by Opportunity</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_top_customers(recs_df), use_container_width=True)

    st.markdown('<p class="section-header">Customer × Product Purchase Matrix</p>', unsafe_allow_html=True)
    st.plotly_chart(
        chart_heatmap_customer_product(cp_matrix, customers_df, products_df),
        use_container_width=True,
    )


# ── TAB 3: Customer Detail View ────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Customer Detail View</p>', unsafe_allow_html=True)

    customer_names = sorted(recs_df["CustomerName"].dropna().unique().tolist())
    sel_customer = st.selectbox("Select Customer", customer_names)

    if sel_customer:
        cust_recs = recs_df[recs_df["CustomerName"] == sel_customer].copy()
        cust_id = cust_recs["CustomerID"].iloc[0]
        cust_meta = customers_df[customers_df["CustomerID"] == cust_id]

        # Customer profile
        if not cust_meta.empty:
            m = cust_meta.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Segment", m.get("Segment", "—"))
            col2.metric("Region", m.get("Region", "—"))
            col3.metric("Industry", m.get("Industry", "—"))
            col4.metric("Annual Revenue", f"€{m.get('AnnualRevenue', 0):,.0f}" if pd.notna(m.get("AnnualRevenue")) else "—")

        total_cust_opp = cust_recs["PredictedRevenue"].sum()
        st.markdown(f"**Total Opportunity: `€{total_cust_opp:,.2f}`** across {len(cust_recs)} recommendations")

        # Products currently owned
        if cust_id in cp_matrix.index:
            owned_pids = cp_matrix.loc[cust_id]
            owned_pids = owned_pids[owned_pids == 1].index.tolist()
            owned_names = products_df[products_df["ProductID"].isin(owned_pids)]["ProductName"].tolist()
            if owned_names:
                st.markdown("**Currently Owns:** " + " · ".join([f"`{n}`" for n in owned_names]))

        st.markdown("---")
        st.markdown("#### Recommendations")

        for _, row in cust_recs.sort_values("PredictedRevenue", ascending=False).iterrows():
            level_class = f"badge-l{row['Level']}"
            level_label = LEVEL_LABELS.get(row["Level"], "")
            with st.expander(f"**{row['ProductName']}** — €{row['PredictedRevenue']:,.2f} predicted"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Propensity Score", f"{row['PropensityScore']}%")
                c2.metric("Avg Unit Price", f"€{row['AvgPrice']:,.2f}")
                c3.metric("Avg Quantity", f"{row['AvgQty']:.1f}")
                st.markdown(f"**Reason:** {row['Reason']}")
                st.markdown(f"**Type:** {row['Type']}  |  **Level:** {level_label}")
                st.markdown(f"**Product Line:** {row['ProductLine']}  ·  **Group:** {row['ProductGroup']}  ·  **Class:** {row['ProductClass']}")


# ── TAB 4: Data Explorer ──────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">Raw Data Explorer</p>', unsafe_allow_html=True)

    subtab1, subtab2, subtab3, subtab4 = st.tabs(["Transactions", "Products", "Customers", "Merged"])

    with subtab1:
        st.dataframe(st.session_state["merged_df"][["TransactionID", "CustomerID", "ProductID", "Quantity", "UnitPrice", "Revenue"]].head(200) if "TransactionID" in st.session_state["merged_df"].columns else st.session_state["merged_df"].head(200), use_container_width=True)
    with subtab2:
        st.dataframe(products_df, use_container_width=True)
    with subtab3:
        st.dataframe(customers_df, use_container_width=True)
    with subtab4:
        st.dataframe(st.session_state["merged_df"].head(200), use_container_width=True)
