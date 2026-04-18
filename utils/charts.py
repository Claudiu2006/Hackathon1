"""
Plotly chart builders for the Cross-Sell Intelligence dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

PALETTE = {
    "primary": "#0F4C81",
    "accent": "#E8630A",
    "success": "#1B8A5A",
    "warn": "#C9A800",
    "bg": "#0D1117",
    "surface": "#161B22",
    "text": "#E6EDF3",
    "muted": "#8B949E",
    "level1": "#2563EB",
    "level2": "#7C3AED",
    "level3": "#0891B2",
    "level4": "#D97706",
}

LEVEL_COLORS = {
    1: PALETTE["level1"],
    2: PALETTE["level2"],
    3: PALETTE["level3"],
    4: PALETTE["level4"],
}

LEVEL_LABELS = {
    1: "L1 – Product Line",
    2: "L2 – Product Group",
    3: "L3 – Product Class",
    4: "New Idea (CF)",
}


def dark_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE["text"], size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["muted"], size=12),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["muted"])),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    )
    return fig


def chart_revenue_by_region(recs_df: pd.DataFrame) -> go.Figure:
    grouped = (
        recs_df.groupby("Region")["PredictedRevenue"]
        .sum()
        .reset_index()
        .sort_values("PredictedRevenue", ascending=True)
    )
    fig = go.Figure(go.Bar(
        x=grouped["PredictedRevenue"],
        y=grouped["Region"],
        orientation="h",
        marker_color=PALETTE["accent"],
        text=grouped["PredictedRevenue"].apply(lambda v: f"€{v:,.0f}"),
        textposition="outside",
        textfont=dict(color=PALETTE["text"]),
    ))
    dark_layout(fig, "Revenue Opportunity by Region", height=350)
    fig.update_xaxes(title_text="Predicted Revenue (€)")
    return fig


def chart_revenue_by_segment(recs_df: pd.DataFrame) -> go.Figure:
    grouped = (
        recs_df.groupby("Segment")["PredictedRevenue"]
        .sum()
        .reset_index()
    )
    fig = px.pie(
        grouped, names="Segment", values="PredictedRevenue",
        color_discrete_sequence=[PALETTE["primary"], PALETTE["accent"], PALETTE["success"]],
        hole=0.55,
    )
    fig.update_traces(textfont_color=PALETTE["text"], pull=[0.03] * len(grouped))
    dark_layout(fig, "Revenue Split by Customer Segment", height=350)
    return fig


def chart_propensity_distribution(recs_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=recs_df["PropensityScore"],
        nbinsx=20,
        marker_color=PALETTE["primary"],
        marker_line_color=PALETTE["accent"],
        marker_line_width=1,
        opacity=0.85,
    ))
    dark_layout(fig, "Distribution of Propensity Scores (%)", height=320)
    fig.update_xaxes(title_text="Propensity Score (%)")
    fig.update_yaxes(title_text="# Recommendations")
    return fig


def chart_scatter_propensity_vs_revenue(recs_df: pd.DataFrame) -> go.Figure:
    sample = recs_df.copy()
    sample["LevelLabel"] = sample["Level"].map(LEVEL_LABELS).fillna("Other")
    sample["LevelColor"] = sample["Level"].map(LEVEL_COLORS).fillna(PALETTE["muted"])

    fig = px.scatter(
        sample,
        x="PropensityScore",
        y="PredictedRevenue",
        color="LevelLabel",
        size="PredictedRevenue",
        size_max=30,
        hover_data=["CustomerID", "ProductName", "Reason"],
        color_discrete_map={v: LEVEL_COLORS[k] for k, v in LEVEL_LABELS.items()},
        labels={"PropensityScore": "Propensity Score (%)", "PredictedRevenue": "Predicted Revenue (€)"},
    )
    dark_layout(fig, "Conversion Probability vs. Deal Size", height=400)
    return fig


def chart_top_customers(recs_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    top = (
        recs_df.groupby(["CustomerID", "CustomerName"])["PredictedRevenue"]
        .sum()
        .reset_index()
        .nlargest(top_n, "PredictedRevenue")
        .sort_values("PredictedRevenue")
    )
    label = top.apply(
        lambda r: r["CustomerName"] if pd.notna(r.get("CustomerName")) else r["CustomerID"], axis=1
    )
    fig = go.Figure(go.Bar(
        x=top["PredictedRevenue"],
        y=label,
        orientation="h",
        marker=dict(
            color=top["PredictedRevenue"],
            colorscale=[[0, PALETTE["primary"]], [1, PALETTE["accent"]]],
            showscale=False,
        ),
        text=top["PredictedRevenue"].apply(lambda v: f"€{v:,.0f}"),
        textposition="outside",
        textfont=dict(color=PALETTE["text"]),
    ))
    dark_layout(fig, f"Top {top_n} Customers by Revenue Opportunity", height=400)
    fig.update_xaxes(title_text="Total Predicted Revenue (€)")
    return fig


def chart_opportunity_by_level(recs_df: pd.DataFrame) -> go.Figure:
    grouped = (
        recs_df.groupby("Level")["PredictedRevenue"]
        .sum()
        .reset_index()
    )
    grouped["LevelLabel"] = grouped["Level"].map(LEVEL_LABELS)
    colors = grouped["Level"].map(LEVEL_COLORS)

    fig = go.Figure(go.Bar(
        x=grouped["LevelLabel"],
        y=grouped["PredictedRevenue"],
        marker_color=colors,
        text=grouped["PredictedRevenue"].apply(lambda v: f"€{v:,.0f}"),
        textposition="outside",
        textfont=dict(color=PALETTE["text"]),
    ))
    dark_layout(fig, "Revenue Opportunity by Recommendation Level", height=350)
    fig.update_yaxes(title_text="Predicted Revenue (€)")
    return fig


def chart_heatmap_customer_product(
    cp_matrix: pd.DataFrame,
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    max_customers: int = 15,
    max_products: int = 16,
) -> go.Figure:
    sub_cust = cp_matrix.index[:max_customers]
    sub_prod = cp_matrix.columns[:max_products]
    mat = cp_matrix.loc[sub_cust, sub_prod]

    cust_labels = [
        customers_df[customers_df["CustomerID"] == c]["CustomerName"].values[0]
        if c in customers_df["CustomerID"].values else c
        for c in sub_cust
    ]
    prod_labels = [
        products_df[products_df["ProductID"] == p]["ProductName"].values[0]
        if p in products_df["ProductID"].values else p
        for p in sub_prod
    ]

    fig = go.Figure(go.Heatmap(
        z=mat.values,
        x=prod_labels,
        y=cust_labels,
        colorscale=[[0, PALETTE["surface"]], [1, PALETTE["accent"]]],
        showscale=False,
        xgap=2,
        ygap=2,
    ))
    dark_layout(fig, "Customer × Product Purchase Matrix", height=450)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    return fig
