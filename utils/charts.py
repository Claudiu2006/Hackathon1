"""
Plotly chart builders - uses real column schema.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

PALETTE = {
    "primary": "#0F4C81", "accent": "#E8630A", "success": "#1B8A5A",
    "warn": "#C9A800",   "bg": "#0D1117",      "surface": "#161B22",
    "text": "#E6EDF3",   "muted": "#8B949E",
    "level1": "#2563EB", "level2": "#7C3AED",
    "level3": "#0891B2", "level4": "#D97706",
}
LEVEL_COLORS = {1: PALETTE["level1"], 2: PALETTE["level2"],
                3: PALETTE["level3"], 4: PALETTE["level4"]}
LEVEL_LABELS = {1: "L1 – Product Line", 2: "L2 – Product Group",
                3: "L3 – Product Class", 4: "New Idea (CF)"}


def dark_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE["text"], size=16)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["muted"], size=12), height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["muted"])),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    )
    return fig


def chart_revenue_by_region(recs_df):
    col = "danfoss_region_2"
    if col not in recs_df.columns:
        return go.Figure()
    grouped = (recs_df.groupby(col)["PredictedRevenue"].sum()
               .reset_index().sort_values("PredictedRevenue", ascending=True))
    fig = go.Figure(go.Bar(
        x=grouped["PredictedRevenue"], y=grouped[col], orientation="h",
        marker_color=PALETTE["accent"],
        text=grouped["PredictedRevenue"].apply(lambda v: f"€{v:,.0f}"),
        textposition="outside", textfont=dict(color=PALETTE["text"]),
    ))
    dark_layout(fig, "Revenue Opportunity by Region", 380)
    fig.update_xaxes(title_text="Predicted Revenue (€)")
    return fig


def chart_revenue_by_segment(recs_df):
    col = "sf_primary_segment"
    if col not in recs_df.columns:
        return go.Figure()
    grouped = recs_df.groupby(col)["PredictedRevenue"].sum().reset_index()
    grouped = grouped[grouped[col] != "Not Assigned"]
    fig = px.pie(grouped, names=col, values="PredictedRevenue",
                 color_discrete_sequence=[PALETTE["primary"], PALETTE["accent"],
                                          PALETTE["success"], PALETTE["warn"],
                                          "#8B5CF6", "#06B6D4"], hole=0.55)
    fig.update_traces(textfont_color=PALETTE["text"], pull=[0.03]*len(grouped))
    dark_layout(fig, "Revenue by Primary Segment", 380)
    return fig


def chart_revenue_by_customer_type(recs_df):
    col = "customer_type_2"
    if col not in recs_df.columns:
        return go.Figure()
    grouped = (recs_df.groupby(col)["PredictedRevenue"].sum()
               .reset_index().sort_values("PredictedRevenue", ascending=True).tail(15))
    fig = go.Figure(go.Bar(
        x=grouped["PredictedRevenue"], y=grouped[col], orientation="h",
        marker_color=PALETTE["primary"],
        text=grouped["PredictedRevenue"].apply(lambda v: f"€{v:,.0f}"),
        textposition="outside", textfont=dict(color=PALETTE["text"]),
    ))
    dark_layout(fig, "Revenue Opportunity by Customer Type", 420)
    fig.update_xaxes(title_text="Predicted Revenue (€)")
    return fig


def chart_revenue_by_country(recs_df):
    col = "text_country"
    if col not in recs_df.columns:
        return go.Figure()
    grouped = (recs_df.groupby(col)["PredictedRevenue"].sum()
               .reset_index().nlargest(15, "PredictedRevenue")
               .sort_values("PredictedRevenue", ascending=True))
    fig = go.Figure(go.Bar(
        x=grouped["PredictedRevenue"], y=grouped[col], orientation="h",
        marker=dict(color=grouped["PredictedRevenue"],
                    colorscale=[[0, PALETTE["primary"]], [1, PALETTE["accent"]]],
                    showscale=False),
        text=grouped["PredictedRevenue"].apply(lambda v: f"€{v:,.0f}"),
        textposition="outside", textfont=dict(color=PALETTE["text"]),
    ))
    dark_layout(fig, "Top 15 Countries by Revenue Opportunity", 450)
    fig.update_xaxes(title_text="Predicted Revenue (€)")
    return fig


def chart_propensity_distribution(recs_df):
    fig = go.Figure(go.Histogram(
        x=recs_df["PropensityScore"], nbinsx=25,
        marker_color=PALETTE["primary"],
        marker_line_color=PALETTE["accent"], marker_line_width=1, opacity=0.85,
    ))
    dark_layout(fig, "Distribution of Propensity Scores (%)", 320)
    fig.update_xaxes(title_text="Propensity Score (%)")
    fig.update_yaxes(title_text="# Recommendations")
    return fig


def chart_scatter_propensity_vs_revenue(recs_df):
    sample = recs_df.copy()
    sample["LevelLabel"] = sample["Level"].map(LEVEL_LABELS).fillna("Other")
    fig = px.scatter(
        sample, x="PropensityScore", y="PredictedRevenue",
        color="LevelLabel", size="PredictedRevenue", size_max=28,
        hover_data=["soldto", "ProductLine", "Reason"],
        color_discrete_map={v: LEVEL_COLORS[k] for k, v in LEVEL_LABELS.items()},
        labels={"PropensityScore": "Propensity Score (%)",
                "PredictedRevenue": "Predicted Revenue (€)"},
    )
    dark_layout(fig, "Conversion Probability vs. Deal Size", 420)
    return fig


def chart_top_customers(recs_df, top_n=15):
    name_col = "soldto" if "soldto" in recs_df.columns else "CustomerKey"
    top = (recs_df.groupby(["CustomerKey", name_col])["PredictedRevenue"]
           .sum().reset_index()
           .nlargest(top_n, "PredictedRevenue")
           .sort_values("PredictedRevenue"))
    fig = go.Figure(go.Bar(
        x=top["PredictedRevenue"], y=top[name_col], orientation="h",
        marker=dict(color=top["PredictedRevenue"],
                    colorscale=[[0, PALETTE["primary"]], [1, PALETTE["accent"]]],
                    showscale=False),
        text=top["PredictedRevenue"].apply(lambda v: f"€{v:,.0f}"),
        textposition="outside", textfont=dict(color=PALETTE["text"]),
    ))
    dark_layout(fig, f"Top {top_n} Customers by Revenue Opportunity", 480)
    fig.update_xaxes(title_text="Total Predicted Revenue (€)")
    return fig


def chart_opportunity_by_level(recs_df):
    grouped = recs_df.groupby("Level")["PredictedRevenue"].sum().reset_index()
    grouped["LevelLabel"] = grouped["Level"].map(LEVEL_LABELS)
    fig = go.Figure(go.Bar(
        x=grouped["LevelLabel"], y=grouped["PredictedRevenue"],
        marker_color=[LEVEL_COLORS.get(l, PALETTE["muted"]) for l in grouped["Level"]],
        text=grouped["PredictedRevenue"].apply(lambda v: f"€{v:,.0f}"),
        textposition="outside", textfont=dict(color=PALETTE["text"]),
    ))
    dark_layout(fig, "Revenue Opportunity by Recommendation Level", 360)
    fig.update_yaxes(title_text="Predicted Revenue (€)")
    return fig


def chart_pl_gap_heatmap(recs_df):
    """Show which PLs are most commonly missing across customers (L1 only)."""
    l1 = recs_df[recs_df["Level"] == 1]
    if l1.empty:
        return go.Figure()

    pivot = l1.groupby(["danfoss_region_2" if "danfoss_region_2" in l1.columns else "CustomerKey",
                         "ProductLine"])["PredictedRevenue"].sum().unstack(fill_value=0)
    # Limit to top regions/customers
    pivot = pivot.head(12)

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0, PALETTE["surface"]], [1, PALETTE["accent"]]],
        showscale=True, xgap=2, ygap=2,
        colorbar=dict(tickfont=dict(color=PALETTE["muted"])),
    ))
    dark_layout(fig, "Revenue Gap Heatmap — Region × Product Line", 420)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    return fig
