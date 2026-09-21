"""
Cereal Market Intelligence & Market Entry Assistant
Single-file Streamlit app.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push this file + requirements.txt + the .xlsx data file to a GitHub repo,
    then point Streamlit Cloud at app.py. Set HUGGINGFACEHUB_API_TOKEN (and/or
    OPENAI_API_KEY) under the app's "Secrets" panel -- NEVER hardcode a token
    in this file. This app reads credentials only from st.secrets / env vars.
"""

import os
from typing import Union, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # picks up a local .env file for local dev (HUGGINGFACEHUB_API_TOKEN=..., etc.)


def get_secret(key: str, default=None):
    """Read a credential from Streamlit Cloud's secrets.toml if it exists, otherwise
    fall back to an environment variable (populated by .env locally via load_dotenv()).
    Safe to call even when no secrets.toml exists at all (normal for local dev) --
    st.secrets raises StreamlitSecretNotFoundError in that case, which we catch here."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

# ============================================================================
# PAGE CONFIG + LIGHT STYLING
# ============================================================================
st.set_page_config(
    page_title="Cereal Market Intelligence",
    page_icon="🥣",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.5rem; max-width: 1250px;}
    .metric-card {
        background: #f7f7fa; border: 1px solid #eaeaf0; border-radius: 10px;
        padding: 14px 18px; text-align: center;
    }
    .metric-card h3 {margin: 0; font-size: 0.85rem; color: #6b6b7b; font-weight: 600;}
    .metric-card p {margin: 4px 0 0 0; font-size: 1.6rem; font-weight: 700; color: #1f2430;}
    .insight-box {
        background: #eef6ff; border-left: 4px solid #2E75B6; border-radius: 6px;
        padding: 10px 16px; margin: 10px 0; font-size: 0.95rem;
    }
    .opportunity-box {
        background: #fff3e6; border-left: 4px solid #e8792a; border-radius: 6px;
        padding: 10px 16px; margin: 10px 0; font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_FILE = "USA_Breakfast_Cereal_FMCG_L52W_Synthetic_Dataset_Updated.xlsx"
VECTOR_STORE_DIR = "review_vector_store"


def metric_card(label: str, value: str, col):
    col.markdown(f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>', unsafe_allow_html=True)


# ============================================================================
# DATA LOADING + CLEANING  (cached so the app doesn't reload on every click)
# ============================================================================
@st.cache_data(show_spinner="Loading and cleaning category data...")
def load_data():
    sheets = pd.read_excel(DATA_FILE, sheet_name=None)

    product_df = sheets["Product_Master"].copy()
    perf_df = sheets["L52W_Performance"].copy()
    survey_df = sheets["Consumer_Survey"].copy()
    reviews_df = sheets["Consumer_Reviews"].copy()
    launches_df = sheets["Product_Launches"].copy()

    # --- bug fix: brand names with stray whitespace (e.g. 'Morning Peak ') ---
    perf_df["Brand"] = perf_df["Brand"].astype(str).str.strip()

    perf_df.rename(columns={
        "Base Price ($)": "Base_Price",
        "Avg Price ($)": "Avg_Price",
        "Revenue ($)": "Revenue",
        "Stores Selling": "Stores_Selling",
        "Stores Available": "Stores_Available",
        "Item Name": "Item_name",
    }, inplace=True)

    product_df.rename(columns={
        "Item Name": "Item_name",
        "Pack Size": "Pack_Size",
        "Pack UOM": "Pack_UOM",
        "Base Price ($)": "Base_Price",
    }, inplace=True)

    # Discount % column's raw scale is unreliable (see project data-QA notes) -> derive it
    perf_df["Discount_calc_%"] = (1 - perf_df["Avg_Price"] / perf_df["Base_Price"]) * 100

    # Flavor casing fix ('cinnamon' vs 'Cinnamon') so it lines up with the survey exactly
    product_df["Flavor"] = product_df["Flavor"].astype(str).str.strip().str.title()

    product_df = build_price_tiers(product_df)

    return product_df, perf_df, survey_df, reviews_df, launches_df


# ============================================================================
# ANALYTICAL FUNCTION LIBRARY  (same 10 functions from the project notebook)
# ============================================================================
def get_brand_performance(perf_df: pd.DataFrame, brand: Optional[str] = None) -> dict:
    total_units = perf_df["Units"].sum()
    total_revenue = perf_df["Revenue"].sum()
    scope_df = perf_df if brand is None else perf_df[perf_df["Brand"] == brand]
    if brand is not None and scope_df.empty:
        raise ValueError(f"Brand not found in data: {brand!r}")

    units = scope_df["Units"].sum()
    revenue = scope_df["Revenue"].sum()
    avg_selling_price = revenue / units if units else 0.0
    avg_distribution_acv = scope_df["ACV Reach %"].mean() * 100
    revenue_per_acv_point = revenue / avg_distribution_acv if avg_distribution_acv else 0.0
    units_per_acv_point = units / avg_distribution_acv if avg_distribution_acv else 0.0

    return {
        "Brand": brand if brand else "TOTAL CATEGORY",
        "SKU_Count": int(scope_df["SKU"].nunique()),
        "Units": int(units),
        "Revenue_$": round(float(revenue), 0),
        "Unit_Market_Share_%": round(units / total_units * 100, 2),
        "Revenue_Market_Share_%": round(revenue / total_revenue * 100, 2),
        "Avg_Selling_Price_$": round(float(avg_selling_price), 2),
        "Avg_Distribution_%ACV": round(float(avg_distribution_acv), 1),
        "Revenue_per_ACV_point_$": round(float(revenue_per_acv_point), 0),
        "Units_per_ACV_point": round(float(units_per_acv_point), 1),
        "Promo_Incidence_%": round(float(scope_df["Promo Flag"].mean() * 100), 1),
    }


def compare_brands(perf_df: pd.DataFrame, brands: list) -> pd.DataFrame:
    rows = [get_brand_performance(perf_df, b) for b in brands]
    return pd.DataFrame(rows).set_index("Brand")


def analyze_distribution_productivity(perf_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    all_brands = perf_df["Brand"].unique()
    table = compare_brands(perf_df, list(all_brands))
    table = table.sort_values("Revenue_per_ACV_point_$", ascending=False)
    return table[["SKU_Count", "Units", "Revenue_$", "Avg_Distribution_%ACV", "Revenue_per_ACV_point_$"]].head(top_n)


def analyze_price_performance(perf_df: pd.DataFrame, n_tiers: int = 4) -> pd.DataFrame:
    df = perf_df.copy()
    df["Price_Band"] = pd.qcut(df["Avg_Price"], q=n_tiers, duplicates="drop")
    out = df.groupby("Price_Band", observed=True).agg(
        SKU_Count=("SKU", "nunique"),
        Units=("Units", "sum"),
        Revenue=("Revenue", "sum"),
        Avg_Distribution_ACV=("ACV Reach %", lambda x: round(x.mean() * 100, 1)),
    )
    out["Units_per_SKU"] = (out["Units"] / out["SKU_Count"]).round(0)
    out["Revenue_per_SKU"] = (out["Revenue"] / out["SKU_Count"]).round(0)
    out.index = out.index.astype(str)
    return out


def build_price_tiers(product_df: pd.DataFrame, price_col: str = "Base_Price") -> pd.DataFrame:
    out = product_df.copy()
    out["Price_Tier"] = pd.qcut(out[price_col], q=3, labels=["Value", "Mainstream", "Premium"])
    return out


def analyze_demand_vs_supply(consumer_df: pd.DataFrame, product_df: pd.DataFrame,
                              demand_col: str, supply_col: str) -> pd.DataFrame:
    demand_share = consumer_df[demand_col].value_counts(normalize=True).mul(100).round(1)
    supply_share = product_df[supply_col].value_counts(normalize=True).mul(100).round(1)
    combined = pd.DataFrame({"Demand_%": demand_share, "Supply_%": supply_share}).fillna(0)
    combined["Gap_(Demand-Supply)"] = (combined["Demand_%"] - combined["Supply_%"]).round(1)

    d_med, s_med = combined["Demand_%"].median(), combined["Supply_%"].median()

    def quadrant(row):
        if row["Demand_%"] >= d_med and row["Supply_%"] < s_med:
            return "High Demand / Low Supply (OPPORTUNITY)"
        elif row["Demand_%"] >= d_med and row["Supply_%"] >= s_med:
            return "High Demand / High Supply (Contested)"
        elif row["Demand_%"] < d_med and row["Supply_%"] >= s_med:
            return "Low Demand / High Supply (Oversupplied)"
        return "Low Demand / Low Supply (Niche)"

    combined["Quadrant"] = combined.apply(quadrant, axis=1)
    combined.index.name = "Category"
    return combined.sort_values("Gap_(Demand-Supply)", ascending=False)


def analyze_claims_landscape(product_df: pd.DataFrame, perf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    exploded = product_df.assign(Claim=product_df["Claims"].str.split(";")).explode("Claim")
    exploded["Claim"] = exploded["Claim"].str.strip()

    sku_counts = exploded.groupby("Claim")["SKU"].nunique().sort_values(ascending=False)
    result = sku_counts.to_frame("SKU_Count")
    result["%_of_SKUs"] = (result["SKU_Count"] / product_df["SKU"].nunique() * 100).round(1)

    if perf_df is not None:
        claim_to_skus = exploded.groupby("Claim")["SKU"].apply(set)
        revenue_by_claim = {c: perf_df[perf_df["SKU"].isin(skus)]["Revenue"].sum() for c, skus in claim_to_skus.items()}
        result["Revenue_$"] = pd.Series(revenue_by_claim).round(0)

    return result.sort_values("SKU_Count", ascending=False)


def find_competitive_whitespace(product_df: pd.DataFrame, perf_df: pd.DataFrame,
                                 dim1: str = "Format", dim2: str = "Positioning") -> pd.DataFrame:
    grid = product_df.groupby([dim1, dim2], observed=True).agg(SKU_Count=("SKU", "nunique")).reset_index()

    dims_needed = [d for d in [dim1, dim2] if d not in perf_df.columns]
    if dims_needed:
        lookup = product_df.set_index("SKU")[dims_needed]
        perf_annotated = perf_df.join(lookup, on="SKU")
    else:
        perf_annotated = perf_df.copy()

    rev = perf_annotated.groupby([dim1, dim2], observed=True).agg(
        Revenue_=("Revenue", "sum"),
        Units_=("Units", "sum"),
        Avg_Distribution_ACV=("ACV Reach %", lambda x: round(x.mean() * 100, 1)),
    ).reset_index()

    out = grid.merge(rev, on=[dim1, dim2], how="left").fillna(0)
    out["Revenue_per_SKU"] = (out["Revenue_"] / out["SKU_Count"]).round(0)
    return out.sort_values("SKU_Count")


def analyze_promotion_effectiveness(perf_df: pd.DataFrame) -> dict:
    df = perf_df.copy()
    promo_on = df[df["Promo Flag"] == 1]
    promo_off = df[df["Promo Flag"] == 0]

    summary = pd.DataFrame({
        "Promo_OFF": {
            "Rows": len(promo_off),
            "Avg_Units_per_Row": round(promo_off["Units"].mean(), 1),
            "Avg_Discount_%": round(promo_off["Discount_calc_%"].mean(), 2),
        },
        "Promo_ON": {
            "Rows": len(promo_on),
            "Avg_Units_per_Row": round(promo_on["Units"].mean(), 1),
            "Avg_Discount_%": round(promo_on["Discount_calc_%"].mean(), 2),
            "Avg_Promo_Lift_%": round((promo_on["Promo Lift %"].mean() - 1) * 100, 1),
        },
    })
    lift_vs_discount_corr = promo_on["Discount_calc_%"].corr(promo_on["Incremental Lift %"])
    return {"summary_table": summary, "discount_vs_incremental_lift_correlation": round(lift_vs_discount_corr, 3)}


def summarize_review_themes(reviews_df: pd.DataFrame, brand: Optional[str] = None) -> pd.DataFrame:
    df = reviews_df if brand is None else reviews_df[reviews_df["Brand"] == brand]
    if brand is not None and df.empty:
        raise ValueError(f"Brand not found in reviews: {brand!r}")
    theme_sentiment = pd.crosstab(df["Theme"], df["Sentiment"])
    theme_sentiment["Total"] = theme_sentiment.sum(axis=1)
    if "negative" in theme_sentiment.columns:
        theme_sentiment["Negative_%"] = (theme_sentiment["negative"] / theme_sentiment["Total"] * 100).round(1)
    return theme_sentiment.sort_values("Negative_%", ascending=False)


# ============================================================================
# LOAD DATA
# ============================================================================
product_df, perf_df, survey_df, reviews_df, launches_df = load_data()
ALL_BRANDS = sorted(perf_df["Brand"].unique())

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## 🥣 Cereal Market Intelligence")
    st.caption("AI-Powered Market Entry Assistant — U.S. Breakfast Cereal (xAOC, L52 Weeks, synthetic data)")
    st.markdown("---")
    st.markdown(
        f"""
        **Category snapshot**
        - {product_df['SKU'].nunique():,} SKUs
        - {len(ALL_BRANDS)} brands
        - {perf_df['Channel'].nunique()} channels
        - {len(reviews_df):,} consumer reviews
        - {len(survey_df):,} survey respondents
        """
    )
    st.markdown("---")
    st.caption(
        "Built with pandas analytics + a LangChain RAG/Agent layer. "
        "Numbers on this page always come from direct calculation — "
        "the AI Analyst tab is the only place an LLM generates text, "
        "and it is restricted to reporting only what its tools return."
    )

# ============================================================================
# HEADER
# ============================================================================
st.title("🥣 Cereal Market Intelligence & Market Entry Assistant")
st.markdown(
    "Explore the U.S. breakfast cereal category end-to-end: category size, competitive "
    "landscape, pricing, distribution, promotion, consumer demand gaps, and real review "
    "evidence — then ask the AI Market Analyst a direct question."
)

tabs = st.tabs([
    "🏠 Category Overview",
    "🏷️ Brand Explorer",
    "💰 Price & Promotion",
    "🎯 Demand vs Supply",
    "🧩 Competitive White Space",
    "🌿 Claims Landscape",
    "💬 Reviews & Sentiment",
    "🤖 AI Market Analyst",
])

# ============================================================================
# TAB 1 — CATEGORY OVERVIEW
# ============================================================================
with tabs[0]:
    cat = get_brand_performance(perf_df, brand=None)
    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card("Total Revenue", f"${cat['Revenue_$']:,.0f}", c1)
    metric_card("Total Units", f"{cat['Units']:,}", c2)
    metric_card("SKUs", f"{cat['SKU_Count']:,}", c3)
    metric_card("Avg. Distribution (%ACV)", f"{cat['Avg_Distribution_%ACV']}%", c4)
    metric_card("Promo Incidence", f"{cat['Promo_Incidence_%']}%", c5)

    st.markdown("###")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue by Channel")
        channel_rev = perf_df.groupby("Channel")["Revenue"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(channel_rev, x="Channel", y="Revenue", text_auto=".2s", color="Channel")
        fig.update_layout(showlegend=False, yaxis_title="Revenue ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("SKU Mix by Format")
        format_mix = product_df["Format"].value_counts().reset_index()
        format_mix.columns = ["Format", "SKU_Count"]
        fig = px.pie(format_mix, names="Format", values="SKU_Count", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Revenue by Price Tier")
        tier_rev = (
            perf_df.merge(product_df[["SKU", "Price_Tier"]], on="SKU", how="left")
            .groupby("Price_Tier", observed=True)["Revenue"].sum().reset_index()
        )
        fig = px.bar(tier_rev, x="Price_Tier", y="Revenue", color="Price_Tier", text_auto=".2s")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Top 10 Brands by Revenue")
        brand_rev = perf_df.groupby("Brand")["Revenue"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(brand_rev.sort_values("Revenue"), x="Revenue", y="Brand", orientation="h", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2 — BRAND EXPLORER
# ============================================================================
with tabs[1]:
    st.subheader("Compare Brands Side by Side")
    default_brands = list(perf_df.groupby("Brand")["Revenue"].sum().sort_values(ascending=False).head(3).index)
    selected_brands = st.multiselect("Select brands to compare", ALL_BRANDS, default=default_brands)

    if selected_brands:
        cmp_df = compare_brands(perf_df, selected_brands)
        st.dataframe(cmp_df, use_container_width=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            fig = px.bar(cmp_df.reset_index(), x="Brand", y="Revenue_Market_Share_%", color="Brand", text_auto=True)
            fig.update_layout(showlegend=False, title="Revenue Market Share (%)")
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            fig = px.bar(cmp_df.reset_index(), x="Brand", y="Revenue_per_ACV_point_$", color="Brand", text_auto=True)
            fig.update_layout(showlegend=False, title="Revenue per Point of Distribution ($)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one brand above to see its KPIs.")

    st.markdown("---")
    st.subheader("Distribution Productivity Leaderboard")
    st.caption("Which brands generate the most revenue per point of ACV distribution — efficiency, not just size.")
    top_n = st.slider("Show top N brands", 5, 20, 10)
    prod_table = analyze_distribution_productivity(perf_df, top_n)
    st.dataframe(prod_table, use_container_width=True)
    fig = px.bar(prod_table.reset_index(), x="Brand", y="Revenue_per_ACV_point_$", text_auto=".2s")
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3 — PRICE & PROMOTION
# ============================================================================
with tabs[2]:
    st.subheader("Price-Performance")
    n_tiers = st.slider("Number of price bands", 2, 6, 4)
    price_perf = analyze_price_performance(perf_df, n_tiers)
    st.dataframe(price_perf, use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        fig = px.bar(price_perf.reset_index(), x="Price_Band", y="Revenue_per_SKU", text_auto=".2s")
        fig.update_layout(title="Revenue per SKU by Price Band", xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        fig = px.bar(price_perf.reset_index(), x="Price_Band", y="Units_per_SKU", text_auto=".2s")
        fig.update_layout(title="Units per SKU by Price Band", xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box">Note: this shows association, not causation — price is not '
        'randomly assigned in this data, so "higher price = higher productivity" here should be '
        'read as a pattern to investigate further, not a proven price recommendation.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Promotion Effectiveness")
    promo = analyze_promotion_effectiveness(perf_df)
    st.dataframe(promo["summary_table"], use_container_width=True)
    st.metric("Correlation: discount depth vs. incremental lift", promo["discount_vs_incremental_lift_correlation"])
    st.caption(
        "Uses a price-derived discount (Discount_calc_%), not the dataset's raw 'Discount %' "
        "column, whose scale did not match the README's stated encoding during data QA."
    )

# ============================================================================
# TAB 4 — DEMAND VS SUPPLY
# ============================================================================
with tabs[3]:
    st.subheader("Where Does Consumer Demand Outpace What's on Shelf?")
    st.caption("Compares the % share of survey respondents preferring a category value against the % share of SKUs offering it.")

    dim_map = {
        "Format": ("Preferred Format", "Format"),
        "Price Tier": ("Preferred Price Tier", "Price_Tier"),
        "Flavor": ("Preferred Flavor", "Flavor"),
    }
    dim_choice = st.selectbox("Compare demand vs. supply for:", list(dim_map.keys()))
    demand_col, supply_col = dim_map[dim_choice]

    result = analyze_demand_vs_supply(survey_df, product_df, demand_col, supply_col)

    fig = go.Figure()
    max_val = max(result["Demand_%"].max(), result["Supply_%"].max()) * 1.15
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                              line=dict(dash="dash", color="gray"), name="Demand = Supply", showlegend=True))
    fig.add_trace(go.Scatter(
        x=result["Supply_%"], y=result["Demand_%"], mode="markers+text",
        text=result.index, textposition="top center",
        marker=dict(size=14, color=result["Gap_(Demand-Supply)"], colorscale="RdYlGn",
                    colorbar=dict(title="Gap"), cmid=0),
        name="Category values",
    ))
    fig.update_layout(
        xaxis_title="Supply Share (% of SKUs)", yaxis_title="Demand Share (% of survey respondents)",
        title=f"Demand vs. Supply — {dim_choice}", height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(result, use_container_width=True)

    top_gap = result.iloc[0]
    if top_gap["Gap_(Demand-Supply)"] > 0:
        st.markdown(
            f'<div class="opportunity-box"><b>Largest gap:</b> {result.index[0]} shows '
            f'{top_gap["Demand_%"]}% consumer demand vs. only {top_gap["Supply_%"]}% of current SKU supply '
            f'— a +{top_gap["Gap_(Demand-Supply)"]} point gap worth investigating further.</div>',
            unsafe_allow_html=True,
        )

# ============================================================================
# TAB 5 — COMPETITIVE WHITE SPACE
# ============================================================================
with tabs[4]:
    st.subheader("Competitive White-Space Grid")
    st.caption("Low SKU-count / low-revenue cells are potential white space — but only worth pursuing if Tab 4 also shows demand support.")

    dims_available = ["Format", "Positioning", "Price_Tier", "Brand"]
    cc1, cc2, cc3 = st.columns(3)
    dim1 = cc1.selectbox("Dimension 1", dims_available, index=0)
    dim2 = cc2.selectbox("Dimension 2", dims_available, index=1)
    metric_choice = cc3.selectbox("Color by", ["SKU_Count", "Revenue_", "Revenue_per_SKU"])

    if dim1 == dim2:
        st.warning("Choose two different dimensions.")
    else:
        ws = find_competitive_whitespace(product_df, perf_df, dim1, dim2)
        pivot = ws.pivot(index=dim1, columns=dim2, values=metric_choice)

        # Built with go.Heatmap (not px.imshow) -- px.imshow's internal dtype check
        # (`img.dtype == np.bool`) breaks on newer numpy where that alias was removed.
        # go.Heatmap avoids that code path entirely.
        z = pivot.values
        text = np.where(pd.isna(z), "", np.round(z, 0).astype(str))
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            hoverongaps=False,
        ))
        fig.update_layout(title=f"{metric_choice} — {dim1} x {dim2}", height=550)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Lowest-SKU-count combinations (potential white space):**")
        st.dataframe(ws.sort_values("SKU_Count").head(10), use_container_width=True)

# ============================================================================
# TAB 6 — CLAIMS LANDSCAPE
# ============================================================================
with tabs[5]:
    st.subheader("Product Claims Landscape")
    st.caption("Each SKU can carry multiple claims (e.g. 'Organic; Vegan; Whole Grain') — this counts SKU coverage per individual claim.")

    claims = analyze_claims_landscape(product_df, perf_df)
    cc1, cc2 = st.columns(2)
    with cc1:
        fig = px.bar(claims.reset_index(), x="%_of_SKUs", y="Claim", orientation="h", text_auto=True)
        fig.update_layout(title="% of SKUs Carrying Each Claim", yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        fig = px.bar(claims.reset_index(), x="Revenue_$", y="Claim", orientation="h", text_auto=".2s")
        fig.update_layout(title="Revenue Associated with Each Claim", yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(claims, use_container_width=True)

# ============================================================================
# TAB 7 — REVIEWS & SENTIMENT
# ============================================================================
with tabs[6]:
    st.subheader("Consumer Review Themes")
    brand_filter = st.selectbox("Filter by brand (optional)", ["All Brands"] + ALL_BRANDS)
    scoped_brand = None if brand_filter == "All Brands" else brand_filter

    themes = summarize_review_themes(reviews_df, scoped_brand)
    st.dataframe(themes, use_container_width=True)

    sentiment_cols = [c for c in ["positive", "neutral", "negative"] if c in themes.columns]
    fig = px.bar(themes.reset_index(), x="Theme", y=sentiment_cols, barmode="stack",
                 color_discrete_map={"positive": "#2ca02c", "neutral": "#bbbbbb", "negative": "#d62728"})
    fig.update_layout(title="Sentiment Breakdown by Theme", xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Semantic Search over Real Reviews (RAG)")
    st.caption(
        "Searches actual review text using the vector store built in the project notebook's "
        "RAG section. Requires the `review_vector_store/` folder to exist alongside this app."
    )

    query = st.text_input("Search consumer reviews, e.g. 'complaints about packaging'")
    if query:
        try:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            from langchain_community.vectorstores import FAISS

            hf_token = get_secret("HUGGINGFACEHUB_API_TOKEN")
            embedding_model = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                huggingfacehub_api_token=hf_token,
            )
            vector_store = FAISS.load_local(
                VECTOR_STORE_DIR, embedding_model, allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            results = retriever.invoke(query)

            for doc in results:
                st.markdown(
                    f"**{doc.metadata['Brand']}** | {doc.metadata['Sentiment']} | "
                    f"{doc.metadata['Theme']} | Rating: {doc.metadata['Rating']}/5"
                )
                st.write(doc.page_content.split("Review: ")[-1])
                st.markdown("---")
        except FileNotFoundError:
            st.info(
                f"No vector store found at `{VECTOR_STORE_DIR}/`. Run Part 4 of the project "
                "notebook first to build and save it, then copy that folder next to this app."
            )
        except Exception as e:
            st.error(f"Review search unavailable right now: {type(e).__name__}: {e}")

# ============================================================================
# TAB 8 — AI MARKET ANALYST (Agent)
# ============================================================================
with tabs[7]:
    st.subheader("🤖 Ask the AI Market Analyst")
    st.caption(
        "Answers are generated by an LLM Agent that can call the analytical functions above as "
        "tools, and retrieve real review text via RAG. It is instructed to never state a number "
        "that didn't come from a tool call."
    )

    with st.expander("How this works / why numbers here are trustworthy"):
        st.markdown(
            "- **Calculations** (revenue, share, distribution, etc.) always come from the same "
            "Python functions used in the tabs above — the Agent calls them as tools, it does "
            "not calculate or guess numbers itself.\n"
            "- **Consumer opinions** are retrieved from real review text via the RAG retriever, "
            "not generated from the LLM's general knowledge.\n"
            "- **The LLM only orchestrates**: it decides which tool(s) a question needs, calls "
            "them, and writes the final summary from their results."
        )

    @st.cache_resource(show_spinner="Setting up the AI Analyst (first question only)...")
    def build_agent():
        from langchain_core.tools import tool
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        hf_token = get_secret("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "No HUGGINGFACEHUB_API_TOKEN found. Add it under Streamlit Cloud 'Secrets' "
                "or your local .env file -- never hardcode it in this file."
            )

        model_repo_id = get_secret("HF_MODEL_REPO_ID", "openai/gpt-oss-120b")
        llm = HuggingFaceEndpoint(
            repo_id=model_repo_id, task="text-generation", temperature=0.1,
            huggingfacehub_api_token=hf_token,
        )
        model = ChatHuggingFace(llm=llm)

        @tool
        def get_brand_performance_tool(brand: str) -> dict:
            """Get KPI performance for one cereal brand: units, revenue, market share, effective
            selling price, distribution (%ACV), revenue per point of distribution, and promo
            incidence. Use this whenever the user asks how a specific named brand is performing."""
            return get_brand_performance(perf_df, brand)

        @tool
        def compare_brands_tool(brands: list) -> dict:
            """Compare KPI performance across multiple named cereal brands side by side."""
            return compare_brands(perf_df, brands).reset_index().to_dict(orient="records")

        @tool
        def get_category_overview_tool() -> dict:
            """Get total category-level KPIs (all brands combined)."""
            return get_brand_performance(perf_df, brand=None)

        @tool
        def analyze_distribution_productivity_tool(top_n: int = 10) -> dict:
            """Rank brands by revenue generated per point of distribution (ACV %)."""
            return analyze_distribution_productivity(perf_df, top_n).reset_index().to_dict(orient="records")

        @tool
        def analyze_price_performance_tool(n_tiers: int = 4) -> dict:
            """Break the category into price bands and compare units/revenue productivity per band."""
            return analyze_price_performance(perf_df, n_tiers).reset_index().astype(str).to_dict(orient="records")

        @tool
        def analyze_demand_vs_supply_tool(demand_column: str, supply_column: str) -> dict:
            """Compare consumer preference share against SKU supply share for a matching category.
            Valid pairs: demand_column='Preferred Format'/supply_column='Format',
            'Preferred Price Tier'/'Price_Tier', or 'Preferred Flavor'/'Flavor'."""
            return analyze_demand_vs_supply(survey_df, product_df, demand_column, supply_column).reset_index().to_dict(orient="records")

        @tool
        def analyze_claims_landscape_tool() -> dict:
            """Get SKU coverage and revenue for each individual product claim."""
            return analyze_claims_landscape(product_df, perf_df).reset_index().to_dict(orient="records")

        @tool
        def find_competitive_whitespace_tool(dim1: str = "Format", dim2: str = "Positioning") -> dict:
            """Get a grid of SKU count, revenue, and distribution across two product dimensions
            (choose from: Format, Positioning, Price_Tier, Brand)."""
            return find_competitive_whitespace(product_df, perf_df, dim1, dim2).to_dict(orient="records")

        @tool
        def analyze_promotion_effectiveness_tool() -> dict:
            """Compare promoted vs non-promoted sales performance."""
            res = analyze_promotion_effectiveness(perf_df)
            return {"summary": res["summary_table"].to_dict(), "discount_vs_lift_correlation": res["discount_vs_incremental_lift_correlation"]}

        @tool
        def summarize_review_themes_tool(brand: str = None) -> dict:
            """Get a breakdown of review themes by sentiment, ranked by % negative."""
            return summarize_review_themes(reviews_df, brand).reset_index().to_dict(orient="records")

        @tool
        def retrieve_consumer_reviews_tool(query: str) -> str:
            """Retrieve the most relevant real consumer review excerpts for a qualitative
            question. Always use this for what consumers said/felt/complained about -- never
            invent review content."""
            try:
                from langchain_huggingface import HuggingFaceEndpointEmbeddings
                from langchain_community.vectorstores import FAISS
                embedding_model = HuggingFaceEndpointEmbeddings(
                    model="sentence-transformers/all-MiniLM-L6-v2", huggingfacehub_api_token=hf_token,
                )
                vector_store = FAISS.load_local(VECTOR_STORE_DIR, embedding_model, allow_dangerous_deserialization=True)
                docs = vector_store.as_retriever(search_kwargs={"k": 5}).invoke(query)
                return "\n\n".join(
                    f"[{d.metadata['Brand']} | {d.metadata['Sentiment']} | {d.metadata['Theme']}] "
                    + d.page_content.split("Review: ")[-1] for d in docs
                )
            except Exception:
                return "Review vector store not available -- build it via the project notebook's RAG section first."

        all_tools = [
            get_brand_performance_tool, compare_brands_tool, get_category_overview_tool,
            analyze_distribution_productivity_tool, analyze_price_performance_tool,
            analyze_demand_vs_supply_tool, analyze_claims_landscape_tool,
            find_competitive_whitespace_tool, analyze_promotion_effectiveness_tool,
            summarize_review_themes_tool, retrieve_consumer_reviews_tool,
        ]

        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Cereal Market Analyst Agent helping a team evaluate a U.S.
breakfast cereal market-entry opportunity.

STRICT RULES:
1. NEVER state a number (revenue, units, %, price, count) unless it came directly from a tool
   result in this conversation. If you don't have a tool result for a number, say you don't
   have that data rather than estimating it.
2. For any claim about what consumers think, prefer, or complain about, use
   retrieve_consumer_reviews_tool or summarize_review_themes_tool -- do not invent consumer
   opinions from general knowledge.
3. If a question needs multiple pieces of evidence, call multiple tools and combine results.
4. Be explicit about which tool produced which fact when you summarize your answer."""),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(model, all_tools, agent_prompt)
        return AgentExecutor(agent=agent, tools=all_tools, verbose=True, max_iterations=8,
                              max_execution_time=60, handle_parsing_errors=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    user_question = st.chat_input("Ask about brands, price, distribution, promotions, or consumer opinions...")

    if user_question:
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            try:
                agent_executor = build_agent()
                with st.spinner("Thinking (calling tools)..."):
                    result = agent_executor.invoke({"input": user_question})
                answer = result["output"]
                st.write(answer)
                st.session_state.chat_history.append(("assistant", answer))
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                err_text = str(e)
                if "402" in err_text or "Payment Required" in err_text or "credits" in err_text.lower():
                    st.error(
                        "The AI Analyst's free HuggingFace inference credits are exhausted for "
                        "this period. The rest of this app (all other tabs) is unaffected since "
                        "it runs on direct calculation, not the LLM. Options: wait for credits to "
                        "reset, switch HF_MODEL_REPO_ID to a smaller free model, or add an "
                        "OPENAI_API_KEY and swap the LLM."
                    )
                else:
                    st.error(f"Agent error ({type(e).__name__}): {err_text}")

    st.markdown("---")
    st.caption("Example questions: \"How is Harvest Grove performing compared to Bright Bowl?\" · "
               "\"Where does consumer demand for price tier not match the market?\" · "
               "\"What do consumers complain about regarding convenience?\"")
