"""
app.py — Stock Direction Forecast App (Advanced Streamlit)

Capstone-level, comprehensive, professional web interface.

Run:
    streamlit run app.py

Features:
    - Session state persistence across reruns
    - Progress bar with step-by-step updates
    - Date range presets (1Y, 3Y, 5Y, Max)
    - Classification-only model comparison
    - Probability-based direction visualization
    - Radar chart for model comparison
    - Interactive forecast controls (model picker, CI slider)
    - Font Awesome icons (no emojis)
    - Glassmorphism CSS with animations
    - Tooltips on all inputs and metrics
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from data_loader import (
    download_stock_data,
    get_stock_info,
    download_external_data,
)
from features import build_features, get_feature_columns
from models import train_and_evaluate, get_best_model_name, get_model_selection_score
from forecast import forecast_future, get_confidence_band
from utils import (
    POPULAR_STOCKS, COLORS, MODEL_COLORS,
    format_number, format_currency, DISCLAIMER,
    CONFIDENCE_LEVELS,
)

# ─── Page Config ───
st.set_page_config(
    page_title="Stock Direction Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "### Stock Direction Forecast\n\n"
            "Direction-first stock analysis with classification-focused prediction.\n\n"
            "**Models:** Logistic Regression, Random Forest, XGBoost, LightGBM\n\n"
            "**Focus:** 20-day direction classification with technical, market, VIX, and sector features\n\n"
            "**Projection:** Calibrated probability outlook with a tuned decision threshold and optional price-path scenario band\n\n"
            "**Technologies:** Scikit-learn, XGBoost, LightGBM, Plotly, Streamlit, yfinance\n\n"
            "**Author:** Younes Rahebi"
        ),
    },
)

# ─── Font Awesome + Custom CSS ───
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    /* ─── HIDE Streamlit auto-generated header anchor links (broken X icons) ─── */
    a.headerlink,
    .stMarkdown a[href^="#"],
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a,
    [data-testid="stMarkdownContainer"] a.headerlink,
    .element-container a.headerlink {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        opacity: 0 !important;
    }
    /* Also hide the anchor icon that Streamlit injects */
    .stMarkdown h1::after, .stMarkdown h2::after, .stMarkdown h3::after,
    .stMarkdown h4::after, .stMarkdown h5::after, .stMarkdown h6::after {
        display: none !important;
    }
    /* ─── Glassmorphism Metric Cards ─── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg,
            rgba(26, 31, 46, 0.9) 0%,
            rgba(37, 43, 59, 0.8) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow:
            0 12px 40px rgba(0, 212, 170, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #9CA3AF !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #FAFAFA, #D1D5DB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ─── Tab Styles ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(26, 31, 46, 0.5);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 170, 0.1);
    }

    /* ─── Gradient Header ─── */
    .main-header {
        text-align: center;
        padding: 10px 0 24px 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #00D4AA 0%, #3B82F6 50%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        animation: fadeInDown 0.8s ease;
    }
    .main-header h1 i {
        -webkit-text-fill-color: #00D4AA;
        margin-right: 10px;
    }
    .main-header p {
        color: #6B7280;
        font-size: 1rem;
        animation: fadeInUp 0.8s ease 0.2s both;
    }

    /* ─── Stock Info Card ─── */
    .stock-card {
        background: linear-gradient(135deg,
            rgba(26, 31, 46, 0.95) 0%,
            rgba(37, 43, 59, 0.9) 100%);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 18px 26px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
        animation: fadeIn 0.6s ease;
    }
    .stock-card .ticker {
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00D4AA, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stock-card .info {
        color: #6B7280;
        margin-left: 14px;
        font-size: 0.9rem;
    }

    /* ─── Section Headers with FA Icons ─── */
    .fa-section {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin: 0.8rem 0 0.4rem 0;
        font-size: 1.25rem;
        font-weight: 600;
        color: #FAFAFA;
    }
    .fa-section i {
        color: #00D4AA;
        font-size: 1.1rem;
    }
    /* Kill the ::before pseudo-element that renders a broken X box */
    .fa-section::before,
    .fa-section::after {
        display: none !important;
        content: none !important;
    }
    .sidebar-section i {
        color: #00D4AA;
        margin-right: 6px;
    }

    /* ─── Fade Animations ─── */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════

def init_session_state():
    """Initialize session state — results survive page reruns."""
    defaults = {
        "analysis_done": False,
        "df": None,
        "df_feat": None,
        "feature_cols": None,
        "results": None,
        "stock_info": None,
        "params": None,
        "external_df": None,
        "deep_analysis_table": None,
        "deep_analysis_best": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ═══════════════════════════════════════════════
# SIDEBAR — User Inputs
# ═══════════════════════════════════════════════

def render_sidebar():
    """Sidebar: stock selection, date range, model and forecast settings."""
    with st.sidebar:
        st.markdown('<div class="fa-section" style="font-size:1.5rem;"><i class="fa-solid fa-sliders sidebar-section"></i> Settings</div>', unsafe_allow_html=True)
        st.markdown("---")

        # ─ Stock Selection ─
        st.markdown('<div class="fa-section"><i class="fa-solid fa-chart-pie sidebar-section"></i> Stock Selection</div>', unsafe_allow_html=True)
        input_method = st.radio(
            "Stock selection method",
            ["Select from List", "Manual Input"],
            horizontal=True,
            label_visibility="collapsed",
            help="Pick from popular stocks or enter a custom ticker symbol.",
        )

        if input_method == "Select from List":
            market = st.selectbox(
                "Exchange", list(POPULAR_STOCKS.keys()),
                help="Which stock exchange to browse?",
            )
            stocks = POPULAR_STOCKS[market]
            ticker_options = [f"{code} — {name}" for code, name in stocks.items()]
            selected = st.selectbox("Stock", ticker_options)
            ticker = selected.split(" — ")[0]
        else:
            ticker = st.text_input(
                "Ticker Symbol", value="THYAO.IS",
                placeholder="e.g. AAPL, MSFT, THYAO.IS",
                help="Enter a valid Yahoo Finance ticker symbol.",
            )

        st.markdown("---")

        # ─ Date Range ─
        st.markdown('<div class="fa-section"><i class="fa-regular fa-calendar sidebar-section"></i> Date Range</div>', unsafe_allow_html=True)

        # Preset buttons
        st.caption("Quick Select:")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        preset_start = None
        with pcol1:
            if st.button("1Y", use_container_width=True, help="Last 1 year"):
                preset_start = datetime.now() - timedelta(days=365)
        with pcol2:
            if st.button("3Y", use_container_width=True, help="Last 3 years"):
                preset_start = datetime.now() - timedelta(days=3*365)
        with pcol3:
            if st.button("5Y", use_container_width=True, help="Last 5 years"):
                preset_start = datetime.now() - timedelta(days=5*365)
        with pcol4:
            if st.button("Max", use_container_width=True, help="All available data"):
                preset_start = datetime(2000, 1, 1)

        # Save preset to session state
        if preset_start is not None:
            st.session_state["preset_start"] = preset_start.date()

        default_start = st.session_state.get(
            "preset_start",
            (datetime.now() - timedelta(days=5*365)).date()
        )

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start", value=default_start,
                max_value=datetime.now(),
                help="Data download start date.",
            )
        with col2:
            end_date = st.date_input(
                "End", value=datetime.now(),
                min_value=start_date,
                max_value=datetime.now(),
                help="Data download end date.",
            )

        st.markdown("---")

        # ─ Prediction Settings ─
        st.markdown('<div class="fa-section"><i class="fa-solid fa-bullseye sidebar-section"></i> Prediction Settings</div>', unsafe_allow_html=True)

        st.caption("Classification-only setup")

        target_horizon = st.select_slider(
            "Target Horizon (days)",
            options=[1, 5, 10, 20],
            value=20,
            help="How many business days ahead to predict direction. 20 days is the default because it has been the strongest setup in your tests.",
        )

        st.caption("External factor groups")
        include_market = st.checkbox(
            "Market Benchmark",
            value=True,
            help="Adds broad market trend and relative-strength features versus the main benchmark index. This has been the most consistently useful external factor group.",
        )
        include_vix = st.checkbox(
            "VIX (Fear Index)",
            value=True,
            help="Adds volatility-regime and fear features from the VIX. Useful for capturing market stress and risk appetite shifts.",
        )
        include_sector = st.checkbox(
            "Sector ETF Proxy",
            value=False,
            help="Adds sector-relative strength and sector-trend context using an ETF proxy such as XLK or XLF when the stock sector is known.",
        )

        use_external = any([
            include_market,
            include_vix,
            include_sector,
        ])

        training_window = st.selectbox(
            "Training Window",
            ["Auto (Recent 3Y)", "All History", "Last 3 Years", "Last 2 Years", "Last 1 Year"],
            index=0,
            help="Use more recent history to reduce regime drift. Auto keeps the last 3 years when enough data exists.",
        )

        st.markdown("---")

        # ─ Model Selection ─
        st.markdown('<div class="fa-section"><i class="fa-solid fa-microchip sidebar-section"></i> Model Selection</div>', unsafe_allow_html=True)

        model_options = [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "LightGBM",
        ]
        default_models = [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "LightGBM",
        ]

        selected_models = st.multiselect(
            "Models",
            model_options,
            default=default_models,
            help="Choose which classification models to compare. Logistic Regression is the simple benchmark; tree models usually perform best.",
        )

        # Hyperparameter tuning
        do_tuning = st.checkbox(
            "Hyperparameter Tuning",
            value=True,
            help="Automatic parameter optimization via RandomizedSearchCV for the selected models, including Logistic Regression. Increases training time.",
        )

        st.markdown("---")

        # ─ Projection Settings ─
        st.markdown('<div class="fa-section"><i class="fa-solid fa-wand-magic-sparkles sidebar-section"></i> Projection</div>', unsafe_allow_html=True)
        forecast_days = st.slider(
            "Days Ahead", 1, 30, 5,
            help="Projection length. Uncertainty grows with more days.",
        )

        st.markdown("---")

        # ─ Run Button ─
        run_analysis = st.button(
            "Run Analysis",
            type="primary",
            use_container_width=True,
        )
        deep_analyze = st.button(
            "Deep Analyze Best Setup",
            use_container_width=True,
            help="Benchmarks all 1Y/2Y/3Y training windows, all horizon options (1/5/10/20), and all external-factor combinations. It uses fast comparison mode first, then refits only the best setup.",
        )

        # About is in the native Streamlit hamburger menu (set_page_config)

    return {
        "ticker": ticker.strip().upper(),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "selected_models": selected_models,
        "forecast_days": forecast_days,
        "run_analysis": run_analysis,
        "deep_analyze": deep_analyze,
        "do_tuning": do_tuning,
        "training_window": training_window,
        "target_horizon": target_horizon,
        "use_external": use_external,
        "include_market": include_market,
        "include_vix": include_vix,
        "include_sector": include_sector,
    }


# ═══════════════════════════════════════════════
# SPARKLINE — Mini chart helper
# ═══════════════════════════════════════════════

def apply_training_window(df: pd.DataFrame, training_window: str) -> pd.DataFrame:
    """Limit the training set to more recent history when requested."""
    window_map = {
        "Last 1 Year": 252,
        "Last 2 Years": 504,
        "Last 3 Years": 756,
    }

    if training_window == "All History":
        return df
    if training_window == "Auto (Recent 3Y)":
        return df.tail(756) if len(df) > 756 else df

    bars = window_map.get(training_window)
    if bars is None:
        return df
    return df.tail(bars) if len(df) > bars else df


def get_selected_factor_label(params: dict) -> str:
    """Build a readable label for the selected external factor groups."""
    labels = []
    if params.get("include_market", False):
        labels.append("Market")
    if params.get("include_vix", False):
        labels.append("VIX")
    if params.get("include_sector", False):
        labels.append("Sector")
    return " + ".join(labels) if labels else "Technical Only"


def get_deep_factor_configs() -> list[dict]:
    """Return all external-factor combinations used by deep analysis."""
    return [
        {"label": "Technical Only", "include_market": False, "include_vix": False, "include_sector": False},
        {"label": "Market Only", "include_market": True, "include_vix": False, "include_sector": False},
        {"label": "VIX Only", "include_market": False, "include_vix": True, "include_sector": False},
        {"label": "Sector Only", "include_market": False, "include_vix": False, "include_sector": True},
        {"label": "Market + VIX", "include_market": True, "include_vix": True, "include_sector": False},
        {"label": "Market + Sector", "include_market": True, "include_vix": False, "include_sector": True},
        {"label": "VIX + Sector", "include_market": False, "include_vix": True, "include_sector": True},
        {"label": "Market + VIX + Sector", "include_market": True, "include_vix": True, "include_sector": True},
    ]


def get_available_external_groups(external_df: pd.DataFrame | None) -> set[str]:
    """Detect which external factor groups are actually available for a ticker."""
    if external_df is None or external_df.empty:
        return set()

    available = set()
    columns = set(external_df.columns)
    if any(col.startswith("Market_") for col in columns):
        available.add("market")
    if any(col.startswith("VIX_") for col in columns):
        available.add("vix")
    if any(col.startswith("Sector_") for col in columns):
        available.add("sector")
    return available


def subset_external_df(
    external_df: pd.DataFrame | None,
    include_market: bool,
    include_vix: bool,
    include_sector: bool,
) -> pd.DataFrame | None:
    """Return only the external columns needed for a specific factor combination."""
    if external_df is None or external_df.empty:
        return None

    selected_prefixes = []
    if include_market:
        selected_prefixes.append("Market_")
    if include_vix:
        selected_prefixes.append("VIX_")
    if include_sector:
        selected_prefixes.append("Sector_")

    if not selected_prefixes:
        return None

    cols = [
        col for col in external_df.columns
        if any(col.startswith(prefix) for prefix in selected_prefixes)
    ]
    if not cols:
        return None

    return external_df[cols].copy()


def prepare_feature_set(
    df_raw: pd.DataFrame,
    start_date_str: str,
    horizon: int,
    external_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build and trim a feature frame for a specific horizon and factor set."""
    df_feat = build_features(df_raw, horizon=horizon, external_df=external_df)
    df_feat = df_feat.loc[start_date_str:]
    feature_cols = get_feature_columns(df_feat)
    return df_feat, feature_cols


def train_configuration(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    training_window: str,
    selected_models: list[str],
    do_tuning: bool,
    horizon: int,
    progress_callback=None,
) -> tuple[dict, pd.DataFrame]:
    """Train and evaluate one configuration safely."""
    df_train = apply_training_window(df_feat, training_window)
    if len(df_train) < 100:
        raise ValueError(
            "Not enough data. Please select a wider date range or a longer training window "
            "(roughly 100+ trading days, and more for longer horizons)."
        )

    X = df_train[feature_cols]
    y = df_train["Target"]
    results = train_and_evaluate(
        X,
        y,
        selected_models,
        do_tuning=do_tuning,
        horizon=horizon,
        progress_callback=progress_callback,
    )
    return results, df_train


def run_deep_analysis(
    df_raw: pd.DataFrame,
    start_date_str: str,
    selected_models: list[str],
    do_tuning: bool,
    external_df_all: pd.DataFrame | None,
    progress_callback,
) -> tuple[pd.DataFrame, dict]:
    """Benchmark many realistic setups and refit the strongest one."""
    horizons = [1, 5, 10, 20]
    training_windows = ["Last 1 Year", "Last 2 Years", "Last 3 Years"]
    all_factor_configs = get_deep_factor_configs()
    available_groups = get_available_external_groups(external_df_all)

    factor_configs = []
    for config in all_factor_configs:
        required_groups = {
            name for name, enabled in [
                ("market", config["include_market"]),
                ("vix", config["include_vix"]),
                ("sector", config["include_sector"]),
            ]
            if enabled
        }
        if required_groups and not required_groups.issubset(available_groups):
            continue
        factor_configs.append(config)

    if not factor_configs:
        factor_configs = [all_factor_configs[0]]

    feature_cache = {}
    for horizon in horizons:
        for config in factor_configs:
            ext_subset = subset_external_df(
                external_df_all,
                config["include_market"],
                config["include_vix"],
                config["include_sector"],
            )
            df_feat, feature_cols = prepare_feature_set(
                df_raw,
                start_date_str,
                horizon=horizon,
                external_df=ext_subset,
            )
            feature_cache[(horizon, config["label"])] = {
                "df_feat": df_feat,
                "feature_cols": feature_cols,
                "external_df": ext_subset,
            }

    total_runs = len(horizons) * len(training_windows) * len(factor_configs)
    completed_runs = 0
    benchmark_rows = []
    best_row = None

    for horizon in horizons:
        for config in factor_configs:
            cache_entry = feature_cache[(horizon, config["label"])]
            df_feat = cache_entry["df_feat"]
            feature_cols = cache_entry["feature_cols"]

            for training_window in training_windows:
                completed_runs += 1
                progress_callback(
                    completed_runs,
                    total_runs + 1,
                    f"Deep analyze: {config['label']} | {training_window} | {horizon}-day",
                )
                try:
                    results, df_train = train_configuration(
                        df_feat,
                        feature_cols,
                        training_window,
                        selected_models,
                        do_tuning=False,
                        horizon=horizon,
                    )
                except ValueError:
                    continue

                best_model_name = get_best_model_name(results)
                best_model_metrics = results[best_model_name]["metrics"]
                row = {
                    "Target Horizon (days)": horizon,
                    "Training Window": training_window,
                    "Factors": config["label"],
                    "Best Model": best_model_name,
                    "Rows Used": len(df_train),
                    **best_model_metrics,
                }
                benchmark_rows.append(row)

                candidate_score = get_model_selection_score(best_model_metrics)
                if best_row is None or candidate_score > best_row["score"]:
                    best_row = {
                        "score": candidate_score,
                        "config": config,
                        "training_window": training_window,
                        "horizon": horizon,
                    }

    if not benchmark_rows or best_row is None:
        raise ValueError(
            "Deep analysis could not find a valid setup. Please widen the date range "
            "or reduce the number of selected models."
        )

    best_cache = feature_cache[(best_row["horizon"], best_row["config"]["label"])]
    progress_callback(
        total_runs + 1,
        total_runs + 1,
        "Refitting the best deep-analysis setup...",
    )
    best_results, _ = train_configuration(
        best_cache["df_feat"],
        best_cache["feature_cols"],
        best_row["training_window"],
        selected_models,
        do_tuning=do_tuning,
        horizon=best_row["horizon"],
    )

    best_model_name = get_best_model_name(best_results)
    best_metrics = best_results[best_model_name]["metrics"]
    best_summary = {
        "label": best_row["config"]["label"],
        "training_window": best_row["training_window"],
        "horizon": best_row["horizon"],
        "include_market": best_row["config"]["include_market"],
        "include_vix": best_row["config"]["include_vix"],
        "include_sector": best_row["config"]["include_sector"],
        "best_model": best_model_name,
        "metrics": best_metrics,
        "results": best_results,
        "feature_cols": best_cache["feature_cols"],
        "df_feat": best_cache["df_feat"],
        "external_df": best_cache["external_df"],
    }

    benchmark_df = pd.DataFrame(benchmark_rows).sort_values(
        by=["AUC-ROC (%)", "Balanced Accuracy (%)", "MCC", "Accuracy (%)", "Macro F1-Score (%)"],
        ascending=False,
    ).reset_index(drop=True)

    return benchmark_df, best_summary


def create_sparkline(values, color="#00D4AA", height=40, width=120):
    """Create a tiny trend line (sparkline) chart."""
    fig = go.Figure(go.Scatter(
        y=values, mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
    ))
    fig.update_layout(
        height=height, width=width,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════
# TAB 1 — Data Overview
# ═══════════════════════════════════════════════

def render_data_tab(df: pd.DataFrame, stock_info: dict):
    """Data table and price/volume charts."""

    # ─ Top KPI cards with sparklines ─
    cols = st.columns(5)
    last_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    change = last_close - prev_close
    change_pct = (change / prev_close) * 100

    cols[0].metric(
        "Last Close", format_currency(last_close),
        delta=f"{change_pct:+.2f}%",
        help="Most recent closing price",
    )
    cols[1].metric(
        "All-Time High", format_currency(df["High"].max()),
        help="Highest price in the selected range",
    )
    cols[2].metric(
        "All-Time Low", format_currency(df["Low"].min()),
        help="Lowest price in the selected range",
    )
    cols[3].metric(
        "Avg Volume", format_number(df["Volume"].mean()),
        help="Average daily trading volume",
    )
    cols[4].metric(
        "Trading Days", f"{len(df):,}",
        help="Total business days in dataset",
    )

    # Sparkline previews
    spark_cols = st.columns(5)
    spark_cols[0].plotly_chart(
        create_sparkline(df["Close"].tail(30).values, COLORS["primary"]),
        use_container_width=True, config={"displayModeBar": False},
    )
    spark_cols[1].plotly_chart(
        create_sparkline(df["High"].tail(30).values, COLORS["success"]),
        use_container_width=True, config={"displayModeBar": False},
    )
    spark_cols[2].plotly_chart(
        create_sparkline(df["Low"].tail(30).values, COLORS["danger"]),
        use_container_width=True, config={"displayModeBar": False},
    )
    spark_cols[3].plotly_chart(
        create_sparkline(df["Volume"].tail(30).values, COLORS["info"]),
        use_container_width=True, config={"displayModeBar": False},
    )

    st.markdown("---")

    # ─ Price Chart (Candlestick + MA) ─
    st.markdown('<div class="fa-section"><i class="fa-solid fa-chart-line"></i> Price Chart</div>', unsafe_allow_html=True)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=("", "Volume"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color=COLORS["success"],
        decreasing_line_color=COLORS["danger"],
    ), row=1, col=1)

    # SMA lines
    for period, color in [(7, COLORS["info"]), (21, COLORS["accent"]), (50, COLORS["secondary"])]:
        sma = df["Close"].rolling(period).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma, name=f"SMA {period}",
            line=dict(color=color, width=1.2, dash="dot"),
        ), row=1, col=1)

    # Volume bars
    colors_vol = [COLORS["success"] if c >= o else COLORS["danger"]
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors_vol, opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(
        height=600, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True, legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─ Data Table ─
    st.markdown('<div class="fa-section"><i class="fa-solid fa-table"></i> Data Table</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Last 20 Trading Days**")
        display_df = df.tail(20).copy()
        display_df.index = display_df.index.strftime("%Y-%m-%d")
        display_df = display_df.round(2)
        st.dataframe(display_df, use_container_width=True, height=400)
    with col_b:
        st.markdown("**Statistical Summary**")
        st.dataframe(df.describe().round(2), use_container_width=True, height=400)


# ═══════════════════════════════════════════════
# TAB 2 — Technical Analysis
# ═══════════════════════════════════════════════

def render_technical_tab(df_feat: pd.DataFrame, df_original: pd.DataFrame = None):
    """Technical indicator charts and correlation heatmap."""

    # Use original df for Close price (full date range); fall back to df_feat
    df_price = df_original if df_original is not None else df_feat

    # ─ Technical Indicators Chart ─
    st.markdown('<div class="fa-section"><i class="fa-solid fa-chart-area"></i> Technical Indicators</div>', unsafe_allow_html=True)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.35, 0.20, 0.20, 0.25],
        subplot_titles=("Price + Bollinger Bands", "RSI", "MACD", "Stochastic Oscillator"),
    )

    # Price — use full original data for complete range
    fig.add_trace(go.Scatter(
        x=df_price.index, y=df_price["Close"], name="Close",
        line=dict(color=COLORS["primary"], width=1.5),
    ), row=1, col=1)

    if "BB_Upper" in df_feat.columns:
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat["BB_Upper"], name="BB Upper",
            line=dict(color=COLORS["text_muted"], width=0.8, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat["BB_Lower"], name="BB Lower",
            line=dict(color=COLORS["text_muted"], width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(156,163,175,0.1)",
        ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat["RSI"], name="RSI",
        line=dict(color=COLORS["secondary"], width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["danger"],
                  line_width=0.8, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["success"],
                  line_width=0.8, row=2, col=1)

    # MACD
    hist_colors = [COLORS["success"] if v >= 0 else COLORS["danger"]
                   for v in df_feat["MACD_Hist"]]
    fig.add_trace(go.Bar(
        x=df_feat.index, y=df_feat["MACD_Hist"], name="MACD Hist",
        marker_color=hist_colors, opacity=0.6,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat["MACD"], name="MACD",
        line=dict(color=COLORS["info"], width=1.2),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat["MACD_Signal"], name="Signal",
        line=dict(color=COLORS["accent"], width=1.2),
    ), row=3, col=1)

    # Stochastic Oscillator
    if "Stoch_K" in df_feat.columns:
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat["Stoch_K"], name="%K",
            line=dict(color=COLORS["info"], width=1.2),
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat["Stoch_D"], name="%D",
            line=dict(color=COLORS["accent"], width=1.2),
        ), row=4, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color=COLORS["danger"],
                      line_width=0.8, row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color=COLORS["success"],
                      line_width=0.8, row=4, col=1)

    fig.update_layout(
        height=900, template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    # Show full date range on all subplots
    fig.update_xaxes(range=[df_price.index.min(), df_price.index.max()])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ─ ATR & OBV Chart ─
    if "ATR" in df_feat.columns and "OBV_Zscore" in df_feat.columns:
        st.markdown('<div class="fa-section"><i class="fa-solid fa-wave-square"></i> ATR & OBV</div>', unsafe_allow_html=True)
        fig_extra = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("ATR (Average True Range)", "OBV Z-Score"),
        )

        fig_extra.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat["ATR"], name="ATR",
            line=dict(color=COLORS["warning"], width=1.5),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.1)",
        ), row=1, col=1)

        fig_extra.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat["OBV_Zscore"], name="OBV Z-Score",
            line=dict(color=COLORS["secondary"], width=1.5),
        ), row=2, col=1)
        fig_extra.add_hline(y=0, line_dash="dash", line_color=COLORS["text_muted"],
                            line_width=0.5, row=2, col=1)

        fig_extra.update_layout(
            height=400, template="plotly_dark",
            showlegend=True, legend=dict(orientation="h", y=1.05),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_extra, use_container_width=True)

    st.markdown("---")

    # ─ Correlation Heatmap ─
    st.markdown('<div class="fa-section"><i class="fa-solid fa-fire"></i> Feature Correlation Matrix</div>', unsafe_allow_html=True)
    feature_cols = get_feature_columns(df_feat)
    corr_cols = feature_cols[:15]  # Show at most 15 features for readability
    corr = df_feat[corr_cols].corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 9},
    ))
    fig_corr.update_layout(
        height=500, template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 3 — Model Results
# ═══════════════════════════════════════════════

def render_model_tab(results: dict):
    """Classification-focused results view."""

    if not results:
        st.warning("Please select at least 1 model.")
        return

    saved_params = st.session_state.get("params", {})
    horizon = saved_params.get("target_horizon", 20)
    training_window = saved_params.get("training_window", "All History")
    factor_set = get_selected_factor_label(saved_params)
    st.caption(
        f"Mode: **Classification (Up/Down)** | Target Horizon: **{horizon}-day** | "
        f"Training Window: **{training_window}** | Factors: **{factor_set}**"
    )

    st.markdown('<div class="fa-section"><i class="fa-solid fa-scale-balanced"></i> Model Comparison</div>', unsafe_allow_html=True)
    comparison = pd.DataFrame({name: res["metrics"] for name, res in results.items()}).T
    comparison.index.name = "Model"
    highlight_max_cols = [
        c for c in ["Accuracy (%)", "Balanced Accuracy (%)", "MCC", "AUC-ROC (%)"]
        if c in comparison.columns
    ]
    classification_formats = {col: "{:.2f}" for col in comparison.columns}
    if "MCC" in classification_formats:
        classification_formats["MCC"] = "{:.3f}"
    st.dataframe(
        comparison.style.format(classification_formats).highlight_max(
            subset=highlight_max_cols,
            color="#10B98130",
        ),
        use_container_width=True,
    )

    st.markdown("---")

    st.markdown('<div class="fa-section"><i class="fa-solid fa-diagram-project"></i> Radar Chart - Model Comparison</div>', unsafe_allow_html=True)
    radar_metrics = ["Accuracy (%)", "Balanced Accuracy (%)", "Macro F1-Score (%)", "MCC", "AUC-ROC (%)"]
    radar_labels = ["Accuracy", "Bal. Acc.", "Macro F1", "MCC", "AUC-ROC"]

    fig_radar = go.Figure()
    for model_name, res in results.items():
        values = []
        for metric in radar_metrics:
            value = res["metrics"].get(metric, 0)
            if metric == "MCC":
                values.append((max(-1.0, min(1.0, value)) + 1.0) / 2.0)
            else:
                values.append(value / 100)

        color = MODEL_COLORS.get(model_name, COLORS["info"])
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=radar_labels + [radar_labels[0]],
            name=model_name,
            line=dict(color=color, width=2),
            fill="toself",
            fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.1)",
        ))

    fig_radar.update_layout(
        height=450,
        template="plotly_dark",
        polar=dict(
            bgcolor="rgba(26,31,46,0.5)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=60, r=60, t=30, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    best_name = get_best_model_name(results)
    best = results[best_name]
    decision_threshold_pct = best.get("decision_threshold", 0.5) * 100.0
    is_calibrated = best.get("is_calibrated", False)
    st.markdown(f'<div class="fa-section"><i class="fa-solid fa-trophy"></i> Best Model: <strong>{best_name}</strong></div>', unsafe_allow_html=True)
    st.caption("Models are ranked by AUC-ROC, then Balanced Accuracy, MCC, Accuracy, and Macro F1.")
    st.caption(
        f"{'Calibrated' if is_calibrated else 'Raw'} probabilities with a "
        f"{decision_threshold_pct:.1f}% decision threshold."
    )

    m = best["metrics"]
    cols = st.columns(6)
    cols[0].metric("Accuracy", f"{m.get('Accuracy (%)', 0):.1f}%", help="Overall direction accuracy")
    cols[1].metric("Bal. Accuracy", f"{m.get('Balanced Accuracy (%)', 0):.1f}%", help="Average recall across up and down classes")
    cols[2].metric("Macro F1", f"{m.get('Macro F1-Score (%)', 0):.1f}%", help="Macro-averaged F1 across up and down classes")
    cols[3].metric("MCC", f"{m.get('MCC', 0):.3f}", help="Matthews correlation coefficient; more reliable when classes are imbalanced")
    cols[4].metric("AUC-ROC", f"{m.get('AUC-ROC (%)', 50):.1f}%", help="Probability ranking quality (50%=random, 100%=perfect)")
    cols[5].metric("Threshold", f"{decision_threshold_pct:.1f}%", help="Probability cutoff used for Bullish vs Bearish classification")

    st.markdown("---")

    st.markdown('<div class="fa-section"><i class="fa-solid fa-code-compare"></i> Prediction Confidence Timeline</div>', unsafe_allow_html=True)
    st.caption("Each point is one past prediction. Higher points mean the model was more confident the stock would go up. Green points actually finished up; red points finished down.")

    dates = pd.DatetimeIndex(best["test_dates"])
    actual = pd.Series(best["y_test"], index=dates).astype(float)
    y_prob = best.get("y_prob")
    if y_prob is not None and len(y_prob) == len(dates):
        prob_up = pd.Series(np.asarray(y_prob, dtype=float) * 100.0, index=dates)
    else:
        prob_up = pd.Series(np.asarray(best["y_pred"], dtype=float) * 100.0, index=dates)

    actual_up_mask = actual >= 0.5
    actual_down_mask = ~actual_up_mask
    predicted_up_mask = prob_up >= decision_threshold_pct

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=decision_threshold_pct, fillcolor=COLORS["danger"], opacity=0.05, line_width=0)
    fig.add_hrect(y0=decision_threshold_pct, y1=100, fillcolor=COLORS["success"], opacity=0.05, line_width=0)
    fig.add_trace(go.Scatter(
        x=prob_up.index,
        y=prob_up.values,
        name=f"{best_name} Probability",
        line=dict(color=MODEL_COLORS.get(best_name, COLORS["primary"]), width=1.8),
        mode="lines",
        opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=prob_up.index[actual_up_mask],
        y=prob_up[actual_up_mask],
        mode="markers",
        name="Actual Up",
        marker=dict(
            color=COLORS["success"],
            size=9,
            symbol="circle",
            line=dict(color="#FFFFFF", width=0.6),
        ),
        customdata=np.column_stack([
            np.where(predicted_up_mask[actual_up_mask], "Bullish", "Bearish"),
            np.full(actual_up_mask.sum(), "Up"),
        ]),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Probability of Up: %{y:.1f}%<br>"
            "Model Call: %{customdata[0]}<br>"
            "Actual Result: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=prob_up.index[actual_down_mask],
        y=prob_up[actual_down_mask],
        mode="markers",
        name="Actual Down",
        marker=dict(
            color=COLORS["danger"],
            size=9,
            symbol="circle",
            line=dict(color="#FFFFFF", width=0.6),
        ),
        customdata=np.column_stack([
            np.where(predicted_up_mask[actual_down_mask], "Bullish", "Bearish"),
            np.full(actual_down_mask.sum(), "Down"),
        ]),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Probability of Up: %{y:.1f}%<br>"
            "Model Call: %{customdata[0]}<br>"
            "Actual Result: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.add_hline(
        y=decision_threshold_pct,
        line_dash="dash",
        line_color=COLORS["text_muted"],
        annotation_text=f"Decision threshold: {decision_threshold_pct:.1f}%",
    )
    fig.update_layout(
        height=500,
        template="plotly_dark",
        yaxis_title="Probability of Up (%)",
        yaxis=dict(range=[-5, 105], tickvals=[0, 25, 50, 75, 100]),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(l=0, r=0, t=70, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown('<div class="fa-section"><i class="fa-solid fa-ranking-star"></i> Feature Importance</div>', unsafe_allow_html=True)
    importance_results = {n: r for n, r in results.items() if r["feature_importance"] is not None}

    if importance_results:
        tabs_fi = st.tabs(list(importance_results.keys()))
        for tab, (name, res) in zip(tabs_fi, importance_results.items()):
            with tab:
                imp = res["feature_importance"].head(15)
                fig_imp = go.Figure(go.Bar(
                    x=imp.values,
                    y=imp.index,
                    orientation="h",
                    marker=dict(
                        color=imp.values,
                        colorscale=[[0, COLORS["info"]], [1, COLORS["primary"]]],
                    ),
                ))
                fig_imp.update_layout(
                    height=400,
                    template="plotly_dark",
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Importance Score",
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Selected models do not support feature importance.")



def render_forecast_tab(
    results: dict,
    df_original: pd.DataFrame,
    feature_cols: list,
    forecast_days: int,
):
    """Classification-based future projection chart and table."""

    def get_outlook_label(probability: float) -> str:
        if probability >= 70:
            return "Strong Bullish"
        if probability >= 60:
            return "Bullish"
        if probability >= 40:
            return "Neutral"
        if probability >= 30:
            return "Bearish"
        return "Strong Bearish"

    def get_outlook_color(probability: float) -> str:
        if probability >= 70:
            return COLORS["success"]
        if probability >= 60:
            return COLORS["primary"]
        if probability >= 40:
            return COLORS["accent"]
        if probability >= 30:
            return COLORS["warning"]
        return COLORS["danger"]

    if not results:
        st.warning("Please run the analysis first.")
        return

    saved_params = st.session_state.get("params", {})
    horizon = saved_params.get("target_horizon", 20)
    external_df = st.session_state.get("external_df", None)

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])

    with ctrl_col1:
        default_model = get_best_model_name(results)
        forecast_model_name = st.selectbox(
            "Projection Model",
            list(results.keys()),
            index=list(results.keys()).index(default_model),
            help="Select the classification model used for the future projection.",
            key="forecast_model_select",
        )

    with ctrl_col2:
        confidence_label = st.selectbox(
            "Scenario Range",
            list(CONFIDENCE_LEVELS.keys()),
            index=2,
            help="Controls the secondary price-scenario range below. 99% is wider, 80% is narrower.",
            key="forecast_confidence_select",
        )
        confidence_level = CONFIDENCE_LEVELS[confidence_label]

    with ctrl_col3:
        if "forecast_custom_days" not in st.session_state:
            st.session_state["forecast_custom_days"] = forecast_days
        custom_days = st.number_input(
            "Days Ahead",
            min_value=1,
            max_value=30,
            help="Number of business days to project forward.",
            key="forecast_custom_days",
        )

    forecast_model = results[forecast_model_name]["model"]
    decision_threshold_pct = results[forecast_model_name].get("decision_threshold", 0.5) * 100.0
    is_calibrated = results[forecast_model_name].get("is_calibrated", False)

    st.markdown(f'<div class="fa-section"><i class="fa-solid fa-wand-magic-sparkles"></i> {custom_days}-Day Future Projection</div>', unsafe_allow_html=True)
    training_window = saved_params.get("training_window", "All History")
    factor_set = get_selected_factor_label(saved_params)
    projection_input = apply_training_window(df_original, training_window)
    st.caption(
        f"Model: **{forecast_model_name}** | Scenario Range: **{confidence_label}** | "
        f"Horizon: **{horizon}-day** | Training Window: **{training_window}** | "
        f"Factors: **{factor_set}** | Method: Probability-Based Projection"
    )
    st.caption("This path is derived from direction probabilities and historical move size, not from direct price regression.")
    st.caption(
        f"{'Calibrated' if is_calibrated else 'Raw'} probabilities with a "
        f"{decision_threshold_pct:.1f}% decision threshold for Bullish/Bearish calls."
    )

    with st.spinner("Computing projection..."):
        forecast_df = forecast_future(
            model=forecast_model,
            df_original=projection_input,
            feature_cols=feature_cols,
            days=custom_days,
            horizon=horizon,
            external_df=external_df,
        )

    if forecast_df.empty:
        st.error("Could not compute the projection. There may not be enough data.")
        return

    daily_vol = projection_input["Close"].pct_change().std()
    forecast_df = get_confidence_band(forecast_df, daily_vol, confidence_level)

    last_close = df_original["Close"].iloc[-1]
    probabilities = forecast_df["Probability (%)"].astype(float)
    forecast_df["Signal"] = np.where(probabilities >= decision_threshold_pct, "▲ Bullish", "▼ Bearish")
    forecast_df["Outlook"] = probabilities.map(get_outlook_label)
    marker_colors = probabilities.map(get_outlook_color).tolist()
    final_probability = float(probabilities.iloc[-1])
    final_outlook = forecast_df["Outlook"].iloc[-1]
    final_signal = forecast_df["Signal"].iloc[-1]
    final_projected_price = float(forecast_df["Projected Price"].iloc[-1])
    final_projected_change = float(forecast_df["Projected Change (%)"].iloc[-1])

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=decision_threshold_pct, fillcolor=COLORS["danger"], opacity=0.06, line_width=0)
    fig.add_hrect(y0=decision_threshold_pct, y1=100, fillcolor=COLORS["success"], opacity=0.06, line_width=0)
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=probabilities,
        name="Probability of Up",
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=10, color=marker_colors, line=dict(color="#FFFFFF", width=0.8)),
        customdata=np.column_stack([
            forecast_df["Signal"].values,
            forecast_df["Outlook"].values,
            forecast_df["Projected Change (%)"].astype(float).values,
            forecast_df["Projected Price"].astype(float).values,
        ]),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Probability of Up: %{y:.1f}%<br>"
            "Signal: %{customdata[0]}<br>"
            "Outlook: %{customdata[1]}<br>"
            "Projected Change: %{customdata[2]:+.2f}%<br>"
            "Projected Price: %{customdata[3]:.2f}<extra></extra>"
        ),
    ))
    fig.add_hline(
        y=decision_threshold_pct,
        line_dash="dash",
        line_color=COLORS["text"],
        annotation_text=f"Decision threshold: {decision_threshold_pct:.1f}%",
    )
    fig.update_layout(
        height=460,
        template="plotly_dark",
        yaxis_title="Probability of Up (%)",
        yaxis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100]),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="fa-section"><i class="fa-solid fa-list-ol"></i> Projection Details</div>', unsafe_allow_html=True)
    display_forecast = forecast_df.copy()
    display_forecast["Date"] = display_forecast["Date"].dt.strftime("%Y-%m-%d (%A)")
    display_forecast = display_forecast.rename(columns={"Probability (%)": "Probability of Up (%)"})
    preferred_order = ["Date", "Probability of Up (%)", "Signal", "Outlook", "Projected Price", "Projected Change (%)"]
    ordered_cols = [col for col in preferred_order if col in display_forecast.columns]
    ordered_cols += [col for col in display_forecast.columns if col not in ordered_cols]
    display_forecast = display_forecast[ordered_cols]

    st.dataframe(display_forecast, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", format_currency(last_close), help="Last closing price")
    col2.metric(
        f"Day {custom_days} Up Probability",
        f"{final_probability:.1f}%",
        delta=f"{final_probability - decision_threshold_pct:+.1f} pts",
        help=f"Probability that the stock will be up after {custom_days} business days",
    )
    col3.metric(
        "Final Outlook",
        final_outlook,
        delta=final_signal,
        help="Probability-based direction label for the final future day",
    )
    if "Upper Bound" in forecast_df.columns:
        col4.metric(
            f"Scenario Range ({confidence_label})",
            f"{forecast_df['Lower Bound'].iloc[-1]:.2f} - {forecast_df['Upper Bound'].iloc[-1]:.2f}",
            delta=f"{final_projected_change:+.2f}%",
            help=f"Projected price range and projected change at {confidence_label} confidence",
        )
    else:
        col4.metric(
            f"Day {custom_days} Scenario Price",
            format_currency(final_projected_price),
            delta=f"{final_projected_change:+.2f}%",
            help=f"Projected price path {custom_days} business days from now",
        )

    st.markdown("---")
    st.warning(DISCLAIMER)


def render_deep_analysis_tab():
    """Show the grid-style benchmark that searches for the best setup."""
    benchmark_df = st.session_state.get("deep_analysis_table")
    best_summary = st.session_state.get("deep_analysis_best")

    if benchmark_df is None or benchmark_df.empty or best_summary is None:
        st.info("Run **Deep Analyze Best Setup** from the sidebar to benchmark all windows, horizons, and factor combinations.")
        return

    st.markdown('<div class="fa-section"><i class="fa-solid fa-magnifying-glass-chart"></i> Best Setup From Deep Analysis</div>', unsafe_allow_html=True)
    st.caption("Deep analysis benchmarks all 1Y / 2Y / 3Y windows, all horizon options, and all external-factor combinations in fast mode, then refits the strongest setup for the main tabs.")

    collapsed_horizons = []
    if {"Target Horizon (days)", "Training Window", "Rows Used"}.issubset(benchmark_df.columns):
        rows_by_window = (
            benchmark_df.groupby(["Target Horizon (days)", "Training Window"])["Rows Used"]
            .median()
            .unstack()
        )
        collapsed_horizons = [
            int(horizon)
            for horizon, row in rows_by_window.iterrows()
            if row.dropna().nunique() <= 1
        ]
    if collapsed_horizons:
        joined_horizons = ", ".join(f"{h}d" for h in collapsed_horizons)
        st.info(
            "For horizon(s) "
            f"{joined_horizons}, the selected date range produced the same usable history "
            "for the 1Y / 2Y / 3Y windows. In those cases, identical scores across training "
            "windows are expected because the model is seeing the same rows."
        )
    
    best_metrics = best_summary["metrics"]
    top_cols = st.columns(3)
    top_cols[0].metric("Best Horizon", f"{best_summary['horizon']} days")
    top_cols[1].metric("Best Window", best_summary["training_window"])
    top_cols[2].metric("Best Factors", best_summary["label"])

    bottom_cols = st.columns(3)
    bottom_cols[0].metric("Best Model", best_summary["best_model"])
    bottom_cols[1].metric("AUC-ROC", f"{best_metrics.get('AUC-ROC (%)', 50):.1f}%")
    bottom_cols[2].metric("Bal. Accuracy", f"{best_metrics.get('Balanced Accuracy (%)', 50):.1f}%")

    st.markdown("---")
    st.markdown('<div class="fa-section"><i class="fa-solid fa-table-cells-large"></i> Deep Analysis Ranking</div>', unsafe_allow_html=True)
    display_df = benchmark_df.copy()
    numeric_cols = [
        "Accuracy (%)",
        "Balanced Accuracy (%)",
        "Macro Precision (%)",
        "Macro Recall (%)",
        "Macro F1-Score (%)",
        "AUC-ROC (%)",
        "Decision Threshold (%)",
    ]
    format_map = {col: "{:.2f}" for col in numeric_cols if col in display_df.columns}
    if "MCC" in display_df.columns:
        format_map["MCC"] = "{:.3f}"

    st.dataframe(
        display_df.style.format(format_map).highlight_max(
            subset=[col for col in ["AUC-ROC (%)", "Balanced Accuracy (%)", "Accuracy (%)"] if col in display_df.columns],
            color="#10B98130",
        ),
        use_container_width=True,
        height=520,
    )


def main():
    """Main Streamlit application."""

    # Initialize session state
    init_session_state()

    # Header
    st.markdown("""
    <div class="main-header">
        <div style="background: linear-gradient(135deg, #00D4AA 0%, #3B82F6 50%, #7C3AED 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px; animation: fadeInDown 0.8s ease;"><i class="fa-solid fa-chart-line" style="-webkit-text-fill-color: #00D4AA; margin-right: 10px;"></i> Stock Direction Forecast</div>
        <p>Direction Classification &bull; Probability-Based Projection</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    params = render_sidebar()
    params = params.copy()

    # Validation
    if not params["ticker"]:
        st.info("Select a stock from the sidebar and click **Run Analysis** or **Deep Analyze Best Setup**.")
        return

    if not params["selected_models"]:
        st.warning("Please select at least 1 model.")
        return

    if params["start_date"] > params["end_date"]:
        st.error("Start date must be on or before the end date.")
        return

    # ─── RUN ANALYSIS ───
    # Placeholder to keep the DOM tree stable so st.tabs doesn't lose its state
    progress_container = st.empty()

    if params["run_analysis"] or params["deep_analyze"]:
        # Reset forecast tab widgets so they use fresh values
        for key in ["forecast_custom_days", "forecast_model_select", "forecast_confidence_select"]:
            if key in st.session_state:
                del st.session_state[key]

        # Progress bar
        progress = progress_container.progress(0, text="Starting analysis...")

        # Download data
        progress.progress(5, text=f"Downloading {params['ticker']} data...")
        
        # Determine fetch start date (add 90 days padding for moving averages)
        start_dt = datetime.strptime(params["start_date"], "%Y-%m-%d").date()
        fetch_start = max(
            datetime(2000, 1, 1).date(),
            start_dt - timedelta(days=90)
        )
        # Convert back to string for yfinance
        fetch_start_str = fetch_start.strftime("%Y-%m-%d")

        try:
            df = download_stock_data(
                params["ticker"], fetch_start_str, params["end_date"]
            )
        except ValueError as e:
            st.error(str(e))
            return

        progress.progress(15, text="Fetching stock info...")
        stock_info = get_stock_info(params["ticker"])

        # Download external data
        external_df_all = None
        should_download_all_external = params["deep_analyze"]
        if should_download_all_external or params.get("use_external", False):
            if should_download_all_external:
                factor_text = "Market + VIX + Sector combinations"
            else:
                selected_factor_names = []
                if params.get("include_market", False):
                    selected_factor_names.append("Market")
                if params.get("include_vix", False):
                    selected_factor_names.append("VIX")
                if params.get("include_sector", False):
                    selected_factor_names.append("Sector")
                factor_text = " + ".join(selected_factor_names) if selected_factor_names else "selected external factors"

            progress.progress(20, text=f"Downloading external data ({factor_text})...")
            try:
                external_df_all = download_external_data(
                    fetch_start,
                    params["end_date"],
                    ticker=params["ticker"],
                    sector=stock_info.get("sector", ""),
                    include_vix=True if should_download_all_external else params.get("include_vix", False),
                    include_market=True if should_download_all_external else params.get("include_market", False),
                    include_sector=True if should_download_all_external else params.get("include_sector", False),
                )
            except Exception:
                external_df_all = None

        start_date_str = params["start_date"]
        df_display = df.loc[start_date_str:]

        if params["deep_analyze"]:
            progress.progress(28, text="Preparing deep-analysis benchmark...")

            def update_deep_progress(current, total, message):
                pct = 28 + int((current / max(total, 1)) * 62)
                progress.progress(min(pct, 92), text=message)

            try:
                benchmark_df, best_summary = run_deep_analysis(
                    df,
                    start_date_str,
                    params["selected_models"],
                    params["do_tuning"],
                    external_df_all,
                    update_deep_progress,
                )
            except ValueError as e:
                st.error(str(e))
                return

            progress.progress(95, text="Preparing best setup from deep analysis...")

            best_params = params.copy()
            best_params["training_window"] = best_summary["training_window"]
            best_params["target_horizon"] = best_summary["horizon"]
            best_params["include_market"] = best_summary["include_market"]
            best_params["include_vix"] = best_summary["include_vix"]
            best_params["include_sector"] = best_summary["include_sector"]
            best_params["use_external"] = any([
                best_summary["include_market"],
                best_summary["include_vix"],
                best_summary["include_sector"],
            ])

            st.session_state["analysis_done"] = True
            st.session_state["df"] = df_display
            st.session_state["df_feat"] = best_summary["df_feat"]
            st.session_state["feature_cols"] = best_summary["feature_cols"]
            st.session_state["results"] = best_summary["results"]
            st.session_state["stock_info"] = stock_info
            st.session_state["params"] = best_params
            st.session_state["external_df"] = best_summary["external_df"]
            st.session_state["deep_analysis_table"] = benchmark_df
            st.session_state["deep_analysis_best"] = {
                "label": best_summary["label"],
                "training_window": best_summary["training_window"],
                "horizon": best_summary["horizon"],
                "best_model": best_summary["best_model"],
                "metrics": best_summary["metrics"],
            }
        else:
            external_df = subset_external_df(
                external_df_all,
                params.get("include_market", False),
                params.get("include_vix", False),
                params.get("include_sector", False),
            )

            horizon = params.get("target_horizon", 1)
            feat_label = "technical + external factors" if external_df is not None and not external_df.empty else "technical factors only"
            progress.progress(30, text=f"Feature engineering ({feat_label})...")

            try:
                df_feat, feature_cols = prepare_feature_set(
                    df,
                    start_date_str,
                    horizon=horizon,
                    external_df=external_df,
                )
                progress.progress(40, text="Training models (walk-forward validation)...")

                def update_progress(current, total, message):
                    pct = 40 + int((current / max(total, 1)) * 50)
                    progress.progress(min(pct, 90), text=message)

                results, _ = train_configuration(
                    df_feat,
                    feature_cols,
                    params.get("training_window", "Auto (Recent 3Y)"),
                    params["selected_models"],
                    params["do_tuning"],
                    horizon=horizon,
                    progress_callback=update_progress,
                )
            except ValueError as e:
                st.error(str(e))
                return

            progress.progress(95, text="Preparing results...")

            st.session_state["analysis_done"] = True
            st.session_state["df"] = df_display
            st.session_state["df_feat"] = df_feat
            st.session_state["feature_cols"] = feature_cols
            st.session_state["results"] = results
            st.session_state["stock_info"] = stock_info
            st.session_state["params"] = params
            st.session_state["external_df"] = external_df
            st.session_state["deep_analysis_table"] = None
            st.session_state["deep_analysis_best"] = None

        progress.progress(100, text="Analysis complete!")
        progress_container.empty()  # Clear progress bar cleanly after done

    # ─── DISPLAY RESULTS ───
    if not st.session_state["analysis_done"]:
        st.info("Select your settings in the sidebar and click **Run Analysis** or **Deep Analyze Best Setup**.")
        return

    # Read from session state
    df = st.session_state["df"]
    df_feat = st.session_state["df_feat"]
    feature_cols = st.session_state["feature_cols"]
    results = st.session_state["results"]
    stock_info = st.session_state["stock_info"]
    saved_params = st.session_state["params"]

    # Stock info card
    st.markdown(f"""
    <div class="stock-card">
        <span class="ticker">
            <i class="fa-solid fa-building-columns" style="-webkit-text-fill-color: #00D4AA; margin-right: 8px;"></i>
            {saved_params['ticker']}
        </span>
        <span class="info">
            {stock_info['name']} | {stock_info.get('sector', '')}
            | {stock_info.get('exchange', '')}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ─── TABS ───
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview",
        "Technical Analysis",
        "Model Results",
        "Future Projection",
        "Deep Analysis",
    ])

    with tab1:
        render_data_tab(df, stock_info)
    with tab2:
        render_technical_tab(df_feat, df)
    with tab3:
        render_model_tab(results)
    with tab4:
        render_forecast_tab(results, df, feature_cols, saved_params["forecast_days"])
    with tab5:
        render_deep_analysis_tab()


if __name__ == "__main__":
    main()
