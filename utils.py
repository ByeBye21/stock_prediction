"""
utils.py - Helper functions and constants
"""

POPULAR_STOCKS = {
    "BIST (Borsa Istanbul)": {
        "THYAO.IS": "Turkish Airlines",
        "ASELS.IS": "Aselsan",
        "SISE.IS": "Sisecam",
        "EREGL.IS": "Eregli Iron & Steel",
        "KCHOL.IS": "Koc Holding",
        "GARAN.IS": "Garanti BBVA",
        "AKBNK.IS": "Akbank",
        "BIMAS.IS": "BIM",
        "TUPRS.IS": "Tupras",
        "SAHOL.IS": "Sabanci Holding",
    },
    "US (NYSE / NASDAQ)": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet (Google)",
        "AMZN": "Amazon",
        "TSLA": "Tesla",
        "NVDA": "NVIDIA",
        "META": "Meta (Facebook)",
        "NFLX": "Netflix",
        "JPM": "JPMorgan Chase",
        "V": "Visa",
    },
    "Europe": {
        "VOW3.DE": "Volkswagen",
        "SAP.DE": "SAP",
        "SIE.DE": "Siemens",
        "MC.PA": "LVMH",
        "OR.PA": "L'Oreal",
    },
}


COLORS = {
    "primary": "#00D4AA",
    "secondary": "#7C3AED",
    "accent": "#F59E0B",
    "success": "#10B981",
    "danger": "#EF4444",
    "info": "#3B82F6",
    "warning": "#F97316",
    "bg_dark": "#0E1117",
    "bg_card": "#1A1F2E",
    "text": "#FAFAFA",
    "text_muted": "#9CA3AF",
}

MODEL_COLORS = {
    "Logistic Regression": "#3B82F6",
    "Random Forest": "#10B981",
    "XGBoost": "#F59E0B",
    "LightGBM": "#8B5CF6",
}


CONFIDENCE_LEVELS = {
    "80%": 0.80,
    "90%": 0.90,
    "95%": 0.95,
    "99%": 0.99,
}


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number into a human-readable abbreviated string."""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.{decimals}f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    return f"{value:.{decimals}f}"


def format_currency(value: float, currency: str = "") -> str:
    """Format a number as currency."""
    return f"{currency}{value:,.2f}"


def format_percent(value: float) -> str:
    """Format a number as percentage."""
    return f"{value:.2f}%"


DISCLAIMER = """
**Disclaimer**: This application is for **educational and research purposes only**.
The predictions presented here are **not investment advice**.
**Do not** base your financial decisions on this data.
Past performance is not a guarantee of future results.
"""
