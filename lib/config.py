"""Shared configuration, constants, and the compliance disclosure string."""

import math
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Stock Market Analyst"

DISCLOSURE = (
    "This dashboard is for educational and informational purposes only. "
    "It is not financial advice, not a recommendation to buy or sell any security, "
    "and is not personalized to your situation. Consult a licensed advisor before "
    "making investment decisions."
)


def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)


def get_anthropic_key():
    return (get_secret("ANTHROPIC_API_KEY") or "").strip() or None


def get_fred_key():
    return (get_secret("FRED_API_KEY") or "").strip() or None


def get_gemini_key():
    return (get_secret("GEMINI_API_KEY") or "").strip() or None


def get_groq_key():
    return (get_secret("GROQ_API_KEY") or "").strip() or None


def _is_invalid_number(value) -> bool:
    """True if value is None, NaN, or not something that can be treated as a number."""
    if value is None or not isinstance(value, (int, float)):
        return True
    try:
        return math.isnan(value)
    except TypeError:
        return True


def fmt_large_number(value, currency="$"):
    """Format a large number with B/M/T suffixes, e.g. 1234567890 -> '$1.23B'."""
    if _is_invalid_number(value):
        return "—"
    try:
        abs_value = abs(value)
        sign = "-" if value < 0 else ""
        if abs_value >= 1e12:
            return f"{sign}{currency}{abs_value / 1e12:.2f}T"
        if abs_value >= 1e9:
            return f"{sign}{currency}{abs_value / 1e9:.2f}B"
        if abs_value >= 1e6:
            return f"{sign}{currency}{abs_value / 1e6:.2f}M"
        if abs_value >= 1e3:
            return f"{sign}{currency}{abs_value / 1e3:.2f}K"
        return f"{sign}{currency}{abs_value:,.2f}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(value, decimals=2):
    """Format a fraction (0.123) as a percentage string ('12.30%')."""
    if _is_invalid_number(value):
        return "—"
    try:
        return f"{value * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_num(value, decimals=2):
    if _is_invalid_number(value):
        return "—"
    try:
        return f"{value:,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def render_footer():
    """Render the standard disclosure footer. Import and call at the bottom of every page."""
    import streamlit as st

    st.divider()
    st.caption(DISCLOSURE)


_STRUCTURAL_CSS = """
    .main .block-container {
        font-size: 16px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }
    h1, h2, h3, h4 { font-weight: 600; letter-spacing: -0.01em; }
    h1 { font-size: 1.9rem !important; }
    h2, .stApp h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.1rem !important; }
    hr { margin: 0.9rem 0; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.6rem; }
    [data-testid="stMetric"] {
        border-radius: 10px;
        padding: 0.7rem 1rem 0.6rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        font-family: 'SFMono-Regular', 'Consolas', 'Menlo', monospace;
        font-size: 1.45rem;
        font-weight: 600;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'SFMono-Regular', 'Consolas', 'Menlo', monospace;
        font-size: 0.85rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {
        border-radius: 12px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 4px; }
    [data-testid="stTabs"] [data-baseweb="tab"] { font-weight: 500; font-size: 0.9rem; }
    [data-testid="stTextInput"] input,
    [data-baseweb="select"] > div { border-radius: 8px; }
    [data-testid="stSegmentedControl"] { border-radius: 8px; padding: 2px; }
    [data-testid="stPlotlyChart"] { border-radius: 10px; overflow: hidden; }
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .stButton > button { border-radius: 8px; font-weight: 500; }
    [data-testid="stAlert"] { border-radius: 10px; }
"""



def get_plotly_theme() -> dict:
    """Return Plotly update_layout() kwargs for the TradingView dark theme."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(color="#d1d4dc"),
    )


def inject_base_style():
    """Inject structural CSS and render the sidebar header."""
    with st.sidebar:
        st.markdown("### 📈 Market Analyst")

    st.markdown(f"<style>{_STRUCTURAL_CSS}</style>", unsafe_allow_html=True)
