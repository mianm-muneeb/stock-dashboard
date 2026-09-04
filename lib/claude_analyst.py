"""Groq-powered analysis: bull/bear cases, deep dives, macro pulse-checks,
and portfolio commentary.

All prompts instruct the model to stay descriptive/educational and to avoid
buy/hold/sell language, consistent with the rest of the app.
"""

import streamlit as st

from lib.config import get_groq_key

MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are the analysis engine for 'Stock Market Analyst', a personal, "
    "educational stock research dashboard. Your job is to summarize and "
    "contextualize publicly available data for a curious individual "
    "investor.\n\n"
    "Hard rules:\n"
    "- Never give a buy, sell, hold, or any other investment recommendation, "
    "directive, or rating.\n"
    "- Never tell the user what they 'should' do.\n"
    "- Use neutral, descriptive language (e.g. 'revenue grew', 'valuation is "
    "elevated relative to history', 'momentum is firm') rather than "
    "directive language.\n"
    "- Present balanced perspectives - when discussing positives, also "
    "surface relevant risks or counterpoints, and vice versa.\n"
    "- You may discuss historical data, trends, ratios, and general "
    "educational concepts.\n"
    "- Keep responses concise, well-organized with short headers or bullet "
    "points, and grounded only in the data provided to you."
)


@st.cache_resource(show_spinner=False)
def _get_client():
    key = get_groq_key()
    if not key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=key)
    except Exception:
        return None


def is_configured() -> bool:
    return _get_client() is not None


def _stream(user_prompt: str, max_tokens: int = 1500):
    """Yield text chunks from Groq, or raise if not configured."""
    client = _get_client()
    if client is None:
        raise RuntimeError("Groq API key is not configured.")

    stream = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def bull_bear_case(ticker: str, fundamentals: dict, chips: list, technical_score, fundamental_score, news_items: list):
    """Stream a balanced bull case / bear case writeup for a stock."""
    chip_text = "\n".join(f"- {label}: {value}" for label, value in chips)
    news_text = "\n".join(f"- {n['title']} ({n['publisher']})" for n in news_items[:5]) or "None available"

    prompt = (
        f"Write a balanced 'Bull Case' and 'Bear Case' for {ticker} "
        f"({fundamentals.get('name')}), based only on the data below. Use two "
        f"clear sections with headers '### Bull Case' and '### Bear Case', "
        f"3-5 bullet points each. Do not recommend any action.\n\n"
        f"Sector: {fundamentals.get('sector')} / {fundamentals.get('industry')}\n"
        f"Technical strength score (0-100, descriptive): {technical_score}\n"
        f"Fundamental quality score (0-100, descriptive): {fundamental_score}\n"
        f"At a glance:\n{chip_text}\n\n"
        f"Key metrics: P/E {fundamentals.get('trailing_pe')}, Forward P/E "
        f"{fundamentals.get('forward_pe')}, Beta {fundamentals.get('beta')}, "
        f"ROE {fundamentals.get('return_on_equity')}, Profit margin "
        f"{fundamentals.get('profit_margin')}, Revenue growth "
        f"{fundamentals.get('revenue_growth')}, Debt/Equity "
        f"{fundamentals.get('debt_to_equity')}.\n\n"
        f"Recent headlines:\n{news_text}"
    )
    return _stream(prompt, max_tokens=1200)


def deep_analysis(ticker: str, fundamentals: dict, history_summary: str, news_items: list):
    """Stream a deeper multi-section analysis for a stock."""
    news_text = "\n".join(f"- {n['title']} ({n['publisher']})" for n in news_items[:8]) or "None available"

    prompt = (
        f"Provide a deeper educational analysis of {ticker} "
        f"({fundamentals.get('name')}) covering: business overview, recent "
        f"price/volume behavior, valuation context relative to typical "
        f"ranges, profitability and balance-sheet posture, and notable "
        f"risks or watch items suggested by recent news. Organize with short "
        f"markdown headers. Do not recommend any action - this is for "
        f"personal research and education only.\n\n"
        f"Sector: {fundamentals.get('sector')} / {fundamentals.get('industry')}\n"
        f"Business summary: {fundamentals.get('summary')}\n\n"
        f"Price history summary: {history_summary}\n\n"
        f"Key metrics: Market cap {fundamentals.get('market_cap')}, P/E "
        f"{fundamentals.get('trailing_pe')}, Forward P/E "
        f"{fundamentals.get('forward_pe')}, P/B {fundamentals.get('price_to_book')}, "
        f"Beta {fundamentals.get('beta')}, ROE {fundamentals.get('return_on_equity')}, "
        f"ROA {fundamentals.get('return_on_assets')}, Profit margin "
        f"{fundamentals.get('profit_margin')}, Operating margin "
        f"{fundamentals.get('operating_margin')}, Revenue growth "
        f"{fundamentals.get('revenue_growth')}, Earnings growth "
        f"{fundamentals.get('earnings_growth')}, Debt/Equity "
        f"{fundamentals.get('debt_to_equity')}, Current ratio "
        f"{fundamentals.get('current_ratio')}.\n\n"
        f"Recent headlines:\n{news_text}"
    )
    return _stream(prompt, max_tokens=2000)


def macro_pulse_check(snapshot: dict, yield_curve_summary: str):
    """Stream a macro 'pulse check' summary from FRED data."""
    lines = []
    for label, data in snapshot.items():
        prior = f" (prior: {data['prior_value']:.2f})" if data.get("prior_value") is not None else ""
        lines.append(f"- {label}: {data['value']:.2f} {data['units']} as of {data['date'].date()}{prior}")
    indicators_text = "\n".join(lines) or "No indicators available"

    prompt = (
        "Provide a short 'macro pulse check' summarizing what these "
        "indicators suggest about the current economic environment, in "
        "plain educational language. Cover growth, inflation, labor market, "
        "and interest-rate/yield-curve context in short sections. Do not "
        "give investment recommendations or directives - describe the "
        "macro picture only.\n\n"
        f"Indicators:\n{indicators_text}\n\n"
        f"Yield curve: {yield_curve_summary}"
    )
    return _stream(prompt, max_tokens=1200)


def portfolio_analysis(holdings_summary: str, risk_score, sector_breakdown: dict):
    """Stream an educational analysis of the user's portfolio composition."""
    sector_text = "\n".join(f"- {sector}: {pct:.1f}%" for sector, pct in sector_breakdown.items()) or "Unknown"

    prompt = (
        "Provide an educational review of this personal portfolio's "
        "composition: diversification, sector concentration, and overall "
        "risk posture relative to the descriptive risk score provided. "
        "Use short markdown sections. Do not recommend buying, selling, "
        "or rebalancing any specific position - describe characteristics "
        "and tradeoffs only, for the holder's own research.\n\n"
        f"Holdings:\n{holdings_summary}\n\n"
        f"Portfolio risk score (0-100, descriptive): {risk_score}\n\n"
        f"Sector breakdown:\n{sector_text}"
    )
    return _stream(prompt, max_tokens=1500)
