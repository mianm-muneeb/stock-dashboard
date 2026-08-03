import { useState } from 'react'
import { TIMEFRAME_LABELS } from '../lib/timeframes'

export const INDICATOR_OPTIONS = [
  { key: 'EMA20', label: 'EMA 20' },
  { key: 'EMA50', label: 'EMA 50' },
  { key: 'EMA200', label: 'EMA 200' },
  { key: 'BB', label: 'Bollinger' },
  { key: 'VWAP', label: 'VWAP' },
  { key: 'Ichimoku', label: 'Ichimoku' },
  { key: 'Volume', label: 'Volume' },
  { key: 'RSI', label: 'RSI' },
  { key: 'MACD', label: 'MACD' },
]

export default function Toolbar({ symbol, onSymbolChange, timeframe, onTimeframeChange, indicators, onToggleIndicator }) {
  const [input, setInput] = useState(symbol)

  const submitSymbol = (e) => {
    e.preventDefault()
    const next = input.trim().toUpperCase()
    if (next) onSymbolChange(next)
  }

  return (
    <div className="toolbar">
      <form className="symbol-search" onSubmit={submitSymbol}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Symbol (e.g. CELH)"
          spellCheck={false}
          autoCapitalize="characters"
        />
        <button type="submit">Go</button>
      </form>

      <div className="toolbar-divider" />

      <div className="timeframe-bar">
        {TIMEFRAME_LABELS.map((tf) => (
          <button
            key={tf}
            className={`tf-btn ${tf === timeframe ? 'active' : ''}`}
            onClick={() => onTimeframeChange(tf)}
          >
            {tf}
          </button>
        ))}
      </div>

      <div className="toolbar-divider" />

      <div className="indicator-bar">
        {INDICATOR_OPTIONS.map((opt) => (
          <button
            key={opt.key}
            className={`ind-btn ${indicators.has(opt.key) ? 'active' : ''}`}
            onClick={() => onToggleIndicator(opt.key)}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  )
}
