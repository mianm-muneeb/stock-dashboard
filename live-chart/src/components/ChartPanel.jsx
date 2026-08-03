import { useEffect, useRef } from 'react'
import {
  createChart,
  ColorType,
  CrosshairMode,
  LineStyle,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  AreaSeries,
} from 'lightweight-charts'
import { ema, bollingerBands, rsi, macd, vwap, ichimoku } from '../lib/indicators'
import { DrawingPrimitive } from '../lib/drawingPrimitive'

const BG = '#131722'
const GRID = '#1e222d'
const BORDER = '#2a2e39'
const CROSSHAIR = '#758696'
const TEXT = '#d1d4dc'

const EMA_COLORS = { EMA20: '#FF9800', EMA50: '#2962FF', EMA200: '#F23645' }
const SUB_PANE_HEIGHT = 130

const PRICE_AXIS_HOT_ZONE = 50

const DEFAULT_HANDLE_SCALE = {
  axisPressedMouseMove: { time: true, price: true },
  mouseWheel: true,
  pinch: true,
}
const DEFAULT_HANDLE_SCROLL = {
  mouseWheel: true,
  pressedMouseMove: true,
  horzTouchDrag: true,
  vertTouchDrag: false,
}

export default function ChartPanel({ bars, indicators, livePrice, liveBar, activeTool, onToolChange }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const candleSeriesRef = useRef(null)
  const volumeSeriesRef = useRef(null)
  const priceLineRef = useRef(null)
  const extraSeriesRef = useRef([])

  const primitiveRef = useRef(null)
  const drawingsRef = useRef([])
  const pendingRef = useRef(null)
  const draggingRef = useRef(false)
  const activeToolRef = useRef(activeTool)
  const onToolChangeRef = useRef(onToolChange)

  useEffect(() => {
    activeToolRef.current = activeTool
    onToolChangeRef.current = onToolChange
  }, [activeTool, onToolChange])

  // Create the chart once.
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: BG },
        textColor: TEXT,
      },
      grid: {
        vertLines: { color: GRID },
        horzLines: { color: GRID },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: CROSSHAIR, labelBackgroundColor: CROSSHAIR, width: 1 },
        horzLine: { color: CROSSHAIR, labelBackgroundColor: CROSSHAIR, width: 1 },
      },
      timeScale: {
        borderColor: BORDER,
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: BORDER,
      },
      handleScale: DEFAULT_HANDLE_SCALE,
      handleScroll: DEFAULT_HANDLE_SCROLL,
      autoSize: true,
    })

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    })

    chartRef.current = chart
    candleSeriesRef.current = candleSeries

    // Drawing overlay primitive (trend lines, rectangles, text labels).
    const primitive = new DrawingPrimitive(() => ({
      drawings: drawingsRef.current,
      pending: pendingRef.current,
    }))
    candleSeries.attachPrimitive(primitive)
    primitiveRef.current = primitive

    // Double-click resets zoom to fit all data.
    const handleDoubleClick = () => {
      chart.timeScale().fitContent()
    }

    const getPoint = (e) => {
      const series = candleSeriesRef.current
      if (!series || !containerRef.current) return null
      const rect = containerRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const time = chart.timeScale().coordinateToTime(x)
      const price = series.coordinateToPrice(y)
      if (time === null || price === null) return null
      return { time, price }
    }

    const handleMouseDown = (e) => {
      const tool = activeToolRef.current
      if (tool === 'cursor') return
      const point = getPoint(e)
      if (!point) return

      if (tool === 'hline') {
        candleSeriesRef.current?.createPriceLine({
          price: point.price,
          color: '#2962ff',
          lineWidth: 1,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: true,
          title: point.price.toFixed(2),
        })
        onToolChangeRef.current?.('cursor')
        return
      }

      if (tool === 'trend') {
        if (!pendingRef.current) {
          pendingRef.current = { type: 'trend', p1: point, p2: point }
          primitiveRef.current?.requestUpdate()
        } else {
          drawingsRef.current = [
            ...drawingsRef.current,
            { type: 'trend', p1: pendingRef.current.p1, p2: point },
          ]
          pendingRef.current = null
          primitiveRef.current?.requestUpdate()
          onToolChangeRef.current?.('cursor')
        }
        return
      }

      if (tool === 'rect') {
        draggingRef.current = true
        pendingRef.current = { type: 'rect', p1: point, p2: point }
        primitiveRef.current?.requestUpdate()
        return
      }

      if (tool === 'text') {
        const text = window.prompt('Label text:')
        if (text) {
          drawingsRef.current = [...drawingsRef.current, { type: 'text', p: point, text }]
          primitiveRef.current?.requestUpdate()
        }
        onToolChangeRef.current?.('cursor')
      }
    }

    const handleMouseMove = (e) => {
      const tool = activeToolRef.current
      if (tool === 'trend' && pendingRef.current) {
        const point = getPoint(e)
        if (!point) return
        pendingRef.current = { ...pendingRef.current, p2: point }
        primitiveRef.current?.requestUpdate()
        return
      }
      if (tool === 'rect' && draggingRef.current && pendingRef.current) {
        const point = getPoint(e)
        if (!point) return
        pendingRef.current = { ...pendingRef.current, p2: point }
        primitiveRef.current?.requestUpdate()
      }
    }

    const handleMouseUp = (e) => {
      const tool = activeToolRef.current
      if (tool === 'rect' && draggingRef.current) {
        draggingRef.current = false
        const point = getPoint(e) || pendingRef.current?.p2
        const start = pendingRef.current?.p1
        pendingRef.current = null
        if (start && point) {
          drawingsRef.current = [...drawingsRef.current, { type: 'rect', p1: start, p2: point }]
        }
        primitiveRef.current?.requestUpdate()
        onToolChangeRef.current?.('cursor')
      }
    }

    const handleKeyDown = (e) => {
      if (e.key !== 'Escape') return
      pendingRef.current = null
      draggingRef.current = false
      primitiveRef.current?.requestUpdate()
      onToolChangeRef.current?.('cursor')
    }

    // Scrolling over the price axis (right edge of the container) zooms the
    // price scale vertically only. lightweight-charts has no built-in wheel
    // handler for the price axis, so we adjust the visible price range
    // manually; scrolling anywhere else falls through to the chart's own
    // mouseWheel handling, which zooms the time axis.
    const handleWheel = (e) => {
      const series = candleSeriesRef.current
      if (!series || !containerRef.current) return

      const rect = containerRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const overPriceAxis = x >= rect.width - PRICE_AXIS_HOT_ZONE
      if (!overPriceAxis) return

      e.preventDefault()
      e.stopPropagation()

      const priceScale = series.priceScale()
      const range = priceScale.getVisibleRange()
      if (!range) return
      const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1
      const center = (range.from + range.to) / 2
      priceScale.applyOptions({ autoScale: false })
      priceScale.setVisibleRange({
        from: center + (range.from - center) * factor,
        to: center + (range.to - center) * factor,
      })
    }

    // Prevent the browser context menu so right-click drag can be used to
    // stretch/compress the price scale.
    const handleContextMenu = (e) => {
      e.preventDefault()
    }

    const el = containerRef.current
    el.addEventListener('dblclick', handleDoubleClick)
    el.addEventListener('mousedown', handleMouseDown)
    el.addEventListener('mousemove', handleMouseMove)
    el.addEventListener('mouseup', handleMouseUp)
    el.addEventListener('wheel', handleWheel, { passive: false })
    el.addEventListener('contextmenu', handleContextMenu)
    window.addEventListener('keydown', handleKeyDown)

    return () => {
      el.removeEventListener('dblclick', handleDoubleClick)
      el.removeEventListener('mousedown', handleMouseDown)
      el.removeEventListener('mousemove', handleMouseMove)
      el.removeEventListener('mouseup', handleMouseUp)
      el.removeEventListener('wheel', handleWheel)
      el.removeEventListener('contextmenu', handleContextMenu)
      window.removeEventListener('keydown', handleKeyDown)
      chart.remove()
      chartRef.current = null
      candleSeriesRef.current = null
      volumeSeriesRef.current = null
      priceLineRef.current = null
      extraSeriesRef.current = []
      primitiveRef.current = null
    }
  }, [])

  // Toggle pan/zoom handling based on the active drawing tool, and update
  // the cursor to give a visual hint of the active tool.
  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    if (activeTool === 'cursor') {
      chart.applyOptions({ handleScale: DEFAULT_HANDLE_SCALE, handleScroll: DEFAULT_HANDLE_SCROLL })
    } else {
      chart.applyOptions({ handleScale: false, handleScroll: false })
      pendingRef.current = null
      draggingRef.current = false
      primitiveRef.current?.requestUpdate()
    }

    if (containerRef.current) {
      containerRef.current.style.cursor = activeTool === 'cursor' ? 'default' : 'crosshair'
    }
  }, [activeTool])

  // Rebuild series whenever the bar set or active indicators change.
  useEffect(() => {
    const chart = chartRef.current
    const candleSeries = candleSeriesRef.current
    if (!chart || !candleSeries || !bars.length) return

    candleSeries.setData(bars.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })))

    // Remove previous overlay/sub-pane series.
    for (const series of extraSeriesRef.current) {
      chart.removeSeries(series)
    }
    extraSeriesRef.current = []
    volumeSeriesRef.current = null

    // Drop any leftover sub-panes so RSI/MACD/Volume get clean, freshly
    // indexed panes below the main chart instead of reusing stale ones.
    for (let i = chart.panes().length - 1; i >= 1; i--) {
      chart.removePane(i)
    }

    if (priceLineRef.current) {
      candleSeries.removePriceLine(priceLineRef.current)
      priceLineRef.current = null
    }

    // -- Overlay indicators (main pane) ------------------------------------
    for (const key of ['EMA20', 'EMA50', 'EMA200']) {
      if (!indicators.has(key)) continue
      const period = Number(key.replace('EMA', ''))
      const data = ema(bars, period)
      if (!data.length) continue
      const series = chart.addSeries(LineSeries, {
        color: EMA_COLORS[key],
        lineWidth: 1,
        title: key,
        priceLineVisible: false,
        lastValueVisible: false,
      })
      series.setData(data)
      extraSeriesRef.current.push(series)
    }

    if (indicators.has('BB') && bars.length >= 20) {
      const { upper, middle, lower } = bollingerBands(bars, 20, 2)
      for (const [data, title] of [[upper, 'BB Upper'], [middle, 'BB Mid'], [lower, 'BB Lower']]) {
        const series = chart.addSeries(LineSeries, {
          color: 'rgba(255,255,255,0.6)',
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          title,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        series.setData(data)
        extraSeriesRef.current.push(series)
      }
    }

    if (indicators.has('VWAP')) {
      const series = chart.addSeries(LineSeries, {
        color: '#b388ff',
        lineWidth: 1.5,
        title: 'VWAP',
        priceLineVisible: false,
        lastValueVisible: false,
      })
      series.setData(vwap(bars))
      extraSeriesRef.current.push(series)
    }

    if (indicators.has('Ichimoku') && bars.length >= 52) {
      const { tenkan, kijun, spanA, spanB, chikou } = ichimoku(bars)

      const cloudBullish = chart.addSeries(AreaSeries, {
        lineVisible: false,
        topColor: 'rgba(38,166,154,0.15)',
        bottomColor: 'rgba(38,166,154,0.02)',
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      })
      cloudBullish.setData(spanA.filter((p, i) => spanB[i] && p.value >= spanB[i].value))
      extraSeriesRef.current.push(cloudBullish)

      const cloudBearish = chart.addSeries(AreaSeries, {
        lineVisible: false,
        topColor: 'rgba(239,83,80,0.15)',
        bottomColor: 'rgba(239,83,80,0.02)',
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      })
      cloudBearish.setData(spanB.filter((p, i) => spanA[i] && spanA[i].value < p.value))
      extraSeriesRef.current.push(cloudBearish)

      const tenkanSeries = chart.addSeries(LineSeries, {
        color: '#ef5350',
        lineWidth: 1,
        title: 'Tenkan',
        priceLineVisible: false,
        lastValueVisible: false,
      })
      tenkanSeries.setData(tenkan)
      extraSeriesRef.current.push(tenkanSeries)

      const kijunSeries = chart.addSeries(LineSeries, {
        color: '#2962ff',
        lineWidth: 1,
        title: 'Kijun',
        priceLineVisible: false,
        lastValueVisible: false,
      })
      kijunSeries.setData(kijun)
      extraSeriesRef.current.push(kijunSeries)

      const spanASeries = chart.addSeries(LineSeries, {
        color: 'rgba(38,166,154,0.6)',
        lineWidth: 1,
        title: 'Span A',
        priceLineVisible: false,
        lastValueVisible: false,
      })
      spanASeries.setData(spanA)
      extraSeriesRef.current.push(spanASeries)

      const spanBSeries = chart.addSeries(LineSeries, {
        color: 'rgba(239,83,80,0.6)',
        lineWidth: 1,
        title: 'Span B',
        priceLineVisible: false,
        lastValueVisible: false,
      })
      spanBSeries.setData(spanB)
      extraSeriesRef.current.push(spanBSeries)

      const chikouSeries = chart.addSeries(LineSeries, {
        color: '#9aff4d',
        lineWidth: 1,
        title: 'Chikou',
        priceLineVisible: false,
        lastValueVisible: false,
      })
      chikouSeries.setData(chikou)
      extraSeriesRef.current.push(chikouSeries)
    }

    // -- Sub-panes -----------------------------------------------------------
    let nextPane = 1
    const extraPanes = []

    if (indicators.has('Volume')) {
      const paneIndex = nextPane++
      const volumeSeries = chart.addSeries(
        HistogramSeries,
        { priceFormat: { type: 'volume' }, priceLineVisible: false, lastValueVisible: false },
        paneIndex,
      )
      volumeSeries.setData(
        bars.map((b) => ({
          time: b.time,
          value: b.volume,
          color: b.close >= b.open ? 'rgba(38,166,154,0.5)' : 'rgba(239,83,80,0.5)',
        })),
      )
      extraSeriesRef.current.push(volumeSeries)
      volumeSeriesRef.current = volumeSeries
      extraPanes.push(paneIndex)
    }

    if (indicators.has('RSI')) {
      const paneIndex = nextPane++
      const rsiSeries = chart.addSeries(
        LineSeries,
        { color: '#a78bfa', lineWidth: 1.5, title: 'RSI', priceLineVisible: false, lastValueVisible: false },
        paneIndex,
      )
      rsiSeries.setData(rsi(bars, 14))
      rsiSeries.createPriceLine({ price: 70, color: 'rgba(239,83,80,0.5)', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false })
      rsiSeries.createPriceLine({ price: 30, color: 'rgba(38,166,154,0.5)', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false })
      extraSeriesRef.current.push(rsiSeries)
      extraPanes.push(paneIndex)
    }

    if (indicators.has('MACD')) {
      const paneIndex = nextPane++
      const { macdLine, signalLine, histogram } = macd(bars, 12, 26, 9)

      const histSeries = chart.addSeries(
        HistogramSeries,
        { priceLineVisible: false, lastValueVisible: false, title: 'MACD Hist' },
        paneIndex,
      )
      histSeries.setData(histogram.map((p) => ({ time: p.time, value: p.value, color: p.value >= 0 ? 'rgba(38,166,154,0.5)' : 'rgba(239,83,80,0.5)' })))
      extraSeriesRef.current.push(histSeries)

      const macdSeries = chart.addSeries(
        LineSeries,
        { color: '#5dade2', lineWidth: 1.5, title: 'MACD', priceLineVisible: false, lastValueVisible: false },
        paneIndex,
      )
      macdSeries.setData(macdLine)
      extraSeriesRef.current.push(macdSeries)

      const signalSeries = chart.addSeries(
        LineSeries,
        { color: '#f1c40f', lineWidth: 1.5, title: 'Signal', priceLineVisible: false, lastValueVisible: false },
        paneIndex,
      )
      signalSeries.setData(signalLine)
      extraSeriesRef.current.push(signalSeries)

      extraPanes.push(paneIndex)
    }

    // Size panes: main pane gets the remaining height after fixed-size sub-panes.
    requestAnimationFrame(() => {
      const panes = chart.panes()
      const total = containerRef.current?.clientHeight ?? 650
      const mainHeight = Math.max(200, total - extraPanes.length * SUB_PANE_HEIGHT)
      if (panes[0]) panes[0].setHeight(mainHeight)
      extraPanes.forEach((paneIndex) => {
        if (panes[paneIndex]) panes[paneIndex].setHeight(SUB_PANE_HEIGHT)
      })
    })

    chart.timeScale().fitContent()
  }, [bars, indicators])

  // Live price line.
  useEffect(() => {
    const candleSeries = candleSeriesRef.current
    if (!candleSeries || livePrice === null || livePrice === undefined) return

    if (priceLineRef.current) {
      priceLineRef.current.applyOptions({ price: livePrice })
    } else {
      priceLineRef.current = candleSeries.createPriceLine({
        price: livePrice,
        color: '#f1c40f',
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: 'Last',
      })
    }
  }, [livePrice])

  // Live bar update (1m timeframe streaming).
  useEffect(() => {
    if (!liveBar) return
    const candleSeries = candleSeriesRef.current
    if (!candleSeries) return

    candleSeries.update({
      time: liveBar.time,
      open: liveBar.open,
      high: liveBar.high,
      low: liveBar.low,
      close: liveBar.close,
    })

    volumeSeriesRef.current?.update({
      time: liveBar.time,
      value: liveBar.volume,
      color: liveBar.close >= liveBar.open ? 'rgba(38,166,154,0.5)' : 'rgba(239,83,80,0.5)',
    })
  }, [liveBar])

  return <div ref={containerRef} className="chart-container" />
}
