// Indicator calculations operating on an array of bars:
//   { time, open, high, low, close, volume }
// All functions return arrays of { time, value } (or value sets) aligned to
// `bars`, skipping points where the window isn't yet full.

export function sma(bars, period, key = 'close') {
  const out = []
  let sum = 0
  for (let i = 0; i < bars.length; i++) {
    sum += bars[i][key]
    if (i >= period) sum -= bars[i - period][key]
    if (i >= period - 1) {
      out.push({ time: bars[i].time, value: sum / period })
    }
  }
  return out
}

export function ema(bars, period, key = 'close') {
  const out = []
  const k = 2 / (period + 1)
  let prev = null
  for (let i = 0; i < bars.length; i++) {
    const price = bars[i][key]
    if (prev === null) {
      if (i >= period - 1) {
        // seed with SMA of the first `period` values
        let sum = 0
        for (let j = i - period + 1; j <= i; j++) sum += bars[j][key]
        prev = sum / period
        out.push({ time: bars[i].time, value: prev })
      }
    } else {
      prev = price * k + prev * (1 - k)
      out.push({ time: bars[i].time, value: prev })
    }
  }
  return out
}

export function bollingerBands(bars, period = 20, mult = 2) {
  const middle = sma(bars, period)
  const upper = []
  const lower = []
  const offset = bars.length - middle.length

  middle.forEach((point, idx) => {
    const i = idx + offset
    let sumSq = 0
    for (let j = i - period + 1; j <= i; j++) {
      sumSq += (bars[j].close - point.value) ** 2
    }
    const std = Math.sqrt(sumSq / period)
    upper.push({ time: point.time, value: point.value + mult * std })
    lower.push({ time: point.time, value: point.value - mult * std })
  })

  return { middle, upper, lower }
}

export function rsi(bars, period = 14) {
  const out = []
  if (bars.length < period + 1) return out

  let gainSum = 0
  let lossSum = 0
  for (let i = 1; i <= period; i++) {
    const change = bars[i].close - bars[i - 1].close
    if (change >= 0) gainSum += change
    else lossSum -= change
  }
  let avgGain = gainSum / period
  let avgLoss = lossSum / period
  out.push(rsiPoint(bars[period].time, avgGain, avgLoss))

  for (let i = period + 1; i < bars.length; i++) {
    const change = bars[i].close - bars[i - 1].close
    const gain = change >= 0 ? change : 0
    const loss = change < 0 ? -change : 0
    avgGain = (avgGain * (period - 1) + gain) / period
    avgLoss = (avgLoss * (period - 1) + loss) / period
    out.push(rsiPoint(bars[i].time, avgGain, avgLoss))
  }
  return out
}

function rsiPoint(time, avgGain, avgLoss) {
  if (avgLoss === 0) return { time, value: 100 }
  const rs = avgGain / avgLoss
  return { time, value: 100 - 100 / (1 + rs) }
}

/** Volume-weighted average price, accumulated from the start of each session (UTC day). */
export function vwap(bars) {
  const out = []
  let cumPV = 0
  let cumVol = 0
  let currentDay = null

  for (const bar of bars) {
    const day = Math.floor(bar.time / 86400)
    if (day !== currentDay) {
      currentDay = day
      cumPV = 0
      cumVol = 0
    }
    const typicalPrice = (bar.high + bar.low + bar.close) / 3
    cumPV += typicalPrice * bar.volume
    cumVol += bar.volume
    out.push({ time: bar.time, value: cumVol > 0 ? cumPV / cumVol : bar.close })
  }

  return out
}

function highestLowest(bars, period, idx) {
  let hh = -Infinity
  let ll = Infinity
  for (let j = idx - period + 1; j <= idx; j++) {
    if (bars[j].high > hh) hh = bars[j].high
    if (bars[j].low < ll) ll = bars[j].low
  }
  return { hh, ll }
}

/**
 * Ichimoku Kinko Hyo. Senkou Span A/B are projected `displacement` periods
 * into the future and Chikou Span is the close shifted `displacement`
 * periods into the past, all using the bars' own time spacing.
 */
export function ichimoku(bars, conversionPeriod = 9, basePeriod = 26, spanBPeriod = 52, displacement = 26) {
  const tenkan = []
  const kijun = []
  const spanA = []
  const spanB = []
  const chikou = []

  if (bars.length < 2) return { tenkan, kijun, spanA, spanB, chikou }

  const interval = bars[1].time - bars[0].time

  for (let i = 0; i < bars.length; i++) {
    if (i >= conversionPeriod - 1) {
      const { hh, ll } = highestLowest(bars, conversionPeriod, i)
      tenkan.push({ time: bars[i].time, value: (hh + ll) / 2 })
    }
    if (i >= basePeriod - 1) {
      const { hh, ll } = highestLowest(bars, basePeriod, i)
      kijun.push({ time: bars[i].time, value: (hh + ll) / 2 })
    }
    if (i >= spanBPeriod - 1) {
      const { hh, ll } = highestLowest(bars, spanBPeriod, i)
      spanB.push({ time: bars[i].time + displacement * interval, value: (hh + ll) / 2 })
    }
    chikou.push({ time: bars[i].time - displacement * interval, value: bars[i].close })
  }

  const tenkanMap = new Map(tenkan.map((p) => [p.time, p.value]))
  for (const k of kijun) {
    const t = tenkanMap.get(k.time)
    if (t !== undefined) {
      spanA.push({ time: k.time + displacement * interval, value: (t + k.value) / 2 })
    }
  }

  return { tenkan, kijun, spanA, spanB, chikou }
}

export function macd(bars, fast = 12, slow = 26, signalPeriod = 9) {
  const emaFast = ema(bars, fast)
  const emaSlow = ema(bars, slow)

  const fastMap = new Map(emaFast.map((p) => [p.time, p.value]))
  const slowMap = new Map(emaSlow.map((p) => [p.time, p.value]))

  const macdLine = []
  for (const bar of bars) {
    if (fastMap.has(bar.time) && slowMap.has(bar.time)) {
      macdLine.push({ time: bar.time, value: fastMap.get(bar.time) - slowMap.get(bar.time) })
    }
  }

  // EMA of the MACD line for the signal line
  const k = 2 / (signalPeriod + 1)
  const signalLine = []
  let prev = null
  macdLine.forEach((point, i) => {
    if (prev === null) {
      if (i >= signalPeriod - 1) {
        let sum = 0
        for (let j = i - signalPeriod + 1; j <= i; j++) sum += macdLine[j].value
        prev = sum / signalPeriod
        signalLine.push({ time: point.time, value: prev })
      }
    } else {
      prev = point.value * k + prev * (1 - k)
      signalLine.push({ time: point.time, value: prev })
    }
  })

  const signalMap = new Map(signalLine.map((p) => [p.time, p.value]))
  const histogram = macdLine
    .filter((p) => signalMap.has(p.time))
    .map((p) => ({ time: p.time, value: p.value - signalMap.get(p.time) }))

  return { macdLine, signalLine, histogram }
}
