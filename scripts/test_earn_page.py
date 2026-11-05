import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lib import kronos_utils as ku
for t in ("CELH", "VST"):
    b = ku.fetch_bars(t, "1d")
    ei = ku.earnings_info(t, b)
    print(t, "next", ei["next"], ei["next_timing"], "past", len(ei["hist"]),
          "moves%", [round(m * 100, 1) for m in ei["hist"]["move"].tail(8)])
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(str(ROOT / "pages" / "16_🔮_Kronos_Forecast.py"), default_timeout=300)
at.session_state["kr_params"] = dict(ticker="CELH", interval="1d", mode="Kronos-small", horizon=63, lookback=400, paths=10,
    temp=1.0, top_p=0.9, use_anchor=True, use_rescale=True, drift_keep=0.5, show_raw=False, holdout=False, use_earn=True)
at.run()
print("exceptions:", [e.value for e in at.exception])
print("info:", [i.value for i in at.info])
print("warning:", [w.value for w in at.warning])
print("metrics:", [(m.label, m.value) for m in at.metric])
