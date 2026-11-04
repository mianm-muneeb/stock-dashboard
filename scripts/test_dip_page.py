"""Headless check of the Dip Scanner page with real data."""
import glob

from streamlit.testing.v1 import AppTest

page = glob.glob("pages/17_*Dip_Scanner.py")[0]
at = AppTest.from_file(page, default_timeout=600)
at.run()
print("EXC", [e.value for e in at.exception])
for m in at.markdown[:2]:
    print("MD", m.value)
at.text_area[0].input("DT, 2026-11-03").run()
print("EXC2", [e.value for e in at.exception])
print(at.dataframe[-1].value.to_string())
