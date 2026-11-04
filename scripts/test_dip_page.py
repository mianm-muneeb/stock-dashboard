"""Headless check of the Dip Scanner page with real data."""
import glob
import sys

from streamlit.testing.v1 import AppTest

page = glob.glob("pages/17_*Dip_Scanner.py")[0]
at = AppTest.from_file(page, default_timeout=600)
at.run()
print("exceptions:", [e.value for e in at.exception])
print("errors:", [e.value for e in at.error])
print("captions:", [c.value for c in at.caption if "Universe" in c.value])
print("subheaders:", [h.value for h in at.subheader])
for df in at.dataframe[:1]:
    print(df.value.head(10).to_string())
at.text_area[0].input("VST, 2026-11-03\nCELH, 2026-11-04").run()
print("exceptions after sell check:", [e.value for e in at.exception])
print(at.dataframe[-1].value.to_string())
