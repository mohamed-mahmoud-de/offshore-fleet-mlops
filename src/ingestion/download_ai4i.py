"""
Reproducible data acquisition for the AI4I 2020 Predictive Maintenance dataset.
Source: UCI Machine Learning Repository (dataset #601). License: CC BY 4.0.

DE principle: never "manually download a file once." Script the pull so anyone
(a teammate, a server, future-you) can reproduce the exact dataset with one command.

Run from the project root:
    python -m src.ingestion.download_ai4i
"""
import io
import zipfile
import urllib.request
from pathlib import Path

URL = ("https://archive.ics.uci.edu/static/public/601/"
       "ai4i+2020+predictive+maintenance+dataset.zip")
OUT = Path("data/raw/ai4i2020.csv")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)          # ensure data/raw/ exists
    print("Downloading AI4I 2020 from UCI ...")
    with urllib.request.urlopen(URL) as resp:              # fetch the zip into memory
        blob = resp.read()
    print(f"  downloaded {len(blob) / 1024:.0f} KB")

    with zipfile.ZipFile(io.BytesIO(blob)) as z:           # open the zip without saving it
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        OUT.write_bytes(z.read(csv_name))                  # extract just the CSV
    print(f"  saved -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
