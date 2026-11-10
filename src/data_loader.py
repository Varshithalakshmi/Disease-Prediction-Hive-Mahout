
from pathlib import Path
from urllib.request import urlopen, Request
import pandas as pd
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "diabetes.csv"

DATA_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"


def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to UCI dataset...", flush=True)

    request = Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        with urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(OUTPUT_FILE, "wb") as file:
                while True:
                    chunk = response.read(1024 * 64)

                    if not chunk:
                        break

                    file.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        percent = downloaded * 100 / total
                        print(
                            f"\rDownloaded: {percent:.1f}%",
                            end="",
                            flush=True
                        )

        print("\nDownload completed!", flush=True)

        df = pd.read_csv(OUTPUT_FILE)

        print("Dataset shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst five records:")
        print(df.head())

        print("\nSaved to:", OUTPUT_FILE)

    except Exception as error:
        print("\nDownload failed:", error)
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
