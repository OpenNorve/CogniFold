import os
import sys

import requests

DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
OUTPUT_FILE = "locomo10.json"


def download_data():
    if os.path.exists(OUTPUT_FILE):
        print(f"Dataset {OUTPUT_FILE} already exists. Skipping download.")
        return

    print(f"Downloading dataset from {DATASET_URL}...")
    try:
        response = requests.get(DATASET_URL, stream=True)
        response.raise_for_status()

        with open(OUTPUT_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure we are in the correct directory or path is relative
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    download_data()
