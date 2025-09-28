import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

DUMPS = {
    "classification": "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/50x3_3_100k_classification.h5",
    "regression": "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/50x3_1280k_regression.h5",
}
TARGET_DIR = Path(__file__).resolve().parents[1] / "other" / "dumps"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", choices=list(DUMPS), nargs="*")
    args = parser.parse_args()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for key in args.download or DUMPS:
        url = DUMPS[key]
        dest = TARGET_DIR / Path(urlparse(url).path).name
        if dest.exists():
            print(f"already exists, skipping: {dest}")
            continue
        try:
            urlretrieve(url, dest)
            print(f"saved to: {dest}")
        except Exception as e:
            print(f"failed to download {url}: {e}")
            if dest.exists():
                dest.unlink()
            sys.exit(1)


if __name__ == "__main__":
    main()
