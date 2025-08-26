# scripts/dumps.py

import argparse
import os
import sys
from urllib.request import urlretrieve

from utils import get_target_dir, pxp

CLASSIFICATION_DUMP_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/50x3_3_100k_classification.h5"
REGRESSION_DUMP_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/50x3_1280k_regression.h5"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        choices=["classification", "regression"],
        nargs="*",
        help="which dumps to download; if omitted, both are downloaded",
    )
    return parser.parse_args()


def download_dump(url, dest_path):
    if os.path.exists(dest_path):
        pxp(f"already exists, skipping: {dest_path}")
        return
    try:
        urlretrieve(url, dest_path)
        pxp(f"saved to: {dest_path}")
    except Exception as e:
        pxp(f"failed to download {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        sys.exit(1)


def main():
    args = parse_args()

    target_path = get_target_dir("other", "dumps", verbose=False)

    targets = args.download if args.download else ["classification", "regression"]

    if "classification" in targets:
        dest = os.path.join(target_path, "50x3_3_100k_classification.h5")
        download_dump(CLASSIFICATION_DUMP_URL, dest)

    if "regression" in targets:
        dest = os.path.join(target_path, "50x3_1280k_regression.h5")
        download_dump(REGRESSION_DUMP_URL, dest)


if __name__ == "__main__":
    main()
