# scripts/utils.py

import os
import uuid
from datetime import datetime


def pxp(x):
    print()
    print(x)
    print()


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_uuid4():
    return str(uuid.uuid4())


def get_target_dir(*subdirs, verbose=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    if verbose:
        pxp(f"root dir: {root_dir}")

    target_dir = os.path.join(root_dir, *subdirs)

    os.makedirs(target_dir, exist_ok=True)

    if verbose:
        pxp(f"target dir: {target_dir}")

    return target_dir
