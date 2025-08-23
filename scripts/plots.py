import argparse
import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MinMaxScaler

from nanotabpfn.interface import NanoTabPFNRegressor


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="auto",
        choices=["auto", "uniform", "normal"],
    )
    parser.add_argument(
        "--vs-tabpfn",
        action="store_true",
    )

    args = parser.parse_args()

    return args


def pxp(x):
    print()
    print(x)
    print()


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_datasets(n=50, seed=42, lower=-8, upper=8, sampling="auto"):
    rng = np.random.default_rng(seed)
    scaler = MinMaxScaler((lower, upper))

    match sampling:
        case "auto" | "uniform":
            X = rng.uniform(lower, upper, n).astype(np.float32)
        case "normal":
            X = rng.normal(0, 1, (n, 1)).astype(np.float32)
            X = scaler.fit_transform(X)
            X = X.flatten()
        case _:
            raise ValueError(f"unknown sampling: {sampling}")

    y_linear = 2 * X + 1 + rng.normal(0, 1, n)
    y_poly = 0.5 * X**2 + X - 20 + rng.normal(0, 0.1, n)
    y_sin = np.sin(X) + rng.normal(0, 0.1, n)

    X = X.reshape(-1, 1)

    datasets = [
        ("linear", (X, y_linear)),
        ("polynomial", (X, y_poly)),
        ("sine", (X, y_sin)),
    ]
    return datasets


def fit_predict_plot(ax, X, y, models, title=""):
    ax.scatter(X, y, color="gray", alpha=0.5, label="Data")

    x_lim = max(abs(X.flatten().min()), abs(X.flatten().max()))
    y_lim = max(abs(y.min()), abs(y.max()))

    m = 1

    x_lim += x_lim * 0.25 + m
    y_lim += y_lim * 0.25 + m

    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.1)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.1)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    X_plot = np.linspace(-x_lim, x_lim, 500).reshape(-1, 1)

    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X_plot)
        ax.plot(X_plot, preds, label=name)

    ax.set_title(title)


def visualise_and_save(seed, sampling, vs_tabpfn, plot_filepath):
    ds = generate_datasets(
        seed=seed,
        sampling=sampling,
    )
    ms = [
        ("NanoTabPFN", NanoTabPFNRegressor()),
    ]
    if vs_tabpfn:
        try:
            from tabpfn import TabPFNRegressor

            ms.append(("TabPFN", TabPFNRegressor()))
        except Exception as e:
            pxp(f"TabPFN unavailable: {e}")

    rows, cols = len(ds), len(ms)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols + 4, 3.2 * rows + 3))

    axes = np.atleast_1d(axes)
    if axes.ndim == 1:
        axes = axes.reshape(rows, cols)

    for ir, (title, (X, y)) in enumerate(ds):
        for ic, (name, model) in enumerate(ms):
            ax = axes[ir][ic]
            fit_predict_plot(ax, X, y, {name: model}, f"{title} - {name}")

    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.close(fig)

    pxp(f"plot saved to: {plot_filepath}")


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    pxp(f"root dir: {root_dir}")

    timestamp = get_timestamp()

    other_dir = "other"
    plots_dir = "plots"
    plot_filename = f"{timestamp}_plot.png"

    plot_dirpath = os.path.join(root_dir, other_dir, plots_dir)
    plot_filepath = os.path.join(plot_dirpath, plot_filename)
    os.makedirs(plot_dirpath, exist_ok=True)

    visualise_and_save(
        seed=args.seed,
        sampling=args.sampling,
        vs_tabpfn=args.vs_tabpfn,
        plot_filepath=plot_filepath,
    )


if __name__ == "__main__":
    main()
