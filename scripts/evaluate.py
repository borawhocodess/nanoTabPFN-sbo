import argparse

from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
    make_regression,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
import torch

from nanotabpfn.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from nanotabpfn.utils import get_default_device
from utils import pxp


def build_eval_datasets(task_type: str, seed: int):
    datasets = []

    if task_type == "regression":
        datasets.append(
            train_test_split(
                *load_diabetes(return_X_y=True),
                test_size=0.5,
                random_state=seed,
            )
        )

        X, y = fetch_california_housing(return_X_y=True)
        X_sub, _, y_sub, _ = train_test_split(X, y, train_size=500, random_state=seed)
        datasets.append(
            train_test_split(
                X_sub,
                y_sub,
                test_size=0.5,
                random_state=seed,
            )
        )

        X, y = make_regression(n_samples=500, n_features=10, random_state=seed)
        datasets.append(
            train_test_split(
                X,
                y,
                test_size=0.5,
                random_state=seed,
            )
        )
    else:
        datasets.append(
            train_test_split(
                *load_iris(return_X_y=True),
                test_size=0.5,
                random_state=seed,
            )
        )
        datasets.append(
            train_test_split(
                *load_wine(return_X_y=True),
                test_size=0.5,
                random_state=seed,
            )
        )
        datasets.append(
            train_test_split(
                *load_breast_cancer(return_X_y=True),
                test_size=0.5,
                random_state=seed,
            )
        )

    return datasets


def run_evaluation(task_type: str, model, dist, device, eval_datasets):
    scores = []

    if task_type == "regression":
        regressor = NanoTabPFNRegressor(model, dist, device)
        for X_train, X_test, y_train, y_test in eval_datasets:
            regressor.fit(X_train, y_train)
            predictions = regressor.predict(X_test)
            scores.append(r2_score(y_test, predictions))
    else:
        classifier = NanoTabPFNClassifier(model, device)
        for X_train, X_test, y_train, y_test in eval_datasets:
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            scores.append(accuracy_score(y_test, predictions))

    return sum(scores) / len(scores)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="path to artifact file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_default_device()

    seed = 42

    task_type = None
    
    obj = torch.load(args.model, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and isinstance(obj.get("meta"), dict) and obj["meta"].get("task_type"):
        task_type = obj["meta"]["task_type"]
    elif (
        isinstance(obj, dict)
        and isinstance(obj.get("model"), dict)
        and obj["model"].get("bucket_edges") is not None
    ):
        task_type = "regression"
    else:
        task_type = "classification"

    eval_datasets = build_eval_datasets(task_type=task_type, seed=seed)

    score = run_evaluation(task_type, args.model, None, device, eval_datasets)

    label = "avg_r2" if task_type == "regression" else "avg_accuracy"

    pxp(f"{label}: {score:5.2f}")


if __name__ == "__main__":
    main()
