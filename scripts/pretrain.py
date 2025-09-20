import argparse
import json
import os
import sys

import sklearn
import torch

from pfns.model.bar_distribution import FullSupportBarDistribution
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
from torch import nn

from nanotabpfn.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.priors import PriorDumpDataLoader
from nanotabpfn.train import train
from nanotabpfn.utils import (
    get_default_device,
    make_global_bucket_edges,
    set_randomness_seed,
)

from utils import get_timestamp, get_uuid4, pxp


class PretrainMetadataLogger:
    def __init__(self, args, device, metric_field):
        self.timestamp = get_timestamp()
        self.uuid = get_uuid4()

        self.metadata_path = f"other/metadata/pretrain_metadata_{self.timestamp}.json"
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        self.metric_field = metric_field

        self.entry = {
            "meta": {
                "timestamp": self.timestamp,
                "uuid4": self.uuid,
                "version": "0.0",
                "completed": False,
                "task_type": args.type,
            },
            "args": vars(args),
            "env": {
                "device": str(device),
                "python": sys.version.split()[0],
                "torch": getattr(torch, "__version__", None),
                "sklearn": getattr(sklearn, "__version__", None),
            },
            "epochs": {},
        }

        self._write()

    def log_epoch(self, epoch, epoch_time, mean_loss, metric_value):
        epoch_key = str(int(epoch))
        self.entry["epochs"][epoch_key] = {
            "epoch": int(epoch),
            "epoch_time": float(epoch_time),
            "mean_loss": float(mean_loss),
            self.metric_field: float(metric_value),
        }
        self._write()

    def mark_completed(self, expected_final_epoch):
        recorded_epochs = {int(epoch_key) for epoch_key in self.entry["epochs"].keys()}
        completed = expected_final_epoch in recorded_epochs

        self.entry["meta"]["completed"] = completed
        self._write()

    def _write(self):
        with open(self.metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(self.entry, metadata_file, indent=2, sort_keys=False)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-type",
        type=str,
        choices=("regression", "classification"),
        default="regression",
        help="pretraining task type",
    )
    parser.add_argument(
        "-priordump",
        type=str,
        default=None,
        help="path to the prior dump",
    )
    parser.add_argument(
        "-saveweights",
        type=str,
        default=None,
        help="path to save the trained model to",
    )
    parser.add_argument(
        "-heads",
        type=int,
        default=6,  # 4
        help="number of attention heads",
    )
    parser.add_argument(
        "-embeddingsize",
        type=int,
        default=192,  # 512
        help="the size of the embeddings used for the cells",
    )
    parser.add_argument(
        "-hiddensize",
        type=int,
        default=768,  # 1024
        help="size of the hidden layer of the mlps",
    )
    parser.add_argument(
        "-layers",
        type=int,
        default=6,  # 12
        help="number of transformer layers",
    )
    parser.add_argument(
        "-batchsize",
        type=int,
        default=1,
        help="batch size used during training (before gradient accumulation)",
    )
    parser.add_argument(
        "-accumulate",
        type=int,
        default=1,
        help="number of gradients to accumulate before updating the weights",
    )
    parser.add_argument(
        "-lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "-steps",
        type=int,
        default=100,
        help="number of steps that constitute one epoch (important for lr scheduler)",
    )
    parser.add_argument(
        "-epochs",
        type=int,
        default=10000,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "-n_buckets",
        type=int,
        default=None,
        help="number of buckets for the data loader",
    )
    parser.add_argument(
        "-loadcheckpoint",
        type=str,
        default=None,
        help="checkpoint path",
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    defaults = {
        "regression": {
            "priordump": "other/dumps/50x3_1280k_regression.h5",
            "saveweights": "other/model/nanotabpfn_regressor.pth",
            "n_buckets": 100,  # 5000
        },
        "classification": {
            "priordump": "other/dumps/50x3_3_100k_classification.h5",
            "saveweights": "other/model/nanotabpfn_classifier.pth",
        },
    }

    type_defaults = defaults[args.type]

    if args.priordump is None:
        args.priordump = type_defaults["priordump"]

    if args.saveweights is None:
        args.saveweights = type_defaults["saveweights"]

    if args.seed is None:
        args.seed = type_defaults["seed"]

    if args.type == "regression" and args.n_buckets is None:
        args.n_buckets = type_defaults["n_buckets"]

    return args


def build_eval_datasets(task_type, seed):
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


def run_evaluation(task_type, model, dist, device, eval_datasets):
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


def save_artifact(trained_model, bucket_edges, args, metadata_logger, num_outputs):
    artifact = {
        "state_dict": trained_model.to("cpu").state_dict(),
        "arch": {
            "num_attention_heads": args.heads,
            "embedding_size": args.embeddingsize,
            "mlp_hidden_size": args.hiddensize,
            "num_layers": args.layers,
            "num_outputs": num_outputs,
        },
        "meta": {
            "timestamp": metadata_logger.timestamp,
            "uuid4": metadata_logger.uuid,
            "version": "0.0",
            "task_type": args.type,
        },
    }

    if bucket_edges is not None:
        artifact["bucket_edges"] = bucket_edges.to("cpu")

    base, ext = os.path.splitext(args.saveweights)
    ext = ext if ext in {".pth", ".pt"} else ".pth"
    save_path = f"{base}_{metadata_logger.timestamp}{ext}"

    torch.save(artifact, save_path)


def main():
    args = parse_arguments()

    assert args.steps % args.accumulate == 0, "steps MUST be divisible by accumulate!"

    os.makedirs(os.path.dirname(args.saveweights), exist_ok=True)

    set_randomness_seed(args.seed)

    device = get_default_device()

    pxp(f"device: {device}")

    ckpt = None

    if args.loadcheckpoint:
        ckpt = torch.load(args.loadcheckpoint)

    if not args.priordump or not os.path.isfile(args.priordump):
        raise FileNotFoundError(f"Prior dump not found at {args.priordump!r}")

    prior = PriorDumpDataLoader(
        filename=args.priordump,
        num_steps=args.steps,
        batch_size=args.batchsize,
        device=device,
        starting_index=args.steps * (ckpt["epoch"] if ckpt else 0),
    )

    if args.type == "regression":
        bucket_edges = make_global_bucket_edges(
            filename=args.priordump,
            n_buckets=args.n_buckets,
            device=device,
        )
        criterion = FullSupportBarDistribution(bucket_edges)
        num_outputs = args.n_buckets
        metric_name = "avg_r2"
    else:
        bucket_edges = None
        criterion = nn.CrossEntropyLoss()
        num_outputs = prior.max_num_classes
        metric_name = "avg_accuracy"

    model = NanoTabPFNModel(
        num_attention_heads=args.heads,
        embedding_size=args.embeddingsize,
        mlp_hidden_size=args.hiddensize,
        num_layers=args.layers,
        num_outputs=num_outputs,
    )

    if ckpt:
        model.load_state_dict(ckpt["model"])

    eval_datasets = build_eval_datasets(task_type=args.type, seed=args.seed)

    metadata_logger = PretrainMetadataLogger(args, device, metric_field=metric_name)

    def epoch_callback(epoch, epoch_time, mean_loss, model, dist=None):
        avg_score = run_evaluation(args.type, model, dist, device, eval_datasets)
        if args.type == "regression":
            metric_label = "μr2"
            metric_display = f"{avg_score:5.2f}"
        else:
            metric_label = "μacc"
            metric_display = f"{avg_score:5.2f}"
        print(
            f"e {epoch:5d} | t: {epoch_time:5.2f}s | μl: {mean_loss:5.2f} | {metric_label}: {metric_display}",
            flush=True,
        )
        metadata_logger.log_epoch(epoch, epoch_time, mean_loss, avg_score)

    trained_model, loss = train(
        model=model,
        prior=prior,
        criterion=criterion,
        epochs=args.epochs,
        accumulate_gradients=args.accumulate,
        lr=args.lr,
        device=device,
        epoch_callback=epoch_callback,
        ckpt=ckpt,
    )

    save_artifact(trained_model, bucket_edges, args, metadata_logger, num_outputs)

    metadata_logger.mark_completed(args.epochs)

    pxp(f"metadata logged to: {metadata_logger.metadata_path}")


if __name__ == "__main__":
    main()
