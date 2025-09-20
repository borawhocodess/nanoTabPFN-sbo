import argparse

from pfns.model.bar_distribution import FullSupportBarDistribution
from torch import nn

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.priors import PriorDumpDataLoader
from nanotabpfn.train import train
from nanotabpfn.utils import (
    get_default_device,
    make_global_bucket_edges,
    set_randomness_seed,
)

# these are from other scripts
from utils import pxp, RunManager
from evaluate import build_eval_datasets, run_evaluation


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-type",
        type=str,
        choices=("regression", "classification"),
        default="classification",
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
    parser.add_argument(
        "-verbose",
        action="store_true",
    )

    args = parser.parse_args()

    defaults = {
        "classification": {
            "priordump": "other/dumps/50x3_3_100k_classification.h5",
            "saveweights": "other/model/nanotabpfn_classifier.pth",
        },
        "regression": {
            "priordump": "other/dumps/50x3_1280k_regression.h5",
            "saveweights": "other/model/nanotabpfn_regressor.pth",
            "n_buckets": 100,  # 5000
        },
    }

    type_defaults = defaults[args.type]

    if args.priordump is None:
        args.priordump = type_defaults["priordump"]

    if args.saveweights is None:
        args.saveweights = type_defaults["saveweights"]

    if args.type == "regression" and args.n_buckets is None:
        args.n_buckets = type_defaults["n_buckets"]

    return args


def main():
    args = parse_arguments()

    set_randomness_seed(args.seed)

    device = get_default_device()

    manager = RunManager(args=args, device=device)

    ckpt = manager.resume_checkpoint

    start_epoch = manager.resume_epoch

    prior = PriorDumpDataLoader(
        filename=args.priordump,
        num_steps=args.steps,
        batch_size=args.batchsize,
        device=device,
        starting_index=args.steps * start_epoch,
    )

    saved_bucket_edges = manager.bucket_edges

    if args.type == "classification":
        bucket_edges = None
        criterion = nn.CrossEntropyLoss()
        num_outputs = prior.max_num_classes
        metric_name = "avg_accuracy"
        metric = "μacc"
    else:
        bucket_edges = (
            saved_bucket_edges
            if saved_bucket_edges is not None
            else make_global_bucket_edges(
                filename=args.priordump,
                n_buckets=args.n_buckets,
                device=device,
            )
        )
        criterion = FullSupportBarDistribution(bucket_edges)
        num_outputs = args.n_buckets
        metric_name = "avg_r2"
        metric = "μr2"

    manager.update_bucket_edges(bucket_edges)

    arch = manager.arch or {
        "num_attention_heads": args.heads,
        "embedding_size": args.embeddingsize,
        "mlp_hidden_size": args.hiddensize,
        "num_layers": args.layers,
        "num_outputs": num_outputs,
    }

    manager.update_arch(arch)

    model = NanoTabPFNModel(
        num_attention_heads=arch["num_attention_heads"],
        embedding_size=arch["embedding_size"],
        mlp_hidden_size=arch["mlp_hidden_size"],
        num_layers=arch["num_layers"],
        num_outputs=arch["num_outputs"],
    )

    if ckpt:
        model.load_state_dict(ckpt["model"]["state_dict"])

    eval_datasets = build_eval_datasets(task_type=args.type, seed=args.seed)

    def epoch_callback(epoch, epoch_time, mean_loss, model, optimizer_state, dist=None):
        score = run_evaluation(args.type, model, dist, device, eval_datasets)
        
        print(f"e {epoch:5d} | t: {epoch_time:5.2f}s | μl: {mean_loss:5.2f} | {metric}: {score:5.2f}", flush=True)

        manager.metadatahandler.log_epoch(epoch, epoch_time, mean_loss, metric_name, score)

        manager.artifacthandler.save_checkpoint(model, optimizer_state, epoch)

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

    manager.artifacthandler.save_artifact(trained_model)

    manager.metadatahandler.mark_completed(args.epochs)

    pxp(
        "metadata logged to: "
        f"{manager.metadatahandler.metadata_path}\n"
        "checkpoint saved to: "
        f"{manager.latest_checkpoint_path}\n"
        "artifact saved to: "
        f"{manager.latest_artifact_path}",
        on=args.verbose,
    )


if __name__ == "__main__":
    main()
