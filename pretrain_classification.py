import argparse
import os

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn

from nanotabpfn.callbacks import ConsoleLoggerCallback, WandbLoggerCallback
from nanotabpfn.evaluation import get_openml_predictions, TOY_TASKS_CLASSIFICATION
from nanotabpfn.interface import NanoTabPFNClassifier
from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.priors import PriorDumpDataLoader
from nanotabpfn.train import train
from nanotabpfn.utils import get_default_device, set_randomness_seed

parser = argparse.ArgumentParser()

parser.add_argument("-priordump", type=str, default="other/dumps/50x3_3_100k_classification.h5", help="path to the prior dump")
parser.add_argument("-saveweights", type=str, default="other/model/nanotabpfn_weights.pth", help="path to save the trained model to")
parser.add_argument("-heads", type=int, default=6, help="number of attention heads")
parser.add_argument("-embeddingsize", type=int, default=192, help="the size of the embeddings used for the cells")
parser.add_argument("-hiddensize", type=int, default=768, help="size of the hidden layer of the mlps")
parser.add_argument("-layers", type=int, default=6, help="number of transformer layers")
parser.add_argument("-batchsize", type=int, default=1, help="batch size used during training (before gradient accumulation)")
parser.add_argument("-accumulate", type=int, default=1, help="number of gradients to accumulate before updating the weights")
parser.add_argument("-lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-steps", type=int, default=100, help="number of steps that constitute one epoch (important for lr scheduler)")
parser.add_argument("-epochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("-loadcheckpoint", type=str, default=None, help="checkpoint from which to continue training")

args = parser.parse_args()

set_randomness_seed(2402)

device = get_default_device()

os.makedirs(os.path.dirname(args.saveweights), exist_ok=True)

ckpt = None
if args.loadcheckpoint:
    ckpt = torch.load(args.loadcheckpoint)

prior = PriorDumpDataLoader(filename=args.priordump, num_steps=args.steps, batch_size=args.batchsize, device=device, starting_index=args.steps*(ckpt['epoch'] if ckpt else 0))

criterion = nn.CrossEntropyLoss()

model = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=prior.max_num_classes,
)

if ckpt:
    model.load_state_dict(ckpt['model'])


class ToyEvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks):
        self.tasks = tasks

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        classifier = NanoTabPFNClassifier(model, device)
        predictions = get_openml_predictions(model=classifier, tasks=self.tasks)
        scores = []
        for dataset_name, (y_true, y_pred, y_proba) in predictions.items():
            scores.append(accuracy_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        print(f'epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg accuracy {avg_score:.3f}',
              flush=True)


class ProductionEvaluationLoggerCallback(WandbLoggerCallback):
    def __init__(self, project: str, name: str = None, config: dict = None, log_dir: str = None):
        super().__init__(project, name, config, log_dir)

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        classifier = NanoTabPFNClassifier(model, device)
        predictions = get_openml_predictions(model=classifier, classification=True)
        scores = []
        for dataset_name, (y_true, y_pred, y_proba) in predictions.items():
            scores.append(roc_auc_score(y_true, y_proba, multi_class='ovr'))
        avg_score = sum(scores) / len(scores)
        self.wandb.log({
            'epoch': epoch,
            'epoch_time': epoch_time,
            'mean_loss': loss,
            'tabarena_avg_roc_auc': avg_score
        })
        print(f'epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg roc auc {avg_score:.3f}',
              flush=True)


#callbacks = [ProductionEvaluationLoggerCallback('nanoTFM', 'nanoTabPFN-1')]
callbacks = [ToyEvaluationLoggerCallback(TOY_TASKS_CLASSIFICATION)]

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=criterion,
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=callbacks,
    ckpt=ckpt,
    multi_gpu=args.multigpu
)

torch.save(trained_model.to('cpu').state_dict(), args.saveweights)
