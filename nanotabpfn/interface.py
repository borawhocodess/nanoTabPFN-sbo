import os

import numpy as np
import requests
import torch
import torch.nn.functional as F

from pfns.model.bar_distribution import FullSupportBarDistribution

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.utils import get_default_device


def init_model_from_file(file_path, device=torch.device("cpu")):
    """
    Loads a NanoTabPFN model from a file.
    """
    obj = torch.load(file_path, map_location=device, weights_only=False)

    if not (
        isinstance(obj, dict)
        and isinstance(obj.get("model"), dict)
        and "state_dict" in obj["model"]
        and "arch" in obj
    ):
        raise ValueError("File wrong.")

    arch = obj["arch"]
    model = NanoTabPFNModel(
        num_attention_heads=arch["num_attention_heads"],
        embedding_size=arch["embedding_size"],
        mlp_hidden_size=arch["mlp_hidden_size"],
        num_layers=arch["num_layers"],
        num_outputs=arch["num_outputs"],
    )
    model.load_state_dict(obj["model"]["state_dict"])

    bucket_edges = obj["model"].get("bucket_edges", None)

    return model, bucket_edges


class NanoTabPFNClassifier:
    """
    scikit-learn like interface
    """

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        device=None,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            model = "nanotabpfn.pth" # TODO: fix this
            if not os.path.isfile(model):
                print("No cached model found, downloading model artifact.")
                response = requests.get("https://salihboraozturk.com/artifacts/nanotabpfn_classifier.pth") # TODO: fix link
                with open(model, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model, _ = init_model_from_file(model, device=torch.device("cpu"))
        self.model = model.to(device)
        self.device = device

    def fit(self, X_train: np.array, y_train: np.array):
        """
        stores X_train and y_train for later use,
        also computes the highest class number occuring in num_classes
        """
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

    def predict(self, X_test: np.array) -> np.array:
        """
        calls predit_proba and picks the class
        with the highest probability for each datapoint
        """
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test: np.array) -> np.array:
        """
        creates (x,y), runs it through our PyTorch Model,
        cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)  # introduce batch size 1
            y = torch.as_tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.model((x, y), single_eval_pos=len(self.X_train)).squeeze(0)  # remove batch size 1
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, : self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.cpu().numpy()


class NanoTabPFNRegressor:
    """
    scikit-learn like interface
    """

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        dist: FullSupportBarDistribution | None = None,
        device=None,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            model = "nanotabpfn_regressor.pth" # TODO: fix this
            if not os.path.isfile(model):
                print("No cached model found, downloading model artifact.")
                response = requests.get("https://salihboraozturk.com/artifacts/nanotabpfn_regressor.pth")  # TODO: fix link
                with open(model, "wb") as f:
                    f.write(response.content)

        if isinstance(model, str):
            model, bucket_edges = init_model_from_file(model, device=torch.device("cpu"))
            dist = FullSupportBarDistribution(bucket_edges).float()

        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.normalized_dist = None  # Used after fit()

    def fit(self, X_train: np.array, y_train: np.array):
        """
        Stores X_train and y_train for later use.
        Computes target normalization.
        Builds normalized bar distribution from existing self.dist.
        """
        self.X_train = X_train
        self.y_train = y_train

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train) + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std

        # Convert base distribution to original scale for output
        bucket_edges = self.dist.borders
        bucket_edges_denorm = bucket_edges * self.y_train_std + self.y_train_mean
        self.normalized_dist = FullSupportBarDistribution(bucket_edges_denorm).float()

    def predict(self, X_test: np.array) -> np.array:
        """
        Performs in-context learning using X_train and y_train.
        Predicts the means of the output distributions for X_test.
        """
        X = np.concatenate((self.X_train, X_test))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.model((X_tensor, y_tensor), single_eval_pos=len(self.X_train)).squeeze(0)
            preds = self.normalized_dist.mean(logits)

        return preds.cpu().numpy()
