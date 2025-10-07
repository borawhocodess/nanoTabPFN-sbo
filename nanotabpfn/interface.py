import os

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.utils import FullSupportBarDistribution, get_default_device


def init_model_from_state_dict_file(file_path):
    """
    infers model architecture from state dict, instantiates the architecture and loads the weights
    """
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    embedding_size = state_dict['feature_encoder.linear_layer.weight'].shape[0]
    mlp_hidden_size = state_dict['decoder.linear1.weight'].shape[0]
    num_outputs = state_dict['decoder.linear2.weight'].shape[0]
    num_layers = sum('self_attn_between_datapoints.in_proj_weight' in k for k in state_dict)
    num_heads = state_dict['transformer_encoder.transformer_blocks.0.self_attn_between_datapoints.in_proj_weight'].shape[1]//64
    model = NanoTabPFNModel(
        num_attention_heads=num_heads,
        embedding_size=embedding_size,
        mlp_hidden_size=mlp_hidden_size,
        num_layers=num_layers,
        num_outputs=num_outputs,
    )
    model.load_state_dict(torch.load(file_path, map_location='cpu'))
    return model

def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum() # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(lambda x: pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x)), # to apply pd.to_numeric of pandas
        ("to_numeric", FunctionTransformer(lambda x: x.apply(pd.to_numeric, errors='coerce').to_numpy())), # in case numeric columns are stored as strings
        ('imputer', SimpleImputer(strategy='mean')) # median might be better because of outliers
    ])
    cat_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_mask),
            ('cat', cat_transformer, cat_mask)
        ]
    )
    return preprocessor


class NanoTabPFNClassifier():
    """ scikit-learn like interface """

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        device: None | str | torch.device = None,
        num_mem_chunks: int = 8,
        n_estimators: int = 1,
        feature_permutation: bool = True,
        label_permutation: bool = True,
        temperature: float = 1.0,
        random_state: int | None = None,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            model = 'nanotabpfn.pth'
            if not os.path.isfile(model):
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/nanotabpfn_classifier.pth')
                with open(model, 'wb') as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)
        self.model = model.to(device)
        self.device = device
        self.num_mem_chunks = num_mem_chunks
        if n_estimators < 1:
            raise ValueError('n_estimators must be at least 1')
        if temperature <= 0:
            raise ValueError('temperature must be greater than 0')
        self.n_estimators = n_estimators
        self.feature_permutation = feature_permutation
        self.label_permutation = label_permutation
        self.temperature = temperature
        self.random_state = random_state

        self._rng = np.random.default_rng(self.random_state)
        self._ensemble_configs: list[dict[str, np.ndarray | None]] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """ stores X_train and y_train for later use, also computes the highest class number occuring in num_classes """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = np.asarray(y_train).astype(int)
        self.num_classes = int(max(set(self.y_train)) + 1)

        self._rng = np.random.default_rng(self.random_state)
        self._ensemble_configs = []
        num_features = self.X_train.shape[1]
        for _ in range(self.n_estimators):
            feature_perm = None
            if self.feature_permutation and num_features > 1:
                feature_perm = self._rng.permutation(num_features)
            label_perm = None
            if self.label_permutation and self.num_classes > 1:
                label_perm = self._rng.permutation(self.num_classes)
            self._ensemble_configs.append({'feature_perm': feature_perm, 'label_perm': label_perm})
        if not self._ensemble_configs:
            # ensure at least one default configuration
            self._ensemble_configs.append({'feature_perm': None, 'label_perm': None})

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """ calls predit_proba and picks the class with the highest probability for each datapoint """
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        X_test_transformed = self.feature_preprocessor.transform(X_test)

        ensemble_probabilities = []

        for config in self._ensemble_configs:
            feature_perm = config['feature_perm']
            label_perm = config['label_perm']

            x_concat = np.concatenate((self.X_train, X_test_transformed))
            if feature_perm is not None:
                x_concat = x_concat[:, feature_perm]

            if label_perm is not None:
                y_train = np.take(label_perm, self.y_train)
            else:
                y_train = self.y_train

            with torch.no_grad():
                x_tensor = torch.from_numpy(x_concat).unsqueeze(0).to(torch.float32).to(self.device)
                y_tensor = torch.from_numpy(y_train).unsqueeze(0).to(torch.float32).to(self.device)
                logits = self.model((x_tensor, y_tensor), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks).squeeze(0)
                logits = logits[:, :self.num_classes] / self.temperature
                probabilities = F.softmax(logits, dim=1).to('cpu').numpy()

            if label_perm is not None:
                probabilities = probabilities[:, label_perm]

            ensemble_probabilities.append(probabilities)

        mean_probabilities = np.mean(ensemble_probabilities, axis=0)
        return mean_probabilities


class NanoTabPFNRegressor():
    """ scikit-learn like interface """

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        dist: FullSupportBarDistribution | str | None = None,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
        n_estimators: int = 1,
        feature_permutation: bool = True,
        random_state: int | None = None,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            model = 'nanotabpfn_regressor.pth'
            dist = 'nanotabpfn_regressor_buckets.pth'
            if not os.path.isfile(model):
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/nanotabpfn_regressor.pth')
                with open(model, 'wb') as f:
                    f.write(response.content)
            if not os.path.isfile(dist):
                print('No cached bucket edges found, downloading bucket edges.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/nanotabpfn_regressor_buckets.pth')
                with open(dist, 'wb') as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)

        if isinstance(dist, str):
            bucket_edges = torch.load(dist, map_location=device)
            dist = FullSupportBarDistribution(bucket_edges).float()

        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.num_mem_chunks = num_mem_chunks
        if n_estimators < 1:
            raise ValueError('n_estimators must be at least 1')
        self.n_estimators = n_estimators
        self.feature_permutation = feature_permutation
        self.random_state = random_state

        self._rng = np.random.default_rng(self.random_state)
        self._ensemble_configs: list[dict[str, np.ndarray | None]] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Stores X_train and y_train for later use.
        Computes target normalization.
        """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = np.asarray(y_train)

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train) + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std

        self._rng = np.random.default_rng(self.random_state)
        self._ensemble_configs = []
        num_features = self.X_train.shape[1]
        for _ in range(self.n_estimators):
            feature_perm = None
            if self.feature_permutation and num_features > 1:
                feature_perm = self._rng.permutation(num_features)
            self._ensemble_configs.append({'feature_perm': feature_perm})
        if not self._ensemble_configs:
            self._ensemble_configs.append({'feature_perm': None})

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Performs in-context learning using X_train and y_train.
        Predicts the means of the output distributions for X_test.
        Renormalizes the predictions back to the original target scale.
        """
        X_test_transformed = self.feature_preprocessor.transform(X_test)

        ensemble_predictions = []

        for config in self._ensemble_configs:
            feature_perm = config['feature_perm']

            X_concat = np.concatenate((self.X_train, X_test_transformed))
            if feature_perm is not None:
                X_concat = X_concat[:, feature_perm]

            with torch.no_grad():
                X_tensor = torch.tensor(X_concat, dtype=torch.float32, device=self.device).unsqueeze(0)
                y_tensor = torch.tensor(self.y_train_n, dtype=torch.float32, device=self.device).unsqueeze(0)

                logits = self.model((X_tensor, y_tensor), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks).squeeze(0)
                preds_n = self.dist.mean(logits)
                preds = preds_n * self.y_train_std + self.y_train_mean

            ensemble_predictions.append(preds.cpu().numpy())

        mean_predictions = np.mean(ensemble_predictions, axis=0)
        return mean_predictions
