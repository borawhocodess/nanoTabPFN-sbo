import os
import requests
import numpy as np
import torch
import torch.nn.functional as F

from nanotabpfn.utils import get_default_device
from nanotabpfn.model import NanoTabPFNModel

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

class NanoTabPFNClassifier():
    """ scikit-learn like interface """
    def __init__(self, model: NanoTabPFNModel|str|None = None, device=get_default_device()):
        if model == None:
            model = 'nanotabpfn.pth'
            if not os.path.isfile(model):
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/nanotabpfn.pth')
                with open(model, 'wb') as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)
        self.model = model.to(device)
        self.device = device

    def fit(self, X_train: np.array, y_train: np.array):
        """ stores X_train and y_train for later use, also computes the highest class number occuring in num_classes """
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = max(set(y_train))+1

    def predict(self, X_test: np.array) -> np.array:
        """ calls predit_proba and picks the class with the highest probability for each datapoint """
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test: np.array) -> np.array:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x, y), single_eval_pos=len(self.X_train)).squeeze(0)  # remove batch size 1
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, :self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to('cpu').numpy()


