import copy
import numpy as np
import torch
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

"""
Because pytorch does not include a built-in early stopping function, we must define one of our own. We will use the following EarlyStopping class. 

We can provide several parameters to the EarlyStopping object:

- min_delta: this value should be kept small; it specifies the minimum change that should be considered an improvement. Setting it even smaller will not likely have a great deal of impact.
- patience: how long should the training wait for the validation error to improve?
- restore_best_weights: you should usually set this to true, as it restores the weights to the values they were at when the validation set is the highest

"""

# class definition
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"NO improvement found, in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


"""
Early stopping with classification example using IRIS dataset
"""

# set random seed
np.random.seed(42)
torch.manual_seed(42)

df = pd.read_csv(filepath_or_buffer="data/iris.csv", na_values=["NA", "?"])
le = LabelEncoder()

# extracting the training features
x = df[["sepal_l", "sepal_w", "petal_l", "petal_w"]].values

# processing the labels
y = le.fit_transform(df["species"])
species = le.classes_



