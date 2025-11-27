import math
import os

import torch


class EarlyStopping:
    def __init__(self, dataset_name, delta: float = 0.0, patience: int = 7, verbose: bool = True,
                 results_dir: str = '../Results', path: str = 'checkpoint.pt'):
        self.dataset_name = dataset_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = math.inf
        self.delta = delta
        self.results_dir = results_dir
        self.path = path

        self._ensure_results_dir(dataset_name)

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}). Saving model...')
        path = os.path.join(self.results_dir, self.dataset_name, self.path)
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

    def _ensure_results_dir(self, dataset_name):
        directory = os.path.join(self.results_dir, dataset_name)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
