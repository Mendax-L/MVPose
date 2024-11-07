import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.epochs_without_improvement = 0
        self.stopped_epoch = None

    def check_early_stopping(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.stopped_epoch = np.inf
            return True
        else:
            return False