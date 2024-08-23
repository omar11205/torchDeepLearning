# Defining an early stopping class for PyTorch
import copy


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.patience_counter = 0
        self.status = ""

    def reset(self):
        self.status = ""
        self.patience_counter = 0

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.status = f"Improvement!!!, actual counter {self.patience_counter}"
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            self.status = f"NO improvement in the last {self.patience_counter} epochs"
            if self.patience_counter >= self.patience:
                self.status = f"Early stopping triggered after {self.patience_counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                self.reset()
                return True
        return False
