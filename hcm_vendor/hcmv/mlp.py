"""PyTorch multilayer perceptron wrapped as a scikit-learn binary classifier.

``TorchMLPClassifier`` drops into ``build_pipeline`` and ``GridSearchCV`` like any
sklearn estimator. Inputs are expected to be standardized already (the pipeline's
``StandardScaler`` does this).

Training, all inside the data passed to ``fit`` so it never sees the outer test fold:

1. Carve a stratified validation split (``val_fraction``) from the training rows.
2. Train ``Linear -> ReLU -> Dropout`` x len(hidden) -> ``Linear(1)`` with
   ``BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`` (class balancing) and Adam
   (L2 penalty via ``weight_decay``), in shuffled mini-batches.
3. After every epoch, score the validation loss (class-weighted, dropout off). Stop
   after ``patience`` epochs without improvement and restore the best weights.

Seeding is local: weights, dropout masks, batch order and the validation split all
derive from ``random_state`` inside ``torch.random.fork_rng``, so repeated fits give
identical predictions and the global torch RNG is left untouched.
"""

import copy

import numpy as np
import torch
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import nn


def build_network(n_features: int, hidden, dropout: float) -> nn.Sequential:
    """``Linear -> ReLU -> Dropout`` per hidden layer, then a single-logit output."""
    layers, width = [], n_features
    for units in hidden:
        layers += [nn.Linear(width, int(units)), nn.ReLU(), nn.Dropout(dropout)]
        width = int(units)
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Binary MLP classifier with early stopping on an internal validation split.

    Parameters
    ----------
    hidden : tuple of int
        Units per hidden layer, e.g. ``(32, 16)``.
    dropout : float
        Dropout probability after each hidden layer.
    weight_decay : float
        Adam L2 penalty.
    lr : float
        Adam learning rate.
    batch_size : int or None
        Mini-batch size; ``None`` trains full-batch.
    max_epochs, patience : int
        Epoch cap and early-stopping patience (epochs without validation improvement).
    val_fraction : float
        Stratified share of the training rows held out for early stopping. If a
        class has too few rows to split, early stopping is skipped and the network
        trains for ``max_epochs`` on all rows.
    random_state : int
        Seed for weights, dropout, batch order and the validation split.
    device : str
        Torch device; ``"cpu"`` by default.
    n_threads : int
        Torch intra-op threads during fit/predict. 1 keeps parallel CV workers from
        oversubscribing cores.
    """

    def __init__(self, hidden=(32, 16), dropout=0.2, weight_decay=1e-3, lr=1e-3, batch_size=32,
                 max_epochs=1000, patience=50, val_fraction=0.2, random_state=0, device="cpu",
                 n_threads=1):
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device
        self.n_threads = n_threads

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    # ------------------------------------------------------------------ helpers
    def _tensor(self, array):
        return torch.as_tensor(np.asarray(array, dtype=np.float32), device=self.device)

    def _split(self, X, y):
        """(X_train, y_train, X_val, y_val); validation is None if it cannot be stratified."""
        counts = np.bincount(y, minlength=2)
        n_val = int(np.ceil(self.val_fraction * len(y)))
        if self.val_fraction <= 0 or counts.min() < 2 or n_val < 2 or len(y) - n_val < 2:
            return X, y, None, None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_fraction, stratify=y, random_state=self.random_state)
        return X_tr, y_tr, X_val, y_val

    # ---------------------------------------------------------------------- API
    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float32)
        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(f"Only binary classification is supported. The type of the target is {y_type}.")
        self.classes_, y_index = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Classifier can't train when only one class is present.")
        y_index = y_index.astype(np.int64)

        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(self.n_threads)
        torch.use_deterministic_algorithms(True)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.random_state)
                self._train(X, y_index)
        finally:
            torch.set_num_threads(previous_threads)
            torch.use_deterministic_algorithms(previous_deterministic)
        return self

    def _train(self, X, y):
        X_tr, y_tr, X_val, y_val = self._split(X, y)
        n_pos = max(int(y_tr.sum()), 1)
        pos_weight = torch.tensor([(len(y_tr) - n_pos) / n_pos], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        network = build_network(X.shape[1], self.hidden, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        X_tr_t, y_tr_t = self._tensor(X_tr), self._tensor(y_tr)
        has_val = X_val is not None
        if has_val:
            X_val_t, y_val_t = self._tensor(X_val), self._tensor(y_val)
        batch = len(y_tr) if not self.batch_size else int(self.batch_size)

        best_loss, best_state, best_epoch, waited = np.inf, None, 0, 0
        self.loss_curve_, self.validation_curve_ = [], []
        for epoch in range(1, self.max_epochs + 1):
            network.train()
            order = torch.randperm(len(y_tr), device=self.device)
            epoch_loss = 0.0
            for start in range(0, len(y_tr), batch):
                rows = order[start:start + batch]
                optimizer.zero_grad()
                loss = loss_fn(network(X_tr_t[rows]).squeeze(1), y_tr_t[rows])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(rows)
            self.loss_curve_.append(epoch_loss / len(y_tr))
            if not has_val:
                continue
            network.eval()
            with torch.no_grad():
                val_loss = loss_fn(network(X_val_t).squeeze(1), y_val_t).item()
            self.validation_curve_.append(val_loss)
            if val_loss < best_loss:
                best_loss, best_epoch, waited = val_loss, epoch, 0
                best_state = copy.deepcopy(network.state_dict())
            else:
                waited += 1
                if waited >= self.patience:
                    break

        if has_val:
            network.load_state_dict(best_state)
        network.eval()
        self.network_ = network
        self.n_epochs_ = epoch
        self.best_epoch_ = best_epoch if has_val else epoch
        self.best_validation_loss_ = float(best_loss) if has_val else None
        self.early_stopping_used_ = has_val

    def decision_function(self, X):
        """Raw logits for the positive class (``classes_[1]``)."""
        check_is_fitted(self, "network_")
        X = validate_data(self, X, dtype=np.float32, reset=False)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(self.n_threads)
        try:
            with torch.no_grad():
                logits = self.network_(self._tensor(X)).squeeze(1)
        finally:
            torch.set_num_threads(previous_threads)
        return logits.cpu().numpy().astype(np.float64)

    def predict_proba(self, X):
        check_is_fitted(self)
        positive = expit(self.decision_function(X))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        check_is_fitted(self)
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
