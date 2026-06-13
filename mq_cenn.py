# =============================================================================
# mq_cenn.py
# Multi-Quantum Cellular Neural Network (MQ-CeNN)
# Framework classique quantiquement informé par distillation de connaissances.
# =============================================================================

from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit


def _as_float_array(x):
    arr = np.asarray(x, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Input contains non-finite values.")
    return arr


def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QuantumRFFRegressor:
    """
    Régresseur basé sur des caractéristiques de Fourier aléatoires (RFF)
    approximant un noyau quantique gaussien dans un espace de Hilbert.
    """
    def __init__(
        self,
        n_features: int = 1024,
        alpha: float = 1.0,
        gamma: float = 1.0,
        random_state: int = 42,
    ):
        self.n_features = int(n_features)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.random_state = int(random_state)
        self.W_ = None
        self.b_ = None
        self.beta_ = None

    def _project(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("The teacher must be fitted before projection.")
        z = X @ self.W_ + self.b_
        return np.sqrt(2.0 / self.n_features) * np.cos(z)

    def fit(self, X, y):
        X = _as_float_array(X)
        y = _as_float_array(y).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch.")
        rng = np.random.default_rng(self.random_state)
        d = X.shape[1]
        scale = np.sqrt(2.0 * self.gamma)
        self.W_ = rng.normal(loc=0.0, scale=scale, size=(d, self.n_features))
        self.b_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)
        Z = self._project(X)
        A = Z.T @ Z
        A.flat[:: A.shape[0] + 1] += self.alpha
        rhs = Z.T @ y
        try:
            self.beta_ = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.lstsq(A, rhs, rcond=None)[0]
        return self

    def predict(self, X):
        X = _as_float_array(X)
        if self.beta_ is None:
            raise RuntimeError("The teacher must be fitted before predict().")
        return self._project(X) @ self.beta_


class QuantumEntityPool:
    """
    Comité d'experts quantiques indépendants entraînés avec différentes 
    seeds de projection. Sélection chronologique par TimeSeriesSplit.
    """
    def __init__(
        self,
        n_experts: int = 5,
        n_features: int = 1024,
        gamma: float = 1.0,
        alpha_grid: Optional[Sequence[float]] = None,
        base_seed: int = 42,
        n_splits: int = 3,
    ):
        self.n_experts = int(n_experts)
        self.n_features = int(n_features)
        self.gamma = float(gamma)
        self.alpha_grid = list(alpha_grid or [1e-3, 1e-2, 1e-1, 1.0, 10.0])
        self.base_seed = int(base_seed)
        self.n_splits = int(n_splits)
        self.best_alpha_ = None
        self.experts_ = []

    def _select_alpha(self, X: np.ndarray, y: np.ndarray) -> float:
        if len(y) < max(30, self.n_splits + 2):
            return float(self.alpha_grid[len(self.alpha_grid) // 2])
        splitter = TimeSeriesSplit(n_splits=self.n_splits)
        best_alpha, best_loss = None, np.inf
        for alpha in self.alpha_grid:
            losses = []
            for train_idx, val_idx in splitter.split(X):
                model = QuantumRFFRegressor(
                    n_features=self.n_features,
                    alpha=alpha,
                    gamma=self.gamma,
                    random_state=self.base_seed,
                )
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[val_idx])
                losses.append(np.mean(np.abs(pred - y[val_idx])))
            loss = float(np.mean(losses))
            if loss < best_loss:
                best_loss, best_alpha = loss, alpha
        return float(best_alpha)

    def fit(self, X, y):
        X = _as_float_array(X)
        y = _as_float_array(y).reshape(-1)
        self.best_alpha_ = self._select_alpha(X, y)
        self.experts_ = []
        for i in range(self.n_experts):
            model = QuantumRFFRegressor(
                n_features=self.n_features,
                alpha=self.best_alpha_,
                gamma=self.gamma,
                random_state=self.base_seed + 997 * i,
            )
            model.fit(X, y)
            self.experts_.append(model)
        return self

    def predict_pool(self, X):
        if not self.experts_ :
            raise RuntimeError("The pool must be fitted before predict_pool().")
        X = _as_float_array(X)
        return np.column_stack([m.predict(X) for m in self.experts_])

    def predict_mean(self, X):
        return self.predict_pool(X).mean(axis=1)


class CeNNGatingEngine(nn.Module):
    """
    Réseau de Neurones Cellulaire (CeNN) calculant localement les poids
    d'attention locaux via des convolutions 1D temporelles.
    """
    def __init__(
        self,
        input_dim: int,
        n_experts: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.local_cell = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.Tanh(),
        )
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_experts),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_seq):
        x = x_seq.permute(0, 2, 1)
        h = self.local_cell(x)
        pooled = h.mean(dim=-1)
        logits = self.state_head(pooled)
        return self.softmax(logits)


class TrainingTrace:
    """Structure standard pour le suivi des métriques d'entraînement."""
    def __init__(self, best_val_loss: float, epochs_ran: int):
        self.best_val_loss = float(best_val_loss)
        self.epochs_ran = int(epochs_ran)


class MQCeNNRegressor(BaseEstimator, RegressorMixin):
    """
    Framework Multi-Quantum Cellular Neural Network (MQ-CeNN) unifié.
    Gère la reconstruction inverse glissante locale anti-leakage.
    """
    def __init__(
        self,
        n_quantum_features: int = 1024,
        n_experts: int = 5,
        gamma: float = 1.0,
        cenn_hidden: int = 64,
        cenn_kernel: int = 3,
        cenn_epochs: int = 40,
        cenn_lr: float = 1e-3,
        batch_size: int = 512,
        patience: int = 6,
        stationarize: bool = False,
        last_value_index: Optional[int] = None,
        base_seed: int = 42,
        device: Optional[str] = None,
    ):
        self.n_quantum_features = int(n_quantum_features)
        self.n_experts = int(n_experts)
        self.gamma = float(gamma)
        self.cenn_hidden = int(cenn_hidden)
        self.cenn_kernel = int(cenn_kernel)
        self.cenn_epochs = int(cenn_epochs)
        self.cenn_lr = float(cenn_lr)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.stationarize = bool(stationarize)
        self.last_value_index = last_value_index
        self.base_seed = int(base_seed)
        self.device = device
        self.pool_ = None
        self.cenn_ = None
        self.trace_ = None

    def _device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _target_for_training(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.stationarize:
            return y
        if self.last_value_index is None:
            raise ValueError("last_value_index is required when stationarize=True.")
        return y - X[:, int(self.last_value_index)]

    def _reconstruct_prediction(self, X: np.ndarray, pred: np.ndarray) -> np.ndarray:
        if not self.stationarize:
            return pred
        return pred + X[:, int(self.last_value_index)]

    def _make_context(self, X: np.ndarray, context_width: int) -> np.ndarray:
        n, d = X.shape
        width = max(1, min(context_width, d))
        usable = X[:, :width]
        return usable.reshape(n, width, 1)

    def _split_indices(self, n: int):
        cut = max(1, int(n * 0.85))
        idx = np.arange(n)
        return idx[:cut], idx[cut:]

    def fit(self, X, y):
        set_torch_seed(self.base_seed)
        X = _as_float_array(X)
        y = _as_float_array(y).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch.")
        y_fit = self._target_for_training(X, y)
        self.pool_ = QuantumEntityPool(
            n_experts=self.n_experts,
            n_features=self.n_quantum_features,
            gamma=self.gamma,
            base_seed=self.base_seed,
        ).fit(X, y_fit)
        pool_preds = self.pool_.predict_pool(X)
        context_width = int(self.last_value_index + 1) if self.last_value_index is not None else X.shape[1]
        X_seq = self._make_context(X, context_width)
        train_idx, val_idx = self._split_indices(len(y_fit))
        self._fit_cenn(X_seq, pool_preds, y_fit, train_idx, val_idx)
        return self

    def _fit_cenn(self, X_seq, pool_preds, y_fit, train_idx, val_idx):
        dev = self._device()
        x_t = torch.as_tensor(X_seq, dtype=torch.float32, device=dev)
        p_t = torch.as_tensor(pool_preds, dtype=torch.float32, device=dev)
        y_t = torch.as_tensor(y_fit, dtype=torch.float32, device=dev)
        self.cenn_ = CeNNGatingEngine(
            input_dim=x_t.shape[2],
            n_experts=self.n_experts,
            hidden_dim=self.cenn_hidden,
            kernel_size=self.cenn_kernel,
        ).to(dev)
        opt = optim.AdamW(self.cenn_.parameters(), lr=self.cenn_lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        best_state, best_val = None, np.inf
        no_gain = 0
        val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=dev)
        for epoch in range(self.cenn_epochs):
            self.cenn_.train()
            order = train_idx.copy()
            np.random.default_rng(self.base_seed + epoch).shuffle(order)
            for start in range(0, len(order), self.batch_size):
                ids = torch.as_tensor(order[start:start + self.batch_size], dtype=torch.long, device=dev)
                opt.zero_grad()
                alpha = self.cenn_(x_t[ids])
                pred = (alpha * p_t[ids]).sum(dim=1)
                loss = loss_fn(pred, y_t[ids])
                loss.backward()
                nn.utils.clip_grad_norm_(self.cenn_.parameters(), 1.0)
                opt.step()
            self.cenn_.eval()
            with torch.no_grad():
                pred_val = (self.cenn_(x_t[val_idx_t]) * p_t[val_idx_t]).sum(dim=1)
                val_loss = float(loss_fn(pred_val, y_t[val_idx_t]).detach().cpu())
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.cenn_.state_dict().items()}
                no_gain = 0
            else:
                no_gain += 1
                if no_gain >= self.patience:
                    break
        if best_state is not None:
            self.cenn_.load_state_dict(best_state)
        self.trace_ = TrainingTrace(best_val_loss=float(best_val), epochs_ran=epoch + 1)

    def predict(self, X):
        if self.pool_ is None or self.cenn_ is None:
            raise RuntimeError("The model must be fitted before predict().")
        X = _as_float_array(X)
        pool_preds = self.pool_.predict_pool(X)
        context_width = int(self.last_value_index + 1) if self.last_value_index is not None else X.shape[1]
        X_seq = self._make_context(X, context_width)
        dev = self._device()
        self.cenn_.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(X_seq, dtype=torch.float32, device=dev)
            p_t = torch.as_tensor(pool_preds, dtype=torch.float32, device=dev)
            alpha = self.cenn_(x_t)
            pred = (alpha * p_t).sum(dim=1).detach().cpu().numpy()
        return self._reconstruct_prediction(X, pred)

    def predict_teacher_mean(self, X):
        if self.pool_ is None:
            raise RuntimeError("The pool must be fitted before predict_teacher_mean().")
        X = _as_float_array(X)
        pred = self.pool_.predict_mean(X)
        return self._reconstruct_prediction(X, pred)
