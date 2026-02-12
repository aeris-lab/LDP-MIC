import numpy as np
import torch
from typing import Tuple

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_mic_matrix(X: np.ndarray, y: np.ndarray, alpha: float = 0.6, c: float = 15) -> np.ndarray:
    n_features = X.shape[1]
    scores = np.zeros(n_features, dtype=np.float64)
    
    if SKLEARN_AVAILABLE:
        try:
            n_unique = len(np.unique(y))
            if n_unique < 20:
                scores = mutual_info_classif(X, y, random_state=42)
            else:
                scores = mutual_info_regression(X, y, random_state=42)
            scores = np.asarray(scores, dtype=np.float64)
            np.nan_to_num(scores, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            scores = _correlation_scores(X, y, n_features)
    else:
        scores = _correlation_scores(X, y, n_features)
    
    return scores


def _correlation_scores(X: np.ndarray, y: np.ndarray, n_features: int) -> np.ndarray:
    scores = np.zeros(n_features)
    for i in range(n_features):
        try:
            corr = np.corrcoef(X[:, i], y)[0, 1]
            scores[i] = np.abs(corr) if not np.isnan(corr) else 0.0
        except Exception:
            scores[i] = 0.0
    return scores


def compute_mic_weights(X: np.ndarray, y: np.ndarray, alpha: float = 0.6, c: float = 15) -> Tuple[np.ndarray, np.ndarray]:
    importance_scores = compute_mic_matrix(X, y, alpha, c)
    if importance_scores.max() > 0:
        scores_normalized = importance_scores / importance_scores.max()
    else:
        scores_normalized = np.ones_like(importance_scores)
    gamma = scores_normalized + 1e-6
    beta = np.zeros_like(gamma)
    
    return gamma, beta


def compute_mic_for_batch(data_batch: torch.Tensor, labels_batch: torch.Tensor,
                          alpha: float = 0.6, c: float = 15) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(data_batch.shape) > 2:
        data_flat = data_batch.view(data_batch.size(0), -1).cpu().numpy()
    else:
        data_flat = data_batch.cpu().numpy()
    labels_np = labels_batch.cpu().numpy()
    gamma_np, beta_np = compute_mic_weights(data_flat, labels_np, alpha, c)
    if len(data_batch.shape) > 2:
        gamma = torch.from_numpy(gamma_np).reshape(data_batch.shape[1:]).float()
        beta = torch.from_numpy(beta_np).reshape(data_batch.shape[1:]).float()
    else:
        gamma = torch.from_numpy(gamma_np).float()
        beta = torch.from_numpy(beta_np).float()
    
    return gamma, beta

