"""Standard GCN node-classifier and training loop for Phase 7b.

A thin wrapper around PyTorch Geometric's GCN to evaluate the downstream utility
of augmentation strategies. Supports multi-class (Elliptic) and multi-label (PPI)
classification via `task` parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


@dataclass
class ClassifierConfig:
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4
    n_epochs: int = 200
    task: str = "multiclass"  # "multiclass" or "multilabel"
    device: str = "cuda"
    early_stop_patience: int = 30


class GCNClassifier(nn.Module):
    """Simple GCN classifier for node classification."""

    def __init__(self, n_features: int, n_classes: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        from torch_geometric.nn import GCNConv

        dims = [n_features] + [hidden_dim] * (n_layers - 1) + [n_classes]
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def _graph_to_edge_index(graph: nx.Graph, device: str) -> torch.Tensor:
    edges = list(graph.edges())
    if not edges:
        return torch.empty(2, 0, dtype=torch.long, device=device)
    src = [u for u, _ in edges] + [v for _, v in edges]
    dst = [v for _, v in edges] + [u for u, _ in edges]
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def train_classifier(
    graph: nx.Graph,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    config: ClassifierConfig,
    augment_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
    seed: int = 0,
) -> dict:
    """Train a GCN classifier with optional per-epoch augmentation.

    Args:
        graph: networkx Graph (undirected).
        features: [n, d] node features.
        labels: [n] int (multiclass) or [n, c] float (multilabel).
        train_mask, val_mask, test_mask: [n] bool.
        config: ClassifierConfig.
        augment_fn: optional (features, seed) -> features applied each epoch on
            training nodes only.
        seed: seed for model init and augmentation.

    Returns:
        dict with final train/val/test metrics and best test metric by val.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = config.device

    n = features.shape[0]
    edge_index = _graph_to_edge_index(graph, device)

    if config.task == "multiclass":
        n_classes = int(labels.max()) + 1 if labels.size else 1
        y = torch.tensor(labels, dtype=torch.long, device=device)
    else:  # multilabel
        n_classes = int(labels.shape[1])
        y = torch.tensor(labels, dtype=torch.float32, device=device)

    train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool, device=device)
    test_mask_t = torch.tensor(test_mask, dtype=torch.bool, device=device)

    model = GCNClassifier(
        features.shape[1], n_classes, config.hidden_dim, config.n_layers, config.dropout
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val = -1.0
    best_test = 0.0
    patience = 0

    for epoch in range(config.n_epochs):
        model.train()
        if augment_fn is not None:
            features_ep = augment_fn(features, seed + epoch)
        else:
            features_ep = features
        x = torch.tensor(features_ep, dtype=torch.float32, device=device)

        optim.zero_grad()
        logits = model(x, edge_index)

        if config.task == "multiclass":
            loss = F.cross_entropy(logits[train_mask_t], y[train_mask_t])
        else:
            loss = F.binary_cross_entropy_with_logits(logits[train_mask_t], y[train_mask_t])

        loss.backward()
        optim.step()

        # Eval
        model.eval()
        with torch.no_grad():
            x_eval = torch.tensor(features, dtype=torch.float32, device=device)
            logits_eval = model(x_eval, edge_index)
            if config.task == "multiclass":
                preds = logits_eval.argmax(dim=-1)
                val_pred = preds[val_mask_t].cpu().numpy()
                val_true = y[val_mask_t].cpu().numpy()
                test_pred = preds[test_mask_t].cpu().numpy()
                test_true = y[test_mask_t].cpu().numpy()
                val_f1 = f1_score(val_true, val_pred, average="binary", zero_division=0) if n_classes == 2 else f1_score(val_true, val_pred, average="macro", zero_division=0)
                test_f1 = f1_score(test_true, test_pred, average="binary", zero_division=0) if n_classes == 2 else f1_score(test_true, test_pred, average="macro", zero_division=0)
            else:
                probs = torch.sigmoid(logits_eval) > 0.5
                val_pred = probs[val_mask_t].cpu().numpy()
                val_true = y[val_mask_t].cpu().numpy()
                test_pred = probs[test_mask_t].cpu().numpy()
                test_true = y[test_mask_t].cpu().numpy()
                val_f1 = f1_score(val_true, val_pred, average="micro", zero_division=0)
                test_f1 = f1_score(test_true, test_pred, average="micro", zero_division=0)

        if val_f1 > best_val:
            best_val = val_f1
            best_test = test_f1
            patience = 0
        else:
            patience += 1
            if patience > config.early_stop_patience:
                break

    return {
        "best_val_f1": float(best_val),
        "best_test_f1": float(best_test),
        "final_val_f1": float(val_f1),
        "final_test_f1": float(test_f1),
        "n_epochs_run": epoch + 1,
    }
