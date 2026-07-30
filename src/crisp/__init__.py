"""CRISP: Concept Removal via Interpretable Sparse Projections.

Reproduction of Ashuach et al. (ACL 2026), "CRISP: Persistent Concept
Unlearning via Sparse Autoencoders".
"""

from .config import Config
from .features import SelectedFeatures, select_features
from .losses import representation_distance, total_loss, unlearning_loss
from .metrics import overall_score, selection_score
from .sae import SparseAutoencoder, load_saes
from .train import train_crisp

__version__ = "0.1.0"

__all__ = [
    "Config",
    "SelectedFeatures",
    "SparseAutoencoder",
    "load_saes",
    "overall_score",
    "representation_distance",
    "select_features",
    "selection_score",
    "total_loss",
    "train_crisp",
    "unlearning_loss",
]
