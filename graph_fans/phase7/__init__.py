"""Phase 7: Multi-dataset validation of spectral augmentation for real-world tasks.

Real-world datasets (Elliptic Bitcoin, Stanford PPI) with continuous features and
downstream classification labels. Extends Phase 6 infrastructure to new domains.
"""

from .real_dataset_loader import (
    load_real_dataset,
    load_elliptic,
    load_ppi,
)

__all__ = ["load_real_dataset", "load_elliptic", "load_ppi"]
