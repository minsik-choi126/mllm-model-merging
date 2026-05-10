"""E-Pull: Entropy-gated Pull for cross-modality model merging.

See README.md / paper Section: Method.
"""

from .merge import epull_merge_state_dicts, EpullConfig
from .covariance import collect_input_grams
from .joint_diag import joint_diagonalize, per_direction_eigenvalues

__all__ = [
    "EpullConfig",
    "epull_merge_state_dicts",
    "collect_input_grams",
    "joint_diagonalize",
    "per_direction_eigenvalues",
]
