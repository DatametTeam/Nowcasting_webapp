"""
Model evaluation module.
Provides metrics computation and visualization for model performance evaluation.
"""

from nwc_webapp.evaluation.metrics import compute_CSI
from nwc_webapp.evaluation.plots import generate_metrics_plot

__all__ = [
    'compute_CSI',
    'generate_metrics_plot',
]