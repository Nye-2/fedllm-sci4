"""SPECTRA-FedCore method utilities.

This package contains the experiment-facing implementation skeleton for public-spectral
federated core adaptation: SVD basis construction, rank allocation, adapter cores,
FedAvg-style aggregation helpers, metrics, and conservative client-level DP accounting.
"""

from spectra.metrics import classification_metrics
from spectra.experiment_plan import build_run_specs, load_experiment_config, materialize_run_queue, validate_experiment_inputs

try:
    from spectra.adapter import SpectralCoreAdapter
    from spectra.basis import SVDBasis, allocate_layer_ranks, build_truncated_svd, retained_energy
    from spectra.fl import flatten_core_state, load_core_state_, weighted_average_updates
    from spectra.privacy import (
        DEFAULT_RDP_ORDERS,
        calibrate_noise_std,
        clip_by_l2,
        compose_gaussian_rdp,
        gaussian_rdp,
        rdp_to_epsilon,
    )
except ModuleNotFoundError:
    # The base dev environment can run data/metric tests without torch. Server
    # experiment environments should install the `[exp]` extra to enable these.
    pass

__all__ = [
    "DEFAULT_RDP_ORDERS",
    "SVDBasis",
    "SpectralCoreAdapter",
    "allocate_layer_ranks",
    "build_truncated_svd",
    "calibrate_noise_std",
    "classification_metrics",
    "build_run_specs",
    "clip_by_l2",
    "compose_gaussian_rdp",
    "flatten_core_state",
    "gaussian_rdp",
    "load_core_state_",
    "load_experiment_config",
    "materialize_run_queue",
    "rdp_to_epsilon",
    "retained_energy",
    "validate_experiment_inputs",
    "weighted_average_updates",
]
