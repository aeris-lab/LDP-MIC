"""
LDP-MIC: Correlation-Aware Local Differential Privacy for Federated Learning

This package implements the LDP-MIC framework for privacy-preserving federated
learning under untrusted aggregation using Maximum Information Coefficient (MIC)
for adaptive noise calibration.

Main modules:
- FedAverage.py: Main entry point for federated learning
- FedUser.py: Client-side LDP/CDP implementation
- FedServer.py: Server-side aggregation
- modelUtil.py: Model architectures with MIC-based normalization
- mic_utils.py: MIC computation utilities
- datasets.py: Dataset loading and non-IID partitioning
"""

__version__ = "1.0.0"
__author__ = "Anonymous"
