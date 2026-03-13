"""Benchmark utilities for comparing model variants.

Usage example
-------------
>>> from models.benchmark import run_benchmark
>>> results = run_benchmark(test_loader, scalar, device, model_dir="model_save/Spring/best")
>>> for name, metrics in results.items():
...     print(name, metrics)
"""

import os
import dill
import torch
from utils.tools import evaluate
import torch.nn as nn


def load_model(path: str, device: torch.device):
    """Load a serialised model from *path* and move it to *device*."""
    with open(path, "rb") as f:
        model = torch.load(f, pickle_module=dill, weights_only=False)
    return model.to(device)


def run_benchmark(test_loader, scalar, device, model_dir: str, model_names=None):
    """Evaluate every saved model found in *model_dir* and return their metrics.

    Parameters
    ----------
    test_loader : DataLoader
        Test-set data loader.
    scalar : sklearn scaler
        Inverse-transform scaler (output of ``split_data_cnn``).
    device : torch.device
        Device to run inference on.
    model_dir : str
        Directory that contains ``*.pt`` model files.
    model_names : list[str] | None
        Optional explicit list of model file names to load.  When *None*,
        all ``*.pt`` files found in *model_dir* are used.

    Returns
    -------
    dict[str, list]
        Mapping from model name (without ``.pt``) to ``[MAE, RMSE, R2, MBE]``.
    """
    criterion = nn.L1Loss(reduction="sum").to(device)

    if model_names is None:
        model_names = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

    results = {}
    for name in sorted(model_names):
        path = os.path.join(model_dir, name)
        model = load_model(path, device)
        _, metrics = evaluate(data=test_loader, model=model,
                               criterion=criterion, scalar=scalar)
        key = name[:-3] if name.endswith(".pt") else name
        results[key] = metrics
        print(f"{key:50s}  MAE={metrics[0]:.4f}  RMSE={metrics[1]:.4f}"
              f"  R²={metrics[2]:.4f}  MBE={metrics[3]:.4f}")

    return results
