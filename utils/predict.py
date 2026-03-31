"""Test-set inference and ``predictions.csv`` export."""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Set

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.data import H5BinaryDataset
from utils.constants import DEVICE, TEST_IMAGES_PATH


def predict_test(
    run_dir: str,
    model: nn.Module,
    transform: Callable[[torch.Tensor], torch.Tensor],
    threshold: float,
    test_outlier_ids: Optional[Set[str]] = None,
    subset_ids: Optional[List[str]] = None,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Run inference on ``test.h5`` and write ``predictions.csv`` under ``run_dir``.

    Outliers (``test_outlier_ids``): probability forced to 0, no forward pass.
    ``subset_ids``: optional list of patch keys (e.g. smoke-test fraction).
    ``num_workers``: DataLoader workers; ``transform`` must be picklable (e.g. ``PreprocessingTransform``).
    """
    test_ds = H5BinaryDataset(
        TEST_IMAGES_PATH,
        transform=transform,
        mode="test",
        outlier_ids=test_outlier_ids,
        subset_ids=subset_ids,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
    )

    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False, desc="predict"):
            if len(batch) == 3:
                x, img_id, is_out = batch
                is_out = bool(is_out.item())
            else:
                x, img_id = batch
                is_out = False
            x = x.to(DEVICE)
            if is_out:
                prob = 0.0
            else:
                # Multi-view support: (1,V,C,H,W) -> mean prob over views.
                if isinstance(x, torch.Tensor) and x.ndim == 5:
                    b, v = int(x.shape[0]), int(x.shape[1])
                    x2 = x.reshape(b * v, *x.shape[2:])
                    logits = model(x2).view(b, v, -1)
                    prob = torch.sigmoid(logits).mean().item()
                else:
                    logits = model(x)
                    prob = torch.sigmoid(logits).item()
            rows.append({"ID": int(img_id.item()), "Pred": int(prob > threshold)})

    out = pd.DataFrame(rows).set_index("ID")
    out_path = os.path.join(run_dir, "predictions.csv")
    out.to_csv(out_path)
    return out
