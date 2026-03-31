"""Metrics, CSV logging, early stopping, and the training loop."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.common import ensure_dir, safe_json
from utils.config import EarlyStoppingConfig, TrainConfig
from utils.constants import DEVICE


class CSVLogger:
    """Append-only CSV logger: one row per (epoch, split) with metrics and extra columns."""

    def __init__(self, csv_path: str, extra_fieldnames: List[str], metric_fieldnames: List[str]) -> None:
        self.csv_path = csv_path
        self.extra_fieldnames = list(extra_fieldnames)
        self.metric_fieldnames = list(metric_fieldnames)
        ensure_dir(os.path.dirname(self.csv_path))

        if not os.path.exists(self.csv_path):
            header = ["run_name", "epoch", "split"] + self.metric_fieldnames + self.extra_fieldnames
            pd.DataFrame(columns=header).to_csv(self.csv_path, index=False)

    def log(self, run_name: str, epoch: int, split: str, metrics: Dict[str, Any], extra: Dict[str, Any]) -> None:
        row = {
            "run_name": run_name,
            "epoch": int(epoch),
            "split": str(split),
        }
        for k in self.metric_fieldnames:
            row[k] = safe_json(metrics.get(k))
        for k in self.extra_fieldnames:
            row[k] = safe_json(extra.get(k))
        pd.DataFrame([row]).to_csv(self.csv_path, mode="a", header=False, index=False)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_prauc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Binary metrics: accuracy, F1, ROC-AUC, PR-AUC."""
    y_true = y_true.astype(int).reshape(-1)
    y_prob = y_prob.astype(float).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": _safe_auc(y_true, y_prob),
        "prauc": _safe_prauc(y_true, y_prob),
    }


def save_curves_npz(out_path: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Save ROC and PR curve arrays to ``.npz``."""
    y_true = y_true.astype(int).reshape(-1)
    y_prob = y_prob.astype(float).reshape(-1)
    try:
        fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    except Exception:
        fpr, tpr, roc_thr = np.array([]), np.array([]), np.array([])
    try:
        prec, rec, pr_thr = precision_recall_curve(y_true, y_prob)
    except Exception:
        prec, rec, pr_thr = np.array([]), np.array([]), np.array([])
    ensure_dir(os.path.dirname(out_path))
    np.savez(
        out_path,
        fpr=fpr,
        tpr=tpr,
        roc_thresholds=roc_thr,
        precision=prec,
        recall=rec,
        pr_thresholds=pr_thr,
    )


class EarlyStopping:
    """Stop training when the monitored validation metric does not improve."""

    def __init__(self, cfg: EarlyStoppingConfig) -> None:
        self.cfg = cfg
        self.best: Optional[float] = None
        self.best_epoch = -1
        self.num_bad = 0

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.cfg.mode == "min":
            return value < (self.best - self.cfg.min_delta)
        return value > (self.best + self.cfg.min_delta)

    def step(self, epoch: int, metrics: Dict[str, Any]) -> bool:
        """Return True if training should stop."""
        v = metrics.get(self.cfg.monitor)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            self.num_bad += 1
            return self.num_bad >= self.cfg.patience
        v = float(v)
        if self._is_better(v):
            self.best = v
            self.best_epoch = epoch
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad >= self.cfg.patience


class Trainer:
    """Epoch loop with per-center metrics, CSV logging, and checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        train_cfg: TrainConfig,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        logger: CSVLogger,
        run_name: str,
        ckpt_dir: str,
        curves_dir: str,
        centers_in_train: List[int],
        threshold: float = 0.5,
        checkpoint_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.train_cfg = train_cfg
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger
        self.run_name = run_name
        self.ckpt_dir = ckpt_dir
        self.curves_dir = curves_dir
        self.centers_in_train = sorted(list(set(int(c) for c in centers_in_train if int(c) >= 0)))
        self.threshold = float(threshold)
        self.checkpoint_extra = dict(checkpoint_extra) if checkpoint_extra else {}
        ensure_dir(self.ckpt_dir)
        ensure_dir(self.curves_dir)

    def _epoch_pass(self, loader: DataLoader, train: bool) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        losses = []
        y_true_all, y_prob_all, centers_all = [], [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            it = tqdm(loader, leave=False, desc="train" if train else "eval")
            for batch in it:
                if len(batch) == 4:
                    x, y, center, is_out = batch
                    is_out = is_out.to(DEVICE).bool()
                elif len(batch) == 3:
                    x, y, center = batch
                    is_out = torch.zeros(x.shape[0], dtype=torch.bool, device=DEVICE)
                else:
                    x, y = batch
                    center = torch.full((x.shape[0],), -1, dtype=torch.long)
                    is_out = torch.zeros(x.shape[0], dtype=torch.bool, device=DEVICE)

                # Support multi-view inputs: (B,V,C,H,W) -> flatten to (B*V,C,H,W)
                if isinstance(x, torch.Tensor) and x.ndim == 5:
                    b, v = int(x.shape[0]), int(x.shape[1])
                    x = x.reshape(b * v, *x.shape[2:])
                    y = y.view(-1).repeat_interleave(v)
                    center = center.view(-1).repeat_interleave(v)
                    is_out = is_out.view(-1).repeat_interleave(v)

                x = x.to(DEVICE)
                y = y.to(DEVICE).view(-1).float()
                center = center.to(DEVICE).view(-1).long()

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                logits = torch.zeros(x.shape[0], device=DEVICE)
                if (~is_out).any():
                    logits[~is_out] = self.model(x[~is_out]).view(-1)
                prob = torch.sigmoid(logits)
                prob = prob.clone()
                prob[is_out] = 0.0

                if train:
                    if is_out.any():
                        m = ~is_out
                        if m.any():
                            loss = self.criterion(logits[m].view(-1, 1), y[m].view(-1, 1))
                        else:
                            loss = torch.tensor(0.0, device=DEVICE)
                    else:
                        loss = self.criterion(logits.view(-1, 1), y.view(-1, 1))
                else:
                    if is_out.any():
                        m = ~is_out
                        if m.any():
                            loss = self.criterion(logits[m].view(-1, 1), y[m].view(-1, 1))
                        else:
                            loss = torch.tensor(0.0, device=DEVICE)
                    else:
                        loss = self.criterion(logits.view(-1, 1), y.view(-1, 1))

                if train:
                    loss.backward()
                    self.optimizer.step()

                losses.append(float(loss.detach().item()))
                y_true_all.append(y.detach().cpu().numpy())
                y_prob_all.append(prob.detach().cpu().numpy())
                centers_all.append(center.detach().cpu().numpy().astype(int).reshape(-1))

        y_true = np.concatenate(y_true_all, axis=0)
        y_prob = np.concatenate(y_prob_all, axis=0)
        centers = np.concatenate(centers_all, axis=0)

        metrics_global = compute_binary_metrics(y_true, y_prob, threshold=self.threshold)
        out: Dict[str, Any] = {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "accuracy": metrics_global["accuracy"],
            "f1": metrics_global["f1"],
            "auc": metrics_global["auc"],
            "prauc": metrics_global["prauc"],
        }

        for c in self.centers_in_train:
            m = centers == int(c)
            if m.sum() == 0:
                out[f"center_{c}_accuracy"] = float("nan")
                out[f"center_{c}_f1"] = float("nan")
                out[f"center_{c}_auc"] = float("nan")
                out[f"center_{c}_prauc"] = float("nan")
            else:
                cm = compute_binary_metrics(y_true[m], y_prob[m], threshold=self.threshold)
                out[f"center_{c}_accuracy"] = cm["accuracy"]
                out[f"center_{c}_f1"] = cm["f1"]
                out[f"center_{c}_auc"] = cm["auc"]
                out[f"center_{c}_prauc"] = cm["prauc"]

        return out, y_true, y_prob

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, extra: Dict[str, Any]) -> Dict[str, Any]:
        es = EarlyStopping(self.train_cfg.early_stopping)
        best_epoch = -1
        history = []

        for epoch in range(self.train_cfg.num_epochs):
            tr_metrics, _, _ = self._epoch_pass(train_loader, train=True)
            va_metrics, va_y, va_p = self._epoch_pass(val_loader, train=False)

            curves_path = os.path.join(self.curves_dir, f"val_epoch_{epoch:03d}.npz")
            save_curves_npz(curves_path, va_y, va_p)

            log_train = dict(tr_metrics)
            log_val = dict(va_metrics)
            log_train["rocpr_path"] = ""
            log_val["rocpr_path"] = curves_path

            self.logger.log(self.run_name, epoch, "train", log_train, extra)
            self.logger.log(self.run_name, epoch, "val", log_val, extra)

            ckpt_payload = {"model_state": self.model.state_dict(), **self.checkpoint_extra}
            torch.save(ckpt_payload, os.path.join(self.ckpt_dir, "last.pt"))

            monitored = {
                "val_loss": log_val.get("loss"),
                "val_accuracy": log_val.get("accuracy"),
                "val_f1": log_val.get("f1"),
                "val_auc": log_val.get("auc"),
                "val_prauc": log_val.get("prauc"),
            }

            stop = es.step(epoch, monitored)
            if es.best_epoch == epoch:
                best_epoch = epoch
                torch.save(ckpt_payload, os.path.join(self.ckpt_dir, "best.pt"))

            history.append({"epoch": epoch, "train": tr_metrics, "val": va_metrics, "monitor": monitored})
            if stop:
                break

        best_val: Optional[float] = None
        if es.best is not None:
            best_val = float(es.best)

        return {
            "best_epoch": int(best_epoch),
            "early_stopping_monitor": self.train_cfg.early_stopping.monitor,
            "best_monitor_value": best_val,
            "history": history,
        }
