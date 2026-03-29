"""Orchestration: outlier filtering, DINO precompute, training, optional test prediction."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.common import ensure_dir, safe_json, seed_everything
from utils.config import ModuleSpec, RunConfig, save_run_config
from utils.data import (
    CenterProportionalBatchSampler,
    EmbeddingTensorDataset,
    EmbeddingValDataset,
    H5BinaryDataset,
    build_preprocessing,
)
from utils.constants import DEVICE, RUNS_DIR, TEST_IMAGES_PATH, TRAIN_IMAGES_PATH, VAL_IMAGES_PATH
from utils.model import (
    FullModel,
    HeadOnly,
    build_binary_head,
    load_frozen_backbone,
    trainable_parameters,
)
from utils.outliers import ExtractOutlier
from utils.predict import predict_test
from utils.training import CSVLogger, Trainer


def _load_centers_from_h5(h5_path: str, image_ids: List[str]) -> List[int]:
    """Load center ids from H5 ``metadata[0]`` for each patch key."""
    centers = []
    with h5py.File(h5_path, "r") as f:
        for k in image_ids:
            g = f[k]
            if "metadata" not in g:
                centers.append(-1)
                continue
            meta = np.array(g.get("metadata")).reshape(-1)
            centers.append(int(meta[0]) if meta.size else -1)
    return centers


def _estimate_center_proportions(centers: List[int]) -> Dict[int, float]:
    """Empirical center proportions (non-negative centers only)."""
    c = np.array([int(x) for x in centers if int(x) >= 0], dtype=int)
    if c.size == 0:
        return {}
    uniq, cnt = np.unique(c, return_counts=True)
    p = cnt / cnt.sum()
    return {int(u): float(v) for u, v in zip(uniq, p)}


def _subsample_id_list(ids: List[str], fraction: float, rng: np.random.Generator) -> List[str]:
    """Random subset of ``ids`` of size ``max(1, int(len * fraction))`` (smoke tests)."""
    if fraction >= 1.0 or len(ids) == 0:
        return ids
    n = max(1, int(len(ids) * fraction))
    n = min(n, len(ids))
    pick = rng.choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(pick)]


def precompute_backbone_features(
    loader: DataLoader,
    backbone: nn.Module,
    device: torch.device,
    desc: str = "precompute_embeddings",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward per batch; skip backbone on outlier rows (``is_out`` True). Returns CPU tensors."""
    backbone.eval()
    zs, ys, cs, ios = [], [], [], []
    nf = int(backbone.num_features)
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc=desc):
            if len(batch) == 4:
                x, y, c, io = batch
            else:
                x, y, c = batch
                io = torch.zeros(x.shape[0], dtype=torch.bool)
            x = x.to(device)
            io = io.to(device).bool()
            z = torch.zeros(x.shape[0], nf, device=device, dtype=torch.float32)
            if (~io).any():
                z[~io] = backbone(x[~io])
            zs.append(z.cpu())
            ys.append(y.float())
            cs.append(c.long())
            ios.append(io.cpu())
    return torch.cat(zs, 0), torch.cat(ys, 0), torch.cat(cs, 0), torch.cat(ios, 0)


def run_experiment(run_cfg: RunConfig) -> Dict[str, Any]:
    """Run one experiment (possibly multiple seeds): config JSON, metrics, checkpoints, optional test preds."""
    base_run_dir = os.path.join(RUNS_DIR, run_cfg.run_name)
    ensure_dir(base_run_dir)
    save_run_config(run_cfg, base_run_dir)

    transform = build_preprocessing(run_cfg.processing)
    seed_results = []

    for seed in run_cfg.seeds:
        seed_everything(int(seed))
        rng = np.random.default_rng(int(seed) + 42)

        run_dir = os.path.join(base_run_dir, f"seed_{int(seed)}")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        curves_dir = os.path.join(run_dir, "curves")
        metrics_path = os.path.join(run_dir, "metrics.csv")

        ensure_dir(run_dir)
        ensure_dir(ckpt_dir)
        ensure_dir(curves_dir)

        train_probe = H5BinaryDataset(TRAIN_IMAGES_PATH, transform=transform, mode="train")
        val_probe = H5BinaryDataset(VAL_IMAGES_PATH, transform=transform, mode="val")
        train_ids = list(train_probe.image_ids)
        val_ids = list(val_probe.image_ids)

        val_outlier_ids: Optional[Set[str]] = None
        if run_cfg.outlier_params is not None:
            eo = ExtractOutlier(run_cfg.outlier_params)
            train_scan = eo.scan(
                TRAIN_IMAGES_PATH,
                train_ids,
                batch_size=run_cfg.outlier_scan_batch_size,
                num_workers=run_cfg.outlier_scan_workers,
                desc=f"outliers train seed={seed}",
            )
            train_out_set = {k for k, v in train_scan.items() if v}
            train_ids = [k for k in train_ids if k not in train_out_set]

            val_scan = eo.scan(
                VAL_IMAGES_PATH,
                val_ids,
                batch_size=run_cfg.outlier_scan_batch_size,
                num_workers=run_cfg.outlier_scan_workers,
                desc=f"outliers val seed={seed}",
            )
            val_outlier_ids = {k for k, v in val_scan.items() if v}

        if run_cfg.data_fraction is not None:
            frac = float(run_cfg.data_fraction)
            if not 0 < frac <= 1:
                raise ValueError("data_fraction must be in (0, 1] or None")
            train_ids = _subsample_id_list(train_ids, frac, rng)
            val_ids = _subsample_id_list(val_ids, frac, rng)

        if len(train_ids) == 0:
            raise RuntimeError("No training patches left after outlier filter / subsampling.")

        train_ds = H5BinaryDataset(TRAIN_IMAGES_PATH, transform=transform, mode="train", subset_ids=train_ids)
        val_ds = H5BinaryDataset(
            VAL_IMAGES_PATH,
            transform=transform,
            mode="val",
            outlier_ids=val_outlier_ids,
            subset_ids=val_ids,
        )
        train_centers = _load_centers_from_h5(TRAIN_IMAGES_PATH, train_ids)
        centers_in_train = sorted(list(set(int(c) for c in train_centers if int(c) >= 0)))

        use_precompute = not run_cfg.model.adapter.enabled

        val_loader_img = DataLoader(
            val_ds,
            batch_size=run_cfg.train.batch_size,
            shuffle=False,
            num_workers=run_cfg.train.num_workers,
            drop_last=False,
        )

        ckpt_extra: Dict[str, Any] = {}

        if use_precompute:
            train_loader_pre = DataLoader(
                train_ds,
                batch_size=run_cfg.train.batch_size,
                shuffle=False,
                num_workers=run_cfg.train.num_workers,
                drop_last=False,
            )
            backbone_enc = load_frozen_backbone(run_cfg.model.backbone_name)
            z_tr, y_tr, c_tr, _ = precompute_backbone_features(
                train_loader_pre,
                backbone_enc,
                DEVICE,
                desc=f"precompute train seed={seed}",
            )
            z_va, y_va, c_va, io_va = precompute_backbone_features(
                val_loader_img,
                backbone_enc,
                DEVICE,
                desc=f"precompute val seed={seed}",
            )
            del backbone_enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_emb = EmbeddingTensorDataset(z_tr, y_tr, c_tr)
            val_emb = EmbeddingValDataset(z_va, y_va, c_va, io_va)

            if run_cfg.train.use_center_balanced_batches:
                proportions = run_cfg.train.center_proportions
                if proportions is None:
                    proportions = _estimate_center_proportions(train_centers)
                batch_sampler = CenterProportionalBatchSampler(
                    centers=train_centers,
                    batch_size=run_cfg.train.batch_size,
                    proportions=proportions,
                    shuffle=True,
                    drop_last=run_cfg.train.drop_last,
                    seed=int(seed),
                )
                train_loader = DataLoader(train_emb, batch_sampler=batch_sampler, num_workers=0)
            else:
                train_loader = DataLoader(
                    train_emb,
                    batch_size=run_cfg.train.batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=run_cfg.train.drop_last,
                )
            val_loader = DataLoader(
                val_emb,
                batch_size=run_cfg.train.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

            backbone = load_frozen_backbone(run_cfg.model.backbone_name)
            in_dim = int(backbone.num_features)
            del backbone
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            head = build_binary_head(run_cfg.model, in_dim=in_dim)
            model = HeadOnly(head).to(DEVICE)
            ckpt_extra = {"head_only": True, "backbone_name": run_cfg.model.backbone_name}
        else:
            if run_cfg.train.use_center_balanced_batches:
                proportions = run_cfg.train.center_proportions
                if proportions is None:
                    proportions = _estimate_center_proportions(train_centers)
                batch_sampler = CenterProportionalBatchSampler(
                    centers=train_centers,
                    batch_size=run_cfg.train.batch_size,
                    proportions=proportions,
                    shuffle=True,
                    drop_last=run_cfg.train.drop_last,
                    seed=int(seed),
                )
                train_loader_img = DataLoader(
                    train_ds,
                    batch_sampler=batch_sampler,
                    num_workers=run_cfg.train.num_workers,
                )
            else:
                train_loader_img = DataLoader(
                    train_ds,
                    batch_size=run_cfg.train.batch_size,
                    shuffle=True,
                    num_workers=run_cfg.train.num_workers,
                    drop_last=run_cfg.train.drop_last,
                )
            train_loader = train_loader_img
            val_loader = val_loader_img
            backbone = load_frozen_backbone(run_cfg.model.backbone_name)
            model = FullModel(backbone, run_cfg.model).to(DEVICE)
            ckpt_extra = {"head_only": False}

        params = trainable_parameters(model)
        if len(params) == 0:
            raise RuntimeError("No trainable parameters found. Make sure adapter/head are enabled.")

        optimizer = optim.Adam(params, lr=run_cfg.train.lr, weight_decay=run_cfg.train.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        extra = {
            "seed": int(seed),
            "outlier_filter": run_cfg.outlier_params is not None,
            "data_fraction": run_cfg.data_fraction,
            "precomputed_embeddings": use_precompute,
            "resize_hw": f"{run_cfg.processing.resize_hw[0]}x{run_cfg.processing.resize_hw[1]}",
            "adapter_enabled": bool(run_cfg.model.adapter.enabled),
            "adapter_cls": safe_json(run_cfg.model.adapter.module_cls),
            "head_cls": safe_json(run_cfg.model.head.module_cls),
            "backbone": run_cfg.model.backbone_name,
            "use_center_balanced_batches": bool(run_cfg.train.use_center_balanced_batches),
            "center_proportions": safe_json(run_cfg.train.center_proportions),
            "early_stopping_monitor": run_cfg.train.early_stopping.monitor,
            "early_stopping_mode": run_cfg.train.early_stopping.mode,
            "early_stopping_patience": run_cfg.train.early_stopping.patience,
        }

        metric_fields = ["loss", "accuracy", "f1", "auc", "prauc", "rocpr_path"]
        for c in centers_in_train:
            metric_fields += [
                f"center_{c}_accuracy",
                f"center_{c}_f1",
                f"center_{c}_auc",
                f"center_{c}_prauc",
            ]

        logger = CSVLogger(metrics_path, extra_fieldnames=list(extra.keys()), metric_fieldnames=metric_fields)

        trainer = Trainer(
            model=model,
            train_cfg=run_cfg.train,
            optimizer=optimizer,
            criterion=criterion,
            logger=logger,
            run_name=run_cfg.run_name,
            ckpt_dir=ckpt_dir,
            curves_dir=curves_dir,
            centers_in_train=centers_in_train,
            threshold=run_cfg.predict_threshold,
            checkpoint_extra=ckpt_extra,
        )

        t0 = time.time()
        fit = trainer.fit(train_loader, val_loader, extra=extra)
        t1 = time.time()

        fit["seed"] = int(seed)
        fit["time_sec"] = float(t1 - t0)
        seed_results.append(fit)

        if run_cfg.do_predict_test:
            ckpt = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu")
            if ckpt.get("head_only"):
                bb = load_frozen_backbone(ckpt["backbone_name"])
                infer_cfg = replace(
                    run_cfg.model,
                    adapter=ModuleSpec(enabled=False, module_cls=None, module_kwargs=None),
                )
                infer_model = FullModel(bb, infer_cfg).to(DEVICE)
                sd = ckpt["model_state"]
                head_sd = {k.replace("head.", "", 1): v for k, v in sd.items() if k.startswith("head.")}
                infer_model.head.load_state_dict(head_sd)
            else:
                infer_model = model
                infer_model.load_state_dict(ckpt["model_state"])
            with h5py.File(TEST_IMAGES_PATH, "r") as _tf:
                test_all_ids = list(_tf.keys())
            test_run_ids = test_all_ids
            if run_cfg.data_fraction is not None:
                test_run_ids = _subsample_id_list(test_all_ids, float(run_cfg.data_fraction), rng)
            test_outlier_ids: Optional[Set[str]] = None
            if run_cfg.outlier_params is not None:
                eo_t = ExtractOutlier(run_cfg.outlier_params)
                test_scan = eo_t.scan(
                    TEST_IMAGES_PATH,
                    test_run_ids,
                    batch_size=run_cfg.outlier_scan_batch_size,
                    num_workers=run_cfg.outlier_scan_workers,
                    desc=f"outliers test seed={seed}",
                )
                test_outlier_ids = {k for k, v in test_scan.items() if v}
            test_subset = test_run_ids if run_cfg.data_fraction is not None else None
            predict_test(
                run_dir,
                infer_model,
                transform,
                threshold=run_cfg.predict_threshold,
                test_outlier_ids=test_outlier_ids,
                subset_ids=test_subset,
                num_workers=run_cfg.train.num_workers,
            )

    return {
        "run_name": run_cfg.run_name,
        "seeds": list(run_cfg.seeds),
        "seed_results": seed_results,
    }
