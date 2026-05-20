import argparse
import logging
import os
from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger

import wandb
from src.datasets.data_utils import build_dataloaders
from src.models.model_utils import build_model
from src.utils import print_args, set_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument(
        "--dataset_name",
        default="P12",
        choices=["P12", "P19", "PAM"],
        type=str,
    )
    parser.add_argument("--split", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--missing_ratio", default=0, type=float)
    parser.add_argument("--remove_style", default="random", type=str)
    # model
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--d_model", default=64, type=int)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ff_expand_factor", default=4, type=int)
    parser.add_argument("--qkv_bias", type=int, default=1)
    parser.add_argument("--norm_first", type=int, default=1)
    parser.add_argument("--ib", default=0, type=int)
    parser.add_argument("--vanilla_dual_att", default=0, type=int)
    parser.add_argument(
        "--distribute_style", default="add", choices=["add", "concat", "gate"], type=str
    )
    # hyperparameter
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--lr_scheduler", default=1, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--w_kl", type=float, default=1)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    # P12 P19: AUPRC, PAM: Accuracy
    parser.add_argument(
        "--monitor_metric",
        default="AUPRC",
        type=str,
        choices=["AUPRC", "AUROC", "Accuracy"],
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--version", type=str)
    parser.add_argument("--dev", type=int, default=0)
    parser.add_argument("--log_exp", type=int, default=0)
    args = parser.parse_args()
    return args


def experiment(args):
    log_dir = f"run_logs/{args.model_name}/{args.dataset_name}/ver-{args.version}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    mlf_logger = (
        MLFlowLogger(
            experiment_name=f"{args.model_name}_{args.dataset_name}",
            tracking_uri="Your MLFlow server URI...",
            tags={"project_name": "DispFormer"},
        )
        if args.log_exp
        else None
    )
    logger = set_logger(log_dir, saving=not bool(args.dev), mlflow_logger=mlf_logger)
    print_args(args)
    seed = L.seed_everything(args.seed)
    logger.info(f"Current PID: {os.getpid()}")
    logger.info(f"Global seed set to: {seed}")
    logger.info(f"CWD:{os.getcwd()}")

    # load data
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        split=args.split,
        dev=bool(args.dev),
    )

    model = build_model(args)

    callbacks = []
    ckp_callback = ModelCheckpoint(
        dirpath=f"{log_dir}/checkpoint",
        monitor=f"val/{args.monitor_metric}",
        mode="max",
    )
    early_stop = EarlyStopping(
        monitor=f"val/{args.monitor_metric}", mode="max", patience=10
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [ckp_callback, early_stop, lr_monitor]

    trainer = L.Trainer(
        default_root_dir=log_dir,
        callbacks=callbacks,
        devices=[args.device],
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        logger=mlf_logger,
        enable_progress_bar=bool(args.dev),
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
