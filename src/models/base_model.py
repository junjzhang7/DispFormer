import logging
from typing import Any, Dict

import lightning as L
import torch
from torch import Tensor, nn
from torchmetrics.classification import (
    Accuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    F1Score,
    Precision,
    Recall,
)

logger = logging.getLogger(__name__)


class BaseModel(L.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.best_monitor_metric = 0

        if args.dataset_name == "P12":
            self.n_channels = 36
            self.n_classes = 2
        elif args.dataset_name == "P19":
            self.n_channels = 34
            self.n_classes = 2
        elif args.dataset_name == "PAM":
            self.n_channels = 17
            self.n_classes = 8

    def training_step(self, batch, batch_idx) -> Tensor | Dict[str, Any]:
        time = batch["time"]
        value = batch["value"]
        indicator = batch["indicator"]
        delta = batch["delta"]
        label = batch["label"]

        outputs = self.forward(time, delta, value, indicator)
        logits = outputs["logits"]

        loss = nn.CrossEntropyLoss()(logits, label)

        self.log("train/loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.y_true_all = []
        self.y_prob_all = []

    def validation_step(self, batch, batch_idx) -> Tensor | Dict[str, Any]:
        time = batch["time"]
        value = batch["value"]
        indicator = batch["indicator"]
        delta = batch["delta"]
        label = batch["label"]

        outputs = self.forward(time, delta, value, indicator)
        logits = outputs["logits"]
        return {"logits": logits}

    def on_validation_batch_end(
        self, outputs: Tensor | Dict[str, Any] | None, batch: Any, batch_idx: int
    ) -> None:
        label = batch["label"]
        if self.n_classes == 2:
            y_prob = torch.sigmoid(outputs["logits"])
        elif self.n_classes > 2:
            y_prob = torch.softmax(outputs["logits"], dim=-1)

        y_true = label.detach()
        y_prob = y_prob.detach()

        self.y_true_all.append(y_true)
        self.y_prob_all.append(y_prob)
        return

    def on_validation_epoch_end(self) -> None:
        y_true_all = torch.concat(self.y_true_all, axis=0)
        y_prob_all = torch.concat(self.y_prob_all, axis=0)
        self.y_true_all.clear()
        self.y_prob_all.clear()

        score_dict = self.calculate_metrics(y_prob_all, y_true_all)
        for key in score_dict.keys():
            self.log(f"val/{key}", score_dict[key])
            logger.info(f"val/{key}: {score_dict[key]}")

        monitor_metric = score_dict[self.args.monitor_metric]
        if monitor_metric > self.best_monitor_metric:
            self.best_monitor_metric = monitor_metric
            logger.info(
                f"New best {self.args.monitor_metric}: {self.best_monitor_metric:.4f} in epoch {self.trainer.current_epoch}"
            )
        self.log("val/best_monitor_metric", self.best_monitor_metric)

    def on_test_epoch_start(self) -> None:
        self.y_true_all = []
        self.y_prob_all = []

    def test_step(self, batch, batch_idx) -> Tensor | Dict[str, Any]:
        time = batch["time"]
        value = batch["value"]
        indicator = batch["indicator"]
        delta = batch["delta"]
        label = batch["label"]

        outputs = self.forward(time, delta, value, indicator)
        logits = outputs["logits"]
        return {"logits": logits}

    def on_test_batch_end(
        self, outputs: Tensor | Dict[str, Any] | None, batch: Any, batch_idx: int
    ) -> None:
        label = batch["label"]
        if self.n_classes == 2:
            y_prob = torch.sigmoid(outputs["logits"])
        elif self.n_classes > 2:
            y_prob = torch.softmax(outputs["logits"], dim=-1)

        y_true = label.detach()
        y_prob = y_prob.detach()

        self.y_true_all.append(y_true)
        self.y_prob_all.append(y_prob)
        return

    def on_test_epoch_end(self) -> None:
        y_true_all = torch.cat(self.y_true_all, dim=0)
        y_prob_all = torch.cat(self.y_prob_all, dim=0)
        self.y_true_all.clear()
        self.y_prob_all.clear()

        score_dict = self.calculate_metrics(y_prob_all, y_true_all)
        for key in score_dict.keys():
            self.log(f"test/{key}", score_dict[key])
            logger.info(f"test/{key}: {score_dict[key]}")
        return

    def calculate_metrics(self, prob, label) -> Dict[str, float]:
        prob = torch.nan_to_num(prob)

        if self.args.dataset_name in ["P12", "P19"]:
            AUROC = BinaryAUROC()(prob[:, 1], label)
            AUPRC = BinaryAveragePrecision()(prob[:, 1], label)

            return {"AUROC": AUROC * 100, "AUPRC": AUPRC * 100}

        elif self.args.dataset_name == "PAM":
            ACC = Accuracy(task="multiclass", num_classes=self.n_classes).to(
                self.device
            )(prob, label)
            precision = Precision(
                task="multiclass", num_classes=self.n_classes, average="macro"
            ).to(self.device)(prob, label)
            recall = Recall(
                task="multiclass", num_classes=self.n_classes, average="macro"
            ).to(self.device)(prob, label)
            F1 = F1Score(
                task="multiclass", num_classes=self.n_classes, average="macro"
            ).to(self.device)(prob, label)

            return {
                "Accuracy": ACC * 100,
                "Precision": precision * 100,
                "Recall": recall * 100,
                "F1": F1 * 100,
            }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        if self.args.lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.args.gamma
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return {"optimizer": optimizer}
