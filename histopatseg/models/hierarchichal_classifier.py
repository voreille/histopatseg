import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

from histopatseg.constants import SUPERCLASS_MAPPING


class ConditionalMultiBranchClassifier(pl.LightningModule):

    def __init__(
            self,
            feature_dim: int,
            num_superclasses: int,  # typically 3: nor, aca, scc
            num_luad_classes: int,  # typically 3: aca_bd, aca_md, aca_pd
            num_lusc_classes: int,  # typically 3: scc_bd, scc_md, scc_pd
            lr: float = 1e-3,
            superclass_weights: torch.Tensor = None,
            luad_weights: torch.Tensor = None,
            lusc_weights: torch.Tensor = None,
            weight_decay: float = 0,
            lambda_diff: float = 1.0):
        super().__init__()
        self.total_num_classes = 1 + num_luad_classes + num_lusc_classes
        # Replace the identity with your shared encoder if needed.
        self.feature_extractor = nn.Identity()

        # Superclass head (applied to every sample)
        self.super_head = nn.Linear(feature_dim, num_superclasses)
        # Differentiation heads for each cancer type:
        self.luad_head = nn.Linear(feature_dim, num_luad_classes)
        self.lusc_head = nn.Linear(feature_dim, num_lusc_classes)

        self.lusc_label = SUPERCLASS_MAPPING["scc"]
        self.luad_label = SUPERCLASS_MAPPING["aca"]
        self.normal_label = SUPERCLASS_MAPPING["nor"]

        # Loss functions for each task.
        self.loss_super = nn.CrossEntropyLoss(weight=superclass_weights)
        self.loss_luad = nn.CrossEntropyLoss(weight=luad_weights,
                                             ignore_index=-1)
        self.loss_lusc = nn.CrossEntropyLoss(weight=lusc_weights,
                                             ignore_index=-1)

        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_diff = lambda_diff  # weighting factor for the differentiation losses

        # For monitoring purposes (optional). Here we monitor the superclass accuracy.
        self.train_super_acc = MulticlassAccuracy(num_classes=num_superclasses,
                                                  average='macro')
        self.val_super_acc = MulticlassAccuracy(num_classes=num_superclasses,
                                                average='macro')

        self.save_hyperparameters()

    def forward(self, x):
        features = self.feature_extractor(x)
        super_logits = self.super_head(features)
        luad_logits = self.luad_head(features)
        lusc_logits = self.lusc_head(features)
        return super_logits, luad_logits, lusc_logits

    def training_step(self, batch, batch_idx):
        # Assume batch = (inputs, (super_labels, diff_labels))
        # diff_labels should encode the differentiation labels for cancer cases.
        inputs, super_labels, diff_labels = batch
        super_logits, luad_logits, lusc_logits = self(inputs)

        # Always compute the superclass loss on all samples.
        loss_super = self.loss_super(super_logits, super_labels)

        # Create masks based on superclass labels.

        mask_luad = (super_labels == self.luad_label)
        mask_lusc = (super_labels == self.lusc_label)

        # Compute the LUAD differentiation loss only for LUAD samples.
        if mask_luad.sum() > 0:
            loss_luad = self.loss_luad(luad_logits[mask_luad],
                                       diff_labels[mask_luad])
        else:
            loss_luad = torch.tensor(0.0, device=self.device)

        # Compute the LUSC differentiation loss only for LUSC samples.
        if mask_lusc.sum() > 0:
            loss_lusc = self.loss_lusc(lusc_logits[mask_lusc],
                                       diff_labels[mask_lusc])
        else:
            loss_lusc = torch.tensor(0.0, device=self.device)

        # Combine losses; you can weight the differentiation losses with lambda_diff.
        total_loss = loss_super + self.lambda_diff * (loss_luad + loss_lusc)

        # Optionally, log individual losses and superclass accuracy.
        self.log("train_loss", total_loss)
        self.log("train_super_loss", loss_super)
        self.log("train_luad_loss", loss_luad)
        self.log("train_lusc_loss", loss_lusc)

        # Calculate and log superclass accuracy
        super_acc = self.train_super_acc(super_logits, super_labels)
        self.log("train_super_acc", super_acc, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, super_labels, diff_labels = batch
        super_logits, luad_logits, lusc_logits = self(inputs)
        loss_super = self.loss_super(super_logits, super_labels)

        mask_luad = (super_labels == self.luad_label)
        mask_lusc = (super_labels == self.lusc_label)

        if mask_luad.sum() > 0:
            loss_luad = self.loss_luad(luad_logits[mask_luad],
                                       diff_labels[mask_luad])
        else:
            loss_luad = torch.tensor(0.0, device=self.device)
        if mask_lusc.sum() > 0:
            loss_lusc = self.loss_lusc(lusc_logits[mask_lusc],
                                       diff_labels[mask_lusc])
        else:
            loss_lusc = torch.tensor(0.0, device=self.device)

        total_loss = loss_super + self.lambda_diff * (loss_luad + loss_lusc)

        # Log validation losses
        self.log("val_loss", total_loss, prog_bar=True)

        super_acc = self.val_super_acc(super_logits, super_labels)
        self.log("val_super_acc", super_acc, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.lr,
                                weight_decay=self.weight_decay)
