import logging

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW, RMSprop
from torchmetrics.classification import MulticlassAccuracy, MultilabelAccuracy

# from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_loss_function(loss_name, **kwargs):
    loss_dict = {
        "BCEWithLogitsLoss": BCEWithLogitsLoss,
        "CrossEntropyLoss": CrossEntropyLoss,
    }

    loss_class = loss_dict.get(loss_name)

    if loss_class is None:
        raise ValueError(f"Loss function '{loss_name}' is not supported.\n"
                         f"The available losses are: {list(loss_dict.keys())}")

    logging.info(f"Using loss function: {loss_name} with arguments: {kwargs}")

    # Check if 'weight' is in kwargs and convert it to a tensor
    if 'weight' in kwargs and not isinstance(kwargs['weight'], torch.Tensor):
        kwargs['weight'] = torch.tensor(
            kwargs['weight'],
            dtype=torch.float,
        )

    return loss_class(**kwargs)


def get_metric(
    num_classes,
    threshold=0.5,
    task="multilabel",
    average="macro",
    multidim_average="global",
    top_k=1,
):
    if task == "multilabel":
        return MultilabelAccuracy(
            num_labels=num_classes,
            threshold=threshold,
            average=average,
            multidim_average=multidim_average,
        )

    return MulticlassAccuracy(
        num_classes=num_classes,
        average=average,
        multidim_average=multidim_average,
        top_k=top_k,
    )


def get_optimizer(parameters, optimizer_name, **kwargs):
    """
    Factory function to create an optimizer.

    Args:
        name (str): Name of the optimizer (e.g., "adam", "sgd").
        parameters: Model's parameters to optimize.
        **kwargs: Additional arguments for the optimizer.

    Returns:
        torch.optim.Optimizer: The instantiated optimizer.
    """
    optimizer_dict = {
        "Adam": Adam,
        "AdamW": AdamW,
        "SGD": SGD,
        "RMSprop": RMSprop
    }

    logging.info(f"== Optimizer: {optimizer_name} ==")

    optimizer_class = optimizer_dict.get(optimizer_name)

    if optimizer_class is None:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported")

    return optimizer_class(parameters, **kwargs)


def get_scheduler(optimizer, name, **kwargs):
    """
    Factory function to create a learning rate scheduler.

    Args:
        name (str): Name of the scheduler (e.g., "StepLR", "CosineAnnealingLR").
        optimizer: Optimizer to attach the scheduler to.
        **kwargs: Additional arguments for the scheduler.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or dict: The instantiated scheduler.
    """

    schedulers_dict = {
        "StepLR": torch.optim.lr_scheduler.StepLR,
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }

    scheduler_class = schedulers_dict.get(name)
    if scheduler_class is None:
        # raise ValueError(f"Unsupported scheduler: {name}")
        return None

    if name == "ReduceLROnPlateau":
        return {
            "scheduler": scheduler_class(optimizer, **kwargs),
            "monitor": kwargs.get("monitor", "val_loss"),
            "interval": kwargs.get("interval", "epoch"),
            "frequency": kwargs.get("frequency", 1),
        }

    return scheduler_class(optimizer, **kwargs)


class BaseAttentionAggregatorPL(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        feature_extractor=None,
        attention_dim=128,
        hidden_dim=128,
        attention_branches=1,
        num_classes=3,
        dropout=0.2,
        optimizer="adam",
        optimizer_kwargs=None,
        scheduler=None,
        scheduler_kwargs=None,
        loss="BCEWithLogitsLoss",
        loss_kwargs=None,
    ):
        super().__init__()  # Corrected initialization
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.attention_branches = attention_branches
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor

        self.feature_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.attention_tanh = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),  # matrix V
            nn.Tanh(),
        )

        self.attention_sigmoid = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),  # matrix U
            nn.Sigmoid(),
        )

        self.attention_weights = nn.Linear(
            self.attention_dim, self.attention_branches
        )  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_dim * self.attention_branches,
                      self.num_classes),
        )

        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.scheduler_name = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.loss_fn = get_loss_function(loss, **(loss_kwargs or {}))

        # Metrics
        self.train_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average='macro',
        )
        self.val_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average='macro',
        )

        self.save_hyperparameters(ignore=["feature_extractor"])

    def forward(self, x):
        raise NotImplementedError("This is an abstract class.")

    def forward_attention(self, x):
        """
        x: (num_patches, input_dim) - Each WSI has a variable number of patches
        """
        x = self.feature_projection(x)  # (num_patches, hidden_dim)

        # Compute Gated Attention Scores
        attention_tanh = self.attention_tanh(x)  # (num_patches, attention_dim)
        attention_sigmoid = self.attention_sigmoid(
            x)  # (num_patches, attention_dim)
        attention_scores = self.attention_weights(
            attention_tanh *
            attention_sigmoid)  # (num_patches, attention_heads)

        # Normalize attention scores
        attention_scores = torch.transpose(attention_scores, 1,
                                           0)  # (attention_heads, num_patches)
        attention_scores = F.softmax(attention_scores,
                                     dim=1)  # Normalize over patches

        # Aggregate patch embeddings using attention
        aggregated_features = torch.mm(attention_scores,
                                       x)  # (attention_heads, hidden_dim)

        # Classification
        prediction = self.classifier(aggregated_features)  # (num_classe,)

        return prediction.squeeze(), attention_scores

    def predict_step(self, batch, batch_idx):
        wsi_ids, embeddings, labels = batch
        logits = []

        for embedding in embeddings:
            output, _ = self(embedding)
            logits.append(output)

        logits = torch.stack(logits)

        preds = logits.argmax(dim=-1)

        probs = torch.sigmoid(logits)  # Shape: (batch_size, 3)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "labels": labels,
            "wsi_ids": wsi_ids,
        }

    def step(self, batch):
        raise NotImplementedError("This is an abstract class.")

    def training_step(self, batch, batch_idx):
        loss, preds, labels, batch_size = self.step(batch)
        self.train_accuracy(preds, labels)

        # Log metrics
        self.log("train_loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=batch_size,
                 prog_bar=True)
        self.log("train_acc",
                 self.train_accuracy.compute(),
                 on_step=True,
                 on_epoch=True,
                 batch_size=batch_size,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, preds, labels, batch_size = self.step(batch)
        self.val_accuracy(preds, labels)
        accuracy = self.val_accuracy.compute()

        # Log metrics
        self.log("val_loss",
                 val_loss,
                 on_epoch=True,
                 batch_size=batch_size,
                 prog_bar=True)
        self.log("val_acc",
                 accuracy,
                 on_epoch=True,
                 batch_size=batch_size,
                 prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels, batch_size = self.step(batch)
        # Log test metrics
        self.log("test_loss", loss, batch_size=batch_size)
        self.log(
            "test_acc",
            MulticlassAccuracy(
                num_classes=self.num_classes,
                average='macro',
            )(preds, labels).compute(),
            batch_size=batch_size,
        )

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.optimizer_name,
                                  **self.optimizer_kwargs)
        scheduler = get_scheduler(optimizer, self.scheduler_name,
                                  **self.scheduler_kwargs)

        if scheduler is None:
            return optimizer

        if isinstance(scheduler, dict):
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return [optimizer], [scheduler]


class AttentionAggregatorFromEmbeddings(BaseAttentionAggregatorPL):

    def forward(self, x):
        return self.forward_attention(x)

    def step(self, batch):
        _, embeddings, labels = batch
        batch_outputs = []
        batch_size = len(labels)

        for embedding in embeddings:
            outputs, _ = self(embedding)
            batch_outputs.append(outputs)

        batch_outputs = torch.stack(batch_outputs)

        if self.loss_fn.__class__.__name__ == "BCEWithLogitsLoss":
            labels_one_hot = F.one_hot(labels,
                                       num_classes=self.num_classes).float()
            loss = self.loss_fn(batch_outputs, labels_one_hot)
        else:
            loss = self.loss_fn(batch_outputs, labels)

        preds = batch_outputs.argmax(dim=-1)

        return loss, preds, labels, batch_size


class AttentionAggregator(BaseAttentionAggregatorPL):

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        return self.forward_attention(x)

    def step(self, batch):
        _, tiles_set, labels = batch
        batch_outputs = []
        batch_size = len(labels)

        for tiles in tiles_set:
            outputs, _ = self(tiles)
            batch_outputs.append(outputs)

        batch_outputs = torch.stack(batch_outputs)

        if self.loss_fn.__class__.__name__ == "BCEWithLogitsLoss":
            labels_one_hot = F.one_hot(labels,
                                       num_classes=self.num_classes).float()
            loss = self.loss_fn(batch_outputs, labels_one_hot)
        else:
            loss = self.loss_fn(batch_outputs, labels)

        preds = batch_outputs.argmax(dim=-1)

        return loss, preds, labels, batch_size
