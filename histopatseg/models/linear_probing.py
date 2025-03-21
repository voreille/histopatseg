import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics.classification import MulticlassAccuracy


class LinearProbingBase(pl.LightningModule):

    def __init__(self, feature_dim, num_classes, lr=1e-3):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.train_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average='macro',
        )
        self.val_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average='macro',
        )
        self.num_classes = num_classes
        self.save_hyperparameters()

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        accuracy = self.train_accuracy(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        accuracy = self.val_accuracy(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        accuracy = MulticlassAccuracy(num_classes=self.num_classes).to(
            inputs.device)(outputs, labels)
        return loss, accuracy

    def configure_optimizers(self):
        # Only optimize the classifier parameters
        optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr)
        return optimizer


class LinearProbingFromEmbeddings(LinearProbingBase):

    def forward(self, x):
        # Pass inputs through the frozen backbone
        return self.classifier(x)


class LinearProbing(LinearProbingBase):

    def __init__(self, backbone, feature_dim, num_classes, lr=0.001):
        super().__init__(feature_dim, num_classes, lr)
        self.backbone = backbone

    def forward(self, x):
        # Pass inputs through the frozen backbone
        with torch.no_grad():
            features = self.backbone(x)
        # Optionally, flatten or pool the features if needed; here we assume a feature vector
        return self.classifier(features)
