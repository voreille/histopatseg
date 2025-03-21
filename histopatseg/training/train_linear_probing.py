import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from histopatseg.constants import CLASS_MAPPING, SUPERCLASS_MAPPING
from histopatseg.data.dataset import EmbeddingDataset
from histopatseg.data.utils import split_data
from histopatseg.models.linear_probing import LinearProbingFromEmbeddings
from histopatseg.utils import get_class_weights

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()


def get_dataloader(embeddings,
                   labels,
                   patient_ids,
                   batch_size,
                   num_workers,
                   random_state=42):
    """Create a DataLoader from embeddings and labels."""

    (train_embeddings, train_labels, _, val_embeddings, val_labels, _,
     test_embeddings, test_labels, _) = split_data(embeddings,
                                                   labels,
                                                   patient_ids,
                                                   random_state=random_state)

    train_dataloader = DataLoader(EmbeddingDataset(train_embeddings,
                                                   train_labels),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(EmbeddingDataset(val_embeddings, val_labels),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    test_dataloader = DataLoader(EmbeddingDataset(test_embeddings,
                                                  test_labels),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataloader


@click.command()
@click.option("--output-path",
              default=None,
              help="Path to store the weight of the linear layer.")
@click.option("--embeddings-path",
              required=True,
              help="Path to the embeddings numpy file.")
@click.option("--num-epochs", default=10, help="Number of epochs to train.")
@click.option("--gpu-id", default=0, help="Name of the model to use.")
@click.option("--batch-size", default=256, help="Batch size for inference.")
@click.option("--lr", default=0.001, show_default=True, help="LR for Adam")
@click.option("--weight-decay",
              default=0,
              type=click.FLOAT,
              show_default=True,
              help="Weight decay for Adam")
@click.option("--num-workers",
              default=0,
              help="Number of workers for dataloader.")
@click.option("--task",
              default="class",
              show_default=True,
              help="superclass or class.")
def main(output_path, embeddings_path, num_epochs, gpu_id, batch_size, lr,
         weight_decay, num_workers, task):
    """Simple CLI program to greet someone"""
    embeddings_path = Path(embeddings_path).resolve()
    # Parse magnification from the embeddings path
    magnification = int(embeddings_path.stem.split("_")[-1].replace("x", ""))

    data_path = Path(os.getenv("LUNGHIST700_PATH"))
    metadata = pd.read_csv(
        data_path /
        f"LungHist700_{magnification}x/metadata.csv").set_index("tile_id")

    with np.load(embeddings_path) as data:
        embeddings = data["embeddings"]
        tile_ids = data["tile_ids"]
        embedding_dim = data["embedding_dim"]

    if task == "superclass":
        labels = metadata.loc[tile_ids,
                              "superclass"].map(SUPERCLASS_MAPPING).values
        num_classes = len(SUPERCLASS_MAPPING)
    elif task == "class":
        labels = metadata.loc[tile_ids, "class_name"].map(CLASS_MAPPING).values
        num_classes = len(CLASS_MAPPING)

    patient_ids = metadata.loc[tile_ids, "patient_id"].values

    train_loader, val_loader, test_loader = get_dataloader(
        embeddings,
        labels,
        patient_ids,
        batch_size,
        num_workers,
    )

    linear_probing = LinearProbingFromEmbeddings(
        embedding_dim,
        num_classes,
        lr=lr,
        weight_decay=weight_decay,
        class_weights=get_class_weights(magnification),
    )
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=[gpu_id],
        # log_every_n_steps=log_every_n,
        precision="16-mixed",
        # check_val_every_n_epoch=10,
    )
    trainer.fit(linear_probing, train_loader, val_loader)
    trainer.test(linear_probing, test_loader)
    if output_path:
        trainer.save_checkpoint(output_path)


if __name__ == "__main__":
    main()
