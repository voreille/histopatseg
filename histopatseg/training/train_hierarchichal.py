import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader

from histopatseg.data.dataset import HierarchicalEmbeddingDataset
from histopatseg.models.hierarchichal_classifier import ConditionalMultiBranchClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()


def split_data(embeddings, tile_ids, metadata, random_state=42):
    """Split the data into train and validation sets."""
    test_splitter = GroupShuffleSplit(n_splits=1,
                                      test_size=0.2,
                                      random_state=random_state)
    patient_ids = metadata.loc[tile_ids, "patient_id"].values
    labels = metadata.loc[tile_ids, "class_name"].values

    train_val_idx, test_idx = next(
        test_splitter.split(embeddings, labels, groups=patient_ids))

    train_val_embeddings = embeddings[train_val_idx]
    train_val_tile_ids = tile_ids[train_val_idx]

    test_embeddings = embeddings[test_idx]
    test_tile_ids = tile_ids[test_idx]

    train_val_labels = metadata.loc[train_val_tile_ids, "class_name"].values
    train_val_patient_ids = metadata.loc[train_val_tile_ids,
                                         "patient_id"].values

    splitter = StratifiedGroupKFold(n_splits=5,
                                    shuffle=True,
                                    random_state=random_state)
    train_idx, val_idx = next(
        splitter.split(train_val_embeddings,
                       train_val_labels,
                       groups=train_val_patient_ids))

    train_embeddings = train_val_embeddings[train_idx]
    train_tile_ids = train_val_tile_ids[train_idx]

    val_embeddings = train_val_embeddings[val_idx]
    val_tile_ids = train_val_tile_ids[val_idx]

    train_patient_ids = metadata.loc[train_tile_ids, "patient_id"].values
    val_patient_ids = metadata.loc[val_tile_ids, "patient_id"].values
    test_patient_ids = metadata.loc[test_tile_ids, "patient_id"].values

    are_splits_disjoint = set(train_patient_ids).isdisjoint(
        test_patient_ids) and set(train_patient_ids).isdisjoint(
            val_patient_ids) and set(val_patient_ids).isdisjoint(
                test_patient_ids)

    if not are_splits_disjoint:
        raise ValueError("Train, validation, and test sets have no overlap.")

    logger.info(f"Train size: {len(train_embeddings)}")
    logger.info(f"Validation size: {len(val_embeddings)}")
    logger.info(f"Test size: {len(test_embeddings)}")

    return (train_embeddings, train_tile_ids, val_embeddings, val_tile_ids,
            test_embeddings, test_tile_ids)


def get_dataloader(embeddings,
                   tile_ids,
                   metadata,
                   batch_size,
                   num_workers,
                   random_state=42):
    """Create a DataLoader from embeddings and labels."""
    (train_embeddings, train_tile_ids, val_embeddings, val_tile_ids,
     test_embeddings, test_tile_ids) = split_data(embeddings,
                                                  tile_ids,
                                                  metadata,
                                                  random_state=random_state)

    train_dataloader = DataLoader(HierarchicalEmbeddingDataset(
        train_embeddings, train_tile_ids, metadata),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(HierarchicalEmbeddingDataset(
        val_embeddings, val_tile_ids, metadata),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    test_dataloader = DataLoader(HierarchicalEmbeddingDataset(
        test_embeddings, test_tile_ids, metadata),
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
def main(output_path, embeddings_path, num_epochs, gpu_id, batch_size, lr,
         weight_decay, num_workers):
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

    train_loader, val_loader, test_loader = get_dataloader(
        embeddings,
        tile_ids,
        metadata,
        batch_size,
        num_workers,
    )

    linear_probing = ConditionalMultiBranchClassifier(
        embedding_dim,
        num_superclasses=3,
        num_luad_classes=3,
        num_lusc_classes=3,
        lr=lr,
        weight_decay=weight_decay,
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
    if output_path:
        trainer.save_checkpoint(output_path)


if __name__ == "__main__":
    main()
