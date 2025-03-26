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
from torchvision import transforms

from histopatseg.constants import CLASS_MAPPING
from histopatseg.data.dataset import LabeledTileDataset
from histopatseg.models.linear_probing import LinearProbing
from histopatseg.models.models import load_model
from histopatseg.utils import get_class_weights

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()


def get_augmentations():
    return [
        transforms.RandomResizedCrop(224,
                                     scale=(0.8,
                                            1.0)),  # random crop and resize
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2,
                               saturation=0.2,
                               hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomSolarize(threshold=128, p=0.1),
    ]


def split_data(tile_ids, metadata, random_state=42):
    """Split the data into train and validation sets."""
    tile_ids = np.array(tile_ids)
    initial_idx = np.arange(len(tile_ids))
    test_splitter = GroupShuffleSplit(n_splits=1,
                                      test_size=0.2,
                                      random_state=random_state)
    patient_ids = metadata.loc[tile_ids, "patient_id"].values
    labels = metadata.loc[tile_ids, "class_name"].values

    train_val_idx, test_idx = next(
        test_splitter.split(tile_ids, labels, groups=patient_ids))

    train_val_tile_ids = tile_ids[train_val_idx]
    train_val_idx = initial_idx[train_val_idx]

    test_tile_ids = tile_ids[test_idx]

    train_val_labels = metadata.loc[train_val_tile_ids, "class_name"].values
    train_val_patient_ids = metadata.loc[train_val_tile_ids,
                                         "patient_id"].values

    splitter = StratifiedGroupKFold(n_splits=5,
                                    shuffle=True,
                                    random_state=random_state)
    train_idx, val_idx = next(
        splitter.split(train_val_tile_ids,
                       train_val_labels,
                       groups=train_val_patient_ids))
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    train_tile_ids = tile_ids[train_idx]
    val_tile_ids = tile_ids[val_idx]

    train_patient_ids = metadata.loc[train_tile_ids, "patient_id"].values
    val_patient_ids = metadata.loc[val_tile_ids, "patient_id"].values
    test_patient_ids = metadata.loc[test_tile_ids, "patient_id"].values

    are_splits_disjoint = set(train_patient_ids).isdisjoint(
        test_patient_ids) and set(train_patient_ids).isdisjoint(
            val_patient_ids) and set(val_patient_ids).isdisjoint(
                test_patient_ids)

    if not are_splits_disjoint:
        raise ValueError("Train, validation, and test sets have no overlap.")

    logger.info(f"Train size: {len(train_tile_ids)}")
    logger.info(f"Validation size: {len(val_tile_ids)}")
    logger.info(f"Test size: {len(test_tile_ids)}")

    return train_idx, val_idx, test_idx


def get_dataloader(tile_paths,
                   metadata,
                   batch_size,
                   num_workers,
                   transform_train=None,
                   transform_eval=None,
                   random_state=42):
    """Create a DataLoader from embeddings and labels."""

    tile_paths = np.array(tile_paths)
    tile_ids = [p.stem for p in tile_paths]
    train_idx, val_idx, test_idx = split_data(tile_ids,
                                              metadata,
                                              random_state=random_state)

    train_tile_paths = tile_paths[train_idx]
    val_tile_paths = tile_paths[val_idx]
    test_tile_paths = tile_paths[test_idx]

    train_dataloader = DataLoader(
        LabeledTileDataset(train_tile_paths,
                           metadata,
                           transform=transform_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        LabeledTileDataset(val_tile_paths, metadata, transform=transform_eval),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        LabeledTileDataset(test_tile_paths, metadata,
                           transform=transform_eval),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dataloader, val_dataloader, test_dataloader


@click.command()
@click.option("--output-path",
              default=None,
              help="Path to store the weight of the linear layer.")
@click.option("--magnification",
              default=20,
              help="Magnification of the tiles.")
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
def main(output_path, magnification, num_epochs, gpu_id, batch_size, lr,
         weight_decay, num_workers):
    """Simple CLI program to greet someone"""
    # Parse magnification from the embeddings path

    data_path = Path(
        os.getenv("LUNGHIST700_PATH")) / f"LungHist700_{magnification}x"
    metadata = pd.read_csv(data_path / "metadata.csv").set_index("tile_id")

    tile_paths = list((data_path / "tiles").glob("*.png"))

    model, transform, embedding_dim = load_model("UNI2", device="cpu")
    linear_probing = LinearProbing(
        embedding_dim,
        num_classes=len(CLASS_MAPPING),
        backbone=model,
        lr=lr,
        weight_decay=weight_decay,
        class_weights=get_class_weights(metadata=metadata, class_column="class_name"),
    )

    train_loader, val_loader, test_loader = get_dataloader(
        tile_paths,
        metadata,
        batch_size,
        num_workers,
        transform_train=transforms.Compose(get_augmentations() + [transform]),
        transform_eval=transform,
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=[gpu_id],
        precision="16-mixed",
    )

    trainer.fit(linear_probing, train_loader, val_loader)
    trainer.test(linear_probing, test_loader)
    if output_path:
        trainer.save_checkpoint(output_path)


if __name__ == "__main__":
    main()
