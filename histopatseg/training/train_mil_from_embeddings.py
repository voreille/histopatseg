import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from histopatseg.constants import CLASS_MAPPING
from histopatseg.data.dataset import EmbeddingDatasetMIL
from histopatseg.data.utils import split_data
from histopatseg.models.mil import AttentionAggregatorFromEmbeddings
from histopatseg.utils import get_class_weights

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()


def get_dataloader(embeddings,
                   tile_ids,
                   metadata,
                   batch_size,
                   num_workers,
                   class_mapping=CLASS_MAPPING,
                   random_state=42):
    """Create a DataLoader from embeddings and labels."""

    train_idx, val_idx, test_idx = split_data(tile_ids,
                                              metadata,
                                              random_state=random_state)
    train_embeddings = embeddings[train_idx]
    val_embeddings = embeddings[val_idx]
    test_embeddings = embeddings[test_idx]

    train_tile_ids = tile_ids[train_idx]
    val_tile_ids = tile_ids[val_idx]
    test_tile_ids = tile_ids[test_idx]
    collate_fn = EmbeddingDatasetMIL.get_collate_fn_ragged()

    train_dataloader = DataLoader(
        EmbeddingDatasetMIL(train_embeddings,
                            train_tile_ids,
                            metadata,
                            class_mapping=class_mapping),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        EmbeddingDatasetMIL(val_embeddings,
                            val_tile_ids,
                            metadata,
                            class_mapping=class_mapping),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        EmbeddingDatasetMIL(test_embeddings,
                            test_tile_ids,
                            metadata,
                            class_mapping=class_mapping),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader, test_dataloader


def filter_data(embeddings, tile_ids, metadata, superclass="all"):
    if superclass == "all":
        num_classes = 7
        return embeddings, tile_ids, num_classes, CLASS_MAPPING

    if superclass == "aca":
        class_mapping = {"aca_bd": 0, "aca_md": 1, "aca_pd": 2}
    elif superclass == "scc":
        class_mapping = {"scc_bd": 0, "scc_md": 1, "scc_pd": 2}

    df = pd.DataFrame(embeddings, index=tile_ids)
    tile_ids_to_keep = metadata[metadata["superclass"] == superclass].index
    df = df.loc[tile_ids_to_keep]
    return df.values, df.index.values, 3, class_mapping


def compute_class_weights(metadata, superclass, class_mapping):
    metadata_per_image = metadata.groupby("original_filename").agg({
        "class_name":
        "first",
        "superclass":
        "first"
    })

    if superclass != "all":
        metadata_per_image = metadata_per_image[
            metadata_per_image["superclass"] == superclass]

    return get_class_weights(metadata_per_image,
                             class_column="class_name",
                             class_mapping=class_mapping)


@click.command()
@click.option("--output-path",
              default=None,
              help="Path to store the weight of the linear layer.")
@click.option("--embeddings-path",
              required=True,
              help="Path to the embeddings numpy file.")
@click.option("--num-epochs", default=10, help="Number of epochs to train.")
@click.option("--gpu-id", default=0, help="Name of the model to use.")
@click.option("--batch-size", default=32, help="Batch size for inference.")
@click.option("--lr", default=0.001, show_default=True, help="LR for Adam")
@click.option("--dropout", default=0.2, show_default=True, help="LR for Adam")
@click.option("--weight-decay",
              default=0,
              type=click.FLOAT,
              show_default=True,
              help="Weight decay for Adam")
@click.option("--num-workers",
              default=0,
              help="Number of workers for dataloader.")
@click.option("--superclass",
              default="aca",
              show_default=True,
              help="Choose on which superclass to train.")
def main(output_path, embeddings_path, num_epochs, gpu_id, batch_size, lr,
         dropout, weight_decay, num_workers, superclass):
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

    embeddings, tile_ids, num_classes, class_mapping = filter_data(
        embeddings, tile_ids, metadata, superclass)

    train_loader, val_loader, test_loader = get_dataloader(
        embeddings,
        tile_ids,
        metadata,
        batch_size,
        num_workers,
        class_mapping=class_mapping,
    )
    class_weights = compute_class_weights(metadata, superclass, class_mapping)

    print(f"Training on {len(train_loader.dataset)} samples.")
    print(f"Validating on {len(val_loader.dataset)} samples.")
    print(f"Testing on {len(test_loader.dataset)} samples.")
    print(f"Class weights: {class_weights}")

    mil_model = AttentionAggregatorFromEmbeddings(
        embedding_dim,
        num_classes=num_classes,
        optimizer="Adam",
        optimizer_kwargs={
            "lr": lr,
            "weight_decay": weight_decay
        },
        # loss="CrossEntropyLoss",
        loss_kwargs={"weight": class_weights},
        dropout=dropout,
    )
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=[gpu_id],
        # log_every_n_steps=log_every_n,
        precision="16-mixed",
        check_val_every_n_epoch=10,
        logger=TensorBoardLogger("tb_logs", name="mil"),
    )
    trainer.fit(mil_model, train_loader, val_loader)
    trainer.test(mil_model, test_loader)
    if output_path:
        trainer.save_checkpoint(output_path)


if __name__ == "__main__":
    main()
