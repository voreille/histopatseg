import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from lightning.pytorch.loggers import TensorBoardLogger
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from histopatseg.constants import CLASS_MAPPING
from histopatseg.data.dataset import TileDatasetMIL
from histopatseg.data.utils import split_data_per_image
from histopatseg.models.mil import AttentionAggregator
from histopatseg.models.models import load_model
from histopatseg.utils import find_tile_path, get_class_weights

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


def get_dataloader(metadata,
                   batch_size=32,
                   num_workers=0,
                   transform_train=None,
                   transform_eval=None,
                   class_mapping=CLASS_MAPPING,
                   random_state=42,
                   cache_data=False):
    """Create a DataLoader from embeddings and labels."""

    metadata_per_image = metadata.groupby("original_filename").agg(
        {key: "first"
         for key in metadata.columns})

    train_image_ids, val_image_ids, test_image_ids = split_data_per_image(
        metadata_per_image, random_state=random_state)

    collate_fn = TileDatasetMIL.get_collate_fn_ragged()

    train_dataloader = DataLoader(
        TileDatasetMIL(train_image_ids,
                       metadata,
                       transform=transform_train,
                       class_mapping=class_mapping,
                       cache_data=cache_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_dataloader = DataLoader(
        TileDatasetMIL(val_image_ids,
                       metadata,
                       transform=transform_eval,
                       class_mapping=class_mapping,
                       cache_data=cache_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_dataloader = DataLoader(
        TileDatasetMIL(test_image_ids,
                       metadata,
                       transform=transform_eval,
                       class_mapping=class_mapping,
                       cache_data=cache_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader, test_dataloader


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


def get_class_mapping(superclass):
    if superclass == "aca":
        return {"aca_bd": 0, "aca_md": 1, "aca_pd": 2}
    elif superclass == "scc":
        return {"scc_bd": 0, "scc_md": 1, "scc_pd": 2}
    return CLASS_MAPPING


@click.command()
@click.option("--output-path",
              default=None,
              help="Path to store the weight of the linear layer.")
@click.option("--model-name",
              default="UNI2",
              show_default=True,
              help="Name of the model to use for feature extraction.")
@click.option("--magnification",
              default=20,
              show_default=True,
              help="Magnification of the tiles.")
@click.option("--num-epochs",
              default=10,
              show_default=True,
              help="Number of epochs to train.")
@click.option("--gpu-id", default=0, help="Name of the model to use.")
@click.option("--batch-size",
              default=32,
              show_default=True,
              help="Batch size for inference.")
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
def main(output_path, model_name, magnification, num_epochs, gpu_id,
         batch_size, lr, dropout, weight_decay, num_workers, superclass):
    # Parse magnification from the embeddings path

    data_path = Path(
        os.getenv("LUNGHIST700_PATH")) / f"LungHist700_{magnification}x/"
    metadata = pd.read_csv(data_path / "metadata.csv").set_index("tile_id")
    tiles_dir = data_path / "tiles"

    if superclass != "all":
        metadata = metadata[metadata["superclass"] == superclass]

    metadata["tile_path"] = "None"
    metadata["tile_path"] = metadata.index.map(
        lambda x: find_tile_path(x, tiles_dir))

    class_mapping = get_class_mapping(superclass)

    model, transform, embedding_dim, _ = load_model(model_name, device="cpu")
    train_loader, val_loader, test_loader = get_dataloader(
        metadata,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_train=transforms.Compose(get_augmentations() + [transform]),
        transform_eval=transform,
        class_mapping=class_mapping,
        cache_data=False,
    )
    class_weights = compute_class_weights(metadata, superclass, class_mapping)

    print(f"Training on {len(train_loader.dataset)} samples.")
    print(f"Validating on {len(val_loader.dataset)} samples.")
    print(f"Testing on {len(test_loader.dataset)} samples.")
    print(f"Class weights: {class_weights}")

    mil_model = AttentionAggregator(
        embedding_dim,
        feature_extractor=model,
        num_classes=len(class_mapping),
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
        # precision="16-mixed",
        precision="bf16",
        check_val_every_n_epoch=10,
        logger=TensorBoardLogger("tb_logs", name="mil_from_tiles"),
    )
    trainer.fit(mil_model, train_loader, val_loader)
    trainer.test(mil_model, test_loader)
    if output_path:
        trainer.save_checkpoint(output_path)


if __name__ == "__main__":
    main()
