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
from histopatseg.data.dataset import LungHist700ImageDataset
from histopatseg.data.utils import split_data_per_image
from histopatseg.models.lora import inject_lora
from histopatseg.models.mil_complete import MILModel
from histopatseg.models.models import load_model
from histopatseg.utils import get_class_weights

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()


def get_torchvision_augmentations():
    return [
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # random crop and resize
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomSolarize(threshold=128, p=0.1),
        transforms.ToTensor(),
    ]


def get_dataloader(
    images_dir,
    metadata,
    batch_size=32,
    num_workers=0,
    transform_train=None,
    transform_eval=None,
    class_mapping=CLASS_MAPPING,
    random_state=42,
    cache_data=False,
):
    """Create a DataLoader from embeddings and labels."""

    train_image_ids, val_image_ids, test_image_ids = split_data_per_image(
        metadata, random_state=random_state
    )

    train_image_paths = [images_dir / f"{image_id}.png" for image_id in train_image_ids]
    val_image_paths = [images_dir / f"{image_id}.png" for image_id in val_image_ids]
    test_image_paths = [images_dir / f"{image_id}.png" for image_id in test_image_ids]

    collate_fn = LungHist700ImageDataset.get_collate_fn_ragged()

    train_dataloader = DataLoader(
        LungHist700ImageDataset(
            train_image_paths,
            metadata,
            transform=transform_train,
            class_mapping=class_mapping,
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_dataloader = DataLoader(
        LungHist700ImageDataset(
            val_image_paths,
            metadata,
            transform=transform_eval,
            class_mapping=class_mapping,
        ),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_dataloader = DataLoader(
        LungHist700ImageDataset(
            test_image_paths,
            metadata,
            transform=transform_eval,
            class_mapping=class_mapping,
        ),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dataloader, val_dataloader, test_dataloader


def compute_class_weights(metadata, superclass, class_mapping):
    metadata_per_image = metadata.groupby("original_filename").agg(
        {"class_name": "first", "superclass": "first"}
    )

    if superclass != "all":
        metadata_per_image = metadata_per_image[metadata_per_image["superclass"] == superclass]

    return get_class_weights(
        metadata_per_image, class_column="class_name", class_mapping=class_mapping
    )


def get_class_mapping(superclass):
    if superclass == "aca":
        return {"aca_bd": 0, "aca_md": 1, "aca_pd": 2}
    elif superclass == "scc":
        return {"scc_bd": 0, "scc_md": 1, "scc_pd": 2}
    return CLASS_MAPPING


@click.command()
@click.option("--output-path", default=None, help="Path to store the weight of the linear layer.")
@click.option(
    "--model-name",
    default="UNI2",
    show_default=True,
    help="Name of the model to use for feature extraction.",
)
@click.option("--magnification", default=20, show_default=True, help="Magnification of the tiles.")
@click.option("--num-epochs", default=10, show_default=True, help="Number of epochs to train.")
@click.option("--gpu-id", default=0, help="Name of the model to use.")
@click.option("--batch-size", default=32, show_default=True, help="Batch size for inference.")
@click.option("--lr", default=0.001, show_default=True, help="LR for Adam")
@click.option("--dropout", default=0.2, show_default=True, help="LR for Adam")
@click.option(
    "--weight-decay", default=0, type=click.FLOAT, show_default=True, help="Weight decay for Adam"
)
@click.option("--num-workers", default=0, help="Number of workers for dataloader.")
@click.option(
    "--superclass", default="aca", show_default=True, help="Choose on which superclass to train."
)
def main(
    output_path,
    model_name,
    magnification,
    num_epochs,
    gpu_id,
    batch_size,
    lr,
    dropout,
    weight_decay,
    num_workers,
    superclass,
):
    # Parse magnification from the embeddings path

    data_path = Path(os.getenv("LUNGHIST700_RESAMPLED_PATH"))
    images_dir = data_path / f"LungHist700_{magnification}x/"
    metadata = pd.read_csv(data_path / "metadata.csv").set_index("filename")

    if superclass != "all":
        metadata = metadata[metadata["superclass"] == superclass]

    class_mapping = get_class_mapping(superclass)

    model, transform, embedding_dim, _ = load_model(model_name, device="cpu")
    train_loader, val_loader, test_loader = get_dataloader(
        images_dir,
        metadata,
        batch_size=batch_size,
        num_workers=num_workers,
        # transform_train=get_train_augmentations(),
        # transform_eval=get_eval_transform(),
        transform_train=transforms.Compose(get_torchvision_augmentations()),
        transform_eval=transforms.ToTensor(),
        class_mapping=class_mapping,
        cache_data=True,
    )
    # class_weights = compute_class_weights(metadata, superclass, class_mapping)

    print(f"Training on {len(train_loader.dataset)} samples.")
    print(f"Validating on {len(val_loader.dataset)} samples.")
    print(f"Testing on {len(test_loader.dataset)} samples.")
    # print(f"Class weights: {class_weights}")

    model = inject_lora(model)
    model.train()

    mil_model = MILModel(
        embedding_dim,
        feature_extractor=model,
        num_classes=len(class_mapping),
        tile_size=224,
        optimizer="Adam",
        optimizer_kwargs={"lr": lr, "weight_decay": weight_decay},
        # loss="CrossEntropyLoss",
        # loss_kwargs={"weight": class_weights},
        dropout=dropout,
        feature_extractor_transform=transform,
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
