from datetime import datetime
import logging
from pathlib import Path

import click
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from histopatseg.constants import CLASS_MAPPING
from histopatseg.data.dataset import MILDataset
from histopatseg.data.utils import split_data_per_image
from histopatseg.models.foundation_models import load_model
from histopatseg.models.mil import AttentionAggregator
from histopatseg.utils import get_class_weights, get_device

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_torchvision_augmentations():
    return [
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # random crop and resize
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomSolarize(threshold=128, p=0.1),
    ]


def get_tile_transform():
    return [
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]


def get_class_mapping(superclass):
    if superclass == "aca":
        return {"aca_bd": 0, "aca_md": 1, "aca_pd": 2}
    elif superclass == "scc":
        return {"scc_bd": 0, "scc_md": 1, "scc_pd": 2}
    return CLASS_MAPPING


def get_dataloader(
    images_dir,
    metadata,
    batch_size=32,
    num_workers=0,
    transform=None,
    augment=None,
    class_mapping=CLASS_MAPPING,
    random_state=42,
    fold=0,
    n_splits=4,
):
    """Create a DataLoader from embeddings and labels."""

    train_image_ids, val_image_ids, test_image_ids = split_data_per_image(
        metadata,
        random_state=random_state,
        fold=fold,
        n_splits=n_splits,
    )

    train_image_paths = [images_dir / f"{image_id}.png" for image_id in train_image_ids]
    val_image_paths = [images_dir / f"{image_id}.png" for image_id in val_image_ids]
    test_image_paths = [images_dir / f"{image_id}.png" for image_id in test_image_ids]

    train_labels = (
        metadata.loc[train_image_ids, "class_name"].map(lambda x: class_mapping[x]).values
    )
    val_labels = metadata.loc[val_image_ids, "class_name"].map(lambda x: class_mapping[x]).values
    test_labels = metadata.loc[test_image_ids, "class_name"].map(lambda x: class_mapping[x]).values

    collate_fn = MILDataset.get_collate_fn_ragged()

    train_dataloader = DataLoader(
        MILDataset(
            train_image_paths,
            train_labels,
            transform=transform,
            augment=augment,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_dataloader = DataLoader(
        MILDataset(
            val_image_paths,
            val_labels,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    test_dataloader = DataLoader(
        MILDataset(
            test_image_paths,
            test_labels,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
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


@click.command()
@click.option("--output-path", default=None, help="Path to store the weight of the linear layer.")
@click.option(
    "--model-name",
    default="UNI2",
    show_default=True,
    help="Name of the model to use for feature extraction.",
)
@click.option(
    "--images-dir",
    default=None,
    type=click.Path(exists=True),
    help="Path to the images directory.",
)
@click.option("--metadata-path", type=click.Path(exists=True), help="Path to the metadata file.")
@click.option("--max-epochs", default=10, show_default=True, help="Max number of epochs to train.")
@click.option("--gpu-id", default=0, help="Name of the model to use.")
@click.option("--batch-size", default=32, show_default=True, help="Batch size for inference.")
@click.option("--lr", default=0.001, show_default=True, help="LR for Adam")
@click.option("--dropout", default=0.2, show_default=True, help="Dropout rate for MIL model")
@click.option(
    "--weight-decay", default=0, type=click.FLOAT, show_default=True, help="Weight decay for Adam"
)
@click.option("--num-workers", default=0, help="Number of workers for dataloader.")
@click.option(
    "--superclass", default="aca", show_default=True, help="Choose on which superclass to train."
)
@click.option("--fold", default=0, help="Fold for cross-validation.")
@click.option("--n-splits", default=4, show_default=True, help="Number of cross-validation folds.")
@click.option(
    "--random-state", default=42, show_default=True, help="Random seed for reproducibility."
)
@click.option("--experiment-name", default=None, help="Custom name for this experiment run.")
def main(
    output_path,
    model_name,
    images_dir,
    metadata_path,
    max_epochs,
    gpu_id,
    batch_size,
    lr,
    dropout,
    weight_decay,
    num_workers,
    superclass,
    fold,
    n_splits,
    random_state,
    experiment_name,
):
    pl.seed_everything(random_state)
    # Parse magnification from the embeddings path
    images_dir = Path(images_dir).resolve()
    metadata = pd.read_csv(metadata_path).set_index("filename")

    if not experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"{model_name}_{superclass}_fold{fold}_{timestamp}"

    if superclass != "all":
        metadata = metadata[metadata["superclass"] == superclass]

    class_mapping = get_class_mapping(superclass)

    model, transform, embedding_dim, _ = load_model(model_name, device=get_device(gpu_id))
    print(f"Model: {model_name}, Transform advised: {transform}")
    used_transform = transforms.Compose(get_tile_transform())
    print(f"Used transform: {used_transform}, don't worry ToTensor is done in the dataset.")

    train_loader, val_loader, test_loader = get_dataloader(
        images_dir,
        metadata,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=transforms.Compose(get_torchvision_augmentations()),
        transform=used_transform,
        class_mapping=class_mapping,
        fold=fold,
        n_splits=n_splits,
        random_state=random_state,
    )
    print(f"Training on fold {fold} with superclass {superclass}.")
    print(f"Training on {len(train_loader.dataset)} samples.")
    print(f"Validating on {len(val_loader.dataset)} samples.")
    print(f"Testing on {len(test_loader.dataset)} samples.")

    mil_model = AttentionAggregator(
        embedding_dim,
        feature_extractor=model,
        num_classes=len(class_mapping),
        optimizer="AdamW",
        optimizer_kwargs={"lr": lr, "weight_decay": weight_decay},
        scheduler="ReduceLROnPlateau",
        scheduler_kwargs={
            "mode": "max",
            "factor": 0.5,  # Reduce LR by half when plateauing
            "patience": 15,  # Wait 15 epochs before reducing
            "threshold": 0.005,
            "verbose": True,
        },
        dropout=dropout,
    )

    # Correctly debug the first batch
    try:
        print("Fetching first batch...")
        sample_tile_bags, labels = next(iter(train_loader))
        print(f"Batch fetched successfully. Tile bags shape: {len(sample_tile_bags)} samples")

        print("Running forward pass on first sample...")
        # Pass the first sample's tile bag to the model
        sample_output, _ = mil_model(sample_tile_bags[0])
        print(f"Forward pass successful! Output shape: {sample_output.shape}")
    except Exception as e:
        print(f"Error during debug forward pass: {e}")
        import traceback

        traceback.print_exc()

    # Replace early stopping with a simple checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/" + experiment_name,
        filename="{epoch:03d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,  # Save the top 3 models instead of just 1
        verbose=True,
    )

    # Add LR monitor to track learning rate changes
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    model_precision_mapping = {
        "UNI2": "bf16-mixed",
        "H-optimus-1": "16-mixed",
        "H-optimus-0": "16-mixed",
    }
    precision = model_precision_mapping.get(model_name, "f16-mixed")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[gpu_id],
        precision=precision,
        check_val_every_n_epoch=1,
        logger=TensorBoardLogger("tb_logs", name="mil_from_tiles", version=experiment_name),
        callbacks=[checkpoint_callback, lr_monitor],
    )

    # Create a dictionary with all hyperparameters
    hparams = {
        # Run environment info
        "run/experiment_name": experiment_name,
        "run/model_name": model_name,
        "run/output_path": str(output_path) if output_path else None,
        "run/gpu_id": gpu_id,
        "run/num_workers": num_workers,
        "run/fold": fold,
        "run/timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        # Model architecture params
        "model/architecture": model_name,
        "model/class": model.__class__.__name__,
        "model/embedding_dim": embedding_dim,
        "model/num_classes": len(class_mapping),
        # Training hyperparams
        "train/optimizer": "AdamW",
        "train/learning_rate": lr,
        "train/weight_decay": weight_decay,
        "train/batch_size": batch_size,
        "train/max_epoch": max_epochs,
        "train/dropout": dropout,
        "train/precision": precision,
        # Data params
        "data/superclass": superclass,
        "data/images_dir": str(images_dir),
        "data/metadata_path": str(metadata_path),
        "data/num_train_samples": len(train_loader.dataset),
        "data/num_val_samples": len(val_loader.dataset),
        "data/num_test_samples": len(test_loader.dataset),
        # Cross-validation params
        "cv/fold_idx": fold,
        "cv/n_splits": n_splits,
        "cv/random_state": random_state,
        # Class info
        "class/mapping": str(class_mapping),
        "class/count": len(class_mapping),
    }

    # Log class distribution
    for cls_name, cls_idx in class_mapping.items():
        train_count = sum(1 for label in train_loader.dataset.labels if label == cls_idx)
        val_count = sum(1 for label in val_loader.dataset.labels if label == cls_idx)
        test_count = sum(1 for label in test_loader.dataset.labels if label == cls_idx)

        hparams[f"class_dist/train_{cls_name}"] = train_count
        hparams[f"class_dist/val_{cls_name}"] = val_count
        hparams[f"class_dist/test_{cls_name}"] = test_count

    # Log hyperparameters
    trainer.logger.log_hyperparams(hparams)

    trainer.fit(mil_model, train_loader, val_loader)
    trainer.test(mil_model, test_loader)
    if output_path:
        trainer.save_checkpoint(output_path)


if __name__ == "__main__":
    main()
