from collections import defaultdict
from pathlib import Path

import click
import h5py
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from histopatseg.data.dataset import TileDataset
from histopatseg.models.foundation_models import load_model
from histopatseg.utils import get_device, seed_everything


@click.command()
@click.option("--tcga-ut-dir", required=True, help="Path to TCGA UT directory.")
@click.option(
    "--n-wsi", default=None, type=int, help="Number of WSI to use to compute prototypes."
)
@click.option("--output-h5", default="cancer_prototypes.h5", help="Path to output h5 file.")
@click.option("--magnification-key", default=0, help="magnification key to use.")
@click.option("--model-name", default="UNI2", help="Foundation model name.")
@click.option("--gpu-id", default=0, help="GPU ID to use.")
@click.option("--batch-size", default=64, help="Batch size for processing images.")
@click.option("--random-state", default=42, help="Random state for reproducibility.")
@click.option("--num-workers", default=24, help="Number of workers for data loading.")
@click.option(
    "--exclude-cancers",
    default="",
    help="Comma-separated list of cancer names to exclude (e.g., 'Lung_normal,Lung_adenocarcinoma').",
)
@click.option(
    "--pre-centercrop-size",
    default=None,
    type=click.INT,
    help="Size of the center crop before applying default transforms. If None, no center crop is applied.",
)
@click.option(
    "--feature-transformation",
    default="centering,normalization",
    help="Feature transformation to apply.",
)
def main(
    tcga_ut_dir,
    n_wsi,
    output_h5,
    magnification_key,
    model_name,
    gpu_id,
    batch_size,
    random_state,
    num_workers,
    exclude_cancers,
    pre_centercrop_size,
    feature_transformation,
):
    """Compute average embeddings for each cancer type in the TCGA-UT dataset"""
    device = get_device(gpu_id=gpu_id)
    seed_everything(seed=random_state)
    excluded_cancers = set(exclude_cancers.split(",")) if exclude_cancers else set()
    feature_transformations = feature_transformation.split(",")

    tcga_ut_dir = Path(tcga_ut_dir).resolve()

    # Load the model
    model, preprocess, embedding_dim, autocast_dtype = load_model(
        model_name=model_name, device=device
    )
    model.eval()
    click.echo(f"Model {model_name} loaded with embedding dimension {embedding_dim}")
    if pre_centercrop_size is not None:
        click.echo(f"Precenter crop size: {pre_centercrop_size}")
        preprocess = transforms.Compose(
            [
                transforms.CenterCrop(pre_centercrop_size),
                preprocess,
            ]
        )

    click.echo(f"Preprocessing function: {preprocess}")

    # Get cancer types (top-level directories)
    cancer_types = [
        d for d in tcga_ut_dir.iterdir() if d.is_dir() and d.name not in excluded_cancers
    ]
    click.echo(f"Computing prototypes for {len(cancer_types)} cancer types")

    # Dictionary to store embeddings for each cancer type
    cancer_embeddings = defaultdict(np.ndarray)

    # Process each cancer type
    for cancer_dir in cancer_types:
        cancer_name = cancer_dir.name
        click.echo(f"Processing cancer type: {cancer_name}")

        # Get magnification directory
        mag_dir = cancer_dir / str(magnification_key)
        if not mag_dir.exists():
            click.echo(f"Magnification {magnification_key} not found for {cancer_name}, skipping")
            continue

        # Get WSI directories
        wsi_dirs = [d for d in mag_dir.iterdir() if d.is_dir()]
        if n_wsi is not None:
            # Randomly select n_wsi if specified
            if len(wsi_dirs) > n_wsi:
                np.random.shuffle(wsi_dirs)
                wsi_dirs = wsi_dirs[:n_wsi]

        click.echo(f"Processing {len(wsi_dirs)} WSIs for {cancer_name}")

        # Process each WSI
        # Get all image files
        image_files = []
        for wsi_dir in wsi_dirs:
            files = list(wsi_dir.glob("*.jpg"))
            if len(files) == 0:
                click.echo(f"No image files found in {wsi_dir}, skipping")
                continue
            image_files.extend(files)

        if not image_files:
            raise ValueError(
                f"No image files found in {cancer_name} at magnification {magnification_key}"
            )

        click.echo(f"Found {len(image_files)} image files for {cancer_name}")

        # Create dataset and dataloader
        dataset = TileDataset(
            image_files,
            transform=preprocess,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
        )

        # Compute embeddings
        with torch.inference_mode():
            wsi_embeddings = []

            for batch in tqdm(dataloader, desc=f"Processing {cancer_name}"):
                images, _ = batch
                images = images.to(device)
                # with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                embeddings = model(images)
                wsi_embeddings.append(embeddings.cpu().numpy())

        wsi_embeddings = np.vstack(wsi_embeddings)
        cancer_embeddings[cancer_name] = wsi_embeddings

    mean_embeddings = np.mean(
        np.vstack([embeddings for embeddings in cancer_embeddings.values()]), axis=0
    )

    # Compute average embeddings for each cancer type
    cancer_prototypes = {}
    for cancer_name, embeddings in cancer_embeddings.items():

        if "centering" in feature_transformations:
            embeddings = embeddings - mean_embeddings
        if "normalization" in feature_transformations:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        cancer_prototypes[cancer_name] = np.mean(embeddings, axis=0)
        click.echo(f"{cancer_name}: {len(embeddings)} tiles")

    # Save prototypes to HDF5 file
    output_h5 = Path(output_h5).resolve()
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("mean_embedding", data=mean_embeddings)
        for cancer_name, prototype in cancer_prototypes.items():
            f.create_dataset(cancer_name, data=prototype)

    click.echo(f"Prototypes saved to {output_h5}")


if __name__ == "__main__":
    main()
