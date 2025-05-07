from collections import defaultdict
from pathlib import Path

import click
import h5py
import numpy as np
import torch
from tqdm import tqdm

from histopatseg.data.dataset import TileOnTheFlyDataset
from histopatseg.models.foundation_models import load_model
from histopatseg.utils import get_device


def collate_fn_ragged(batch):
    """Collate function for variable-length sequences."""
    images = torch.cat(batch, dim=0)
    return images


@click.command()
@click.option("--lunghist700-dir", required=True, help="Path to LungHist700 directory.")
@click.option("--output-h5", default="cancer_prototypes.h5", help="Path to output h5 file.")
@click.option("--model-name", default="UNI2", help="Foundation model name.")
@click.option("--gpu-id", default=0, help="GPU ID to use.")
@click.option("--batch-size", default=8, help="Batch size for processing images.")
@click.option("--num-workers", default=8, help="Number of workers for data loading.")
@click.option(
    "--exclude-cancers",
    default="",
    help="Comma-separated list of cancer names to exclude (e.g., 'scc_pd,scc_md,scc_bd').",
)
def main(
    lunghist700_dir,
    output_h5,
    model_name,
    gpu_id,
    batch_size,
    num_workers,
    exclude_cancers,
):
    """Compute average embeddings for each cancer type in the TCGA-UT dataset"""
    device = get_device(gpu_id=gpu_id)
    excluded_cancers = set(exclude_cancers.split(",")) if exclude_cancers else set()

    lunghist700_dir = Path(lunghist700_dir).resolve()

    # Load the model
    model, preprocess, embedding_dim, autocast_dtype = load_model(
        model_name=model_name, device=device
    )
    model.eval()
    click.echo(f"Model {model_name} loaded with embedding dimension {embedding_dim}")
    click.echo(f"Preprocessing function: {preprocess}")

    # Get cancer types (top-level directories)
    image_paths = list(lunghist700_dir.glob("*.png"))
    cancer_types = list(set([f.stem.split("_40x")[0].split("_20x")[0] for f in image_paths]))
    if excluded_cancers:
        cancer_types = [cancer for cancer in cancer_types if cancer not in excluded_cancers]

    click.echo(f"Computing prototypes for {len(cancer_types)} cancer types")

    # Dictionary to store embeddings for each cancer type
    cancer_embeddings = defaultdict(np.ndarray)

    # Process each cancer type
    for cancer_name in cancer_types:
        image_files = [f for f in image_paths if f.stem.startswith(cancer_name)]

        if len(image_files) == 0:
            raise ValueError(f"No image files found in {cancer_name}")

        click.echo(f"Processing {len(image_files)} images for {cancer_name}")

        dataset = TileOnTheFlyDataset(
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
            collate_fn=collate_fn_ragged,
        )

        # Compute embeddings
        with torch.inference_mode():
            wsi_embeddings = []

            for images in tqdm(dataloader, desc=f"Processing {cancer_name}"):
                images = images.to(device)
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
        embeddings = embeddings - mean_embeddings
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
