import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from histopatseg.data.dataset import TileOnTheFlyDataset
from histopatseg.models.models import load_model
from histopatseg.utils import get_device

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

project_dir = Path(__file__).parents[2].resolve()

load_dotenv()


def compute_embeddings_bag(model, dataloader, device="cuda", autocast_dtype=torch.float16):
    """Compute embeddings for MIL bags (all tiles from each WSI)."""

    model.to(device)
    model.eval()
    all_embeddings = []
    all_image_ids = []

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        with torch.inference_mode():
            for batch_idx, tile_bags in enumerate(tqdm(dataloader, desc="Processing images")):
                image_ids = [f"img_{batch_idx}_{i}" for i in range(len(tile_bags))]

                for i, tiles in enumerate(tile_bags):
                    image_id = image_ids[i]
                    # Process tiles in smaller batches to avoid OOM
                    batch_size = 64
                    embeddings_list = []

                    for j in range(0, len(tiles), batch_size):
                        batch_tiles = tiles[j : j + batch_size].to(device)
                        batch_embeddings = model(batch_tiles).detach().cpu()
                        embeddings_list.append(batch_embeddings)

                    # Combine embeddings for all tiles from this image
                    img_embeddings = torch.cat(embeddings_list, dim=0).numpy()

                    all_embeddings.append(img_embeddings)
                    all_image_ids.append(image_id)

    return all_embeddings, all_image_ids


@click.command()
@click.option(
    "--input-dir", type=click.Path(exists=True), help="Path to the resized LungHist700 dataset."
)
@click.option(
    "--metadata-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the metadata file.",
)
@click.option("--output-file", type=click.Path(), help="Path to store the embeddings.")
@click.option("--model-name", default="UNI2", show_default=True, help="Name of the model to use.")
@click.option(
    "--tile-size",
    type=click.INT,
    default=224,
    show_default=True,
    help="Size of tiles to extract from images.",
)
@click.option("--gpu-id", default=0, show_default=True, help="GPU ID to use.")
@click.option(
    "--batch-size", default=1, show_default=True, help="Batch size for whole images (usually 1)."
)
@click.option(
    "--num-workers", default=1, show_default=True, help="Number of workers for dataloader."
)
def main(
    input_dir, metadata_path, output_file, model_name, tile_size, gpu_id, batch_size, num_workers
):
    """Compute embeddings for LungHist700 dataset using MIL approach with on-the-fly tiling."""

    input_dir = Path(input_dir).resolve()
    output_file = Path(output_file).resolve()
    device = get_device(gpu_id)
    output_file = Path(output_file).resolve()
    magnification = input_dir.name.split("_")[-1].replace("x", "")

    # Get the paths to all images (not tiles)
    image_paths = list(input_dir.rglob("*.png"))

    if not image_paths:
        logger.error(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(image_paths)} images at magnification {magnification}x")

    # Load metadata if available
    metadata = pd.read_csv(metadata_path).set_index("filename")
    logger.info(f"Loaded metadata for {len(metadata)} images")

    logger.info(f"Loading model {model_name} on device {device}.")
    model, transform, embedding_dim, autocast_dtype = load_model(model_name, device)

    # Create a dataset for MIL approach - tiles are created on the fly
    mil_dataset = TileOnTheFlyDataset(
        image_paths=image_paths,
        transform=transform,
        augment=None,
        tile_size=tile_size,
    )

    # Create dataloader with custom collate function for variable-length bags
    dataloader = DataLoader(
        mil_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=TileOnTheFlyDataset.get_collate_fn_ragged(),
    )

    logger.info(f"Computing embeddings for {len(image_paths)} images with on-the-fly tiling.")
    embeddings_list, image_ids = compute_embeddings_bag(
        model, dataloader, device=device, autocast_dtype=autocast_dtype
    )

    # Create mapping from image paths to their IDs
    image_paths_str = [str(p) for p in image_paths]

    logger.info(f"Saving embeddings to {output_file}.")
    output_folder = output_file.parent
    output_folder.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_file,
        embeddings=embeddings_list,
        image_ids=image_ids,
        image_paths=image_paths_str,
        embedding_dim=embedding_dim,
    )

    logger.info(f"Successfully processed {len(image_paths)} images and saved embeddings.")


if __name__ == "__main__":
    main()
