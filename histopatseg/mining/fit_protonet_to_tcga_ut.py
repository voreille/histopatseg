import logging
from pathlib import Path
import random

import click
import numpy as np
import torch
from tqdm import tqdm

from histopatseg.data.dataset import TileDataset
from histopatseg.fewshot.protonet import ProtoNet
from histopatseg.models.foundation_models import load_model
from histopatseg.utils import get_device

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[logging.StreamHandler()],  # Output logs to the console
)

logger = logging.getLogger(__name__)

label_map = {
    "Lung_adenocarcinoma": 0,
    "Lung_squamous_cell_carcinoma": 1,
    "Lung_normal": 2,
}


def get_fitted_protonet(embeddings_train, labels_train, label_map=None):
    protonet = ProtoNet(label_map=label_map)
    protonet.fit(
        torch.tensor(embeddings_train, dtype=torch.float32),
        torch.tensor(labels_train, dtype=torch.long),
    )

    return protonet


def get_label_from_path(path):
    """
    Get the label from the path.
    """
    # Assuming the label is the parent directory of the tile
    return Path(path).parents[2].name


def compute_embeddings(dataloader, model):
    embeddings_all = []
    labels_all = []
    device = next(model.parameters())[0].device

    for batch_idx, (batch, batch_tilepath) in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.inference_mode():
            embeddings = model(batch.to(device, non_blocking=True))
            embeddings = embeddings.cpu().numpy()
            embeddings_all.append(embeddings)
            # batch_tilepath = batch_tilepath.cpu().numpy()
            batch_labels = [label_map[get_label_from_path(x)] for x in batch_tilepath]
            labels_all.append(batch_labels)

    return np.concatenate(embeddings_all, axis=0), np.concatenate(labels_all, axis=0)


def get_tile_paths(tiles_dir, n_wsi=32, magnification_key=0, random_state=42):
    """
    Get all tile paths from the given directory.
    """
    tiles_dir = Path(tiles_dir)
    tile_paths = []
    for cancer_dir in tiles_dir.iterdir():
        if not cancer_dir.is_dir():
            continue

        wsi_dirs = list((cancer_dir / f"{magnification_key}").iterdir())
        if len(wsi_dirs) < n_wsi:
            logger.warning(f"Not enough WSI directories found in {cancer_dir}.")
            continue

        # Randomly select n_wsi
        random.seed(random_state)
        random.shuffle(wsi_dirs)
        wsi_dirs = wsi_dirs[:n_wsi]

        for wsi_dir in wsi_dirs:
            tile_paths.extend(
                [f for f in wsi_dir.iterdir() if f.suffix in [".png", ".jpg", ".jpeg"]]
            )

    return tile_paths


@click.command()
@click.option(
    "--tiles-dir",
    type=click.Path(exists=True),
    help="Path to the directory containing tiles.",
)
@click.option("--output-path", required=True, help="Path to save the output ProtoNet")
@click.option("--model-name", default="UNI2", help="Model name to use for embeddings")
@click.option("--gpu-id", default=0, help="GPU ID to use.")
@click.option("--batch-size", default=256, help="Batch size for embedding computation")
@click.option("--num-workers", default=8, help="Number of workers for data loading.")
@click.option("--n-wsi", default=32, help="Number of WSI to sample from each cancer type")
@click.option("--magnification-key", default=0, help="Magnification key to use")
@click.option("--random-state", default=42, help="Random state for reproducibility")
def main(
    tiles_dir,
    output_path,
    model_name,
    gpu_id,
    batch_size,
    num_workers,
    n_wsi,
    magnification_key,
    random_state,
):
    """Simple CLI program to greet someone"""

    device = get_device(gpu_id=gpu_id)
    model, preprocess, _, _ = load_model(model_name, device)

    logger.info(f"Using model {model_name} for embedding computation")
    logger.info(f"Using preprocess:\n {preprocess}")

    tile_paths = get_tile_paths(
        tiles_dir,
        n_wsi=n_wsi,
        magnification_key=magnification_key,
        random_state=random_state,
    )

    dataset = TileDataset(tile_paths, preprocess=preprocess)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    embeddings, labels = compute_embeddings(dataloader, model)
    unique_labels = np.unique(labels)
    label_map_pruned = {
        label: numeric_label
        for label, numeric_label in label_map.items()
        if numeric_label in unique_labels
    }

    protonet = get_fitted_protonet(embeddings, labels, label_map=label_map_pruned)
    protonet.save(output_path)


if __name__ == "__main__":
    main()
