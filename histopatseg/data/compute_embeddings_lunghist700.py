import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from histopatseg.data.dataset import TileDataset
from histopatseg.models.models import load_model
from histopatseg.utils import get_device

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

project_dir = Path(__file__).parents[2].resolve()

load_dotenv()


def compute_embeddings(model,
                       dataloader,
                       device="cuda",
                       autocast_dtype=torch.float16):
    """Compute embeddings dynamically for the given model."""

    model.to(device)
    model.eval()
    embeddings, tile_ids = [], []

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        with torch.inference_mode():
            for images, batch_tile_ids in tqdm(dataloader,
                                               desc="Computing embeddings"):
                embeddings.append(model(images.to(device)).detach().cpu())
                tile_ids.extend(batch_tile_ids)

    return torch.cat(embeddings, dim=0).numpy(), np.array(tile_ids)


@click.command()
@click.option("--output-file", help="path to store the embeddings.")
@click.option("--model-name", default="UNI2", help="Name of the model to use.")
@click.option("--magnification",
              type=click.INT,
              default=10,
              help="Magnification level of the tiles.")
@click.option("--gpu-id", default=0, help="Name of the model to use.")
@click.option("--batch-size", default=256, help="Batch size for inference.")
@click.option("--num-workers",
              default=0,
              help="Number of workers for dataloader.")
def main(output_file, model_name, magnification, gpu_id, batch_size,
         num_workers):
    """Simple CLI program to greet someone"""

    autocast_dtype_dict = {
        "local": torch.float16,
        "bioptimus": torch.float16,
        "UNI2": torch.bfloat16,
    }
    autocast_dtype = autocast_dtype_dict[model_name]
    device = get_device(gpu_id)

    output_file = Path(output_file).resolve()
    data_path = Path(os.getenv("LUNGHIST700_PATH"))
    tiles_dir = data_path / f"LungHist700_{magnification}x/tiles"
    tile_paths = list(tiles_dir.glob("*.png"))

    logger.info(f"Loading model {model_name} on device {device}.")

    model, transform, embedding_dim = load_model(model_name, device)

    tile_dataset = TileDataset(tile_paths, transform=transform)
    dataloader = DataLoader(tile_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)
    logger.info(f"Computing embeddings for {len(tile_paths)} tiles.")
    embeddings, tile_ids = compute_embeddings(model,
                                              dataloader,
                                              device=device,
                                              autocast_dtype=autocast_dtype)

    logger.info(f"Saving embeddings to {output_file}.")

    output_folder = output_file.parent
    output_folder.mkdir(parents=True, exist_ok=True)
    np.savez(output_file,
             embeddings=embeddings,
             tile_ids=tile_ids,
             embedding_dim=embedding_dim)


if __name__ == "__main__":
    main()
