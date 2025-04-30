from pathlib import Path

import click
from dotenv import load_dotenv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from histopatseg.data.dataset import TileDataset
from histopatseg.models.foundation_models import load_model
from histopatseg.utils import get_device

project_dir = Path(__file__).parents[2].resolve()

load_dotenv()


def compute_embeddings(model, dataloader, device="cuda", autocast_dtype=torch.float16):
    """Compute embeddings dynamically for the given model."""

    model.to(device)
    model.eval()
    embeddings, tile_ids = [], []

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        with torch.inference_mode():
            for images, batch_tile_ids in tqdm(dataloader, desc="Computing embeddings"):
                embeddings.append(model(images.to(device)).cpu())
                tile_ids.extend(batch_tile_ids)

    return torch.cat(embeddings, dim=0).numpy(), np.array(tile_ids)


@click.command()
@click.option("--output-file", help="path to store the embeddings.")
@click.option("--tiles-dir", type=click.Path(exists=True), help="path to store the embeddings.")
@click.option("--model-name", default="UNI2", show_default=True, help="Name of the model to use.")
@click.option("--gpu-id", default=0, show_default=True, help="Name of the model to use.")
@click.option("--batch-size", default=256, show_default=True, help="Batch size for inference.")
@click.option(
    "--num-workers", default=0, show_default=True, help="Number of workers for dataloader."
)
@click.option("--tile-size", default=224, help="Tile size for center crop")
@click.option("--is-raw-data", is_flag=True, help="wether the data is raw or not.")
def main(output_file, tiles_dir, model_name, gpu_id, batch_size, num_workers, tile_size, is_raw_data):
    """Simple CLI program to greet someone"""

    device = get_device(gpu_id)
    tiles_dir = Path(tiles_dir).resolve()

    output_file = Path(output_file).resolve()
    if is_raw_data:
        tile_paths = list(tiles_dir.rglob("*.jpg"))
    else:
        tile_paths = list(tiles_dir.glob("*.png"))

    click.echo(f"Loading model {model_name} on device {device}.")

    model, transform, embedding_dim, autocast_dtype = load_model(model_name, device)
    if model_name == "H-optimus-0" or model_name == "H-optimus-1":
        normalize = transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517),
        )
    else:
        click.echo("Using default ImageNet normalization for tiles.")
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    transform = transforms.Compose(
        [
            transforms.Resize(tile_size),
            transforms.CenterCrop((tile_size, tile_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    tile_dataset = TileDataset(tile_paths, transform=transform)
    dataloader = DataLoader(
        tile_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    click.echo(f"Computing embeddings for {len(tile_paths)} tiles.")
    embeddings, tile_ids = compute_embeddings(
        model, dataloader, device=device, autocast_dtype=autocast_dtype
    )

    click.echo(f"Saving embeddings to {output_file}.")

    output_folder = output_file.parent
    output_folder.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, embeddings=embeddings, tile_ids=tile_ids, embedding_dim=embedding_dim)


if __name__ == "__main__":
    main()
