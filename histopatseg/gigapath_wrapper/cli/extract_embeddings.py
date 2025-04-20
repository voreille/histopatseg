from pathlib import Path

import click
from PIL import Image
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from histopatseg.gigapath_wrapper.data.dataset import TileDatasetFromImage


@click.command()
@click.option(
    "--input-image",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the WSI image to tile",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to store output embeddings",
)
@click.option(
    "--tile-model-name",
    type=str,
    required=True,
    help='Timm model name or HF hub path (e.g. "hf_hub:prov-gigapath/prov-gigapath")',
)
@click.option(
    "--slide-model-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the slide-level encoder .pth file",
)
@click.option("--batch-size", default=64, help="Batch size for tile encoding")
@click.option("--num-workers", default=4, help="Number of workers for DataLoader")
@click.option(
    "--device", default="cuda", help='Device to run the models on (e.g. "cuda" or "cpu")'
)
def main(
    input_image, output_dir, tile_model_name, slide_model_path, batch_size, num_workers, device
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and tile image
    image = Image.open(input_image).convert("RGB")
    dataset = TileDatasetFromImage(image)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Load models
    tile_encoder = timm.create_model(tile_model_name, pretrained=True).to(device)
    tile_encoder.eval()

    slide_encoder = torch.load(slide_model_path, map_location=device)
    slide_encoder.eval()

    all_embeds = []
    all_coords = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding tiles"):
            imgs, coords = batch
            imgs = imgs.to(device)
            embeds = tile_encoder(imgs).cpu()
            all_embeds.append(embeds)
            all_coords.append(coords)

    tile_embeddings = torch.cat(all_embeds)
    coordinates = torch.cat(all_coords).float()

    # Compute WSI embedding
    with torch.no_grad():
        wsi_embedding = (
            slide_encoder(tile_embeddings.to(device), coordinates.to(device)).cpu().squeeze()
        )

    # Save outputs
    torch.save(tile_embeddings, output_dir / "tile_embeddings.pt")
    torch.save(wsi_embedding, output_dir / "wsi_embedding.pt")

    click.echo(f"Saved embeddings to {output_dir}")


if __name__ == "__main__":
    main()
