"""
TODO: Compute first every tile embedding and then compute the WSI embedding
"""

from pathlib import Path

import click
import gigapath.slide_encoder as slide_encoder
import h5py
import numpy as np
from PIL import Image
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm


class TileDatasetFromImage(Dataset):
    def __init__(self, image: Image.Image, tile_size: int = 256):
        self.tile_size = tile_size
        self.image = image
        self.tiles, self.coords = self._tile_image()
        self.transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _center_crop_to_multiple(self, image: Image.Image) -> Image.Image:
        W, H = image.size
        new_W = (W // self.tile_size) * self.tile_size
        new_H = (H // self.tile_size) * self.tile_size
        left = (W - new_W) // 2
        top = (H - new_H) // 2
        right = left + new_W
        bottom = top + new_H
        return image.crop((left, top, right, bottom))

    def _tile_image(self):
        image = self._center_crop_to_multiple(self.image)
        W, H = image.size
        tiles = []
        coords = []
        for y in range(0, H - self.tile_size + 1, self.tile_size):
            for x in range(0, W - self.tile_size + 1, self.tile_size):
                crop = image.crop((x, y, x + self.tile_size, y + self.tile_size))
                tiles.append(crop)
                coords.append([x, y])
        return tiles, coords

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        # Make sure coordinates are in the right format - should be [x, y] in pixels
        coord = torch.tensor(self.coords[idx], dtype=torch.float)
        return self.transform(tile), coord


def load_tile_encoder(device, model_cache_dir=None):
    """
    Get or create a tile encoder model with manual weight caching.

    Args:
        device: Device to place the model on ('cpu' or 'cuda:x')
        model_cache_dir: Optional custom directory to store model weights

    Returns:
        Loaded tile encoder model on the specified device
    """
    # Hardcoded model name for tile encoder
    tile_model_name = "hf_hub:prov-gigapath/prov-gigapath"

    # Define custom weights path if cache dir provided
    weights_filename = "tile_encoder_weights.pt"
    weights_path = None
    if model_cache_dir:
        model_cache_dir = Path(model_cache_dir)
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        weights_path = model_cache_dir / weights_filename
        click.echo(f"Using custom model cache path: {weights_path}")

    # Check if we have cached weights
    if weights_path and weights_path.exists():
        click.echo(f"Loading tile encoder from cached weights: {weights_path}")
        try:
            # Load model directly from saved weights
            tile_encoder = timm.create_model(
                "hf_hub:prov-gigapath/prov-gigapath", pretrained=False
            )
            tile_encoder.load_state_dict(torch.load(weights_path, map_location="cpu"))
            tile_encoder = tile_encoder.to(device)
            tile_encoder.eval()
            click.echo("Successfully loaded tile encoder from cached weights")
            return tile_encoder
        except Exception as e:
            click.echo(f"Error loading from cached weights: {e}")
            click.echo("Will download model from HF hub instead")

    # Otherwise, download from hub
    click.echo(f"Downloading tile encoder model: {tile_model_name}")
    try:
        tile_encoder = timm.create_model(tile_model_name, pretrained=True)
        tile_encoder = tile_encoder.to(device)
        tile_encoder.eval()

        # Save model if cache path specified
        if weights_path:
            click.echo(f"Saving tile encoder weights to: {weights_path}")
            torch.save(tile_encoder.state_dict(), weights_path)

        return tile_encoder

    except Exception as e:
        click.echo(f"Error loading tile encoder: {e}")
        raise RuntimeError(f"Failed to load tile encoder: {e}")


def load_slide_encoder_model(device, global_pool=True, model_cache_dir=None):
    """
    Get or create a slide encoder model with manual weight caching.

    Args:
        device: Device to place the model on ('cpu' or 'cuda:x')
        global_pool: Whether to use global pooling
        model_cache_dir: Optional custom directory to store model weights

    Returns:
        Loaded slide encoder model on the specified device
    """
    # Include global_pool setting in filename
    weights_filename = f"slide_encoder_global_pool_{global_pool}.pth"

    # If custom cache dir provided, use it
    if model_cache_dir:
        model_cache_dir = Path(model_cache_dir)
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a full path to cached weights
        local_path = model_cache_dir / weights_filename

        # If we have cached weights, use them directly
        if local_path.exists():
            click.echo(f"Loading slide encoder from custom cache: {local_path}")
            try:
                # Create model using our cached weights file directly
                slide_encoder_model = slide_encoder.create_model(
                    pretrained=str(local_path),  # Pass our cached file path directly
                    model_arch="gigapath_slide_enc12l768d",
                    in_chans=1536,
                    global_pool=global_pool,
                )
                slide_encoder_model = slide_encoder_model.to(device)
                slide_encoder_model.eval()
                click.echo(
                    f"Successfully loaded slide encoder from custom cache (global_pool={global_pool})"
                )
                return slide_encoder_model
            except Exception as e:
                click.echo(f"Error loading from custom cache: {e}")
                click.echo("Will download from HF hub instead")

    # No custom cache or loading failed, download from HF hub
    try:
        # Set the cache directory for the HF download if specified
        local_dir = str(model_cache_dir) if model_cache_dir else None

        click.echo(f"Downloading slide encoder from HF hub (global_pool={global_pool})")
        slide_encoder_model = slide_encoder.create_model(
            pretrained="hf_hub:prov-gigapath/prov-gigapath",
            model_arch="gigapath_slide_enc12l768d",
            in_chans=1536,
            global_pool=global_pool,
            local_dir=local_dir,  # Pass the custom directory if provided
        )
        slide_encoder_model = slide_encoder_model.to(device)
        slide_encoder_model.eval()

        # If we have a custom cache dir, save a copy there with our specific global_pool name
        if model_cache_dir:
            slide_encoder_path = model_cache_dir / weights_filename
            click.echo(f"Saving a copy to custom cache: {slide_encoder_path}")

            # Get the state dict from the loaded model
            state_dict = {"model": slide_encoder_model.state_dict()}
            torch.save(state_dict, slide_encoder_path)

        return slide_encoder_model

    except Exception as e:
        click.echo(f"Error loading slide encoder: {e}")
        raise RuntimeError(f"Failed to load slide encoder: {e}")


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing WSI images to process",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to store output embeddings",
)
@click.option("--batch-size", default=64, help="Batch size for tile encoding")
@click.option("--num-workers", default=4, help="Number of workers for DataLoader")
@click.option(
    "--gpu-id", type=int, default=0, help="ID of the GPU to use (e.g. 0, 1, 2). Use -1 for CPU."
)
@click.option(
    "--file-extensions",
    default=".jpg,.jpeg,.png,.tif,.tiff",
    help="Comma-separated list of image file extensions to process",
)
@click.option(
    "--output-prefix",
    default="",
    help="Optional prefix for output files",
)
@click.option(
    "--global-pool-tile-encoder",
    type=click.BOOL,  # Change to boolean type
    default=True,
    help="Whether to use global pooling for tile encoder (True/False)",
)
@click.option(
    "--model-cache-dir",
    type=click.Path(path_type=Path),
    default="models/gigapath/pretrained_weights/",
    help="Directory to cache models",
)
def main(
    input_dir,
    output_dir,
    batch_size,
    num_workers,
    gpu_id,
    file_extensions,
    output_prefix,
    global_pool_tile_encoder,
    model_cache_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hardcoded model name for tile encoder since not intended to be changed
    tile_model_name = "hf_hub:prov-gigapath/prov-gigapath"

    # Add global pool setting to output prefix
    output_prefix = f"{output_prefix}global_pool_{global_pool_tile_encoder}_"

    # Set device based on gpu_id
    if gpu_id >= 0 and torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        click.echo(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = "cpu"
        click.echo("Using CPU")

    extensions = file_extensions.split(",")
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(input_dir.glob(f"*{ext}")))

    if not image_paths:
        click.echo(f"No matching image files found in {input_dir}")
        return

    click.echo(f"Found {len(image_paths)} images to process")

    # Load models
    click.echo(f"Loading tile encoder model: {tile_model_name}")

    tile_encoder = load_tile_encoder(device, model_cache_dir)
    click.echo("Tile encoder loaded")

    slide_encoder_model = load_slide_encoder_model(
        device, global_pool=global_pool_tile_encoder, model_cache_dir=model_cache_dir
    )
    click.echo("Slide encoder loaded")

    # Prepare for storing results
    wsi_file = h5py.File(output_dir / f"{output_prefix}wsi_embeddings.h5", "w")
    tile_file = h5py.File(output_dir / f"{output_prefix}tile_embeddings.h5", "w")
    metadata_file = open(output_dir / f"{output_prefix}metadata.csv", "w")

    metadata_file.write("image_name,n_tiles,embedding_idx\n")
    wsi_embeddings_list = []

    # Process each image
    embedding_idx = 0
    for img_path in tqdm(image_paths, desc="Processing images"):
        img_name = img_path.stem
        click.echo(f"\nProcessing: {img_name}")

        try:
            # Load and tile image
            image = Image.open(img_path).convert("RGB")
            dataset = TileDatasetFromImage(image)

            if len(dataset) == 0:
                click.echo(f"No tiles extracted from {img_name}, skipping")
                continue

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True,  # Keep workers alive between iterations
                pin_memory=True,  # Speed up CPU->GPU transfers
                prefetch_factor=2,  # Prefetch batches
            )

            all_embeds = []
            all_coords = []

            # Extract tile embeddings
            with torch.cuda.amp.autocast(dtype=torch.float16):
                with torch.inference_mode():
                    for batch in tqdm(loader, desc=f"Embedding tiles for {img_name}"):
                        imgs, coords = batch
                        imgs = imgs.to(device)
                        embeds = tile_encoder(imgs).cpu()
                        all_embeds.append(embeds)
                        all_coords.append(coords)

            # Combine embeddings and coordinates
            tile_embeddings = torch.cat(all_embeds)
            coordinates = torch.cat(all_coords).float()
            tile_embeddings = tile_embeddings.unsqueeze(0)
            coordinates = coordinates.unsqueeze(0)

            # Compute WSI embedding
            with torch.cuda.amp.autocast(dtype=torch.float16):
                with torch.no_grad():
                    wsi_embedding = (
                        slide_encoder_model(tile_embeddings.to(device), coordinates.to(device))[0]
                        .cpu()
                        .squeeze()
                    )

            # Store tile embeddings
            tile_group = tile_file.create_group(img_name)
            tile_group.create_dataset("embeddings", data=tile_embeddings.numpy())
            tile_group.create_dataset("coordinates", data=coordinates.numpy())

            # Store WSI embedding
            wsi_embeddings_list.append(wsi_embedding.numpy())
            metadata_file.write(f"{img_name},{len(dataset)},{embedding_idx}\n")
            embedding_idx += 1

            click.echo(
                f"Processed {img_name}: {len(dataset)} tiles, WSI shape: {wsi_embedding.shape}"
            )
            torch.cuda.empty_cache()

        except Exception as e:
            click.echo(f"Error processing {img_name}: {e}")

    # Save final outputs
    if wsi_embeddings_list:
        wsi_file.create_dataset("embeddings", data=np.vstack(wsi_embeddings_list))
    wsi_file.close()
    tile_file.close()
    metadata_file.close()
    click.echo(f"Saved H5 embeddings to {output_dir}")


if __name__ == "__main__":
    main()
