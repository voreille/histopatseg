import os
from pathlib import Path
import time

import click
import gigapath.slide_encoder as slide_encoder
import h5py
from huggingface_hub import HfFolder, hf_hub_download
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


def load_slide_encoder_model(
    model_name="hf_hub:prov-gigapath/prov-gigapath",
    model_arch="gigapath_slide_enc12l768d",
    global_pool=True,
    timeout=300,  # 5 minute timeout
):
    """
    Load the slide encoder model with better error handling and timeout.

    Args:
        model_name: HF hub model name or local path
        model_arch: Model architecture name
        global_pool: Whether to use global pooling
        timeout: Maximum time to wait for model download in seconds
    """
    click.echo(f"Loading slide encoder model: {model_name}")

    # Extract hub name if using HF hub
    if model_name.startswith("hf_hub:"):
        hub_name = model_name.split(":")[1]
        local_dir = os.path.join(os.path.expanduser("~"), ".cache/")
        local_path = os.path.join(local_dir, "slide_encoder.pth")

        # Check if file already exists
        if not os.path.exists(local_path):
            click.echo(f"Downloading slide encoder model from HF Hub: {hub_name}")
            start_time = time.time()

            try:
                # Check if we have a token
                token = HfFolder.get_token()
                if token is None:
                    click.echo(
                        "Warning: No Hugging Face token found. If the model is private, download may fail."
                    )

                # Download with progress feedback and timeout
                hf_hub_download(
                    hub_name,
                    filename="slide_encoder.pth",
                    local_dir=local_dir,
                    force_download=False,
                )

                # Check for timeout
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Model download timed out after {timeout} seconds")

                click.echo(f"Download completed in {time.time() - start_time:.1f} seconds")
            except Exception as e:
                click.echo(f"Error downloading model: {str(e)}")
                click.echo("Falling back to randomly initialized model")
                return slide_encoder.gigapath_slide_enc12l768d(
                    in_chans=1536, global_pool=global_pool
                )

    # Now load the model using the original function
    try:
        model = slide_encoder.create_model(model_name, model_arch, 1536, global_pool=global_pool)
        return model
    except Exception as e:
        click.echo(f"Error loading model: {str(e)}")
        click.echo("Falling back to randomly initialized model")
        # Create the model directly without loading weights
        return slide_encoder.gigapath_slide_enc12l768d(in_chans=1536, global_pool=global_pool)


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
    type=click.Choice(["avg", "none"], case_sensitive=False),
    default="avg",
    help="Global pooling method for tile encoder",
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
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hardcoded model name for tile encoder since not intended to be changed
    tile_model_name = "hf_hub:prov-gigapath/prov-gigapath"

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
    # tile_encoder = timm.create_model(tile_model_name, pretrained=True).to(device)
    # tile_encoder.eval()
    # tile_encoder, slide_encoder_model = load_tile_slide_encoder(
    #     global_pool=True if global_pool_tile_encoder == "avg" else False
    # )

    slide_encoder_model = load_slide_encoder_model(
        global_pool=True if global_pool_tile_encoder == "avg" else False
    )
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    # slide_encoder_model = slide_encoder.create_model(
    #     "hf_hub:prov-gigapath/prov-gigapath",
    #     "gigapath_slide_enc12l768d",
    #     1536,
    #     global_pool=True if global_pool_tile_encoder == "avg" else False,
    # )

    tile_encoder.eval()
    slide_encoder_model.eval()
    # slide_encoder_model = slide_encoder.create_model(
    #     "hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536
    # )
    slide_encoder_model.eval()
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

            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

            all_embeds = []
            all_coords = []

            # Extract tile embeddings
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
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
                        slide_encoder_model(tile_embeddings.to(device), coordinates.to(device))
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
