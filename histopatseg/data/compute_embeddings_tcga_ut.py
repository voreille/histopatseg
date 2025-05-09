import json
from pathlib import Path

import click
import h5py
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from histopatseg.data.dataset import TileDataset
from histopatseg.models.foundation_models import load_model
from histopatseg.utils import get_device


def load_hdf5(input_path):
    """
    Load data from an HDF5 file.

    Parameters:
        input_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary containing all datasets and global attributes.
    """
    result = {}

    with h5py.File(input_path, "r") as file:
        # Load global attributes
        global_attrs = {}
        for attr_key, attr_val in file.attrs.items():
            # Deserialize JSON strings if necessary
            if isinstance(attr_val, str) and attr_val.startswith("{"):
                try:
                    attr_val = json.loads(attr_val)
                except json.JSONDecodeError:
                    pass
            global_attrs[attr_key] = attr_val
        result["global_attributes"] = global_attrs

        # Load all datasets
        datasets = {}
        for key in file.keys():
            datasets[key] = file[key][:]
        result["datasets"] = datasets

    return result


def save_hdf5(output_path, asset_dict, global_attr_dict=None, mode="a"):
    """
    Save data to an HDF5 file.

    Parameters:
        output_path (str): Path to the HDF5 file.
        asset_dict (dict): Dictionary of datasets to save.
        global_attr_dict (dict): Dictionary of global attributes for the WSI.
        mode (str): File mode ('w' for write, 'a' for append).
    """
    file = h5py.File(output_path, mode)

    # Add global attributes to the file
    if global_attr_dict is not None:
        for attr_key, attr_val in global_attr_dict.items():
            # Serialize unsupported types to JSON strings
            if isinstance(attr_val, (dict, list)):
                attr_val = json.dumps(attr_val)
            file.attrs[attr_key] = attr_val

    # Add datasets and their attributes
    for key, val in asset_dict.items():
        if val.dtype.kind == "S":  # Check if the data is a string type
            # Use variable-length strings
            dt = h5py.string_dtype(encoding="utf-8")
            data_shape = val.shape
            if key not in file:
                dset = file.create_dataset(
                    key,
                    shape=data_shape,
                    maxshape=(None,) + data_shape[1:],
                    dtype=dt,
                )
                dset[:] = val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0] :] = val
        else:
            # Handle non-string datasets
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (1,) + data_shape[1:]
                maxshape = (None,) + data_shape[1:]
                dset = file.create_dataset(
                    key,
                    shape=data_shape,
                    maxshape=maxshape,
                    chunks=chunk_shape,
                    dtype=data_type,
                )
                dset[:] = val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0] :] = val

    file.close()
    return output_path


@click.command()
@click.option("--tcga-ut-dir", required=True, help="Path to TCGA UT directory.")
@click.option("--output-h5", default="cancer_prototypes.h5", help="Path to output h5 file.")
@click.option("--magnification-key", default=0, type=click.INT, help="magnification key to use.")
@click.option("--model-name", default="UNI2", help="Foundation model name.")
@click.option("--gpu-id", default=0, help="GPU ID to use.")
@click.option("--batch-size", default=64, help="Batch size for processing images.")
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
def main(
    tcga_ut_dir,
    output_h5,
    magnification_key,
    model_name,
    gpu_id,
    batch_size,
    num_workers,
    exclude_cancers,
    pre_centercrop_size,
):
    """Compute average embeddings for each cancer type in the TCGA-UT dataset"""
    device = get_device(gpu_id=gpu_id)
    excluded_cancers = set(exclude_cancers.split(",")) if exclude_cancers else set()

    tcga_ut_dir = Path(tcga_ut_dir).resolve()
    output_h5 = Path(output_h5).resolve()
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    # Load the model
    model, preprocess, embedding_dim, _ = load_model(model_name=model_name, device=device)
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

    # Process each cancer type
    global_attr_dict = {
        "model_name": model_name,
        "cancer_types": [c.name for c in cancer_types],
        "transform": str(preprocess),
    }

    mode = "w"
    for cancer_dir in cancer_types:
        cancer_name = cancer_dir.name
        click.echo(f"Processing cancer type: {cancer_name}")

        # Get magnification directory
        mag_dir = cancer_dir / str(magnification_key)
        if not mag_dir.exists():
            click.echo(f"Magnification {magnification_key} not found for {cancer_name}, skipping")
            continue

        # Get WSI directories
        image_files = list(mag_dir.rglob("*.jpg"))

        # Create dataset and dataloader
        dataset = TileDataset(
            image_files,
            transform=preprocess,
            return_path=True,
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
            for batch in tqdm(dataloader, desc=f"Processing {cancer_name}"):
                images, images_path = batch
                images = images.to(device)
                images_path = [Path(f).relative_to(tcga_ut_dir) for f in images_path]
                image_ids = [str(p.parent.name) + "/" + str(p.stem) for p in images_path]
                wsi_ids = [str(p.parent.name) for p in images_path]
                image_names = [p.name for p in images_path]
                images_path = [str(p) for p in images_path]

                embeddings = model(images)
                asset_dict = {
                    "embeddings": embeddings.cpu().numpy(),
                    "image_paths": np.array(images_path, dtype="S"),
                    "labels": np.array([cancer_name] * len(images_path), dtype="S"),
                    "magnification_key": np.array([magnification_key] * len(images_path)),
                    "wsi_ids": np.array(wsi_ids, dtype="S"),
                    "image_names": np.array(image_names, dtype="S"),
                    "image_ids": np.array(image_ids, dtype="S"),
                }
                save_hdf5(
                    output_h5,
                    asset_dict,
                    global_attr_dict=global_attr_dict,
                    mode=mode,
                )
                mode = "a"
                global_attr_dict = None


if __name__ == "__main__":
    main()
