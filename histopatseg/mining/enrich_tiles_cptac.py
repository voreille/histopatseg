from pathlib import Path
import random

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
from openslide import OpenSlide
import pandas as pd
import torch
from torch.nn.functional import normalize
from tqdm import tqdm

from histopatseg.data.compute_embeddings_tcga_ut import load_hdf5
from histopatseg.fewshot.protonet import ProtoNet

label_map_lunghist700 = {
    "luad_differentiation": {"aca_bd": 0, "aca_md": 1, "aca_pd": 2, "nor": 3},
    "complete": {
        "aca_bd": 0,
        "aca_md": 1,
        "aca_pd": 2,
        "nor": 3,
        "scc_bd": 4,
        "scc_md": 5,
        "scc_pd": 6,
    },
    "nsclc_subtyping": {"aca": 0, "scc": 1, "nor": 2},
}

pattern_list = [
    "Normal",
    "Acinar adenocarcinoma",  # 142
    "Solid adenocarcinoma",  # 43
    "Papillary adenocarcinoma",  # 32
    "Micropapillary adenocarcinoma",  # 9
    "Lepidic adenocarcinoma",  # 6
    # "Mucinous adenocarcinoma",
]
pattern_mapping_to_tumor_types = {
    "Normal": ["Normal"],
    "Acinar adenocarcinoma": ["Acinar adenocarcinoma"],
    "Solid adenocarcinoma": ["Solid adenocarcinoma"],
    "Papillary adenocarcinoma": ["Papillary adenocarcinoma"],
    "Micropapillary adenocarcinoma": ["Micropapillary adenocarcinoma"],
    "Lepidic adenocarcinoma": ["Lepidic adenocarcinoma"],
    "Mucinous adenocarcinoma": ["Mucinous adenocarcinoma", "Invasive mucinous adenocarcinoma"],
}

pattern_mapping_to_lunghist700_class = {
    "Acinar adenocarcinoma": "aca_md",
    "Solid adenocarcinoma": "aca_pd",
    "Papillary adenocarcinoma": "aca_md",
    "Micropapillary adenocarcinoma": "aca_pd",
    "Lepidic adenocarcinoma": "aca_bd",
    "Normal": "nor",
    "Mucinous adenocarcinoma": "aca_bd",  # based on visual inspection
}


def save_heatmaps_figure(
    path,
    heatmaps,
    thumbnail,
    wsi_id,
    tumor_hist_type,
    n_selected_tiles,
    label_map,
):
    # Normalize all heatmaps to the same scale
    vmin = np.min(heatmaps)
    vmax = np.max(heatmaps)
    num_classes = heatmaps.shape[2]

    # Create a figure with a grid layout
    fig = plt.figure(figsize=(15, 20))  # Adjusted height to accommodate the large thumbnail
    grid = plt.GridSpec(3, num_classes + 1, height_ratios=[1, 0.05, 1], hspace=0.5, wspace=0.3)

    # Plot heatmaps in the first row
    heatmaps_list = [heatmaps[:, :, i] for i in range(heatmaps.shape[2])]
    titles = list(label_map.keys())

    for i, (heatmap, title) in enumerate(zip(heatmaps_list, titles)):
        ax = fig.add_subplot(grid[0, i])
        im = ax.imshow(heatmap.squeeze(), cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Add a single colorbar in the last column of the first row
    cbar_ax = fig.add_subplot(grid[0, -1])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Heatmap Intensity", fontsize=10)

    # Plot the thumbnail with overlay in the second row spanning all columns
    thumbnail_ax = fig.add_subplot(grid[2, :])
    thumbnail_ax.imshow(thumbnail)
    thumbnail_ax.set_title(f"Thumbnail with {n_selected_tiles} Selected Tiles", fontsize=12)
    thumbnail_ax.axis("off")

    # Add a main title
    plt.suptitle(f"Heatmaps for WSI {wsi_id} with {tumor_hist_type} Tumor Type", fontsize=16)

    # Save the plot
    plt.savefig(
        path / f"{wsi_id}__{tumor_hist_type}_overlay.png",
        dpi=300,
        bbox_inches="tight",
    )


def compute_distances_to_protypes(embeddings, protonet):
    """
    Compute the similarity between an embedding and a prototype.
    """
    # Normalize the vectors

    feats_query = torch.tensor(embeddings, dtype=torch.float32)

    feats_query = feats_query - protonet.mean
    feats_query = normalize(feats_query, dim=-1, p=2)
    feats_query = feats_query[:, None]  # [N x 1 x D]
    proto_embeddings = protonet.prototype_embeddings[None, :]  # [1 x C x D]
    pw_dist = (feats_query - proto_embeddings).norm(dim=-1, p=2)  # [N x C ]

    return pw_dist.numpy()


def select_tiles(distances, lunghist700_class_index, n_tiles_max=100, cutoff=0.8, random_state=42):
    """
    Select tiles based on the distance to the prototype, limit the number of tiles randomly,
    and keep only tiles within the top percentage of closest distances.

    Args:
        distances (np.ndarray): Pairwise distances to prototypes.
        lunghist700_class_index (int): Index of the target class prototype.
        n_tiles_max (int): Maximum number of tiles to select.
        cutoff (float): Fraction of tiles to keep based on closeness (e.g., 0.8 for top 20%).
        random_state (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Indices of selected tiles.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Get the distances to the target class prototype
    distances_class = distances[:, lunghist700_class_index]

    # Get the minimum distance to all other class prototypes for each tile
    distances_other_classes = np.min(
        np.delete(distances, lunghist700_class_index, axis=1),
        axis=1,
    )

    # Select indices where the distance to the target class is smaller
    selected_indices = np.where(distances_class < distances_other_classes)[0]

    # Apply cutoff to keep only the top percentage of closest distances
    if len(selected_indices) > 0:
        selected_distances = distances_class[selected_indices]
        threshold = np.quantile(selected_distances, 1 - cutoff)  # Compute the cutoff threshold
        selected_indices = selected_indices[selected_distances <= threshold]

    if len(selected_indices) == 0:
        # If no indices are selected, return an empty array
        return np.array([])

    # Randomly sample up to n_tiles_max indices
    if len(selected_indices) > n_tiles_max:
        selected_indices = np.random.choice(selected_indices, n_tiles_max, replace=False)

    return selected_indices


def compute_heatmap_optimized(
    wsi,
    coordinates,
    scores,
    tile_size=224,
    tile_level=0,
    rescale=False,
    selected_indices=None,
):
    # Rescale scores if needed
    if rescale:
        scores = (2 * scores - np.min(scores) - np.max(scores)) / (np.max(scores) - np.min(scores))

    num_classes = scores.shape[1]

    downsample_to_base = wsi.level_downsamples[tile_level]  # From scores_level to level 0

    wsi_dimensions = wsi.level_dimensions[0]
    downsample = downsample_to_base * tile_size
    heatmap_height = np.ceil(wsi_dimensions[0] / downsample).astype(int)
    heatmap_width = np.ceil(wsi_dimensions[1] / downsample).astype(int)
    heatmap = np.zeros(
        (heatmap_width, heatmap_height, num_classes), dtype=np.float32
    )  # Shape should be (height, width)

    # Populate the heatmap
    for i, (x, y) in enumerate(coordinates):
        grid_x = np.floor(x / downsample).astype(int)
        grid_y = np.floor(y / downsample).astype(int)
        heatmap[grid_y, grid_x, :] = scores[i, :]

    # Create an overlay mask for selected tiles
    overlay_mask = np.zeros((heatmap_width, heatmap_height), dtype=np.uint8)
    if selected_indices is not None:
        for i in selected_indices:
            x, y = coordinates[i]
            grid_x = np.floor(x / downsample).astype(int)
            grid_y = np.floor(y / downsample).astype(int)
            overlay_mask[grid_y, grid_x] = 255  # Mark selected tiles

    # Upscale the heatmap and overlay mask to match the thumbnail size
    thumbnail_size = wsi.level_dimensions[-1]  # (height, width)
    heatmap_upscaled = cv2.resize(heatmap, thumbnail_size, interpolation=cv2.INTER_LINEAR)
    overlay_mask_upscaled = cv2.resize(
        overlay_mask, thumbnail_size, interpolation=cv2.INTER_NEAREST
    )

    # Get the thumbnail
    thumbnail = np.array(wsi.get_thumbnail(thumbnail_size).convert("RGB"))

    # Blend the overlay mask with the thumbnail
    overlay = thumbnail.copy()
    overlay[overlay_mask_upscaled > 0] = [0, 255, 0]  # Highlight selected tiles in green
    blended_thumbnail = cv2.addWeighted(thumbnail, 0.7, overlay, 0.3, 0)

    return heatmap_upscaled, blended_thumbnail


def save_tiles(
    wsi,
    selected_indices,
    coordinates,
    output_tiles_dir,
    wsi_id,
    tile_size=256,
    level=0,
):
    """
    Save the selected tiles to the output directory.
    """
    mpp = wsi.properties.get("openslide.mpp-x", "nan")
    if mpp != "nan":
        mpp = float(mpp) * 1000
        mpp_str = f"{int(mpp):04d}"
    else:
        mpp_str = "nan"

    for i in selected_indices:
        coord_x, coord_y = coordinates[i][0], coordinates[i][1]
        tile = wsi.read_region((coord_x, coord_y), level, (tile_size, tile_size))
        tile.save(output_tiles_dir / f"{wsi_id}__x{coord_x}_y{coord_y}__{mpp_str}.png")


@click.command()
@click.option(
    "--csv-cptac-luad",
    default="/mnt/nas6/data/CPTAC/TCIA_CPTAC_LUAD_Pathology_Data_Table.csv",
    help="Path to CPTAC LUAD CSV file.",
)
@click.option("--raw-wsi-dir", help="Path to raw WSI directory.")
@click.option("--cptac-embeddings-path", help="Path to raw WSI directory.")
@click.option("--protonet-path", help="Path to Protonet model.")
@click.option(
    "--lunghist700-task", default="complete", help="Maximum number of tiles to process per WSI."
)
@click.option("--output-dir", help="Path to output tiles directory.")
@click.option("--n-wsi-max", default=32, help="Maximum number of WSIs to process per pattern.")
@click.option("--n-tiles-max", default=32, help="Maximum number of tiles to process per WSI.")
@click.option("--random-state", default=42, help="Random state for reproducibility.")
def main(
    csv_cptac_luad,
    raw_wsi_dir,
    cptac_embeddings_path,
    protonet_path,
    lunghist700_task,
    output_dir,
    n_wsi_max,
    n_tiles_max,
    random_state,
):
    """Simple CLI program to greet someone"""
    metadata_cptac = pd.read_csv(csv_cptac_luad).set_index(("Slide_ID"))
    metadata_cptac = metadata_cptac[metadata_cptac["Embedding_Medium"] == "FFPE"]
    protonet = ProtoNet.load(protonet_path)

    if random_state is not None:
        np.random.seed(random_state)

    output_dir = Path(output_dir).resolve()
    output_tiles_dir = output_dir / "tiles"
    output_heatmaps_dir = output_dir / "heatmaps"

    output_tiles_dir.mkdir(parents=True, exist_ok=True)
    output_heatmaps_dir.mkdir(parents=True, exist_ok=True)

    raw_wsi_dir = Path(raw_wsi_dir).resolve()
    cptac_embeddings_path = Path(cptac_embeddings_path).resolve()

    for pattern_name in pattern_list:
        if pattern_name == "Normal":
            wsi_ids = metadata_cptac[
                metadata_cptac["Specimen_Type"] == "normal_tissue"
            ].index.to_list()
        else:
            wsi_ids = list()
            for tumor_type in pattern_mapping_to_tumor_types[pattern_name]:
                wsi_ids.extend(
                    metadata_cptac[
                        metadata_cptac["Tumor_Histological_Type"] == tumor_type
                    ].index.tolist()
                )

        click.echo(f"Found {len(wsi_ids)} {pattern_name} WSIs.")

        if len(wsi_ids) > n_wsi_max:
            np.random.shuffle(wsi_ids)

        lunghist700_class = pattern_mapping_to_lunghist700_class[pattern_name]
        lunghist700_class_index = label_map_lunghist700[lunghist700_task][lunghist700_class]
        # lunghist700_class_index = 0

        output_tiles_dir_pattern = output_tiles_dir / pattern_name
        output_tiles_dir_pattern.mkdir(parents=True, exist_ok=True)

        output_heatmaps_dir_pattern = output_heatmaps_dir / pattern_name
        output_heatmaps_dir_pattern.mkdir(parents=True, exist_ok=True)

        n_wsi_used = 0
        for idx_wsi, wsi_id in tqdm(
            enumerate(wsi_ids), desc=f"Processing {pattern_name}", total=len(wsi_ids)
        ):
            if n_wsi_used > n_wsi_max:
                break
            try:
                embeddings_dict = load_hdf5(cptac_embeddings_path / f"{wsi_id}.h5")
            except FileNotFoundError as e:
                click.echo(f"File not found: {e}")
                continue

            embeddings = np.squeeze(embeddings_dict["datasets"]["features"])
            coordinates = np.squeeze(embeddings_dict["datasets"]["coords"])

            wsi = OpenSlide(raw_wsi_dir / f"{wsi_id}.svs")
            distances = compute_distances_to_protypes(embeddings, protonet)
            tile_indices = select_tiles(
                distances,
                lunghist700_class_index,
                random_state=random_state,  # maybe change since it is already used
                n_tiles_max=n_tiles_max,
            )
            if len(tile_indices) == 0:
                click.echo(f"No tiles selected for WSI {wsi_id}.")
                continue
            n_wsi_used += 1
            n_selected_tiles = len(tile_indices)
            click.echo(
                f"Selected {n_selected_tiles} tiles for WSI {wsi_id} with {pattern_name} pattern."
            )

            save_tiles(
                wsi=wsi,
                selected_indices=tile_indices,
                coordinates=coordinates,
                output_tiles_dir=output_tiles_dir_pattern,
                wsi_id=wsi_id,
                tile_size=256,
                level=0,
            )

            heatmaps, thumbnail = compute_heatmap_optimized(
                wsi,
                coordinates,
                -distances,
                tile_size=256,
                tile_level=0,
                rescale=True,
                selected_indices=tile_indices,
            )
            save_heatmaps_figure(
                output_heatmaps_dir_pattern,
                heatmaps,
                thumbnail,
                wsi_id,
                pattern_name,
                n_selected_tiles,
                label_map=label_map_lunghist700[lunghist700_task],
            )


if __name__ == "__main__":
    main()
