from concurrent.futures import ProcessPoolExecutor
import functools
import multiprocessing as mp
import os
from pathlib import Path
import re

import click
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

load_dotenv()


def extract_magnification(filename):
    """Extract magnification from filename."""
    match = re.search(r"_([24]0)x_", filename)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Magnification not found in filename: {filename}")


def calculate_stride(image_dim, tile_size):
    """Calculate stride to align tiles symmetrically with borders."""
    if image_dim <= tile_size:
        return 0  # Single tile, no stride needed

    n_tiles = np.ceil(image_dim / tile_size)
    if n_tiles == 1:
        return 0  # Single tile, no stride needed
    total_stride_space = image_dim - tile_size * n_tiles
    stride = tile_size + total_stride_space // (n_tiles - 1)
    return int(stride)


def draw_styled_border(draw, left, top, right, bottom, style):
    """Draw styled borders (solid, dotted, dashed) on the image."""
    color = style["color"]
    width = style["width"]
    border_style = style["style"]

    if border_style == "solid":
        for w in range(width):
            draw.rectangle([left + w, top + w, right - w, bottom - w], outline=color)

    elif border_style == "dotted":
        step = 5  # Distance between dots
        for w in range(width):
            # Top border
            for x in range(left + w, right - w, step):
                draw.point((x, top + w), fill=color)
            # Bottom border
            for x in range(left + w, right - w, step):
                draw.point((x, bottom - w), fill=color)
            # Left border
            for y in range(top + w, bottom - w, step):
                draw.point((left + w, y), fill=color)
            # Right border
            for y in range(top + w, bottom - w, step):
                draw.point((right - w, y), fill=color)

    elif border_style == "dashed":
        dash_length = 10  # Length of dashes
        space_length = 5  # Space between dashes
        for w in range(width):
            # Top border
            for x in range(left + w, right - w, dash_length + space_length):
                draw.line([x, top + w, x + dash_length, top + w], fill=color, width=1)
            # Bottom border
            for x in range(left + w, right - w, dash_length + space_length):
                draw.line([x, bottom - w, x + dash_length, bottom - w], fill=color, width=1)
            # Left border
            for y in range(top + w, bottom - w, dash_length + space_length):
                draw.line([left + w, y, left + w, y + dash_length], fill=color, width=1)
            # Right border
            for y in range(top + w, bottom - w, dash_length + space_length):
                draw.line([right - w, y, right - w, y + dash_length], fill=color, width=1)


def load_metadata(input_folder):
    """Load metadata from the dataset."""
    label_mapping = {
        "aca_bd": 0,
        "aca_md": 1,
        "aca_pd": 2,
        "nor": 3,
        "scc_bd": 4,
        "scc_md": 5,
        "scc_pd": 6,
    }

    try:
        metadata = pd.read_csv(Path(input_folder) / "data/data.csv")

        metadata["filename"] = metadata.apply(
            lambda row: "_".join(
                [
                    str(row[col])
                    for col in ["superclass", "subclass", "resolution", "image_id"]
                    if pd.notna(row[col])
                ]
            ),
            axis=1,
        )
        metadata["class_name"] = metadata.apply(
            lambda row: f"{row['superclass']}_{row['subclass']}"
            if pd.notna(row["subclass"])
            else row["superclass"],
            axis=1,
        )

        metadata["label"] = metadata["class_name"].map(label_mapping)

        click.echo(f"Loaded metadata with {len(metadata)} entries")
        return metadata
    except Exception as e:
        click.echo(f"Failed to load metadata: {e}", err=True)
        return None


def create_tile_metadata(tiles_paths, metadata, output_folder):
    """Create metadata for the generated tiles."""
    if metadata is None:
        click.echo("No original metadata available, skipping tile metadata creation", err=True)
        return None

    rows = []

    for tile_path in tiles_paths:
        try:
            original_filename = tile_path.stem.split("_tile_")[0]
            matching_row = metadata[metadata["filename"] == original_filename]

            if not matching_row.empty:
                row = matching_row.iloc[0]
                rows.append(
                    {
                        "tile_id": tile_path.stem,
                        "patient_id": row.get("patient_id", "unknown"),
                        "superclass": row.get("superclass", "unknown"),
                        "subclass": row.get("subclass", "unknown"),
                        "resolution": row.get("resolution", "unknown"),
                        "image_id": row.get("image_id", "unknown"),
                        "class_name": row.get("class_name", "unknown"),
                        "label": row.get("label", -1),
                        "original_filename": original_filename,
                        "tile_path": str(tile_path),
                    }
                )
        except Exception as e:
            click.echo(f"Error processing tile {tile_path}: {e}", err=True)

    if not rows:
        click.echo("No metadata entries created for tiles", err=True)
        return None

    tile_metadata = pd.DataFrame(rows)
    metadata_path = output_folder / "metadata.csv"
    tile_metadata.to_csv(metadata_path, index=False)
    click.echo(f"Saved metadata for {len(tile_metadata)} tiles to {metadata_path}")

    return tile_metadata


def process_image(file, output_params):
    """Process a single image file to generate tiles."""
    tile_size = output_params["tile_size"]
    desired_magnification = output_params["desired_magnification"]
    generate_outlines = output_params["generate_outlines"]
    tiles_folder = output_params["tiles_folder"]
    outline_folder = output_params["outline_folder"]
    border_styles = output_params["border_styles"]

    try:
        filename = file.stem
        magnification = extract_magnification(filename)
        resampling_factor = desired_magnification / magnification

        tile_paths = []

        with Image.open(file) as img:
            # Resize the image based on desired magnification
            new_height = int(img.height * resampling_factor)
            new_width = int(img.width * resampling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Calculate strides for symmetrical tiling
            try:
                stride_x = calculate_stride(new_width, tile_size)
                stride_y = calculate_stride(new_height, tile_size)
            except ValueError:
                # Handle images smaller than tile_size
                stride_x = 0
                stride_y = 0

            # Create overlay for tile visualization if needed
            if generate_outlines:
                outlined_image = img.convert("RGBA")
                overlay = Image.new("RGBA", outlined_image.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)

            # Generate tiles
            x_positions = list(range(0, new_width - tile_size + 1, stride_x or tile_size))
            y_positions = list(range(0, new_height - tile_size + 1, stride_y or tile_size))

            # If image is smaller than tile_size in any dimension, use full image
            if not x_positions:
                x_positions = [0]
            if not y_positions:
                y_positions = [0]

            for i, left in enumerate(x_positions):
                for j, top in enumerate(y_positions):
                    right = min(left + tile_size, new_width)
                    bottom = min(top + tile_size, new_height)

                    # Crop the tile
                    tile = img.crop((left, top, right, bottom))

                    # Filter mostly blank tiles (contains <5% non-white pixels)
                    tile_array = np.array(tile)
                    if (np.mean(tile_array > 240) > 0.95) and (tile_array.shape[2] >= 3):
                        continue

                    # Save the tile
                    tile_id = f"tile_{i}_{j}"
                    tile_output_path = tiles_folder / f"{filename}_{tile_id}.png"
                    tile.save(tile_output_path, "PNG")
                    tile_paths.append(tile_output_path)

                    # Draw border on outlined image if required
                    if generate_outlines:
                        style_index = (i + j) % len(border_styles)
                        draw_styled_border(
                            draw, left, top, right, bottom, border_styles[style_index]
                        )

            # Save the outlined image if required
            if generate_outlines:
                outlined_image = Image.alpha_composite(outlined_image, overlay)
                outlined_image = outlined_image.convert("RGB")  # Convert back to RGB for saving
                outlined_image_path = outline_folder / f"{filename}_outlined.jpg"
                outlined_image.save(outlined_image_path, "JPEG")

        return tile_paths

    except Exception as e:
        click.echo(f"Error processing {file}: {e}", err=True)
        return []


@click.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing LungHist700 images",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for tiles and metadata",
)
@click.option("--tile-size", default=224, type=int, help="Size of tiles to extract (default: 224)")
@click.option(
    "--desired-magnification",
    default=20,
    type=int,
    help="Target magnification for tiles (10x or 20x)",
)
@click.option(
    "--generate-outlines", is_flag=True, help="Generate outlined images showing tile positions"
)
@click.option(
    "--save-metadata", is_flag=True, default=True, help="Generate and save metadata for tiles"
)
@click.option(
    "--num-workers",
    default=1,
    type=int,
    help="Number of parallel workers (default: number of CPU cores)",
)
def main(
    input_dir,
    output_dir,
    tile_size,
    desired_magnification,
    generate_outlines,
    save_metadata,
    num_workers,
):
    """
    Tile LungHist700 dataset with overlap and generate metadata.

    This script takes whole slide images from the LungHist700 dataset,
    resizes them to the desired magnification, and tiles them with overlap.
    It can also generate visualizations of the tiling pattern and create
    metadata for the tiles.
    """
    # Set up directories
    input_folder = Path(input_dir)
    output_folder = Path(output_dir)
    tiles_folder = output_folder / "tiles"
    outline_folder = output_folder / "outline"

    tiles_folder.mkdir(parents=True, exist_ok=True)

    if generate_outlines:
        outline_folder.mkdir(parents=True, exist_ok=True)

    # Define border styles for visualization
    border_styles = [
        {"color": "red", "width": 2, "style": "solid"},  # Solid red
        {"color": "blue", "width": 2, "style": "dotted"},  # Dotted blue
        {"color": "green", "width": 3, "style": "dashed"},  # Dashed green
        {"color": "yellow", "width": 2, "style": "solid"},  # Solid yellow
        {"color": "purple", "width": 3, "style": "dotted"},  # Dotted purple
    ]

    # Find all image files
    files = list(input_folder.rglob("*.jpg"))
    if not files:
        click.echo(f"No image files found in {input_folder}", err=True)
        return

    click.echo(f"Found {len(files)} images to process")
    click.echo(f"Using {num_workers} parallel workers")

    # Prepare parameters for parallel processing
    output_params = {
        "tile_size": tile_size,
        "desired_magnification": desired_magnification,
        "generate_outlines": generate_outlines,
        "tiles_folder": tiles_folder,
        "outline_folder": outline_folder,
        "border_styles": border_styles,
    }

    # Process images in parallel
    all_tile_paths = []
    with click.progressbar(length=len(files), label="Tiling images") as bar:
        process_func = functools.partial(process_image, output_params=output_params)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for tile_paths in executor.map(process_func, files):
                all_tile_paths.extend(tile_paths)
                bar.update(1)

    # Create metadata
    if save_metadata:
        click.echo("Loading metadata...")
        metadata = load_metadata(input_folder)

        if all_tile_paths:
            click.echo(f"Generated {len(all_tile_paths)} tiles. Creating metadata...")
            create_tile_metadata(all_tile_paths, metadata, output_folder)

    click.echo(f"Tiling complete! Tiles saved to {tiles_folder}")
    if generate_outlines:
        click.echo(f"Outlined images saved to {outline_folder}")


if __name__ == "__main__":
    main()
