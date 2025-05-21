from pathlib import Path

import click
import numpy as np
from openslide import OpenSlide
import pandas as pd
from PIL import Image, ImageDraw


def get_base_magnification(wsi):
    """Retrieve base magnification from WSI metadata or infer it from MPP."""
    # Check if magnification is available
    magnification_keys = [
        "aperio.AppMag",
        "openslide.objective-power",
        "hamamatsu.XResolution",
        "hamamatsu.ObjectiveMagnification",
    ]
    for key in magnification_keys:
        mag = wsi.properties.get(key)
        if mag:
            return float(mag)

    raise ValueError(
        "Magnification metadata is missing. "
        "Please ensure the WSI has magnification information or "
        "set `raise_error_mag` to False to attempt inference."
    )


def fetch_wsi(wsi_id, raw_wsi_dir):
    """Fetch WSI file path based on WSI ID."""
    wsi_path_matches = list(Path(raw_wsi_dir).rglob(f"{wsi_id}*.svs"))
    if len(wsi_path_matches) == 0:
        raise FileNotFoundError(f"No WSI found for {wsi_id} in {raw_wsi_dir}")
    elif len(wsi_path_matches) > 1:
        raise FileExistsError(f"Multiple WSIs found for {wsi_id} in {raw_wsi_dir}")
    return OpenSlide(wsi_path_matches[0])


def extract_contours(df):
    contours = []
    current = []
    for _, row in df.iterrows():
        if str(row["X_base"]).strip() == "X_base" and str(row["Y_base"]).strip() == "Y_base":
            if current:
                contours.append(current)
                current = []
            continue
        try:
            x = float(row["X_base"])
            y = float(row["Y_base"])
            if not (pd.isna(x) or pd.isna(y)):
                current.append((x, y))
        except Exception:
            continue
    if current:
        contours.append(current)
    return contours


@click.command()
@click.option(
    "--annotations-dir", type=click.Path(exists=True), help="Path to annotations directory."
)
@click.option("--raw-wsi-dir", type=click.Path(exists=True), help="Path to annotations directory.")
@click.option("--output-dir", type=click.Path(), help="Path to the output binary masks.")
@click.option("--mask-magnification", default=1.5, help="Magnification for output mask.")
def main(
    annotations_dir,
    raw_wsi_dir,
    output_dir,
    mask_magnification,
):
    annotation_paths = list(Path(annotations_dir).rglob("*.csv"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for annotation_path in annotation_paths:
        wsi_id = annotation_path.stem
        try:
            wsi = fetch_wsi(wsi_id, raw_wsi_dir)
        except (FileNotFoundError, FileExistsError) as e:
            print(f"{e}")
            continue

        annotations = pd.read_csv(annotation_path)
        annotations.columns = annotations.columns.str.strip()
        annotations_list = extract_contours(annotations)

        base_magnification = get_base_magnification(wsi)
        downsample_factor = base_magnification / mask_magnification
        mask_width = int(wsi.dimensions[0] / downsample_factor)
        mask_height = int(wsi.dimensions[1] / downsample_factor)

        # Create a blank mask
        mask = Image.new("L", (mask_width, mask_height), 0)
        draw = ImageDraw.Draw(mask)

        # Scale coordinates and draw polygon if enough points
        for contour in annotations_list:
            contour = np.array(contour)

            x_scaled = contour[:, 0] / downsample_factor
            y_scaled = contour[:, 1] / downsample_factor
            points = list(zip(x_scaled, y_scaled))

            if len(points) >= 3:
                draw.polygon(points, outline=255, fill=255)
                # for p in points:
                #     draw.circle(p, 10, outline=125, fill=125)
                print(f" Polygon mask created for {wsi_id} with {len(points)} points.")
            else:
                print(f" Not enough points to make a polygon for {wsi_id}.")
                continue

        # Save mask
        mask_path = output_dir / f"{wsi_id}.png"
        mask.save(mask_path)
        print(f"Mask saved to {mask_path}")


if __name__ == "__main__":
    main()
