from functools import partial
import multiprocessing as mp
import os
from pathlib import Path
import re

import click
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()

# Example usage
LungHist700_path = os.getenv("LUNGHIST700_RAW_PATH")


def extract_magnification(filename):
    """Extract magnification from filename."""
    match = re.search(r"_([24]0)x_", filename)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Magnification not found in filename: {filename}")


def process_image(file, output_dir, magnification):
    """Process a single image with the given magnification."""
    filename = file.stem
    try:
        file_magnification = extract_magnification(filename)
        resampling_factor = magnification / file_magnification

        with Image.open(file) as img:
            # Resize the image based on desired magnification
            new_height = int(img.height * resampling_factor)
            new_width = int(img.width * resampling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            output_path = output_dir / f"{filename}.png"
            img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return False


@click.command()
@click.option("--raw-data-path", default=None, help="Path to LungHist700 raw dir.")
@click.option(
    "--output-dir", default="data/processed/LungHist700/LungHist700_20x", help="output dir"
)
@click.option("--magnification", default=20, help="Desired magnification.")
@click.option(
    "--num-workers", default=1, type=int, help="Number of worker processes."
)
def main(raw_data_path, output_dir, magnification, num_workers):
    if raw_data_path is None:
        raw_data_path = LungHist700_path

    raw_data_path = Path(raw_data_path)
    image_paths = list(raw_data_path.rglob("*.jpg"))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Found {len(image_paths)} images. Processing with {num_workers} workers...")

    # Create a partial function with fixed parameters
    process_func = partial(process_image, output_dir=output_dir, magnification=magnification)

    # Use multiprocessing to process images in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_func, image_paths),
                total=len(image_paths),
                desc="Resizing images",
            )
        )

    # Report results
    successful = results.count(True)
    print(f"Processed {successful} out of {len(image_paths)} images successfully.")


if __name__ == "__main__":
    main()
