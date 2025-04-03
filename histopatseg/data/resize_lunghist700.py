import os
from pathlib import Path
import re

import click
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()

# Example usage
LungHist700_path = os.getenv('LUNGHIST700_RAW_PATH')


def extract_magnification(filename):
    """Extract magnification from filename."""
    match = re.search(r"_([24]0)x_", filename)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Magnification not found in filename: {filename}")


@click.command()
@click.option("--data-path", default=None, help="Path to LungHist700 raw dir.")
@click.option("--output-dir",
              default="data/preprocessed/LungHist700_20x",
              help="output dir")
@click.option("--magnification", default=20, help="Desired magnification.")
def main(data_path, output_dir, magnification):
    if data_path is None:
        data_path = LungHist700_path
    data_path = Path(data_path)
    image_paths = list(data_path.rglob("*.jpg"))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    for file in tqdm(image_paths, desc="Resizing images"):
        filename = file.stem
        file_magnification = extract_magnification(filename)
        resampling_factor = magnification / file_magnification
        with Image.open(file) as img:
            # Resize the image based on desired magnification
            new_height = int(img.height * resampling_factor)
            new_width = int(img.width * resampling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            output_path = output_dir / f"{filename}.png"
            img.save(output_path)


if __name__ == "__main__":
    main()
