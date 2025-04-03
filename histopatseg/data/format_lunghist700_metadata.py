import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

load_dotenv()

project_dir = Path(__file__).parents[2].resolve()
data_path = Path(os.getenv('LUNGHIST700_RAW_PATH'))
output_path = project_dir / "data/processed/LungHist700/metadata.csv"


def format_metadata(metadata):
    metadata['filename'] = metadata.apply(
        lambda row: "_".join([
            str(row[col])
            for col in ['superclass', 'subclass', 'resolution', 'image_id']
            if pd.notna(row[col])
        ]),
        axis=1,
    )
    metadata['class_name'] = metadata.apply(
        lambda row: f"{row['superclass']}_{row['subclass']}"
        if pd.notna(row['subclass']) else row['superclass'],
        axis=1,
    )

    return metadata


def main():
    metadata = pd.read_csv(data_path / "data/data.csv")
    format_metadata(metadata).to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
