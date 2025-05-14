from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from histopatseg.data.compute_embeddings_tcga_ut import load_hdf5
from histopatseg.fewshot.protonet import ProtoNet

label_map = {"Lung_adenocarcinoma": 0, "Lung_squamous_cell_carcinoma": 1}


def get_fitted_protonet(
    tcga_ut_embeddings_path,
    n_wsi=32,
):
    hdf5_attributes = load_hdf5(tcga_ut_embeddings_path)
    embeddings = hdf5_attributes["datasets"]["embeddings"]

    labels = hdf5_attributes["datasets"]["labels"]
    labels = [label.decode("utf-8") for label in labels]

    wsi_ids = hdf5_attributes["datasets"]["wsi_ids"]
    wsi_ids = [Path(wsi_id.decode("utf-8")).name for wsi_id in wsi_ids]

    image_ids = hdf5_attributes["datasets"]["image_ids"]
    image_ids = [image_id.decode("utf-8") for image_id in image_ids]

    metadata_df = pd.DataFrame(
        {
            "label": labels,
            "wsi_id": wsi_ids,
            "image_ids": image_ids,
            "mpp": [
                float(image_name.split("/")[-1].split("_")[-1]) / 1000 for image_name in image_ids
            ],
            "embeddings": list(embeddings),  # Add embeddings as a column
            "numeric_label": [label_map[label] for label in labels],
        },
    ).set_index("image_ids")
    filtered_df = metadata_df[(metadata_df["mpp"] >= 0.45) & (metadata_df["mpp"] <= 0.55)]

    # Initialize an empty list to store the selected rows
    selected_rows = []

    # Group by class labels
    for label, group in filtered_df.groupby("label"):
        # Get unique WSI IDs for the current class
        unique_wsi_ids = group["wsi_id"].unique()

        # Randomly shuffle the WSI IDs
        np.random.shuffle(unique_wsi_ids)
        print(f"label: {label}, n_wsi: {len(unique_wsi_ids)}")

        # Select up to n_wsi WSI IDs
        selected_wsi_ids = unique_wsi_ids[:n_wsi]

        # Filter rows corresponding to the selected WSI IDs
        selected_rows.append(group[group["wsi_id"].isin(selected_wsi_ids)])

    # Concatenate the selected rows into a single DataFrame
    result_df = pd.concat(selected_rows)

    embeddings_train = np.stack(result_df["embeddings"].values)
    labels_train = result_df["numeric_label"].values

    protonet = ProtoNet()
    protonet.fit(
        torch.tensor(embeddings_train, dtype=torch.float32),
        torch.tensor(labels_train, dtype=torch.long),
    )

    return protonet


@click.command()
@click.option("--tcga-ut-embeddings-path", help="Path to LungHist700 embeddings.")
@click.option("--output-path", help="Path to save the output ProtoNet")
@click.option("--n-wsi", default=32, help="Number of WSIs to sample per class")
def main(
    tcga_ut_embeddings_path,
    output_path,
    n_wsi=32,
):
    """Simple CLI program to greet someone"""
    protonet = get_fitted_protonet(tcga_ut_embeddings_path=tcga_ut_embeddings_path, n_wsi=n_wsi)

    protonet.save(output_path)


if __name__ == "__main__":
    main()
