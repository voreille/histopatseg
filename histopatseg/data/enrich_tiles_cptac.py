from pathlib import Path

import click
from histopreprocessing.features.foundation_models import load_model
import numpy as np
from openslide import OpenSlide
import pandas as pd
import torch
from torch.nn.functional import normalize

from histopatseg.data.compute_embeddings_tcga_ut import load_hdf5
from histopatseg.fewshot.protonet import ProtoNet
from histopatseg.utils import get_device

label_map_lunghist700 = {"aca_bd": 0, "aca_md": 1, "aca_pd": 2, "nor": 3}

pattern_list = [
    "Acinar adenocarcinoma",  # 142
    "Solid adenocarcinoma",  # 43
    "Papillary adenocarcinoma",  # 32
    "Micropapillary adenocarcinoma",  # 9
    "Lepidic adenocarcinoma",  # 6
]
pattern_mapping_to_lunghist700_class = {
    "Acinar adenocarcinoma": "aca_md",
    "Solid adenocarcinoma": "aca_pd",
    "Papillary adenocarcinoma": "aca_md",
    "Micropapillary adenocarcinoma": "aca_pd",
    "Lepidic adenocarcinoma": "aca_bd",
}


def get_fitted_protonet_lunghist700(
    lunghist700_csv_path,
    lunghist700_embeddings_path,
):
    metadata = pd.read_csv(lunghist700_csv_path).set_index("tile_id")

    data = np.load(lunghist700_embeddings_path)
    embeddings = data["embeddings"]
    tile_ids = data["tile_ids"]

    embeddings_df = pd.DataFrame(
        {
            "tile_id": tile_ids,
            "embeddings": list(embeddings),  # Add embeddings as a column
        }
    ).set_index("tile_id")
    df = pd.concat([embeddings_df, metadata], axis=1)

    df_filtered = df[(df["superclass"] == "aca") | (df["superclass"] == "nor")]

    embeddings_train = np.stack(df_filtered["embeddings"].values)
    labels_train = df_filtered["class_name"].values
    labels_train = np.array([label_map_lunghist700[label] for label in labels_train])

    protonet = ProtoNet()
    protonet.fit(
        torch.tensor(embeddings_train, dtype=torch.float32),
        torch.tensor(labels_train, dtype=torch.long),
    )

    return protonet


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


def select_tiles(distances, lunghist700_class_index, n_tiles_max=100, random_state=42):
    """
    Select tiles based on the distance to the prototype and limit the number of tiles randomly.
    """
    if random_state is not None:
        np.random.seed(random_state)

    distances_class = distances[:, lunghist700_class_index]

    # Get the minimum distance to all other class prototypes for each tile
    distances_other_classes = np.min(
        np.delete(distances, lunghist700_class_index, axis=1),
        axis=1,
    )

    # Select indices where the distance to the target class is smaller
    selected_indices = np.where(distances_class < distances_other_classes)[0]

    # Randomly sample up to n_tiles_max indices
    if len(selected_indices) > n_tiles_max:
        selected_indices = np.random.choice(selected_indices, n_tiles_max, replace=False)

    return selected_indices


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
@click.option("--lunghist700-csv-path", help="Path to LungHist700 embeddings.")
@click.option("--lunghist700-embeddings-path", help="Path to LungHist700 embeddings.")
@click.option("--output-tiles-dir", help="Path to output tiles directory.")
@click.option("--n-wsi-max", default=32, help="Maximum number of WSIs to process per pattern.")
@click.option("--n-tiles-max", default=32, help="Maximum number of tiles to process per WSI.")
@click.option("--output-embeddings-dir", help="Path to output embeddings directory.")
@click.option("--gpu-id", default=0, help="GPU ID to use.")
@click.option("--batch-size", default=8, help="Batch size for processing images.")
@click.option("--random-state", default=42, help="Random state for reproducibility.")
def main(
    csv_cptac_luad,
    raw_wsi_dir,
    cptac_embeddings_path,
    lunghist700_csv_path,
    lunghist700_embeddings_path,
    output_tiles_dir,
    n_wsi_max,
    n_tiles_max,
    output_embeddings_dir,
    gpu_id,
    batch_size,
    random_state,
):
    """Simple CLI program to greet someone"""
    metadata_cptac = pd.read_csv(csv_cptac_luad).set_index(("Slide_ID"))
    protonet = get_fitted_protonet_lunghist700(
        lunghist700_csv_path=lunghist700_csv_path,
        lunghist700_embeddings_path=lunghist700_embeddings_path,
    )
    # model, preprocess, embedding_dim, _ = load_model(
    #     model_name="UNI2",
    #     device=torch.device("cuda:0"),
    # )
    output_tiles_dir = Path(output_tiles_dir).resolve()

    raw_wsi_dir = Path(raw_wsi_dir).resolve()
    cptac_embeddings_path = Path(cptac_embeddings_path).resolve()
    lunghist700_csv_path = Path(lunghist700_csv_path).resolve()
    lunghist700_embeddings_path = Path(lunghist700_embeddings_path).resolve()

    for pattern_name in pattern_list:
        wsi_ids = metadata_cptac[metadata_cptac["Tumor_Histological_Type"] == pattern_name].index

        lunghist700_class = pattern_mapping_to_lunghist700_class[pattern_name]
        lunghist700_class_index = label_map_lunghist700[lunghist700_class]

        output_tiles_dir_pattern = output_tiles_dir / pattern_name
        output_tiles_dir_pattern.mkdir(parents=True, exist_ok=True)

        for idx_wsi, wsi_id in enumerate(wsi_ids):
            if idx_wsi >= n_wsi_max:
                break
            embeddings_dict = load_hdf5(cptac_embeddings_path / f"{wsi_id}.h5")
            embeddings = np.squeeze(embeddings_dict["datasets"]["features"])
            coordinates = np.squeeze(embeddings_dict["datasets"]["coords"])

            wsi = OpenSlide(raw_wsi_dir / f"{wsi_ids[0]}.svs")
            distances = compute_distances_to_protypes(embeddings, protonet)
            tile_indices = select_tiles(
                distances, lunghist700_class_index, random_state=random_state
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


if __name__ == "__main__":
    main()
