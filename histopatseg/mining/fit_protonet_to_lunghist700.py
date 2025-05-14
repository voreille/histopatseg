import click
import numpy as np
import pandas as pd
import torch

from histopatseg.fewshot.protonet import ProtoNet

label_map_lunghist700 = {
    "luad_differentiation": {"aca_bd": 0, "aca_md": 1, "aca_pd": 2, "nor": 3},
    "nsclc_subtyping": {"aca": 0, "scc": 1, "nor": 2},
    "complete": {
        "aca_bd": 0,
        "aca_md": 1,
        "aca_pd": 2,
        "nor": 3,
        "scc_bd": 4,
        "scc_md": 5,
        "scc_pd": 6,
    },
}


def get_fitted_protonet_lunghist700(
    lunghist700_csv_path,
    lunghist700_embeddings_path,
    task="luad_differentiation",
):
    metadata = pd.read_csv(lunghist700_csv_path).set_index("tile_id")

    data = np.load(lunghist700_embeddings_path)
    embeddings = data["embeddings"]
    tile_ids = data["tile_ids"]

    embeddings_df = pd.DataFrame(
        {
            "tile_id": tile_ids,
            "embeddings": list(embeddings),
        }
    ).set_index("tile_id")

    df = pd.concat([embeddings_df, metadata], axis=1)

    if task == "luad_differentiation":
        print("Fitting ProtoNet for LUAD differentiation task")
        df = df[(df["superclass"] == "aca") | (df["superclass"] == "nor")]
        label_col = "class_name"

    if task == "complete":
        print("Fitting ProtoNet for complete differentiation task")
        label_col = "class_name"
    else:
        print("Fitting ProtoNet for NSCLC subtyping task")
        label_col = "superclass"

    embeddings_train = np.stack(df["embeddings"].values)
    labels_train = df[label_col].values
    labels_train = np.array([label_map_lunghist700[task][label] for label in labels_train])

    protonet = ProtoNet()
    protonet.fit(
        torch.tensor(embeddings_train, dtype=torch.float32),
        torch.tensor(labels_train, dtype=torch.long),
    )

    return protonet


@click.command()
@click.option("--lunghist700-csv-path", help="Path to LungHist700 embeddings.")
@click.option("--lunghist700-embeddings-path", help="Path to LungHist700 embeddings.")
@click.option("--output-path", help="Path to save the output ProtoNet")
@click.option(
    "--task",
    default="luad_differentiation",
    type=click.Choice(label_map_lunghist700.keys()),
    help="Task to perform. Options are 'luad_differentiation' so 4 classes aca_[diff_level] + normal or 'nsclc_subtyping'.",
)
def main(
    lunghist700_csv_path,
    lunghist700_embeddings_path,
    output_path,
    task,
):
    """Simple CLI program to greet someone"""
    print(f"Fitting ProtoNet for task: {task}")
    protonet = get_fitted_protonet_lunghist700(
        lunghist700_csv_path=lunghist700_csv_path,
        lunghist700_embeddings_path=lunghist700_embeddings_path,
        task=task,
    )
    print(f"ProtoNet fitted for task: {task}")

    protonet.save(output_path)


if __name__ == "__main__":
    main()
