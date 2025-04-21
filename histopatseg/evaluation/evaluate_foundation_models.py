from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd

from histopatseg.evaluation.utils import (
    aggregate_tile_embeddings,
    run_cross_validation,
    save_evaluation_results,
)


@click.command()
@click.option(
    "--embeddings-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the embeddings directory.",
)
@click.option(
    "--metadata-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the embeddings directory.",
)
@click.option("--task", default="class_name", help="Task to evaluate. class_name or superclass.")
@click.option("--superclass-to-keep", default="all", help="Superclasses to keep.")
@click.option("--aggregation-method", default="none", help="Aggregation method.")
@click.option("--output-dir", default="./results", help="Directory to save results.")
@click.option("--n-splits", default=4, help="Number of splits for cross-validation.")
def main(
    embeddings_path,
    metadata_path,
    task,
    superclass_to_keep,
    aggregation_method,
    output_dir,
    n_splits,
):
    """Simple CLI program to greet someone"""
    embeddings_path = Path(embeddings_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    output_dir = Path(output_dir).resolve()

    model_name = embeddings_path.stem.split("_embeddings")[0]

    # Load the metadata
    if aggregation_method == "centercrop":
        metadata = pd.read_csv(metadata_path).set_index("filename")
    else:
        metadata = pd.read_csv(metadata_path).set_index("tile_id")

    data = np.load(embeddings_path)
    embeddings = data["embeddings"]
    tile_ids = data["tile_ids"]
    embedding_dim = data["embedding_dim"]

    # Check if all embedding tile_ids are in the metadata index
    missing_ids = [id for id in tile_ids if id not in metadata.index]
    if missing_ids:
        print(f"Warning: {len(missing_ids)} tile_ids from embeddings are not in metadata")
        print(f"First few missing IDs: {missing_ids[:5]}")

    metadata = metadata.reindex(tile_ids)
    metadata["subclass"] = metadata.apply(
        lambda row: row["superclass"]
        if pd.isna(row["subclass"]) and row["superclass"] == "nor"
        else row["subclass"],
        axis=1,
    )
    # Print basic information
    click.echo(f"Loaded {len(embeddings)} embeddings with dimensionality {embeddings.shape[1]}")
    click.echo(f"Embedding dimension from model: {embedding_dim}")

    if aggregation_method == "average":
        embeddings, metadata = aggregate_tile_embeddings(
            embeddings, tile_ids, metadata, group_by="original_filename"
        )

    if superclass_to_keep != "all":
        filter_condition = f"superclass == '{superclass_to_keep}'"
    else:
        filter_condition = None

    summary, results, fold_predictions, confusion_matrices = run_cross_validation(
        embeddings,
        metadata,
        "patient_id",
        task,
        n_splits=n_splits,
        verbose=True,
        filter_condition=filter_condition,
    )
    # Save configuration for reproducibility
    config = {
        "model_name": model_name,
        "task": task,
        "superclass_to_keep": superclass_to_keep,
        "aggregation_method": aggregation_method,
        "n_splits": n_splits,
        "embeddings_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results
    summary_path, results_path = save_evaluation_results(
        summary, results, fold_predictions, confusion_matrices, config, output_dir
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_path.name}")
    print(f"Detailed results: {results_path.name}")


if __name__ == "__main__":
    main()
