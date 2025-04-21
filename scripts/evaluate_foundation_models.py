#!/usr/bin/env python

import argparse
import json
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from histopatseg.evaluation.visualization import compare_embeddings, visualize_all_embeddings

# Add the project root to the Python path to import local modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def load_gigapath_embeddings(output_dir, prefix="lunghist700_global_pool_True_"):
    """Load GigaPath embeddings"""
    output_dir = Path(output_dir)

    # Load slide-level WSI embeddings
    wsi_path = output_dir / f"{prefix}wsi_embeddings.h5"
    with h5py.File(wsi_path, "r") as f:
        wsi_embeddings = np.array(f["embeddings"])

    # Load metadata
    metadata_path = output_dir / f"{prefix}metadata.csv"
    metadata = pd.read_csv(metadata_path)

    return wsi_embeddings, metadata


def load_uni2_embeddings(path):
    """Load UNI2 embeddings (adapt as needed)"""
    # This is a placeholder - update with actual loading code
    pass


def load_hoptimus_embeddings(path):
    """Load H-Optimus embeddings (adapt as needed)"""
    # This is a placeholder - update with actual loading code
    pass


def enrich_metadata(metadata, original_metadata_path):
    """Add class and other information from original metadata"""
    original_metadata = pd.read_csv(original_metadata_path)

    # Create mappings
    filename_to_class = dict(zip(original_metadata["filename"], original_metadata["class_name"]))
    filename_to_superclass = dict(
        zip(original_metadata["filename"], original_metadata["superclass"])
    )
    filename_to_resolution = dict(
        zip(original_metadata["filename"], original_metadata["resolution"])
    )

    # Add columns to metadata
    image_names = metadata["image_name"].tolist()
    classes = []
    superclasses = []
    resolutions = []

    for name in image_names:
        if name in filename_to_class:
            classes.append(filename_to_class[name])
            superclasses.append(filename_to_superclass[name])
            resolutions.append(filename_to_resolution[name])
        else:
            # Try to parse from filename (format: superclass_subclass_resolution_id)
            parts = name.split("_")
            if len(parts) >= 4:
                classes.append(f"{parts[0]}_{parts[1]}")
                superclasses.append(parts[0])
                resolutions.append(parts[2])
            else:
                print(f"Warning: {name} not found in original metadata.")
                classes.append("unknown")
                superclasses.append("unknown")
                resolutions.append("unknown")

    metadata["class_name"] = classes
    metadata["superclass"] = superclasses
    metadata["resolution"] = resolutions

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize foundation model embeddings"
    )
    parser.add_argument("--dataset", default="LungHist700", help="Dataset name")
    parser.add_argument(
        "--original-metadata", type=str, required=True, help="Path to original dataset metadata"
    )
    parser.add_argument("--gigapath-dir", type=str, help="Path to GigaPath embeddings directory")
    parser.add_argument("--uni2-dir", type=str, help="Path to UNI2 embeddings directory")
    parser.add_argument("--hoptimus-dir", type=str, help="Path to H-Optimus embeddings directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument(
        "--compare", action="store_true", help="Generate comparison plots between models"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store results
    results = {
        "dataset": args.dataset,
        "models": {},
        "visualizations": {},
    }

    # Store embeddings and metadata for comparison
    all_embeddings = []
    all_metadata = []
    model_names = []

    # Process GigaPath embeddings if provided
    if args.gigapath_dir:
        print("Processing GigaPath embeddings...")
        gigapath_dir = Path(args.gigapath_dir)

        # Process embeddings with global pooling True
        try:
            gigapath_embeddings_true, gigapath_metadata_true = load_gigapath_embeddings(
                gigapath_dir, prefix="lunghist700_global_pool_True_"
            )
            gigapath_metadata_true = enrich_metadata(
                gigapath_metadata_true, args.original_metadata
            )

            # Generate visualizations
            gigapath_true_results = visualize_all_embeddings(
                gigapath_embeddings_true,
                gigapath_metadata_true,
                output_dir / "gigapath" / "global_pool_true",
                "GigaPath-GlobalPool",
                prefix=f"{args.dataset.lower()}_",
            )

            results["models"]["gigapath_global_pool_true"] = {
                "shape": gigapath_embeddings_true.shape,
                "visualizations": gigapath_true_results,
            }

            # Store for comparison
            all_embeddings.append(gigapath_embeddings_true)
            all_metadata.append(gigapath_metadata_true)
            model_names.append("GigaPath-GlobalPool")
        except Exception as e:
            print(f"Error processing GigaPath GlobalPool embeddings: {e}")

        # Process embeddings with global pooling False
        try:
            gigapath_embeddings_false, gigapath_metadata_false = load_gigapath_embeddings(
                gigapath_dir, prefix="lunghist700_global_pool_False_"
            )
            gigapath_metadata_false = enrich_metadata(
                gigapath_metadata_false, args.original_metadata
            )

            # Generate visualizations
            gigapath_false_results = visualize_all_embeddings(
                gigapath_embeddings_false,
                gigapath_metadata_false,
                output_dir / "gigapath" / "global_pool_false",
                "GigaPath-NoGlobalPool",
                prefix=f"{args.dataset.lower()}_",
            )

            results["models"]["gigapath_global_pool_false"] = {
                "shape": gigapath_embeddings_false.shape,
                "visualizations": gigapath_false_results,
            }

            # Store for comparison
            all_embeddings.append(gigapath_embeddings_false)
            all_metadata.append(gigapath_metadata_false)
            model_names.append("GigaPath-NoGlobalPool")
        except Exception as e:
            print(f"Error processing GigaPath NoGlobalPool embeddings: {e}")

    # Add similar blocks for UNI2 and H-Optimus when implemented

    # Generate comparison plots if requested
    if args.compare and len(all_embeddings) > 1:
        print("Generating comparison plots...")

        # Comparison by superclass
        comparison_fig = compare_embeddings(
            all_embeddings,
            all_metadata,
            model_names,
            dim_reduction_methods=["tsne", "umap", "pca"],
            color_by="superclass",
            save_path=output_dir / f"{args.dataset.lower()}_model_comparison_superclass.png",
        )
        plt.close(comparison_fig)

        # Comparison by class
        comparison_fig = compare_embeddings(
            all_embeddings,
            all_metadata,
            model_names,
            dim_reduction_methods=["tsne", "umap", "pca"],
            color_by="class_name",
            save_path=output_dir / f"{args.dataset.lower()}_model_comparison_class.png",
        )
        plt.close(comparison_fig)

        # Comparison by resolution
        comparison_fig = compare_embeddings(
            all_embeddings,
            all_metadata,
            model_names,
            dim_reduction_methods=["tsne", "umap", "pca"],
            color_by="resolution",
            save_path=output_dir / f"{args.dataset.lower()}_model_comparison_resolution.png",
        )
        plt.close(comparison_fig)

        results["visualizations"]["comparisons"] = {
            "superclass": str(
                output_dir / f"{args.dataset.lower()}_model_comparison_superclass.png"
            ),
            "class": str(output_dir / f"{args.dataset.lower()}_model_comparison_class.png"),
            "resolution": str(
                output_dir / f"{args.dataset.lower()}_model_comparison_resolution.png"
            ),
        }

    # Save results summary
    with open(output_dir / f"{args.dataset.lower()}_results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
