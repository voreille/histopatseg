from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_embeddings(
    reduced_data,
    metadata,
    color_by="superclass",
    method_name="t-SNE",
    figsize=(12, 10),
    title=None,
    save_path=None,
    palette_name="tab10",
):
    """
    Visualize pre-computed reduced embeddings with various metadata labelings.

    Args:
        reduced_data: numpy array of reduced embeddings
        metadata: pandas DataFrame containing metadata columns
        color_by: column in metadata to use for coloring points
        method_name: name of reduction method (for axis labels)
        figsize: size of the figure (width, height)
        title: custom title (if None, generated automatically)
        save_path: path to save figure (optional)
        palette_name: seaborn color palette name

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create dataframe for plotting
    plot_df = pd.DataFrame({
        'x': reduced_data[:, 0], 
        'y': reduced_data[:, 1],
        'label': metadata[color_by] if color_by in metadata.columns else ['Unknown'] * len(reduced_data)
    })
    
    # Get unique labels and sort them for consistent colors
    sorted_labels = sorted(plot_df['label'].unique())
    
    # Generate color palette
    palette = sns.color_palette(palette_name, n_colors=len(sorted_labels))
    
    # Create scatter plot
    sns.scatterplot(
        x='x', 
        y='y',
        hue='label',
        hue_order=sorted_labels,
        palette=palette,
        data=plot_df,
        alpha=0.7,
        s=30,
        ax=ax
    )
    
    # Set title and labels
    if title is None:
        title = f"{method_name} Visualization (Colored by {color_by})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=12)
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=12)
    
    # Improve legend
    legend = ax.legend(
        title=color_by.capitalize(),
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=10
    )
    legend.get_title().set_fontsize(12)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_embeddings(
    embedding_sets,
    metadata_sets,
    model_names,
    dim_reduction_methods=["tsne", "umap", "pca"],
    color_by="superclass",
    figsize=(18, 15),
    save_path=None,
):
    """
    Create a grid comparison of different embedding models and dimensionality reduction methods.

    Args:
        embedding_sets: List of embedding arrays from different models
        metadata_sets: List of metadata DataFrames for each model
        model_names: List of model names (e.g., ['GigaPath', 'UNI2', 'H-Optimus'])
        dim_reduction_methods: List of dimensionality reduction methods to use
        color_by: Metadata column to use for coloring
        figsize: Figure size
        save_path: Path to save the combined figure

    Returns:
        matplotlib figure
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Try to import UMAP
    try:
        import umap

        have_umap = True
    except ImportError:
        have_umap = False
        if "umap" in dim_reduction_methods:
            dim_reduction_methods.remove("umap")
            print("UMAP not available, skipping UMAP visualization")

    # Create subplot grid
    n_models = len(model_names)
    n_methods = len(dim_reduction_methods)

    fig, axes = plt.subplots(n_models, n_methods, figsize=figsize)

    # Compute reduced embeddings and create plots
    for i, (embeddings, metadata, model_name) in enumerate(
        zip(embedding_sets, metadata_sets, model_names)
    ):
        for j, method in enumerate(dim_reduction_methods):
            ax = axes[i, j] if n_models > 1 else axes[j]

            # Apply dimensionality reduction
            if method == "tsne":
                reducer = TSNE(
                    n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1)
                )
                reduced_data = reducer.fit_transform(embeddings)
                method_name = "t-SNE"
            elif method == "pca":
                reducer = PCA(n_components=2, random_state=42)
                reduced_data = reducer.fit_transform(embeddings)
                method_name = "PCA"
            elif method == "umap" and have_umap:
                reducer = umap.UMAP(n_components=2, random_state=42)
                reduced_data = reducer.fit_transform(embeddings)
                method_name = "UMAP"

            # Create plot DataFrame
            plot_df = pd.DataFrame(
                {"x": reduced_data[:, 0], "y": reduced_data[:, 1], "label": metadata[color_by]}
            )

            # Get unique labels and sort them
            sorted_labels = sorted(plot_df["label"].unique())
            palette = dict(zip(sorted_labels, sns.color_palette("tab10", len(sorted_labels))))

            # Create scatter plot
            sns.scatterplot(
                x="x",
                y="y",
                hue="label",
                hue_order=sorted_labels,
                palette=palette,
                data=plot_df,
                alpha=0.7,
                ax=ax,
            )

            # Set title and labels
            title = f"{model_name} - {method_name}"
            ax.set_title(title)
            ax.set_xlabel(f"{method_name} Dim 1")
            ax.set_ylabel(f"{method_name} Dim 2")
            ax.legend().remove()  # Remove individual legends

    # Add a single legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title=color_by.capitalize())

    plt.tight_layout()
    fig.subplots_adjust(right=0.85)  # Make room for the legend

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def visualize_all_embeddings(
    embeddings_data,
    metadata_df,
    output_dir,
    model_name,
    prefix="",
    color_columns=["superclass", "class_name", "resolution"],
):
    """
    Generate all visualizations for a single model's embeddings.

    Args:
        embeddings_data: Numpy array of embeddings
        metadata_df: DataFrame with metadata
        output_dir: Directory to save outputs
        model_name: Name of the model (e.g., 'GigaPath')
        prefix: Prefix for filenames
        color_columns: Columns to use for coloring points

    Returns:
        Dictionary of paths to saved images
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store paths to generated visualizations
    result_paths = {}

    # Compute dimensionality reductions once
    print(f"Computing dimensionality reductions for {model_name}...")

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_data) - 1))
    tsne_result = tsne.fit_transform(embeddings_data)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(embeddings_data)

    # UMAP if available
    try:
        import umap

        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        umap_result = umap_reducer.fit_transform(embeddings_data)
        have_umap = True
    except ImportError:
        print("UMAP not installed. Install with 'pip install umap-learn'")
        have_umap = False

    # Create visualizations for each color column and dimensionality reduction method
    for color_col in color_columns:
        if color_col in metadata_df.columns:
            print(f"Creating visualizations by {color_col}...")

            # t-SNE
            fig = plot_embeddings(
                tsne_result,
                metadata_df,
                color_by=color_col,
                method_name="t-SNE",
                title=f"{model_name} t-SNE Embeddings by {color_col}",
                save_path=output_dir / f"{prefix}{model_name.lower()}_tsne_{color_col}.png",
            )
            plt.close(fig)
            result_paths[f"tsne_{color_col}"] = str(
                output_dir / f"{prefix}{model_name.lower()}_tsne_{color_col}.png"
            )

            # PCA
            fig = plot_embeddings(
                pca_result,
                metadata_df,
                color_by=color_col,
                method_name="PCA",
                title=f"{model_name} PCA Embeddings by {color_col}",
                save_path=output_dir / f"{prefix}{model_name.lower()}_pca_{color_col}.png",
            )
            plt.close(fig)
            result_paths[f"pca_{color_col}"] = str(
                output_dir / f"{prefix}{model_name.lower()}_pca_{color_col}.png"
            )

            # UMAP if available
            if have_umap:
                fig = plot_embeddings(
                    umap_result,
                    metadata_df,
                    color_by=color_col,
                    method_name="UMAP",
                    title=f"{model_name} UMAP Embeddings by {color_col}",
                    save_path=output_dir / f"{prefix}{model_name.lower()}_umap_{color_col}.png",
                )
                plt.close(fig)
                result_paths[f"umap_{color_col}"] = str(
                    output_dir / f"{prefix}{model_name.lower()}_umap_{color_col}.png"
                )

    print(f"All visualizations for {model_name} saved to {output_dir}")
    return result_paths
