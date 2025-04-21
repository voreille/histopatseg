from collections import defaultdict
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def aggregate_tile_embeddings(embeddings, tile_ids, metadata, group_by="original_filename"):
    """
    Aggregate tile-level embeddings into image-level embeddings by averaging.

    Parameters:
    -----------
    embeddings : numpy.ndarray
        Array of embeddings with shape (n_tiles, embedding_dim)
    tile_ids : list or numpy.ndarray
        List of tile IDs corresponding to the embeddings
    metadata : pandas.DataFrame
        DataFrame containing metadata for the tiles, indexed by tile_id
    group_by : str, default='original_filename'
        Column in metadata to group by (typically 'original_filename')

    Returns:
    --------
    tuple: (aggregated_embeddings, aggregated_metadata)
        - aggregated_embeddings: numpy.ndarray with shape (n_images, embedding_dim)
        - aggregated_metadata: pandas.DataFrame with image-level metadata
    """
    # Create a DataFrame with embeddings and tile_ids
    embedding_df = pd.DataFrame(
        embeddings,  # The embedding values
        index=tile_ids,  # Use tile_ids as the index
    )

    # Merge with metadata (metadata should be indexed by tile_id)
    aligned_metadata = metadata.reindex(tile_ids)
    merged_df = embedding_df.join(aligned_metadata)

    # Verify the group_by column exists
    if group_by not in merged_df.columns:
        raise ValueError(f"Column '{group_by}' not found in metadata. Cannot group embeddings.")

    # Identify metadata columns (non-numeric columns after the embeddings)
    embedding_cols = embedding_df.columns
    metadata_cols = [col for col in merged_df.columns if col not in embedding_cols]

    # Build aggregation dictionary
    agg_dict = {
        # Average all embedding columns
        **{col: "mean" for col in embedding_cols},
        # For metadata columns: take first value for categorical, count for one column
        **{col: "first" for col in metadata_cols},
    }

    # Use the last metadata column for counting tiles
    count_col = metadata_cols[-1] if metadata_cols else None
    if count_col:
        agg_dict[count_col] = "count"

    # Group by the specified column and calculate aggregations
    aggregated_df = merged_df.groupby(group_by).agg(agg_dict)

    # Rename the count column
    if count_col:
        aggregated_df.rename(columns={count_col: "tile_count"}, inplace=True)

    # Extract aggregated embeddings and metadata
    aggregated_embeddings = aggregated_df[embedding_cols].values
    aggregated_metadata = aggregated_df.drop(columns=embedding_cols)

    print(
        f"Aggregated {len(embedding_df)} individual tile embeddings into {len(aggregated_df)} {group_by}-level embeddings"
    )

    return aggregated_embeddings, aggregated_metadata


def evaluate_classifier(X_train, y_train, X_test, y_test, clf_type="knn", n_neighbors=5):
    """Train and evaluate a classifier on embeddings.

    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_test: Test embeddings
        y_test: Test labels
        clf_type: Type of classifier ('knn' or 'linear')
        n_neighbors: Number of neighbors for kNN

    Returns:
        Dictionary with performance metrics, trained classifier, and predictions
    """
    start_time = time.time()

    # Create classifier
    if clf_type == "knn":
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors)),
            ]
        )
    else:  # linear
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        C=1.0,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )

    # Train classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "train_time": time.time() - start_time,
    }

    # Calculate per-class metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    for class_name, class_metrics in report.items():
        if isinstance(class_metrics, dict):  # Skip averages
            for metric_name, value in class_metrics.items():
                metrics[f"class_{class_name}_{metric_name}"] = value

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return metrics, clf, y_pred, cm


def custom_balanced_group_kfold(X, y, groups, n_splits=5):
    """Ensure all classes appear in each fold"""
    unique_classes = np.unique(y)
    unique_groups = np.unique(groups)

    # Create a mapping from string labels to integers if needed
    if not np.issubdtype(unique_classes.dtype, np.number):
        class_mapping = {label: i for i, label in enumerate(unique_classes)}
        y_numeric = np.array([class_mapping[label] for label in y])
    else:
        y_numeric = y

    # Group patients by class
    class_to_groups = {c: [] for c in unique_classes}
    for group in unique_groups:
        group_mask = groups == group
        group_classes = y[group_mask]

        # Count occurrences of each class in this group
        if not np.issubdtype(unique_classes.dtype, np.number):
            # For string labels, use value_counts
            counts = pd.Series(group_classes).value_counts()
            most_common_class = counts.idxmax()
        else:
            # For numeric labels, use bincount
            most_common_class = np.bincount(y_numeric[group_mask]).argmax()
            most_common_class = unique_classes[most_common_class]

        class_to_groups[most_common_class].append(group)

    # Create folds ensuring each class is represented
    folds = [[] for _ in range(n_splits)]
    for cls, cls_groups in class_to_groups.items():
        np.random.shuffle(cls_groups)
        for i, group in enumerate(cls_groups):
            fold_idx = i % n_splits
            folds[fold_idx].append(group)

    # Generate train/test indices for each fold
    for i in range(n_splits):
        test_groups = folds[i]
        test_mask = np.isin(groups, test_groups)
        test_indices = np.where(test_mask)[0]
        train_indices = np.where(~test_mask)[0]
        yield train_indices, test_indices


def run_cross_validation(
    embeddings,
    metadata,
    group_col,
    target_col,
    n_splits=5,
    classifiers=None,
    verbose=True,
    filter_condition=None,
):
    """Run cross-validation for embeddings evaluation.

    Args:
        embeddings: Feature vectors
        metadata: DataFrame with metadata
        group_col: Column to use for grouping (patient_id or original_filename)
        target_col: Target column for classification (class_name or superclass)
        n_splits: Number of CV folds
        classifiers: List of classifiers to evaluate ('knn', 'linear')
        verbose: Whether to print progress
        filter_condition: Optional condition to filter the dataset

    Returns:
        Dictionary with performance metrics for each fold and classifier
    """
    if classifiers is None:
        classifiers = ["knn", "linear"]

    # Apply filter if specified
    if filter_condition is not None:
        mask = metadata.eval(filter_condition)
        filtered_embeddings = embeddings[mask]
        filtered_metadata = metadata[mask]

        if verbose:
            print(
                f"Applied filter '{filter_condition}': {len(filtered_embeddings)} out of {len(embeddings)} samples remaining"
            )

        embeddings = filtered_embeddings
        metadata = filtered_metadata

    # Get unique classes for stratification
    y = metadata[target_col].values

    groups = metadata[group_col].values

    # Initialize cross-validator
    cv = custom_balanced_group_kfold(X=embeddings, y=y, groups=groups, n_splits=n_splits)

    # Initialize results
    results = defaultdict(list)
    fold_predictions = {}
    confusion_matrices = {}
    class_distribution = metadata[target_col].value_counts().to_dict()

    class_group_counts = metadata.groupby([target_col, group_col]).size()
    class_group_counts = class_group_counts.groupby(level=0).count()
    print(f"Number of groups per class: {class_group_counts}")

    # Classes appearing in fewer groups than n_splits are at risk
    at_risk_classes = class_group_counts[class_group_counts < n_splits].index.tolist()
    if at_risk_classes:
        print(f"Warning: These classes may be missing in some folds: {at_risk_classes}")

    # Run cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv):
        if verbose:
            print(f"Fold {fold_idx + 1}/{n_splits}")

        # Split data
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Track fold distribution
        train_dist = pd.Series(y_train).value_counts().to_dict()
        test_dist = pd.Series(y_test).value_counts().to_dict()

        # Add detailed class distribution reporting
        if verbose:
            print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

            # Create a DataFrame for pretty-printing the distributions
            class_names = sorted(list(set(y_train) | set(y_test)))
            distribution_data = []

            for class_name in class_names:
                train_count = train_dist.get(class_name, 0)
                test_count = test_dist.get(class_name, 0)
                total_count = train_count + test_count

                train_pct = (train_count / len(y_train) * 100) if len(y_train) > 0 else 0
                test_pct = (test_count / len(y_test) * 100) if len(y_test) > 0 else 0

                distribution_data.append(
                    {
                        "Class": class_name,
                        "Train Count": train_count,
                        "Train %": f"{train_pct:.1f}%",
                        "Test Count": test_count,
                        "Test %": f"{test_pct:.1f}%",
                        "Total": total_count,
                    }
                )

            dist_df = pd.DataFrame(distribution_data)
            print("  Class distribution in this fold:")
            print(f"  {dist_df.to_string(index=False)}")

            # Check for missing classes
            missing_in_train = [c for c in class_names if c not in y_train]
            missing_in_test = [c for c in class_names if c not in y_test]

            if missing_in_train:
                print(f"  ⚠️ WARNING: Classes missing in training set: {missing_in_train}")
            if missing_in_test:
                print(f"  ⚠️ WARNING: Classes missing in test set: {missing_in_test}")

            # Report on patient group distribution
            train_groups = groups[train_idx]
            test_groups = groups[test_idx]

            print(
                f"  Patient groups: {len(set(train_groups))} in training, {len(set(test_groups))} in testing"
            )

            # Count groups per class in this fold
            train_class_groups = {}
            test_class_groups = {}

            for cls in class_names:
                cls_mask_train = y_train == cls
                cls_mask_test = y_test == cls

                train_class_groups[cls] = len(set(train_groups[cls_mask_train]))
                test_class_groups[cls] = len(set(test_groups[cls_mask_test]))

            # Create and display the groups distribution DataFrame
            groups_data = []
            for cls in class_names:
                groups_data.append(
                    {
                        "Class": cls,
                        "Train Groups": train_class_groups[cls],
                        "Test Groups": test_class_groups[cls],
                    }
                )

            groups_df = pd.DataFrame(groups_data)
            print("  Patient groups per class:")
            print(f"  {groups_df.to_string(index=False)}")

            print("  " + "-" * 50)  # Separator line

        # Run each classifier
        for clf_type in classifiers:
            if verbose:
                print(f"  Evaluating {clf_type} classifier...")

            # Evaluate
            metrics, trained_clf, y_pred, cm = evaluate_classifier(
                X_train, y_train, X_test, y_test, clf_type=clf_type
            )

            # Store results
            for metric, value in metrics.items():
                results[f"{clf_type}_{metric}"].append(value)

            # Store predictions and confusion matrix
            fold_predictions[(fold_idx, clf_type)] = {
                "y_true": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "test_indices": test_idx.tolist(),
                "train_indices": train_idx.tolist(),
                "train_dist": train_dist,
                "test_dist": test_dist,
            }

            confusion_matrices[(fold_idx, clf_type)] = cm

    # Aggregate results
    summary = {
        "class_distribution": class_distribution,
        "n_samples": len(embeddings),
        "n_classes": len(class_distribution),
    }

    for metric, values in results.items():
        summary[f"{metric}_mean"] = np.mean(values)
        summary[f"{metric}_std"] = np.std(values)

    return summary, results, fold_predictions, confusion_matrices


def plot_confusion_matrices(confusion_matrices, labels, output_dir=None):
    """Plot confusion matrices for each classifier and fold."""
    clfs = sorted(set([clf for fold, clf in confusion_matrices.keys()]))
    folds = sorted(set([fold for fold, clf in confusion_matrices.keys()]))

    for clf in clfs:
        # Aggregate confusion matrix across folds
        aggregated_cm = np.zeros_like(confusion_matrices[(folds[0], clf)])
        for fold in folds:
            aggregated_cm += confusion_matrices[(fold, clf)]

        # Plot normalized confusion matrix
        plt.figure(figsize=(10, 8))
        cm_normalized = aggregated_cm.astype("float") / aggregated_cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(f"Normalized Confusion Matrix - {clf.upper()} Classifier")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        if output_dir:
            plt.savefig(output_dir / f"confusion_matrix_{clf}.png", dpi=300, bbox_inches="tight")

        plt.close()


def plot_comparison_results(results_dict, metric="accuracy", output_dir=None):
    """Plot comparison between different evaluation configurations."""
    plt.figure(figsize=(12, 6))

    # Set up data for plotting
    data = []

    for config_name, result in results_dict.items():
        for clf in ["knn", "linear"]:
            metric_key = f"{clf}_{metric}_mean"
            std_key = f"{clf}_{metric}_std"

            if metric_key in result["summary"]:
                data.append(
                    {
                        "Configuration": config_name,
                        "Model": clf.upper(),
                        "Score": result["summary"][metric_key],
                        "Std": result["summary"].get(std_key, 0),
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    if len(df) == 0:
        print(f"No data to plot for metric: {metric}")
        return

    # Create plot
    _ = sns.barplot(x="Configuration", y="Score", hue="Model", data=df, palette="Set2")

    # Add error bars
    x_positions = []
    for i, item in enumerate(zip(df["Configuration"], df["Model"])):
        x = i // 2  # Config position
        offset = -0.15 if item[1] == "KNN" else 0.15  # Offset for model type
        x_pos = x + offset
        x_positions.append(x_pos)

    plt.errorbar(
        x=x_positions, y=df["Score"], yerr=df["Std"], fmt="none", ecolor="black", capsize=5
    )

    plt.title(f"Comparison of {metric.replace('_', ' ').title()}")
    plt.ylabel(f"{metric.replace('_', ' ').title()} Score")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / f"comparison_{metric}.png", dpi=300, bbox_inches="tight")

    plt.close()


def save_evaluation_results(
    summary, results, fold_predictions, confusion_matrices, config, output_dir
):
    """
    Save evaluation results to files with proper metadata.

    Args:
        summary: Summary dictionary with aggregated metrics
        results: Dictionary with per-fold metrics
        fold_predictions: Dictionary with predictions for each fold
        confusion_matrices: Dictionary with confusion matrices for each fold
        config: Configuration dictionary with experiment parameters
        output_dir: Directory to save results
    """
    from datetime import datetime
    import json
    from pathlib import Path
    import pickle

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create experiment name from config
    task = config.get("task", "unknown")
    superclass = config.get("superclass_to_keep", "all")
    agg_method = config.get("aggregation_method", "none")
    model_name = config.get("model_name", "unknown")

    experiment_name = f"{model_name}_{task}"
    if superclass != "all":
        experiment_name += f"_{superclass}"
    if agg_method != "none":
        experiment_name += f"_{agg_method}"

    # Save summary as JSON
    summary_path = output_dir / f"{experiment_name}_{timestamp}_summary.json"
    with open(summary_path, "w") as f:
        # Add metadata to summary
        summary_with_meta = {"config": config, "timestamp": timestamp, "results": summary}
        json.dump(summary_with_meta, f, indent=2)

    # Save detailed results as pickle
    results_path = output_dir / f"{experiment_name}_{timestamp}_results.pkl"
    with open(results_path, "wb") as f:
        detailed_results = {
            "config": config,
            "timestamp": timestamp,
            "summary": summary,
            "results": results,
            "fold_predictions": fold_predictions,
            "confusion_matrices": confusion_matrices,
        }
        pickle.dump(detailed_results, f)

    # Save class labels for future reference
    all_classes = []
    for fold_data in fold_predictions.values():
        all_classes.extend(set(fold_data["y_true"]))
    class_labels = sorted(set(all_classes))

    # Generate confusion matrix visualizations
    labels = class_labels
    plot_confusion_matrices(confusion_matrices, labels, output_dir)

    # Generate per-class metrics plots
    plot_class_metrics(results, class_labels, output_dir)

    return summary_path, results_path


def plot_class_metrics(results, class_labels, output_dir):
    """
    Plot per-class metrics across folds.

    Args:
        results: Dictionary with per-fold metrics
        class_labels: List of class labels
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # Extract per-class metrics
    metrics = ["precision", "recall", "f1-score"]
    classifiers = ["knn", "linear"]

    for clf in classifiers:
        for metric in metrics:
            plt.figure(figsize=(10, 6))

            # Prepare data for plotting
            data = []

            for cls in class_labels:
                key = f"{clf}_class_{cls}_{metric}"
                if key in results:
                    for fold_idx, value in enumerate(results[key]):
                        data.append(
                            {"Class": cls, "Fold": fold_idx + 1, "Score": value, "Metric": metric}
                        )

            # Create DataFrame and plot
            df = pd.DataFrame(data)
            if len(df) > 0:
                # Plot per-class metrics
                sns.boxplot(x="Class", y="Score", data=df, palette="viridis")
                plt.title(f"{clf.upper()} - Per-Class {metric.title()} Across Folds")
                plt.ylabel(f"{metric.title()} Score")
                plt.xlabel("Class")
                plt.ylim(0, 1.05)
                plt.grid(axis="y", alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Save figure
                plt.savefig(
                    output_dir / f"{clf}_{metric}_per_class.png", dpi=300, bbox_inches="tight"
                )
                plt.close()


def plot_fold_distributions(fold_predictions, output_dir):
    """
    Plot sample and group distributions across folds.

    Args:
        fold_predictions: Dictionary with predictions for each fold
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Extract fold data
    data = []

    for (fold_idx, clf), fold_info in fold_predictions.items():
        # Skip duplicates (we only need distribution once per fold)
        if clf != "knn":
            continue

        train_dist = fold_info["train_dist"]
        test_dist = fold_info["test_dist"]

        # Add sample distribution data
        for cls, count in train_dist.items():
            data.append({"Fold": fold_idx + 1, "Class": cls, "Count": count, "Set": "Train"})

        for cls, count in test_dist.items():
            data.append({"Fold": fold_idx + 1, "Class": cls, "Count": count, "Set": "Test"})

    # Create DataFrame and plot
    df = pd.DataFrame(data)
    if len(df) > 0:
        # Plot stacked bar chart for train/test distribution
        plt.figure(figsize=(12, 6))
        pivot_df = (
            df.pivot_table(index=["Fold", "Set"], columns="Class", values="Count", aggfunc="sum")
            .fillna(0)
            .reset_index()
        )

        # Plot each class as a segment in stacked bars
        classes = sorted(df["Class"].unique())
        fold_sets = sorted(df[["Fold", "Set"]].apply(tuple, axis=1).unique())

        bar_positions = np.arange(len(fold_sets))
        bottom = np.zeros(len(fold_sets))

        # Create a custom colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

        for i, cls in enumerate(classes):
            heights = [
                pivot_df.loc[(pivot_df["Fold"] == fold) & (pivot_df["Set"] == set), cls].values[0]
                for fold, set in fold_sets
            ]
            plt.bar(bar_positions, heights, bottom=bottom, label=cls, color=colors[i])
            bottom += heights

        # Set labels and title
        plt.title("Class Distribution Across Folds")
        plt.ylabel("Sample Count")
        plt.xticks(bar_positions, [f"{fold}-{set}" for fold, set in fold_sets])
        plt.legend(title="Class")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        # Save figure
        plt.savefig(output_dir / "fold_class_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()
