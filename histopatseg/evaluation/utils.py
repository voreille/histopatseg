from collections import defaultdict
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
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
from sklearn.preprocessing import Normalizer, StandardScaler

from histopatseg.evaluation.group_pipeline import GroupPipeline
from histopatseg.evaluation.prototype_classifier import PrototypeClassifier


class ClassifierEvaluator:
    """Evaluate classifiers with cross-validation, supporting both regular and group-based predictions."""

    def __init__(
        self,
        verbose: bool = True,
    ):
        """
        Initialize evaluator with data.

        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose

        # Results storage
        self.results = defaultdict(list)
        self.fold_predictions = {}
        self.confusion_matrices = {}

    def run_cross_validation(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        target_col: str,
        cv_group_col: str,
        n_splits: int = 5,
        classifiers: Optional[List[str]] = None,
        image_id_col: Optional[str] = None,
    ):
        """
        Run cross-validation evaluation.

        Args:
            embeddings: Feature vectors (n_samples, n_features)
            metadata: DataFrame with sample metadata
            target_col: Column name for target labels
            cv_group_col: Column name for grouping samples for CV splitting
            n_splits: Number of CV folds
            classifiers: List of classifier types to evaluate
            image_id_col: Column name to use for group predictions in prototype classifiers

        Returns:
            Tuple of (summary, results, fold_predictions, confusion_matrices)
        """
        if classifiers is None:
            classifiers = ["knn", "linear", "prototype"]

        # Print class distribution info
        y = metadata[target_col].values
        groups = metadata[cv_group_col].values
        image_ids = metadata[image_id_col].values if image_id_col else None

        self._report_class_distribution(metadata, target_col, cv_group_col, n_splits)

        # Create cross-validation folds
        cv = self._create_folds(embeddings, y, groups, n_splits)

        # Run evaluation for each fold
        for fold_idx, (train_idx, test_idx) in enumerate(cv):
            # Report fold statistics
            if self.verbose:
                self._report_fold_statistics(embeddings, y, groups, train_idx, test_idx)

            self._evaluate_fold(
                embeddings,
                y,
                fold_idx,
                n_splits,
                train_idx,
                test_idx,
                classifiers,
                image_ids=image_ids,
            )

        # Create summary
        summary = self._create_summary(embeddings, metadata, target_col)

        return summary, self.results, self.fold_predictions, self.confusion_matrices

    def _evaluate_fold(
        self,
        embeddings: np.ndarray,
        y: np.ndarray,
        fold_idx: int,
        n_splits: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        classifiers: List[str],
        image_ids: Optional[np.ndarray] = None,
    ):
        """Evaluate all classifiers on one fold."""
        if self.verbose:
            print(f"Fold {fold_idx + 1}/{n_splits}")

        # Split data
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        image_ids_train = None
        image_ids_test = None
        if image_ids is not None:
            image_ids_train = image_ids[train_idx]
            image_ids_test = image_ids[test_idx]

        # Evaluate each classifier
        for clf_type in classifiers:
            if self.verbose:
                print(f"  Evaluating {clf_type} classifier...")

            # Train and evaluate classifier
            metrics, trained_clf, y_pred, cm = evaluate_classifier(
                X_train,
                y_train,
                X_test,
                y_test,
                clf_type=clf_type,
                image_ids_train=image_ids_train,
                image_ids_test=image_ids_test,
            )

            # Store results
            self._store_fold_results(
                fold_idx, clf_type, metrics, y_train, y_test, y_pred, train_idx, test_idx, cm
            )

    def _report_class_distribution(
        self, metadata, target_col, cv_group_col, n_splits: int
    ) -> None:
        """Report class and group distribution."""
        if not self.verbose:
            return

        # Count groups per class
        class_group_counts = metadata.groupby([target_col, cv_group_col]).size()
        class_group_counts = class_group_counts.groupby(level=0).count()
        print(f"Number of groups per class: {class_group_counts}")

        # Classes appearing in fewer groups than n_splits are at risk
        at_risk_classes = class_group_counts[class_group_counts < n_splits].index.tolist()
        if at_risk_classes:
            print(f"Warning: These classes may be missing in some folds: {at_risk_classes}")

    def _create_folds(self, embeddings, y, groups, n_splits: int):
        """Create cross-validation folds."""
        return custom_balanced_group_kfold(X=embeddings, y=y, groups=groups, n_splits=n_splits)

    def _report_fold_statistics(
        self,
        embeddings: np.array,
        y: np.array,
        groups: np.array,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> None:
        """Report detailed statistics about the current fold."""
        if not self.verbose:
            return

        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Distribution of classes
        train_dist = pd.Series(y_train).value_counts().to_dict()
        test_dist = pd.Series(y_test).value_counts().to_dict()

        # Print sample counts
        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        # Create distribution table
        self._print_class_distribution_table(y_train, y_test, train_dist, test_dist)

        # Check for missing classes
        class_names = sorted(list(set(y_train) | set(y_test)))
        missing_in_train = [c for c in class_names if c not in y_train]
        missing_in_test = [c for c in class_names if c not in y_test]

        if missing_in_train:
            print(f"  ⚠️ WARNING: Classes missing in training set: {missing_in_train}")
        if missing_in_test:
            print(f"  ⚠️ WARNING: Classes missing in test set: {missing_in_test}")

        # Report on group distribution
        self._print_group_distribution(y, groups, train_idx, test_idx, class_names)

        print("  " + "-" * 50)  # Separator line

    def _print_class_distribution_table(
        self, y_train: np.ndarray, y_test: np.ndarray, train_dist: Dict, test_dist: Dict
    ) -> None:
        """Print class distribution table."""
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

    def _print_group_distribution(
        self,
        y: np.array,
        groups: np.array,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        class_names: List,
    ) -> None:
        """Print group distribution per class."""
        train_groups = groups[train_idx]
        test_groups = groups[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        print(
            f"  Patient groups: {len(set(train_groups))} in training, {len(set(test_groups))} in testing"
        )

        # Count groups per class
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

    def _store_fold_results(
        self,
        fold_idx: int,
        clf_type: str,
        metrics: Dict,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        cm: np.ndarray,
    ) -> None:
        """Store results for the current fold and classifier."""
        # Store metrics
        for metric, value in metrics.items():
            self.results[f"{clf_type}_{metric}"].append(value)

        # Get distributions
        train_dist = pd.Series(y_train).value_counts().to_dict()
        test_dist = pd.Series(y_test).value_counts().to_dict()

        # Store predictions
        self.fold_predictions[(fold_idx, clf_type)] = {
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "test_indices": test_idx.tolist(),
            "train_indices": train_idx.tolist(),
            "train_dist": train_dist,
            "test_dist": test_dist,
            "is_group_prediction": metrics.get("is_group_prediction", False),
        }

        # Store confusion matrix
        self.confusion_matrices[(fold_idx, clf_type)] = cm

    def _create_summary(
        self, embeddings: np.array, metadata: pd.DataFrame, target_col: str
    ) -> Dict:
        """Create summary statistics from results."""
        class_distribution = metadata[target_col].value_counts().to_dict()

        summary = {
            "class_distribution": class_distribution,
            "n_samples": len(embeddings),
            "n_classes": len(class_distribution),
        }

        for metric, values in self.results.items():
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)

        return summary


def get_classifier(clf_type: str) -> Pipeline:
    """Get a classifier pipeline based on type."""
    if clf_type == "knn":
        requires_groups = False
        clf = Pipeline(
            [
                ("scaler", Normalizer()),
                ("classifier", KNeighborsClassifier(n_neighbors=5, metric="cosine")),
            ]
        )
    elif clf_type == "prototype":
        requires_groups = True
        clf = GroupPipeline(
            [
                ("scaler", Normalizer()),
                ("classifier", PrototypeClassifier(k=5)),
            ]
        )
    elif clf_type == "gaussian_prototype":
        from histopatseg.evaluation.prototype_classifier import GaussianPrototypeClassifier

        requires_groups = True

        clf = GroupPipeline(
            [
                ("scaler", Normalizer()),
                ("classifier", GaussianPrototypeClassifier(k=5, use_gaussian=True)),
            ]
        )
    elif clf_type == "linear":
        requires_groups = False
        clf = Pipeline(
            [
                ("scaler", Normalizer()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42),
                ),
            ]
        )
    elif clf_type == "histogram_cluster":
        from histopatseg.evaluation.prototype_classifier import HistogramClusterClassifier

        requires_groups = True

        clf = GroupPipeline(
            [
                ("scaler", Normalizer()),
                ("classifier", HistogramClusterClassifier(n_clusters=7)),
            ]
        )
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")

    return clf, requires_groups


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, train_time: float) -> Dict:
    """Compute classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "train_time": train_time,
    }

    # Calculate per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    for class_name, class_metrics in report.items():
        if isinstance(class_metrics, dict):  # Skip averages
            for metric_name, value in class_metrics.items():
                metrics[f"class_{class_name}_{metric_name}"] = value

    return metrics


def evaluate_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clf_type: str = "knn",
    image_ids_train: Optional[np.ndarray] = None,
    image_ids_test: Optional[np.ndarray] = None,
) -> Tuple[Dict, BaseEstimator, np.ndarray, np.ndarray]:
    """Train and evaluate a classifier, handling both regular and group predictions."""
    start_time = time.time()

    # Get and train classifier
    clf, requires_groups = get_classifier(clf_type)

    if requires_groups:
        clf.fit(X_train, y_train, image_ids_train)
    else:
        clf.fit(X_train, y_train)

    if requires_groups:
        # Handle group-based prototype classifiers
        metrics, y_pred, cm = evaluate_group_prediction(
            clf, X_test, y_test, image_ids_test, start_time
        )
    else:
        # Standard prediction and evaluation
        y_pred = clf.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, time.time() - start_time)
        cm = confusion_matrix(y_test, y_pred)

    metrics["is_group_prediction"] = requires_groups
    return metrics, clf, y_pred, cm


def evaluate_group_prediction(
    clf: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    image_ids_test: np.ndarray,
    start_time: float,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Evaluate group-based prediction."""
    # Get the actual classifier from the pipeline
    prototype_clf = clf.named_steps["classifier"]

    # Use the scaler from the pipeline
    # TODO: make this more general take everything from the pipeline before the classifier
    X_test_transformed = clf.named_steps["scaler"].transform(X_test)

    # Call predict with groups parameter
    y_pred_by_group, unique_image_ids = prototype_clf.predict(X_test_transformed, image_ids_test)

    # Aggregate ground truth labels by group
    y_test_by_group = []
    for img_id in unique_image_ids:
        mask = np.array(image_ids_test) == img_id
        # Get the most common label in this group
        group_labels = y_test[mask]
        most_common_label = pd.Series(group_labels).value_counts().idxmax()
        y_test_by_group.append(most_common_label)

    # Calculate metrics at the group level
    metrics = compute_metrics(y_test_by_group, y_pred_by_group, time.time() - start_time)
    metrics["n_groups"] = len(unique_image_ids)

    # Create confusion matrix for group-level predictions
    cm = confusion_matrix(y_test_by_group, y_pred_by_group)

    # Map group predictions back to individual samples for consistency
    y_pred = np.zeros(len(y_test), dtype=y_pred_by_group.dtype)
    for i, img_id in enumerate(unique_image_ids):
        mask = np.array(image_ids_test) == img_id
        y_pred[mask] = y_pred_by_group[i]

    return metrics, y_pred, cm


# Wrapper for backward compatibility
def run_cross_validation(
    embeddings,
    metadata,
    group_col,
    target_col,
    n_splits=5,
    classifiers=None,
    verbose=True,
    filter_condition=None,
    image_id_col=None,
):
    """Run cross-validation for embeddings evaluation (backward compatibility wrapper)."""
    # Apply filter if specified
    if filter_condition is not None:
        mask = metadata.eval(filter_condition)
        embeddings = embeddings[mask]
        metadata = metadata[mask]

        if verbose:
            print(f"Applied filter '{filter_condition}': {len(embeddings)} samples remaining")

    evaluator = ClassifierEvaluator(verbose=verbose)

    return evaluator.run_cross_validation(
        embeddings=embeddings,
        metadata=metadata,
        target_col=target_col,
        cv_group_col=group_col,
        n_splits=n_splits,
        classifiers=classifiers,
        image_id_col=image_id_col,
    )


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
