from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class PrototypeClassifier(BaseEstimator, ClassifierMixin):
    """
    Prototype-based classifier for histopathology images.

    For each class, computes a prototype by averaging embeddings.
    During prediction, finds the k-nearest prototypes and performs majority voting.

    Parameters
    ----------
    k : int, default=5
        Number of nearest prototypes to consider for voting
    """

    requires_groups = True

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        """
        Compute class prototypes from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training embeddings
        y : array-like of shape (n_samples,)
            Target class labels

        Returns
        -------
        self : object
            Returns self
        """
        # Check inputs
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        # Store the number of features for validation in predict
        self.n_features_in_ = X.shape[1]

        # Create prototypes for each class by averaging embeddings
        self.prototypes_ = np.zeros((len(self.classes_), X.shape[1]))

        for i, cls in enumerate(self.classes_):
            # Get all embeddings for this class
            class_mask = y == cls
            class_embeddings = X[class_mask]

            # Compute prototype as average embedding
            self.prototypes_[i] = np.mean(class_embeddings, axis=0)

        return self

    def predict(self, X, groups=None):
        """
        Predict class labels for samples in X.

        If groups is provided, predictions are made by aggregating votes
        for each unique group using the k nearest prototypes for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test embeddings
        groups : array-like of shape (n_samples,), optional
            Group identifiers for aggregating samples (e.g., image_ids, patient_ids)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_unique_groups,)
            Predicted class labels. If groups is provided, returns one prediction
            per unique group.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but PrototypeClassifier "
                f"is expecting {self.n_features_in_} features"
            )

        # Compute distances from each sample to all prototypes
        distances = cdist(X, self.prototypes_)

        if groups is None:
            # Sample-level prediction without grouping
            y_pred = np.zeros(X.shape[0], dtype=self.classes_.dtype)

            for i in range(X.shape[0]):
                # Get indices of k nearest prototypes
                nearest_indices = np.argsort(distances[i])[: self.k]
                # Count occurrences of each class among top-k
                votes = Counter(
                    [nearest_indices[j] for j in range(min(self.k, len(nearest_indices)))]
                )

                # Get class with most votes
                most_common = votes.most_common(1)[0][0]
                y_pred[i] = self.classes_[most_common]

            return y_pred
        else:
            # Group-level prediction with aggregation
            unique_groups = np.unique(groups)
            y_pred = np.empty(len(unique_groups), dtype=self.classes_.dtype)

            # For each unique group
            for i, group in enumerate(unique_groups):
                # Get indices of all samples belonging to this group
                sample_indices = np.where(np.array(groups) == group)[0]

                # Initialize votes counter
                votes = Counter()

                # For each sample in this group
                for sample_idx in sample_indices:
                    # Get indices of k nearest prototypes for this sample
                    nearest_indices = np.argsort(distances[sample_idx])[: self.k]

                    # Add votes for the nearest prototypes
                    for proto_idx in nearest_indices:
                        votes[self.classes_[proto_idx]] += 1

                # Get class with most votes
                if votes:
                    most_common = votes.most_common(1)[0][0]
                    y_pred[i] = most_common
                else:
                    # Fallback if no votes (shouldn't happen)
                    y_pred[i] = self.classes_[0]

            return y_pred, unique_groups


class GaussianPrototypeClassifier(BaseEstimator, ClassifierMixin):
    """
    Prototype-based classifier for histopathology images.

    For each class, computes a prototype by averaging embeddings.
    During prediction, finds the k-nearest prototypes and performs majority voting.

    Parameters
    ----------
    k : int, default=5
        Number of nearest prototypes to consider for voting
    use_gaussian : bool, default=True
        Whether to use Gaussian distributions for probability estimation
    reg_covar : float, default=1e-6
        Regularization added to covariance matrices to ensure they are positive definite
    """

    requires_groups = True

    def __init__(self, k=5, use_gaussian=True, reg_covar=1e-6):
        self.k = k
        self.use_gaussian = use_gaussian
        self.reg_covar = reg_covar

    def fit(self, X, y):
        """
        Compute class prototypes from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training embeddings
        y : array-like of shape (n_samples,)
            Target class labels

        Returns
        -------
        self : object
            Returns self
        """
        # Check inputs
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        # Store the number of features for validation in predict
        self.n_features_in_ = X.shape[1]

        # Create prototypes for each class by averaging embeddings
        self.prototypes_ = np.zeros((len(self.classes_), X.shape[1]))

        # If using Gaussian approach, store covariance matrices
        if self.use_gaussian:
            self.covariances_ = []

        for i, cls in enumerate(self.classes_):
            # Get all embeddings for this class
            class_mask = y == cls
            class_embeddings = X[class_mask]

            # Compute prototype as average embedding
            self.prototypes_[i] = np.mean(class_embeddings, axis=0)

            # For Gaussian approach, compute covariance matrix
            if self.use_gaussian:
                if len(class_embeddings) > 1:
                    # Calculate covariance matrix
                    cov = np.cov(class_embeddings, rowvar=False)

                    # Ensure positive definiteness with regularization
                    cov += np.eye(cov.shape[0]) * self.reg_covar
                else:
                    # If only one sample, use identity matrix with regularization
                    cov = np.eye(X.shape[1]) * self.reg_covar

                self.covariances_.append(cov)

        return self

    def predict_proba(self, X, groups=None):
        """
        Predict probability estimates for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test embeddings
        groups : array-like of shape (n_samples,), optional
            Group identifiers for aggregating samples

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes) or tuple
            Probability estimates for each class. If groups is provided, returns
            tuple (proba, unique_groups) where proba has shape (n_unique_groups, n_classes)
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but PrototypeClassifier "
                f"is expecting {self.n_features_in_} features"
            )

        if self.use_gaussian:
            # Compute probabilities using Gaussian distributions
            densities = np.zeros((X.shape[0], len(self.classes_)))

            for i in range(len(self.classes_)):
                # Create multivariate normal distribution
                mvn = multivariate_normal(
                    mean=self.prototypes_[i], cov=self.covariances_[i], allow_singular=True
                )
                # Calculate probability density
                densities[:, i] = mvn.pdf(X)

            # Normalize to get probabilities
            prob_sum = np.sum(densities, axis=1, keepdims=True)
            # Handle cases where all probabilities are very small
            prob_sum = np.maximum(prob_sum, 1e-15)
            proba = densities / prob_sum
        else:
            # Compute distances from each sample to all prototypes
            distances = cdist(X, self.prototypes_)
            proba = np.zeros((X.shape[0], len(self.classes_)))

            for i in range(X.shape[0]):
                # Get indices of k nearest prototypes
                nearest_indices = np.argsort(distances[i])[: self.k]

                # Count votes for each prototype
                votes = Counter()
                for j in range(min(self.k, len(nearest_indices))):
                    votes[nearest_indices[j]] += 1

                # Convert votes to probabilities
                total_votes = sum(votes.values())
                for proto_idx, vote_count in votes.items():
                    proba[i, proto_idx] = vote_count / total_votes

        # Handle group aggregation if groups are provided
        if groups is not None:
            unique_groups = np.unique(groups)
            group_proba = np.zeros((len(unique_groups), len(self.classes_)))

            for i, group in enumerate(unique_groups):
                # Get indices of samples in this group
                group_mask = np.array(groups) == group
                # Average probabilities across all samples in the group
                group_proba[i] = np.mean(proba[group_mask], axis=0)

            return group_proba, unique_groups
        else:
            return proba

    def predict(self, X, groups=None):
        """
        Predict class labels for samples in X.

        If groups is provided, predictions are made by aggregating votes
        for each unique group using the k nearest prototypes for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test embeddings
        groups : array-like of shape (n_samples,), optional
            Group identifiers for aggregating samples (e.g., image_ids, patient_ids)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or tuple
            Predicted class labels. If groups is provided, returns tuple
            (y_pred, unique_groups) where y_pred has shape (n_unique_groups,)
        """
        # If using Gaussian, leverage predict_proba for predictions
        if self.use_gaussian:
            if groups is None:
                proba = self.predict_proba(X)
                return self.classes_[np.argmax(proba, axis=1)]
            else:
                proba, unique_groups = self.predict_proba(X, groups)
                return self.classes_[np.argmax(proba, axis=1)], unique_groups

        # Otherwise, use the original implementation
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but PrototypeClassifier "
                f"is expecting {self.n_features_in_} features"
            )

        # Compute distances from each sample to all prototypes
        distances = cdist(X, self.prototypes_)

        if groups is None:
            # Sample-level prediction without grouping
            y_pred = np.zeros(X.shape[0], dtype=self.classes_.dtype)

            for i in range(X.shape[0]):
                # Get indices of k nearest prototypes
                nearest_indices = np.argsort(distances[i])[: self.k]
                # Count occurrences of each class among top-k
                votes = Counter(
                    [nearest_indices[j] for j in range(min(self.k, len(nearest_indices)))]
                )

                # Get class with most votes
                most_common = votes.most_common(1)[0][0]
                y_pred[i] = self.classes_[most_common]

            return y_pred
        else:
            # Group-level prediction with aggregation
            unique_groups = np.unique(groups)
            y_pred = np.empty(len(unique_groups), dtype=self.classes_.dtype)

            # For each unique group
            for i, group in enumerate(unique_groups):
                # Get indices of all samples belonging to this group
                sample_indices = np.where(np.array(groups) == group)[0]

                # Initialize votes counter
                votes = Counter()

                # For each sample in this group
                for sample_idx in sample_indices:
                    # Get indices of k nearest prototypes for this sample
                    nearest_indices = np.argsort(distances[sample_idx])[: self.k]

                    # Add votes for the nearest prototypes
                    for proto_idx in nearest_indices:
                        votes[self.classes_[proto_idx]] += 1

                # Get class with most votes
                if votes:
                    most_common = votes.most_common(1)[0][0]
                    y_pred[i] = most_common
                else:
                    # Fallback if no votes (shouldn't happen)
                    y_pred[i] = self.classes_[0]

            return y_pred, unique_groups


class HistogramClusterClassifier(BaseEstimator, ClassifierMixin):
    requires_groups = True

    def __init__(self, n_clusters=8, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto"
        )
        self.classifier = LogisticRegression(multi_class="multinomial", max_iter=1000)

    def fit(self, X, y, groups=None):
        # Check inputs
        X, y = check_X_y(X, y)

        if groups is None:
            raise ValueError("Groups parameter is required for ClusterBasedClassifier")

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # Fit KMeans to determine cluster centers
        self.kmeans.fit(X)
        self.cluster_centers_ = self.kmeans.cluster_centers_

        # Assign each sample to a cluster
        X_clustered = self.kmeans.predict(X)

        # Calculate histograms per group
        unique_groups = np.unique(groups)
        X_histograms = np.zeros((len(unique_groups), self.n_clusters))
        y_grouped = np.zeros(len(unique_groups), dtype=y.dtype)

        # Compute histogram for each group and get its label
        for i, group in enumerate(unique_groups):
            # Get samples belonging to this group
            group_mask = groups == group
            group_clusters = X_clustered[group_mask]

            # Create normalized histogram (count of samples in each cluster / total samples)
            counts = np.bincount(group_clusters, minlength=self.n_clusters)
            if counts.sum() > 0:  # Avoid division by zero
                X_histograms[i] = counts / counts.sum()

            # Get the label for this group (should be consistent within a group)
            # Using most common label in case there's any inconsistency
            unique_labels, counts = np.unique(y[group_mask], return_counts=True)
            y_grouped[i] = unique_labels[np.argmax(counts)]

        # Store unique groups for later reference
        self.unique_groups_fit_ = unique_groups

        # Train a classifier on the histograms
        self.classifier.fit(X_histograms, y_grouped)

        return self

    def predict(self, X, groups=None):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test embeddings
        groups : array-like of shape (n_samples,), optional
            Group identifiers for aggregating samples

        Returns
        -------
        y_pred : ndarray or tuple
            If groups is None, returns ndarray of shape (n_samples,) with predictions
            for each sample using the most common cluster assignment per group.
            If groups is provided, returns tuple (y_pred, unique_groups) where
            y_pred has shape (n_unique_groups,).
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but ClusterBasedClassifier "
                f"is expecting {self.n_features_in_} features"
            )

        if groups is None:
            # Without groups, we can't create histograms
            raise ValueError("Groups parameter is required for prediction")

        # Assign each sample to a cluster
        X_clustered = self.kmeans.predict(X)

        # Calculate histograms per group
        unique_groups = np.unique(groups)
        X_histograms = np.zeros((len(unique_groups), self.n_clusters))

        # Compute histogram for each group
        for i, group in enumerate(unique_groups):
            # Get samples belonging to this group
            group_mask = groups == group
            group_clusters = X_clustered[group_mask]

            # Create normalized histogram
            counts = np.bincount(group_clusters, minlength=self.n_clusters)
            if counts.sum() > 0:  # Avoid division by zero
                X_histograms[i] = counts / counts.sum()

        # Predict using the classifier
        y_pred = self.classifier.predict(X_histograms)

        return y_pred, unique_groups

    def predict_proba(self, X, groups=None):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test embeddings
        groups : array-like of shape (n_samples,), optional
            Group identifiers for aggregating samples

        Returns
        -------
        proba : ndarray or tuple
            If groups is None, returns ndarray of shape (n_samples, n_classes).
            If groups is provided, returns tuple (proba, unique_groups) where
            proba has shape (n_unique_groups, n_classes).
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but ClusterBasedClassifier "
                f"is expecting {self.n_features_in_} features"
            )

        if groups is None:
            # Without groups, we can't create histograms
            raise ValueError("Groups parameter is required for prediction")

        # Assign each sample to a cluster
        X_clustered = self.kmeans.predict(X)

        # Calculate histograms per group
        unique_groups = np.unique(groups)
        X_histograms = np.zeros((len(unique_groups), self.n_clusters))

        # Compute histogram for each group
        for i, group in enumerate(unique_groups):
            # Get samples belonging to this group
            group_mask = groups == group
            group_clusters = X_clustered[group_mask]

            # Create normalized histogram
            counts = np.bincount(group_clusters, minlength=self.n_clusters)
            if counts.sum() > 0:  # Avoid division by zero
                X_histograms[i] = counts / counts.sum()

        # Predict probabilities using the classifier
        proba = self.classifier.predict_proba(X_histograms)

        return proba, unique_groups
