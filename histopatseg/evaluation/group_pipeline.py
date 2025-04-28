import inspect

from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if


def _final_estimator_has(attr):
    """Check if the final estimator has the specified attribute."""

    def check(self):
        # Return False if no final estimator
        if not self.steps:
            return False

        # Return False if the final estimator is None
        final_estimator = self.steps[-1][1]
        if final_estimator is None:
            return False

        # Return whether the final estimator has the attribute
        return hasattr(final_estimator, attr)

    return check


class GroupPipeline(Pipeline):
    """Extension of sklearn's Pipeline that supports passing groups to the final estimator."""

    def fit(self, X, y=None, groups=None, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        groups : array-like, optional
            Group labels for the samples used while splitting the dataset into
            train/test set. This parameter is passed to the final estimator
            if it has a 'requires_groups' attribute set to True.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : object
            This estimator
        """
        # Rather than trying to handle all the different versions,
        # let's use the built-in fit method from the parent class
        # to handle the parameter preparation and steps

        # Get the final estimator to check if it needs groups
        final_estimator = self.steps[-1][1] if self.steps else None
        requires_groups = getattr(final_estimator, "requires_groups", False)

        if not requires_groups or groups is None:
            # If groups are not needed or not provided, use the standard pipeline fit
            return super().fit(X, y, **fit_params)
        else:
            # If groups are needed, we need to handle them specially

            # First, use the parent's fit method but with a hack:
            # We'll modify the final step to be None temporarily,
            # so the parent won't fit it
            original_final_step = self.steps[-1]
            self.steps[-1] = (original_final_step[0], None)

            # Now fit all the intermediate steps
            Xt = super().fit_transform(X, y, **fit_params)

            # Restore the final step
            self.steps[-1] = original_final_step

            # Now fit the final estimator with the groups parameter
            final_estimator_params = {}
            final_step_name = self.steps[-1][0]

            # Extract parameters for the final step
            for name, value in fit_params.items():
                if name.startswith(f"{final_step_name}__"):
                    param = name[len(f"{final_step_name}__") :]
                    final_estimator_params[param] = value

            # Fit the final estimator with groups
            self.steps[-1][1].fit(Xt, y, groups, **final_estimator_params)

            return self

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, groups=None, **predict_params):
        """Apply transforms to the data, and predict with the final estimator.

        Parameters
        ----------
        X : iterable
            Data to predict on.
        groups : array-like, optional
            Group labels for prediction.
        **predict_params : dict of string -> object
            Parameters to the ``predict`` method of the final estimator.

        Returns
        -------
        y_pred : ndarray
            Predicted values.
        """
        Xt = X
        for _, _, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)

        # Check if final estimator requires groups for prediction
        final_estimator = self.steps[-1][1]
        requires_groups = getattr(final_estimator, "requires_groups", False)

        if requires_groups and groups is not None:
            return final_estimator.predict(Xt, groups)
        else:
            return final_estimator.predict(Xt, **predict_params)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X, groups=None, **predict_params):
        """Apply transforms, and predict_proba of the final estimator.

        Parameters
        ----------
        X : iterable
            Data to predict on.
        groups : array-like, optional
            Group labels for prediction.
        **predict_params : dict of string -> object
            Parameters to the ``predict_proba`` method of the final estimator.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Probability estimates.
        """
        Xt = X
        for _, _, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)

        final_estimator = self.steps[-1][1]
        requires_groups = getattr(final_estimator, "requires_groups", False)

        if requires_groups and groups is not None:
            return final_estimator.predict_proba(Xt, groups)
        else:
            return final_estimator.predict_proba(Xt, **predict_params)
