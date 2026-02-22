# -*- coding: utf-8 -*-
"""
Swectral model connectors / fittable pipeline tools

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# ruff: noqa: I001
# ruff: noqa: N806

# OS
import os

# Warnings
import warnings

# Basic
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from typing import Any, Annotated, Union, Optional  # noqa: E402
import itertools

# Serialization
import dill

# Model
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin  # noqa: E402
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted  # noqa: E402
from sklearn.metrics import accuracy_score, r2_score  # noqa: E402

# Local
from .specio import simple_type_validator, arraylike_validator, unc_path  # noqa: E402
from .pipeline_validator import _classifier_validator, _regressor_validator, _data_transformer_validator
from .groupstats import (
    regression_performance_marginal_stats,
    performance_metrics_summary,
    classification_performance_marginal_stats,
)


# %% ====== Transformer - Estimator Connectors ======
# Connect data transformer and estimator
# For evaluation of dimension reduction, feature selection or other feature engineering models that requires fitting.

# %% Combinition composers


@simple_type_validator
def factorial_transformer_chains(  # noqa: C901
    *step_transformers: tuple[Union[list[object], dict[str, object], None], ...],
    estimators: Union[list[object], dict[str, object]],
    is_regression: bool = True,
) -> list[object]:
    """
    Combine data transformation models with estimators into chained models using a full-factorial approach.

    Parameters
    ----------
    step_transformers : tuple of (list of object, dict mapping str to object, or None)
        Data transformers of each step.

        Customize transformer name using dictionary input as {custom_name : transformer_model}.

    estimators : list of object or dict mapping str to object
        Estimators for final step.

    is_regression : bool
        Set True if all estimators are regressors, set False if all estimators are classifiers.

        Note: estimators cannot be a mix of regressors and classifiers.

    Returns
    -------
    list of object
        List of combined models.

    Examples
    --------
    Prepare models::

        >>> from sklearn.feature_selection import SelectKBest, f_classif
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> selector5 = SelectKBest(f_classif, k=5)
        >>> selector10 = SelectKBest(f_classif, k=10)
        >>> rf = RandomForestClassifier(n_estimators=10)
        >>> knn = KNeighborsClassifier(n_neighbors=3)

    Without specify labels for component models::

        >>> models = factorial_transformer_chains(
        ...     [selector5, selector10],
        ...     estimators=[knn, rf],
        ...     is_regression=False
        ... )

    Specify labels for component models::

        >>> models = factorial_transformer_chains(
        ...     {'feat5': selector5, 'feat10': selector10},
        ...     estimators={'KNN': knn, 'RF': rf},
        ...     is_regression=False
        ... )
    """
    # Validate given step_transformers
    if step_transformers is None:
        raise ValueError("step_transformers is missing, provide at least one step of transformers.")
    if len(step_transformers) < 1:
        raise ValueError("step_transformers is missing, provide at least one step of transformers.")
    # Validate transformers at each step
    for step_i_transformers in step_transformers:
        if step_i_transformers is None:
            raise ValueError("Provided step_transformers cannot be None.")
        if len(step_i_transformers) < 1:
            raise ValueError("Provided step_transformers cannot be empty.")

    # Get labels
    # Estimator model and model label options
    if type(estimators) is dict:
        estimator_model_options = list(estimators.values())
        estimator_label_options = list(estimators.keys())
    else:
        estimator_model_options = list(estimators)
        estimator_label_options = [model.__class__.__name__ for model in estimator_model_options]
    # Transformaer model and model label options
    chain_model_options: list[list] = []
    chain_label_options: list[list] = []
    for option in step_transformers:
        if isinstance(option, dict):
            chain_model_options.append(list(option.values()))
            chain_label_options.append(list(option.keys()))
        elif isinstance(option, list):
            chain_model_options.append(list(option))
            chain_label_options.append([model.__class__.__name__ for model in option])
        else:
            raise ValueError(f"step_transformers must be dict or list, got type: {type(option)}")

    # Validate labels
    est_labels = []
    for label in estimator_label_options:
        if label not in est_labels:
            est_labels.append(label)
        else:
            raise ValueError(
                f"Duplicate label '{label}' detected for estimators. "
                + "All models must have unique labels.\n"
                + "If labels are not specified, this is a default labeling conflict, "
                + "please explicitly set custom labels using dictionary input."
            )
    trans_labels = []
    for label_options in chain_label_options:
        for label in label_options:
            if label not in trans_labels:
                if label not in est_labels:
                    trans_labels.append(label)
                else:
                    raise ValueError(
                        f"Label '{label}' is already used as an estimator label."
                        + "All models must have unique labels.\n"
                        + "If labels are not specified, this is a default labeling conflict, "
                        + "please explicitly set custom labels using dictionary input."
                    )
            else:
                raise ValueError(
                    f"Duplicate label '{label}' detected for data transformers. "
                    + "All models must have unique labels. "
                    + "If labels are not specified, this is a default labeling conflict, "
                    + "please explicitly set custom labels using dictionary input."
                )

    # Generate model label and model list
    chain_label_list = [list(chain) for chain in itertools.product(*chain_label_options, estimator_label_options)]
    chain_model_list = [list(chain) for chain in itertools.product(*chain_model_options, estimator_model_options)]

    # Combine models
    combined_models = []
    for i in range(len(chain_model_list)):
        # Get transformer list and estimator
        data_transformer = chain_model_list[i][:-1]
        estimator = chain_model_list[i][-1]
        # Get label
        data_transformer_label = chain_label_list[i][:-1]
        estimator_label = chain_label_list[i][-1]
        # Combine models in chain
        if is_regression:
            combined_model = combine_transformer_regressor(
                data_transformer=data_transformer,
                regressor=estimator,
                data_transformer_label=data_transformer_label,
                regressor_label=estimator_label,
            )
        else:
            combined_model = combine_transformer_classifier(
                data_transformer=data_transformer,
                classifier=estimator,
                data_transformer_label=data_transformer_label,
                classifier_label=estimator_label,
            )
        # Append combined model
        combined_models.append(combined_model)

    return combined_models


# %% Combiners / Pipeline tools


# Constructor - combined classifier
@simple_type_validator
def combine_transformer_classifier(  # noqa: C901
    data_transformer: Union[object, list[object]],
    classifier: object,
    data_transformer_label: Union[str, list[str], None] = None,
    classifier_label: Optional[str] = None,
) -> object:
    """
    Combine data transformation models with a classifier into a unified estimator that preserves component names.

    This wrapper functions similarly to scikit-learn's Pipeline but compatible with any transformer and classifier that follows scikit-learn's method conventions.

    Parameters
    ----------
    data_transformer : object or list of object
        Data transformation model, any data transformation or feature selection model.

    classifier : object
        Classification model.

    data_transformer_label : str or list of str or None, optional
        Label(s) for the transformer(s). Defaults to model class names if not specified.

    classifier_label: str or None
        Label for the classifier. Defaults to model class name if not specified.

    Returns
    -------
    object
        Combined classification model.

    Examples
    --------
    Prepare models::

        >>> from sklearn.feature_selection import SelectKBest, f_classif
        >>> from from sklearn.preprocessing import StandardScaler
        >>> from sklearn.neighbors import KNeighborsClassifier

        >>> selector = SelectKBest(f_classif, k=5)
        >>> scaler = StandardScaler()
        >>> knn = KNeighborsClassifier(n_neighbors=3)

    Without specifying model labels::

        >>> combined_model = combine_transformer_classifier([scaler, selector], knn)

    Specify model labels::

        >>> combined_model = combine_transformer_classifier(
        ...     [scaler, selector],
        ...     knn,
        ...     data_transformer_label=['scaler', 'selector'],
        ...     classifier_label='knn'
        ... )
    """  # noqa: E501
    # Validate input models
    if isinstance(data_transformer, list):
        if len(data_transformer) < 1:
            raise ValueError("List of transformers must contain at least 1 transformer, got 0.")
        else:
            data_transformers: list = data_transformer
    else:
        data_transformers = [data_transformer]
    # Validate transformers
    for data_transformer in data_transformers:
        _data_transformer_validator(data_transformer)
    # Validate estimator
    _classifier_validator(classifier)

    # Create combined name
    transformer_name = ""
    transformer_name_list = []
    if data_transformer_label is None:
        for data_transformer in data_transformers:
            transformer_name = transformer_name + f"{data_transformer.__class__.__name__}_"
            transformer_name_list.append(data_transformer.__class__.__name__)
    else:
        if isinstance(data_transformer_label, str):
            data_transformer_label = [data_transformer_label]
        assert isinstance(data_transformer_label, list)
        if len(data_transformers) != len(data_transformer_label):
            raise ValueError(
                f"Got {len(data_transformers)} data transformers, but got {len(data_transformer_label)} label:\
                    {data_transformer_label}"
            )
        for label in data_transformer_label:
            transformer_name = transformer_name + f"{label}_"
            transformer_name_list.append(label)
    if classifier_label is None:
        classifier_label = classifier.__class__.__name__
    combined_name = transformer_name + classifier_label

    # Create new model class to customize name
    class CombinedModel(TransClassifier):

        def __repr__(self) -> str:
            return "TransClassifier_" + combined_name

        def __str__(self) -> str:
            return combined_name

    # Customize name
    CombinedModel.__name__ = combined_name
    CombinedModel.__qualname__ = combined_name

    # Add name attributes
    CombinedModel._transformer_labels = transformer_name_list
    CombinedModel._classifier_label = classifier_label

    # Create model instance
    combined_model = CombinedModel(data_transformers=data_transformers, classifier=classifier)

    return combined_model


# Constructor - combined regressor
@simple_type_validator
def combine_transformer_regressor(  # noqa: C901
    data_transformer: Union[object, list[object]],
    regressor: object,
    data_transformer_label: Optional[list[str]] = None,
    regressor_label: Optional[str] = None,
) -> object:
    """
    Combine data transformation models with a regressor into a unified estimator that preserves component names.

    This wrapper functions similarly to scikit-learn's Pipeline but compatible with any transformer and regressor that follows scikit-learn's method conventions.

    Parameters
    ----------
    data_transformer : object or list of object
        Data transformation model(s), any data transformation or feature selection model(s).

    regressor : object
        Classification model.

    data_transformer_label : str or list of str or None, optional
        Label(s) for the transformer(s). Defaults to model class names if not specified.

    regressor_label: str or None
        Label for the regressor. Defaults to model class name if not specified.

    Returns
    -------
    object
        Combined regression model.

    Examples
    --------
    Prepare models::

        >>> from sklearn.feature_selection import SelectKBest, f_regression
        >>> from from sklearn.preprocessing import StandardScaler
        >>> from sklearn.neighbors import KNeighborsRegressor

        >>> selector = SelectKBest(f_regression, k=5)
        >>> scaler = StandardScaler()
        >>> knn = KNeighborsRegressor(n_neighbors=3)

    Without specifying model labels::

        >>> combined_model = combine_transformer_regressor([scaler, selector], knn)

    Specify model labels::

        >>> combined_model = combine_transformer_regressor(
        ...     [scaler, selector],
        ...     knn,
        ...     data_transformer_label=['scaler', 'selector'],
        ...     regressor_label='knn'
        ... )
    """  # noqa: E501
    # Validate input models
    if isinstance(data_transformer, list):
        if len(data_transformer) < 1:
            raise ValueError("List of transformers must contain at least 1 transformer, got 0.")
        else:
            data_transformers: list = data_transformer
    else:
        data_transformers = [data_transformer]
    # Validate transformers
    for data_transformer in data_transformers:
        _data_transformer_validator(data_transformer)
    # Validate estimator
    _regressor_validator(regressor)

    # Create combined name
    transformer_name = ""
    transformer_name_list = []
    if data_transformer_label is None:
        for data_transformer in data_transformers:
            transformer_name = transformer_name + f"{data_transformer.__class__.__name__}_"
            transformer_name_list.append(data_transformer.__class__.__name__)
    else:
        if len(data_transformers) != len(data_transformer_label):
            raise ValueError(
                f"Got {len(data_transformers)} data transformers, but got {len(data_transformer_label)} label:\
                    {data_transformer_label}"
            )
        for label in data_transformer_label:
            transformer_name = transformer_name + f"{label}_"
            transformer_name_list.append(label)
    if regressor_label is None:
        regressor_label = regressor.__class__.__name__
    combined_name = transformer_name + regressor_label

    # Create new model class to customize name
    class CombinedModel(TransRegressor):

        def __repr__(self) -> str:
            return "TransRegressor_" + combined_name

        def __str__(self) -> str:
            return combined_name

    # Customize name
    CombinedModel.__name__ = combined_name
    CombinedModel.__qualname__ = combined_name

    # Add name attributes
    CombinedModel._transformer_labels = transformer_name_list
    CombinedModel._regressor_label = regressor_label

    # Create model instance
    combined_model = CombinedModel(data_transformers=data_transformers, regressor=regressor)

    return combined_model


# %% Combiner models


# Combined classifier
class TransClassifier(BaseEstimator, ClassifierMixin):
    """
    Combine a chain of data transformation models with a classifier into a unified estimator.
    This wrapper functions similarly to scikit-learn's Pipeline but only requires transformers and classifier to follow scikit-learn's method conventions.

    Attributes
    ----------
    data_transformers : list of object
        List of data transformation models, any data transformation or feature selection model.
    classifier : object
        Classification model.

    Methods
    -------
    fit(X, y)
        Fit the transformer on X, then fit the classifier on transformed X.
    transform(X)
        Transform X using the fitted transformer.
    predict(X)
        Transform X using the fitted transformer, then predict using the fitted classifier.
    predict_proba(X)
        Predict the probability of X using the fitted transformer and classifier.
    score(X, y)
        Compute the accuracy score of the fitted models on the provided X and y.
    """  # noqa: E501

    @simple_type_validator
    def __init__(self, data_transformers: list[object], classifier: object) -> None:
        # Validate transformers
        for transformer in data_transformers:
            _data_transformer_validator(transformer)
        self.data_transformers: list[object] = data_transformers
        # Validate classifiers
        _classifier_validator(classifier)
        self.classifier: object = classifier
        self._is_trans_classifier: bool = True

    @simple_type_validator
    def fit(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
        y: Annotated[Any, arraylike_validator(ndim=1)],
    ) -> 'TransClassifier':
        """
        Fit the transformer on X, then fit the classifier on transformed X.

        Parameters
        ----------
        X : 2D array-like
            Training dataset.
        y : 1D array-like, optional
            Training target values.

        Returns
        -------
        TransRegressor
            The fitted combined model.
        """
        # Validate inputs
        X = np.asarray(X)
        X, y = check_X_y(X, y)
        # Fit transformers and transform X_train
        for transformer in self.data_transformers:
            assert hasattr(transformer, 'fit')
            try:
                transformer.fit(X)  # Try unsupervised
            except Exception:
                transformer.fit(X, y)  # Try supervised
            assert hasattr(transformer, 'transform')
            X = transformer.transform(X)
        # Fit classifier
        assert hasattr(self.classifier, 'fit')
        self.classifier.fit(X, y)
        self.is_fitted_ = True
        if hasattr(self.classifier, 'classes_'):
            self.classes_ = self.classifier.classes_  # Add attr classes_ to outer model wrapper
        else:
            raise ValueError(
                "Invalid classifier without 'classes_', "
                + "fitted classfier must have attribute 'classes_' to interpret 'predict_proba'."
            )
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, 'is_fitted_') and self.is_fitted_

    @simple_type_validator
    def transform(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted transformer.

        Parameters
        ----------
        X : 2D array-like
            Training dataset.

        Returns
        -------
        np.ndarray
            Transformed X.
        """
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X)
        X = check_array(X)
        for transformer in self.data_transformers:
            assert hasattr(transformer, 'transform')
            X = transformer.transform(X)
        result: np.ndarray = np.asarray(X)
        return result

    @simple_type_validator
    def predict(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted transformer, then predict targets using the fitted classifier.

        Parameters
        ----------
        X : 2D array-like
            Dataset to predict.

        Returns
        -------
        np.ndarray
            Predicted target values of X.
        """
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X)
        X = check_array(X)
        # Transform X_pred
        for transformer in self.data_transformers:
            assert hasattr(transformer, 'transform')
            X = transformer.transform(X)
        # Predict using classifier
        assert hasattr(self.classifier, 'predict')
        y_pred = self.classifier.predict(X)
        y_pred_arr: np.ndarray = np.asarray(y_pred)
        return y_pred_arr

    @simple_type_validator
    def predict_proba(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted transformer, then predict the probabilities of the targets using the fitted classifier.

        Parameters
        ----------
        X : 2D array-like
            Dataset to predict.

        Returns
        -------
        np.ndarray
            Predicted target values of X.
        """  # noqa: E501
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X)
        X = check_array(X)
        for transformer in self.data_transformers:
            assert hasattr(transformer, 'transform')
            X = transformer.transform(X)
        assert hasattr(self.classifier, 'predict_proba')
        y_pred_proba = self.classifier.predict_proba(X)
        y_pred_proba_arr: np.ndarray = np.asarray(y_pred_proba)
        return y_pred_proba_arr

    @simple_type_validator
    def score(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
        y: Annotated[Any, arraylike_validator(ndim=1)],
    ) -> float:
        """
        Compute overall accuracy score of the fitted models on the provided X and y.

        Parameters
        ----------
        X : 2D array-like
            Test dataset.
        y : Annotated[Any, arraylike_validator(ndim, optional
            Test target values.

        Returns
        -------
        TransRegressor
            The fitted combined model.
        """
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X)
        for transformer in self.data_transformers:
            assert hasattr(transformer, 'transform')
            X = transformer.transform(X)
        if hasattr(self.classifier, 'score'):
            assert hasattr(self.classifier, 'score')
            overall_accuracy = self.classifier.score(X, y)
        else:
            assert hasattr(self.classifier, 'predict')
            y_pred = self.classifier.predict(X)
            overall_accuracy = accuracy_score(y, y_pred)
        overall_accuracy = float(overall_accuracy)
        return overall_accuracy


# Combined regressor
class TransRegressor(BaseEstimator, RegressorMixin):
    """
    Combine a chain of data transformation models with a regressor into a unified estimator.
    This wrapper functions similarly to scikit-learn's Pipeline but compatible with any transformer and regressor that follows scikit-learn's method conventions.

    Attributes
    ----------
    data_transformers : list
        List of Data transformation models, any data transformation or feature selection model.
    regressor
        Regression model.

    Methods
    -------
    fit(X, y)
        Fit the transformer on X, then fit the regressor on transformed X.
    transform(X)
        Transform X using the fitted transformer.
    predict(X)
        Transform X using the fitted transformer, then predict using the fitted regressor.
    score(X, y)
        Compute the goodness of fit score of the fitted models on the provided X and y.
    """  # noqa: E501

    @simple_type_validator
    def __init__(self, data_transformers: list[object], regressor: object) -> None:
        # Validate transformers
        for transformer in data_transformers:
            _data_transformer_validator(transformer)
        self.data_transformers: list[object] = data_transformers
        # Validate regressor
        _regressor_validator(regressor)
        self.regressor: object = regressor
        self._is_trans_regressor: bool = True

    @simple_type_validator
    def fit(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
        y: Annotated[Any, arraylike_validator(ndim=1)],
    ) -> 'TransRegressor':
        """
        Fit the transformer on X, then fit the regressor on transformed X.

        Parameters
        ----------
        X : 2D array-like
            Training dataset.
        y : Annotated[Any, arraylike_validator(ndim, optional
            Training target values.

        Returns
        -------
        TransRegressor
            The fitted combined model.
        """
        # Validate inputs
        X = np.asarray(X)
        X, y = check_X_y(X, y)
        # Fit transformer and transform X_train
        for transformer in self.data_transformers:
            assert hasattr(transformer, 'fit')
            try:
                transformer.fit(X)  # Try unsupervised
            except Exception:
                transformer.fit(X, y)  # Try supervised
            assert hasattr(transformer, 'transform')
            X = transformer.transform(X)
        # Fit regressor
        assert hasattr(self.regressor, 'fit')
        self.regressor.fit(X, y)
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, 'is_fitted_') and self.is_fitted_

    @simple_type_validator
    def transform(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted transformer.

        Parameters
        ----------
        X : 2D array-like
            Training dataset.

        Returns
        -------
        np.ndarray
            Transformed X.
        """
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X)
        X = check_array(X)
        for transformer in self.data_transformers:
            assert hasattr(transformer, 'transform')
            X = transformer.transform(X)
        result: np.ndarray = np.asarray(X)
        return result

    @simple_type_validator
    def predict(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted transformer, then predict targets using the fitted regressor.

        Parameters
        ----------
        X : 2D array-like
            Dataset to predict.

        Returns
        -------
        np.ndarray
            Predicted target values of X.
        """
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X)
        X = check_array(X)
        # Transform X_pred
        for transformer in self.data_transformers:
            assert hasattr(transformer, 'transform')
            X = transformer.transform(X)
        # Predict using regressor
        assert hasattr(self.regressor, 'predict')
        y_pred = self.regressor.predict(X)
        y_pred_arr: np.ndarray = np.asarray(y_pred)
        return y_pred_arr

    @simple_type_validator
    def score(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
        y: Annotated[Any, arraylike_validator(ndim=1)],
    ) -> float:
        """
        Compute the goodness of fit score of the fitted models on the provided X and y.

        Parameters
        ----------
        X : 2D array-like
            Test dataset.
        y : Annotated[Any, arraylike_validator(ndim, optional
            Test target values.

        Returns
        -------
        TransRegressor
            The fitted combined model.
        """
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X)
        for transformer in self.data_transformers:
            assert hasattr(transformer, 'transform')
            X = transformer.transform(X)
        if hasattr(self.regressor, 'score'):
            assert hasattr(self.regressor, 'score')
            gof_score = self.regressor.score(X, y)
        else:
            assert hasattr(self.regressor, 'predict')
            y_pred = self.regressor.predict(X)
            gof_score = r2_score(y, y_pred)
        gof_score = float(gof_score)
        return gof_score


# %% Identity transformer for passthrough


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    A passthrough transformer that returns the input data unchanged.

    This transformer is useful as a placeholder in pipelines or for enforcing a consistent transformer interface without modifying data.
    """  # noqa: E501

    @simple_type_validator
    def fit(
        self,
        X: Annotated[Any, arraylike_validator()],  # noqa: N803
        y: Optional[Annotated[Any, arraylike_validator()]] = None,
    ) -> 'IdentityTransformer':
        """
        Fit the transformer.

        This method performs no computation and exists to satisfy the scikit-learn estimator interface.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,), optional
            Target values. Ignored.

        Returns
        -------
        IdentityTransformer
            The fitted transformer.
        """
        return self

    @simple_type_validator
    def transform(self, X: Annotated[Any, arraylike_validator()]) -> np.ndarray:  # noqa: N803
        """
        Return the input data unchanged.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        numpy.ndarray
            The input data converted to a NumPy array.
        """
        result: np.ndarray = np.asarray(X)
        return result

    @simple_type_validator
    def fit_transform(
        self,
        X: Annotated[Any, arraylike_validator()],  # noqa: N803
        y: Optional[Annotated[Any, arraylike_validator()]] = None,
    ) -> np.ndarray:  # noqa: N803
        """
        Fit the transformer and return the input data unchanged.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,), optional
            Target values. Ignored.

        Returns
        -------
        numpy.ndarray
            The input data converted to a NumPy array.
        """
        result: np.ndarray = np.asarray(X)
        return result


# %% Combined model statistics


@simple_type_validator
def _convert_metrics_combined_model(metrics_dict: dict, modeleva_report_dir: str) -> dict:  # noqa: C901
    """Convert metrics_dict for marginal performance analysis of combined model components."""
    report_dir = modeleva_report_dir
    # Get paths of model evaluation reports of each chain
    dir_paths = [
        entry.path
        for entry in os.scandir(unc_path(report_dir))
        if entry.is_dir() and "Data_chain_Preprocessing_#" in entry.name and "_Model_" in entry.name
    ]

    # Get chain info with combined model components
    chain_model_components = []
    for path in dir_paths:
        chain_process_info_path = unc_path(
            path + "/.__swectral_dill_data/.__swectral_core_result_Chain_process_info.dill"
        )
        with open(chain_process_info_path, 'rb') as f:
            chain_process_info = dill.load(f)
        combined_model_info_path = unc_path(path + "/.__swectral_dill_data/.__swectral_Combined_model_info.dill")
        if os.path.exists(combined_model_info_path):
            with open(combined_model_info_path, 'rb') as f:
                combined_model_info = dill.load(f)
                model_component_list = list(combined_model_info['model_transformer_labels']) + [
                    combined_model_info['model_estimator_label']
                ]
        else:
            model_component_list = [list(chain_process_info['Chain_in_process_label'])[-1]]  # If not combined model
        chain_model_components.append((list(chain_process_info['Chain_in_process_ID']), model_component_list))

    # Match components of combined model with the chain in metrics_dict
    chain_arr = np.asarray(metrics_dict['chains_in_ID'])
    chain_modelcomp_list = []
    for row_chain in chain_arr:
        for chain_model_component in chain_model_components:
            if list(row_chain) == chain_model_component[0]:
                chain_modelcomp_list.append(chain_model_component[1])

    # Validate model_comp_list length
    if len(chain_modelcomp_list) != len(chain_arr):
        raise ValueError("Process mismatch occurred. Pipeline model evaluation reports may be incomplete.")
    # Validate model_comp_list component numbers
    modelcomp_len: int = max([len(chain_modelcomps) for chain_modelcomps in chain_modelcomp_list])
    chain_modelcomp_list_valid = []
    for i, chain_modelcomps in enumerate(chain_modelcomp_list):
        if len(chain_modelcomps) == modelcomp_len:
            chain_modelcomp_list_valid.append(chain_modelcomps)
        elif len(chain_modelcomps) == 1:
            chain_modelcomp_list_valid.append([''] * (modelcomp_len - 1) + chain_modelcomps)
        else:
            raise ValueError(
                f"Chain combined model have inconsistent number of components,\
                    expected number of components: {modelcomp_len}\
                    chain process ids: {chain_arr[i,:]},\
                    chain model components: {chain_modelcomps}"
            )
    model_comp_df = pd.DataFrame(
        chain_modelcomp_list_valid, columns=[f"Model_step_{i}" for i in range(1, modelcomp_len + 1)]
    )

    # Replace chains_in_ID of metrics_dict
    metrics_dict['chains_in_ID'] = model_comp_df
    if 'regression_metrics' in list(metrics_dict.keys()):
        metrics_dict['regression_metrics'] = pd.concat([metrics_dict['regression_metrics'], model_comp_df], axis=1)
    if 'macro_metrics' in list(metrics_dict.keys()):
        metrics_dict['macro_metrics'] = pd.concat([metrics_dict['macro_metrics'], model_comp_df], axis=1)
    if 'micro_metrics' in list(metrics_dict.keys()):
        metrics_dict['micro_metrics'] = pd.concat([metrics_dict['micro_metrics'], model_comp_df], axis=1)

    return metrics_dict


# Combined model component marginal performance statistics
@simple_type_validator
def combined_model_marginal_stats(
    report_directory: str, metrics_dict: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Compute marginal model performance statistics on combined model components of the performance metrics from SpecPipe model evaluation reports.

    Parameters
    ----------
    pipeline_config_dir : str
        Root of SpecPipe report directory.
    metrics_dict: dict or None
        Dictionary of performance summary from ``performance_metrics_summary``.
        If provided, the summary is skipped. Default is None.

    Returns
    -------
    dict[str, Any]
        Dictionary of marginal model performance statistics on combined model components at each step.
    """  # noqa: E501

    pipeline_config_dir = f"{report_directory}/SpecPipe_configuration/"
    model_evaluation_report_dir = f"{report_directory}/Modeling/Model_evaluation_reports/"

    # Validate metrics_dict
    compute_metrics_dict: bool
    if metrics_dict is None:
        compute_metrics_dict = True
    elif set(metrics_dict.keys()) == {
        "is_regression",
        "chains_in_ID",
        "macro_metrics",
        "micro_metrics",
    }:
        compute_metrics_dict = False
    elif set(metrics_dict.keys()) == {
        "is_regression",
        "chains_in_ID",
        "regression_metrics",
    }:
        compute_metrics_dict = False
    else:
        compute_metrics_dict = True
        warnings.warn(
            f"\n\nCorrupted metrics_dict, got keys: {metrics_dict.keys()}\nThe metrics_dict is recomputed.\n\n",
            stacklevel=2,
        )

    # Summarize performance
    if compute_metrics_dict:
        metrics_dict = performance_metrics_summary(pipeline_config_dir, model_evaluation_report_dir)

    # Add model step info
    metrics_dict_model = _convert_metrics_combined_model(metrics_dict, model_evaluation_report_dir)

    # Compute stats
    assert metrics_dict is not None
    if metrics_dict["is_regression"]:
        marginal_performance_stats = regression_performance_marginal_stats(
            metrics_dict_model, pipeline_config_dir, model_evaluation_report_dir, validate_process=False
        )
    else:
        marginal_performance_stats = classification_performance_marginal_stats(
            metrics_dict_model, pipeline_config_dir, model_evaluation_report_dir, validate_process=False
        )

    return marginal_performance_stats


# %% ====== Estimator Connectors / Stack tools ======
# %%
