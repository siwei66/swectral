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
from collections import defaultdict
import itertools
import copy

# Serialization
import dill

# Model
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone  # noqa: E402
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted  # noqa: E402
from sklearn.metrics import accuracy_score, r2_score  # noqa: E402

# Local
from .specio import simple_type_validator, arraylike_validator, unc_path  # noqa: E402
from .pipeline_validator import (
    _classifier_validator,
    _regressor_validator,
    _data_transformer_validator,
    _resampler_validator,
)
from .groupstats import (
    performance_metrics_summary,
    regression_performance_marginal_stats,
    classification_performance_marginal_stats,
)


# %% ====== Transformer - Estimator Connectors ======
# Connect data transformer and estimator
# For evaluation of dimension reduction, feature selection or other feature engineering models that requires fitting.

# %% Combinition composers


@simple_type_validator
def factorial_model_chains(  # noqa: C901
    *step_trainable_processors: tuple[Union[list[object], dict[str, object], None], ...],
    estimators: Union[list[object], dict[str, object]],
    is_regression: bool = True,
    preserve_train_state: bool = False,
) -> list[object]:
    """
    Combine trainable data preprocessing models of each step with estimators into chained models using a full-factorial approach.

    Parameters
    ----------
    step_trainable_processors : tuple of (list of object, dict mapping str to object, or None)
        Data preprocessing model instance of each step. Valid inputs for each element include:

            - sklearn-style transformers implementing `fit` and `transform`.
            - imblearn-style resamplers implementing `fit_resample`.

        Customize trainable processor name using dictionary input as {custom_name : trainable_processor}.

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
    """  # noqa: E501

    # Validate given step_trainable_processors
    if step_trainable_processors is None:
        raise ValueError("step_trainable_processors is missing, provide at least one step of transformers.")
    if len(step_trainable_processors) < 1:
        raise ValueError("step_trainable_processors is missing, provide at least one step of transformers.")
    # Validate transformers at each step
    for step_i_transformers in step_trainable_processors:
        if step_i_transformers is None:
            raise ValueError("Provided step_trainable_processors cannot be None.")
        if len(step_i_transformers) < 1:
            raise ValueError("Provided step_trainable_processors cannot be empty.")

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
    for option in step_trainable_processors:
        if isinstance(option, dict):
            chain_model_options.append(list(option.values()))
            chain_label_options.append(list(option.keys()))
        elif isinstance(option, list):
            chain_model_options.append(list(option))
            chain_label_options.append([model.__class__.__name__ for model in option])
        else:
            raise ValueError(f"step_trainable_processors must be dict or list, got type: {type(option)}")

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
        trainable_processor = chain_model_list[i][:-1]
        estimator = chain_model_list[i][-1]
        # Get label
        trainable_processor_label = chain_label_list[i][:-1]
        estimator_label = chain_label_list[i][-1]
        # Combine models in chain
        if is_regression:
            combined_model = combine_regressor(
                trainable_processor=trainable_processor,
                regressor=estimator,
                trainable_processor_label=trainable_processor_label,
                regressor_label=estimator_label,
            )
        else:
            combined_model = combine_classifier(
                trainable_processor=trainable_processor,
                classifier=estimator,
                trainable_processor_label=trainable_processor_label,
                classifier_label=estimator_label,
            )
        # Append combined model
        combined_models.append(combined_model)

    return combined_models


# %% Combiners / Pipeline tools


# Constructor - combined classifier
@simple_type_validator
def combine_classifier(  # noqa: C901
    trainable_processor: Union[object, list[object]],
    classifier: object,
    trainable_processor_label: Union[str, list[str], None] = None,
    classifier_label: Optional[str] = None,
    preserve_train_state: bool = False,  # TODO: new
) -> object:
    """
    Combine trainable data preprocessing models with a classifier into a unified estimator that preserves component names.

    This wrapper functions similarly to scikit-learn's Pipeline but compatible with any transformer and classifier that follows scikit-learn's method conventions.

    Parameters
    ----------
    trainable_processor : object or list of object
        Data preprocessing model(s), any trainable data preprocessing, feature selection, or resampling model(s).

    classifier : object
        Classification model.

    trainable_processor_label : str or list of str or None, optional
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
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.neighbors import KNeighborsClassifier

        >>> selector = SelectKBest(f_classif, k=5)
        >>> scaler = StandardScaler()
        >>> knn = KNeighborsClassifier(n_neighbors=3)

    Without specifying model labels::

        >>> combined_model = combine_classifier([scaler, selector], knn)

    Specify model labels::

        >>> combined_model = combine_classifier(
        ...     [scaler, selector],
        ...     knn,
        ...     trainable_processor_label=['scaler', 'selector'],
        ...     classifier_label='knn'
        ... )
    """  # noqa: E501
    # Validate input models
    if isinstance(trainable_processor, list):
        if len(trainable_processor) < 1:
            raise ValueError("List of transformers must contain at least 1 transformer, got 0.")
        else:
            trainable_processors: list = trainable_processor
    else:
        trainable_processors = [trainable_processor]
    # Validate trainable processors
    for trainable_processor in trainable_processors:
        if hasattr(trainable_processor, "fit_resample"):
            _resampler_validator(trainable_processor)
        else:
            _data_transformer_validator(trainable_processor)
    # Validate estimator
    _classifier_validator(classifier)

    # Create combined name
    trainable_processor_name = ""
    trainable_processor_name_list = []
    if trainable_processor_label is None:
        for trainable_processor in trainable_processors:
            trainable_processor_name = trainable_processor_name + f"{trainable_processor.__class__.__name__}_"
            trainable_processor_name_list.append(trainable_processor.__class__.__name__)
    else:
        if isinstance(trainable_processor_label, str):
            trainable_processor_label = [trainable_processor_label]
        assert isinstance(trainable_processor_label, list)
        if len(trainable_processors) != len(trainable_processor_label):
            raise ValueError(
                f"Got {len(trainable_processors)} data transformers, but got {len(trainable_processor_label)} label:\
                    {trainable_processor_label}"
            )
        for label in trainable_processor_label:
            trainable_processor_name = trainable_processor_name + f"{label}_"
            trainable_processor_name_list.append(label)
    if classifier_label is None:
        classifier_label = classifier.__class__.__name__
    combined_name = trainable_processor_name + classifier_label

    # Create new model class to customize name
    class CombinedModel(CombinedClassifier):

        def __repr__(self) -> str:
            return "CombinedClassifier_" + combined_name

        def __str__(self) -> str:
            return combined_name

    # Customize name
    CombinedModel.__name__ = combined_name
    CombinedModel.__qualname__ = combined_name

    # Add name attributes
    CombinedModel._preprocessor_labels = trainable_processor_name_list
    CombinedModel._classifier_label = classifier_label

    # Create model instance
    combined_model = CombinedModel(
        trainable_processors=trainable_processors,
        classifier=classifier,
        preserve_train_state=preserve_train_state,  # TODO: new
    )

    return combined_model


# Constructor - combined regressor
@simple_type_validator
def combine_regressor(  # noqa: C901
    trainable_processor: Union[object, list[object]],
    regressor: object,
    trainable_processor_label: Union[str, list[str], None] = None,
    regressor_label: Optional[str] = None,
    preserve_train_state: bool = False,  # TODO: new
) -> object:
    """
    Combine trainable data preprocessing models with a regressor into a unified estimator that preserves component names.

    This wrapper functions similarly to scikit-learn's Pipeline but compatible with any transformer and regressor that follows scikit-learn's method conventions.

    Parameters
    ----------
    trainable_processor : object or list of object
        Data preprocessing model(s), any trainable data preprocessing, feature selection, or resampling model(s).

    regressor : object
        Regression model.

    trainable_processor_label : str or list of str or None, optional
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
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.neighbors import KNeighborsRegressor

        >>> selector = SelectKBest(f_regression, k=5)
        >>> scaler = StandardScaler()
        >>> knn = KNeighborsRegressor(n_neighbors=3)

    Without specifying model labels::

        >>> combined_model = combine_regressor([scaler, selector], knn)

    Specify model labels::

        >>> combined_model = combine_regressor(
        ...     [scaler, selector],
        ...     knn,
        ...     trainable_processor_label=['scaler', 'selector'],
        ...     regressor_label='knn'
        ... )
    """  # noqa: E501
    # Validate input models
    if isinstance(trainable_processor, list):
        if len(trainable_processor) < 1:
            raise ValueError("List of transformers must contain at least 1 transformer, got 0.")
        else:
            trainable_processors: list = trainable_processor
    else:
        trainable_processors = [trainable_processor]
    # Validate trainable processors
    for trainable_processor in trainable_processors:
        if hasattr(trainable_processor, "fit_resample"):
            _resampler_validator(trainable_processor)
        else:
            _data_transformer_validator(trainable_processor)
    # Validate estimator
    _regressor_validator(regressor)

    # Create combined name
    trainable_processor_name = ""
    trainable_processor_name_list = []
    if trainable_processor_label is None:
        for trainable_processor in trainable_processors:
            trainable_processor_name = trainable_processor_name + f"{trainable_processor.__class__.__name__}_"
            trainable_processor_name_list.append(trainable_processor.__class__.__name__)
    else:
        # TODO: new
        if isinstance(trainable_processor_label, str):
            trainable_processor_label = [trainable_processor_label]
        assert isinstance(trainable_processor_label, list)
        if len(trainable_processors) != len(trainable_processor_label):
            raise ValueError(
                f"Got {len(trainable_processors)} data transformers, but got {len(trainable_processor_label)} label:\
                    {trainable_processor_label}"
            )
        for label in trainable_processor_label:
            trainable_processor_name = trainable_processor_name + f"{label}_"
            trainable_processor_name_list.append(label)
    if regressor_label is None:
        regressor_label = regressor.__class__.__name__
    combined_name = trainable_processor_name + regressor_label

    # Create new model class to customize name
    class CombinedModel(CombinedRegressor):

        def __repr__(self) -> str:
            return "CombinedRegressor_" + combined_name

        def __str__(self) -> str:
            return combined_name

    # Customize name
    CombinedModel.__name__ = combined_name
    CombinedModel.__qualname__ = combined_name

    # Add name attributes
    CombinedModel._preprocessor_labels = trainable_processor_name_list
    CombinedModel._regressor_label = regressor_label

    # Create model instance
    combined_model = CombinedModel(
        trainable_processors=trainable_processors,
        regressor=regressor,
        preserve_train_state=preserve_train_state,  # TODO: new
    )

    return combined_model


# %% Combiner models


# Safe clone a model - if sklearn-compliant use cheaper sklearn clone, or use deepcopy
@simple_type_validator
def safe_clone(model_instance: object, preserve_train_state: bool) -> object:
    """
    Safe clone for ML pipeline components.
    If sklearn-compliant, use cheaper sklearn clone, or use deepcopy.
    """
    if preserve_train_state:
        cloned = copy.deepcopy(model_instance)
    else:
        # Try sklearn ecosystem cloning
        try:
            cloned = clone(model_instance)

            # Optional but highly recommended sanity check
            if hasattr(model_instance, "get_params"):
                original_params = model_instance.get_params()
                cloned_params = cloned.get_params()

                if str(original_params) != str(cloned_params):
                    raise ValueError("Clone parameter reconstruction mismatch.")
            else:
                raise ValueError("Not sklearn-compatible, use deepcopy instead.")

        except Exception:
            # Fallback for non-sklearn-compatible objects
            cloned = copy.deepcopy(model_instance)
    return cloned


# %% Combined classifier


class CombinedClassifier(BaseEstimator, ClassifierMixin):
    """
    Combine a chain of data preprocessing models with a classifier into a unified estimator.
    This wrapper function works similarly to scikit-learn's Pipeline.
    It requires transformers and classifier to follow scikit-learn's method conventions, and requires resamplers to implement method 'fit_resample'.

    Attributes
    ----------
    trainable_processors : list of object
        List of data preprocessing models, any data transformation, feature selection or resampling model.
    classifier : object
        Classification model.

    Methods
    -------
    fit(X, y)
        Fit the trainable processor on X, then fit the classifier on transformed X.
    transform(X)
        Transform X using the fitted trainable processor.
    predict(X)
        Transform X using the fitted trainable processor, then predict using the fitted classifier.
    predict_proba(X)
        Predict the probability of X using the fitted trainable processor and classifier.
    score(X, y)
        Compute the accuracy score of the fitted models on the provided X and y.
    """  # noqa: E501

    @simple_type_validator
    def __init__(
        self,
        trainable_processors: list[object],
        classifier: object,
        preserve_train_state: bool = False,
    ) -> None:
        # Validate transformers and resamplers
        for trainable_processor in trainable_processors:
            if hasattr(trainable_processor, "fit_resample"):
                _resampler_validator(trainable_processor)
            else:
                _data_transformer_validator(trainable_processor)
        self.trainable_processors: list[object] = trainable_processors
        # TODO: new
        self.trainable_processors_: Optional[list[object]] = None
        # Validate classifiers
        _classifier_validator(classifier)
        self.classifier: object = classifier
        # TODO: new
        self.classifier_: Optional[object] = None
        # TODO: new
        self.preserve_train_state: bool = preserve_train_state
        self._is_combined_classifier: bool = True

    # TODO: new
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """get_params for sklearn.GridSearchCV."""
        # Get top-level params automatically via BaseEstimator
        params: dict[str, Any] = super().get_params(deep=False)
        if deep:
            if hasattr(self.classifier, "get_params"):
                for key, value in self.classifier.get_params(deep=True).items():
                    params[f"classifier__{key}"] = value
            for i, proc in enumerate(self.trainable_processors):
                if hasattr(proc, "get_params"):
                    for key, value in proc.get_params(deep=True).items():
                        params[f"processor_{i}__{key}"] = value
        return params

    # TODO: new
    def set_params(self, **params: Any) -> "CombinedClassifier":
        """set_params for sklearn.GridSearchCV."""
        if not params:
            return self
        nested_params: dict[str, dict[str, Any]] = defaultdict(dict)
        for key, value in params.items():
            if "__" in key:
                prefix, sub_key = key.split("__", 1)
                nested_params[prefix][sub_key] = value
            else:
                setattr(self, key, value)
        # Dispatch to the classifier
        if "classifier" in nested_params:
            est = self.classifier
            if hasattr(est, "set_params"):
                est.set_params(**nested_params["classifier"])  # type: ignore
        # Dispatch to the transformers
        for i, proc in enumerate(self.trainable_processors):
            prefix = f"processor_{i}"
            if prefix in nested_params:
                if hasattr(proc, "set_params"):
                    proc.set_params(**nested_params[prefix])  # type: ignore
        return self

    @simple_type_validator
    def fit(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
        y: Annotated[Any, arraylike_validator(ndim=1)],
    ) -> 'CombinedClassifier':
        """
        Fit the trainable processor on X, then fit the classifier on transformed X.

        Parameters
        ----------
        X : 2D array-like
            Training dataset.
        y : 1D array-like, optional
            Training target values.

        Returns
        -------
        CombinedClassifier
            The fitted combined model.
        """
        # TODO: new
        # Clone models
        self.trainable_processors_ = [
            safe_clone(trainable_processor, self.preserve_train_state)
            for trainable_processor in self.trainable_processors
        ]
        self.classifier_ = safe_clone(self.classifier, self.preserve_train_state)
        # Validate inputs
        X = np.asarray(X)
        X, y = check_X_y(X, y)
        # Fit trainable preprocessors and transform X_train
        for trainable_processor in self.trainable_processors_:
            # TODO: new
            # imblearn-style sampler
            if hasattr(trainable_processor, "fit_resample"):
                X, y = trainable_processor.fit_resample(X, y)
            elif hasattr(trainable_processor, "fit") and hasattr(trainable_processor, "transform"):
                # sklearn transformer supporting fit & transform
                try:
                    trainable_processor.fit(X, y)
                except Exception:
                    trainable_processor.fit(X)
                X = trainable_processor.transform(X)
            else:
                related_method = ['fit', 'transform', 'fit_resample', 'predict', 'fit_predict', 'predict_proba']
                existing_methods = [m for m in related_method if hasattr(trainable_processor, m)]
                raise TypeError(
                    f"Transformer '{type(trainable_processor).__name__}' is incompatible. "
                    + "Need 'fit_resample' or ('fit' and 'transform'). "
                    + f"Found methods: {existing_methods}"
                )
        # Fit classifier
        assert hasattr(self.classifier_, 'fit')
        self.classifier_.fit(X, y)
        self.is_fitted_ = True
        if hasattr(self.classifier_, 'classes_'):
            self.classes_ = self.classifier_.classes_  # Add attr classes_ to outer model wrapper
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
        Transform X using the fitted trainable processor.

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
        # TODO: changed
        assert self.trainable_processors_ is not None
        for trainable_processor in self.trainable_processors_:
            if hasattr(trainable_processor, 'transform') and (not hasattr(trainable_processor, 'fit_resample')):
                X = trainable_processor.transform(X)
        result: np.ndarray = np.asarray(X)
        return result

    @simple_type_validator
    def predict(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted trainable processor, then predict targets using the fitted classifier.

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
        # TODO: changed
        assert self.trainable_processors_ is not None
        for trainable_processor in self.trainable_processors_:
            # TODO: changed
            if hasattr(trainable_processor, 'transform') and (not hasattr(trainable_processor, 'fit_resample')):
                X = trainable_processor.transform(X)
        # Predict using classifier
        assert self.classifier_ is not None
        assert hasattr(self.classifier_, 'predict')
        y_pred = self.classifier_.predict(X)
        y_pred_arr: np.ndarray = np.asarray(y_pred)
        return y_pred_arr

    @simple_type_validator
    def predict_proba(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted trainable_processor, then predict the probabilities of the targets using the fitted classifier.

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
        # TODO: changed
        assert self.trainable_processors_ is not None
        for trainable_processor in self.trainable_processors_:
            if hasattr(trainable_processor, 'transform') and (not hasattr(trainable_processor, 'fit_resample')):
                X = trainable_processor.transform(X)
        assert self.classifier_ is not None
        assert hasattr(self.classifier_, 'predict_proba')
        y_pred_proba = self.classifier_.predict_proba(X)
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
        CombinedClassifier
            The fitted combined model.
        """
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X)
        # TODO: changed
        assert self.trainable_processors_ is not None
        for trainable_processor in self.trainable_processors_:
            if hasattr(trainable_processor, 'transform') and (not hasattr(trainable_processor, 'fit_resample')):
                X = trainable_processor.transform(X)
        if hasattr(self.classifier_, 'score'):
            assert self.classifier_ is not None
            assert hasattr(self.classifier_, 'score')
            overall_accuracy = self.classifier_.score(X, y)
        else:
            assert self.classifier_ is not None
            assert hasattr(self.classifier_, 'predict')
            y_pred = self.classifier_.predict(X)
            overall_accuracy = accuracy_score(y, y_pred)
        overall_accuracy = float(overall_accuracy)
        return overall_accuracy


# %% Combined regressor


# Combined regressor
class CombinedRegressor(BaseEstimator, RegressorMixin):
    """
    Combine a chain of data preprocessing models with a regressor into a unified estimator.
    This wrapper function works similarly to scikit-learn's Pipeline.
    It requires transformers and regressor to follow scikit-learn's method conventions, and requires resamplers to implement method 'fit_resample'.

    Attributes
    ----------
    trainable_processors : list of object
        List of data preprocessing models, any data transformation, feature selection or resampling model.
    regressor : object
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
    def __init__(
        self,
        trainable_processors: list[object],
        regressor: object,
        preserve_train_state: bool = False,
    ) -> None:
        # Validate transformers and resamplers
        for trainable_processor in trainable_processors:
            if hasattr(trainable_processor, "fit_resample"):
                _resampler_validator(trainable_processor)
            else:
                _data_transformer_validator(trainable_processor)
        self.trainable_processors: list[object] = trainable_processors
        # TODO: new
        self.trainable_processors_: Optional[list[object]] = None
        # Validate regressors
        _regressor_validator(regressor)
        self.regressor: object = regressor
        # TODO: new
        self.regressor_: Optional[object] = None
        # TODO: new
        self.preserve_train_state: bool = preserve_train_state
        self._is_combined_regressor: bool = True

    # TODO: new
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """get_params for sklearn.GridSearchCV."""
        # Get top-level params automatically via BaseEstimator
        params: dict[str, Any] = super().get_params(deep=False)
        if deep:
            if hasattr(self.regressor, "get_params"):
                for key, value in self.regressor.get_params(deep=True).items():
                    params[f"regressor__{key}"] = value
            for i, proc in enumerate(self.trainable_processors):
                if hasattr(proc, "get_params"):
                    for key, value in proc.get_params(deep=True).items():
                        params[f"processor_{i}__{key}"] = value
        return params

    # TODO: new
    def set_params(self, **params: Any) -> "CombinedRegressor":
        """set_params for sklearn.GridSearchCV."""
        if not params:
            return self
        nested_params: dict[str, dict[str, Any]] = defaultdict(dict)
        for key, value in params.items():
            if "__" in key:
                prefix, sub_key = key.split("__", 1)
                nested_params[prefix][sub_key] = value
            else:
                setattr(self, key, value)
        # Dispatch to the regressor
        if "regressor" in nested_params:
            est = self.regressor
            if hasattr(est, "set_params"):
                est.set_params(**nested_params["regressor"])  # type: ignore
        # Dispatch to the transformers
        for i, proc in enumerate(self.trainable_processors):
            prefix = f"processor_{i}"
            if prefix in nested_params:
                if hasattr(proc, "set_params"):
                    proc.set_params(**nested_params[prefix])  # type: ignore
        return self

    @simple_type_validator
    def fit(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
        y: Annotated[Any, arraylike_validator(ndim=1)],
    ) -> 'CombinedRegressor':
        """
        Fit the trainable processor on X, then fit the regressor on transformed X.

        Parameters
        ----------
        X : 2D array-like
            Training dataset.
        y : 1D array-like, optional
            Training target values.

        Returns
        -------
        CombinedRegressor
            The fitted combined model.
        """
        # TODO: new
        # Clone models
        self.trainable_processors_ = [
            safe_clone(trainable_processor, self.preserve_train_state)
            for trainable_processor in self.trainable_processors
        ]
        self.regressor_ = safe_clone(self.regressor, self.preserve_train_state)
        # Validate inputs
        X = np.asarray(X)
        X, y = check_X_y(X, y)
        # Fit trainable preprocessors and transform X_train
        for trainable_processor in self.trainable_processors_:
            # TODO: new
            # imblearn-style sampler
            if hasattr(trainable_processor, "fit_resample"):
                X, y = trainable_processor.fit_resample(X, y)
            elif hasattr(trainable_processor, "fit") and hasattr(trainable_processor, "transform"):
                # sklearn transformer supporting fit & transform
                try:
                    trainable_processor.fit(X, y)
                except Exception:
                    trainable_processor.fit(X)
                X = trainable_processor.transform(X)
            else:
                related_method = ['fit', 'transform', 'fit_resample', 'predict', 'fit_predict', 'predict_proba']
                existing_methods = [m for m in related_method if hasattr(trainable_processor, m)]
                raise TypeError(
                    f"Transformer '{type(trainable_processor).__name__}' is incompatible. "
                    + "Need 'fit_resample' or ('fit' and 'transform'). "
                    + f"Found methods: {existing_methods}"
                )
        # Fit regressor
        assert hasattr(self.regressor_, 'fit')
        self.regressor_.fit(X, y)
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, 'is_fitted_') and self.is_fitted_

    @simple_type_validator
    def transform(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted trainable processor.

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
        # TODO: changed
        assert self.trainable_processors_ is not None
        for trainable_processor in self.trainable_processors_:
            if hasattr(trainable_processor, 'transform') and (not hasattr(trainable_processor, 'fit_resample')):
                X = trainable_processor.transform(X)
        result: np.ndarray = np.asarray(X)
        return result

    @simple_type_validator
    def predict(self, X: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:  # noqa: N803
        """
        Transform X using the fitted trainable processor, then predict targets using the fitted regressor.

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
        # TODO: changed
        assert self.trainable_processors_ is not None
        for trainable_processor in self.trainable_processors_:
            # TODO: changed
            if hasattr(trainable_processor, 'transform') and (not hasattr(trainable_processor, 'fit_resample')):
                X = trainable_processor.transform(X)
        # Predict using regressor
        assert self.regressor_ is not None
        assert hasattr(self.regressor_, 'predict')
        y_pred = self.regressor_.predict(X)
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
        y : 1D array-like, optional
            Test target values.

        Returns
        -------
        CombinedRegressor
            The fitted combined model.
        """
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X)
        # TODO: changed
        assert self.trainable_processors_ is not None
        for trainable_processor in self.trainable_processors_:
            if hasattr(trainable_processor, 'transform') and (not hasattr(trainable_processor, 'fit_resample')):
                X = trainable_processor.transform(X)
        if hasattr(self.regressor_, 'score'):
            assert self.regressor_ is not None
            assert hasattr(self.regressor_, 'score')
            gof_score = self.regressor_.score(X, y)
        else:
            assert self.regressor_ is not None
            assert hasattr(self.regressor_, 'predict')
            y_pred = self.regressor_.predict(X)
            gof_score = r2_score(y, y_pred)
        gof_score = float(gof_score)
        return gof_score


# %% Identity transformer for passthrough


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    A passthrough scikit-learn-style transformer that returns the input data unchanged.

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


# %% Identity transformer for passthrough


class IdentityResampler(BaseEstimator):
    """
    A passthrough imblearn-style resampler that returns the input data unchanged.

    This resampler is useful as a placeholder in pipelines or for enforcing a consistent resampler interface without modifying data.
    """  # noqa: E501

    @simple_type_validator
    def fit_resample(
        self,
        X: Annotated[Any, arraylike_validator()],  # noqa: N803
        y: Optional[Annotated[Any, arraylike_validator()]] = None,
    ) -> tuple[Any, Any]:  # noqa: N803
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
        tuple[Any, Any]
            The input X and y.
        """
        return X, y


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
                model_component_list = list(combined_model_info['model_preprocessor_labels']) + [
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
    report_directory: str,
    metrics_dict: Optional[dict[str, Any]] = None,
    *,
    _space_wait_timeout: int = 36000,
    _reserve_free_pct: float = 5.0,
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
        metrics_dict = performance_metrics_summary(
            pipeline_config_dir=pipeline_config_dir,
            model_evaluation_report_dir=model_evaluation_report_dir,
            _space_wait_timeout=_space_wait_timeout,
            _reserve_free_pct=_reserve_free_pct,
        )

    # Add model step info
    metrics_dict_model = _convert_metrics_combined_model(metrics_dict, model_evaluation_report_dir)

    # Compute stats
    print("Combined model component decomposition...")
    assert metrics_dict is not None
    if metrics_dict["is_regression"]:
        marginal_performance_stats = regression_performance_marginal_stats(
            metrics_dict=metrics_dict_model,
            pipeline_config_dir=pipeline_config_dir,
            model_evaluation_report_dir=model_evaluation_report_dir,
            validate_process=False,
            _space_wait_timeout=_space_wait_timeout,
            _reserve_free_pct=_reserve_free_pct,
        )
    else:
        marginal_performance_stats = classification_performance_marginal_stats(
            metrics_dict=metrics_dict_model,
            pipeline_config_dir=pipeline_config_dir,
            model_evaluation_report_dir=model_evaluation_report_dir,
            validate_process=False,
            _space_wait_timeout=_space_wait_timeout,
            _reserve_free_pct=_reserve_free_pct,
        )

    return marginal_performance_stats


# %% ====== Estimator Connectors / Stack tools ======
# %%
