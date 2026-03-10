# -*- coding: utf-8 -*-
"""
Test model connectors - transformer-estimator connectors

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import os
import numpy as np
import pytest
import shutil
import sys
import tempfile
from typing import Optional, Any, Annotated

from sklearn.decomposition import PCA  # Unsupervised transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression  # Supervised transformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score

# from imblearn.over_sampling import RandomOverSampler

from swectral.specio import simple_type_validator, arraylike_validator, silent
from swectral.example_data import create_test_spec_exp
from swectral.vegeind.demo_data import create_specind_demo_data
from swectral.pipeline import SpecPipe
from swectral.roistats import roi_mean

from swectral.modelconnector import (
    combine_classifier,
    combine_regressor,
    factorial_model_chains,
    IdentityTransformer,
    IdentityResampler,
    CombinedClassifier,
    CombinedRegressor,
)


# %% Helper - create test data


@simple_type_validator
def create_model_test_data(
    is_classification: bool = True, nsample: int = 20, nfeature: int = 462, seed: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create classification test data."""
    incr_id = nsample // 3
    incr_idt = nsample // 6
    # X train
    X_train = np.array(create_specind_demo_data(nsample=nsample, nband=nfeature))  # noqa: N806
    X_train[0:incr_id] = X_train[0:incr_id] + 5000
    X_train[incr_id : (incr_id * 2)] = X_train[incr_id : (incr_id * 2)] + 10000

    # y train
    if is_classification:
        y_train = np.full((nsample,), 'c')
        y_train[0:incr_id] = 'a'
        y_train[incr_id : (incr_id * 2)] = 'b'
    else:
        y_train = np.zeros((nsample,))
        y_train[0:incr_id] = np.random.random((incr_id,)) + 0.5
        y_train[incr_id : (incr_id * 2)] = np.random.random((incr_id,)) + 1.0

    # Test data
    X_test = np.array(create_specind_demo_data(nsample=nsample // 2, nband=nfeature))  # noqa: N806
    X_test[0:incr_idt] = X_test[0:incr_idt] + 5000
    X_test[incr_idt : (incr_idt * 2)] = X_test[incr_idt : (incr_idt * 2)] + 10000

    # y test
    if is_classification:
        y_test = np.full((nsample // 2,), 'c')
        y_test[0:incr_idt] = 'a'
        y_test[incr_idt : (incr_idt * 2)] = 'b'
    else:
        y_test = np.zeros((nsample // 2,))
        y_test[0:incr_idt] = np.random.random((incr_idt,)) + 0.5
        y_test[incr_idt : (incr_idt * 2)] = np.random.random((incr_idt,)) + 1.0

    return X_train, y_train, X_test, y_test


# Helper - mock user-defined model
class MockUserModel:
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
        y: Annotated[Any, arraylike_validator(ndim=1)],
    ) -> "MockUserModel":
        # Simulated fit logic
        return self

    def predict(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
    ) -> np.ndarray:
        y_pred: np.ndarray = np.zeros(len(X))
        return y_pred

    def predict_proba(
        self,
        X: Annotated[Any, arraylike_validator(ndim=2)],  # noqa: N803
    ) -> np.ndarray:
        proba: np.ndarray = np.column_stack([np.ones(len(X)), np.zeros(len(X))])
        return proba


# %% test functions: combine_classifier


class TestCombineTransformerClassifier:
    """Test combine_classifier functionalities."""

    @staticmethod
    def test_supervised_transformer() -> None:
        """Test supervised transformer."""
        # Data
        test_data = create_model_test_data(nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Create model
        test_transformer = SelectKBest(f_classif, k=5)
        test_classifier = RandomForestClassifier(n_estimators=10)
        combined_model = combine_classifier(test_transformer, test_classifier)

        # Assert modeling
        assert hasattr(combined_model, '_is_combined_classifier')
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'predict_proba')
        assert hasattr(combined_model, 'score')
        combined_model.fit(X_train, y_train)
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (X_test.shape[0], 5)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == y_test.shape
        y_est_proba = combined_model.predict_proba(X_test)
        assert isinstance(y_est_proba, np.ndarray)
        assert y_est_proba.shape == (y_test.shape[0], 3)
        test_score = combined_model.score(X_test, y_test)
        assert test_score > 0

        # Assert dynamic name
        assert combined_model.__class__.__name__ == "SelectKBest_RandomForestClassifier"
        assert combined_model._preprocessor_labels == ['SelectKBest']
        assert combined_model._classifier_label == 'RandomForestClassifier'

    @staticmethod
    def test_unsupervised_transformer() -> None:
        """Test unsupervised transformer."""
        # Data
        test_data = create_model_test_data(nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Create model
        test_transformer = PCA(n_components=5)
        test_classifier = RandomForestClassifier(n_estimators=10)
        combined_model = combine_classifier(test_transformer, test_classifier)

        # Assert modeling
        assert hasattr(combined_model, '_is_combined_classifier')
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'predict_proba')
        assert hasattr(combined_model, 'score')
        combined_model.fit(X_train, y_train)
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (X_test.shape[0], 5)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == y_test.shape
        y_est_proba = combined_model.predict_proba(X_test)
        assert isinstance(y_est_proba, np.ndarray)
        assert y_est_proba.shape == (y_test.shape[0], 3)
        assert list(combined_model.classes_) == list(combined_model.classifier_.classes_)
        test_score = combined_model.score(X_test, y_test)
        assert test_score > 0

        # Assert dynamic name
        assert combined_model.__class__.__name__ == "PCA_RandomForestClassifier"
        assert combined_model._preprocessor_labels == ['PCA']
        assert combined_model._classifier_label == 'RandomForestClassifier'

    @staticmethod
    def test_resampler_classifier() -> None:
        """Test imblearn resampler within CombinedClassifier."""
        # Data
        test_data = create_model_test_data(is_classification=True, nsample=100)
        X_train, y_train, X_test, y_test = test_data  # noqa: N806

        # Create model
        test_resampler = IdentityResampler()
        test_classifier = RandomForestClassifier(n_estimators=10)
        combined_model = combine_classifier(test_resampler, test_classifier)

        # Assert modeling
        assert hasattr(combined_model, '_is_combined_classifier')
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'predict_proba')
        combined_model.fit(X_train, y_train)
        # Assert identity of transform behavior if no transformer is applied
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == X_test.shape
        assert np.array_equal(X_trans, X_test)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == (X_test.shape[0],)
        y_est_proba = combined_model.predict_proba(X_test)
        assert y_est_proba.shape[1] == len(np.unique(y_train))

        # Assert dynamic name
        assert combined_model.__class__.__name__ == "IdentityResampler_RandomForestClassifier"
        assert combined_model._preprocessor_labels == ['IdentityResampler']
        assert combined_model._classifier_label == 'RandomForestClassifier'

    @staticmethod
    def test_multi_preprocessor_chain() -> None:
        """Test a chain of multiple transformers with mixed transformer types."""
        # Data
        test_data = create_model_test_data(nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Create model
        chain_transformers = [IdentityResampler(), PCA(n_components=8), SelectKBest(f_classif, k=5)]
        test_classifier = RandomForestClassifier(n_estimators=10)
        combined_model = combine_classifier(chain_transformers, test_classifier)

        # Assert modeling
        assert hasattr(combined_model, '_is_combined_classifier')
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'predict_proba')
        assert hasattr(combined_model, 'score')
        combined_model.fit(X_train, y_train)
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (X_test.shape[0], 5)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == y_test.shape
        y_est_proba = combined_model.predict_proba(X_test)
        assert isinstance(y_est_proba, np.ndarray)
        assert y_est_proba.shape == (y_test.shape[0], 3)
        assert list(combined_model.classes_) == list(combined_model.classifier_.classes_)
        test_score = combined_model.score(X_test, y_test)
        assert test_score > 0

        # Assert dynamic name and labels
        combined_name = 'IdentityResampler_PCA_SelectKBest_RandomForestClassifier'
        assert combined_model.__class__.__name__ == combined_name
        assert combined_model._preprocessor_labels == ['IdentityResampler', 'PCA', 'SelectKBest']
        assert combined_model._classifier_label == 'RandomForestClassifier'

    @staticmethod
    def test_custom_labels() -> None:
        """Test setting custom labels for transformers and the estimator."""
        # Data
        test_data = create_model_test_data(nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Create model
        # Label mismatch
        with pytest.raises(ValueError, match="Got 2 data transformers, but got 1 label"):
            combined_model = combine_classifier(
                trainable_processor=[StandardScaler(), PCA(n_components=5)],
                classifier=RandomForestClassifier(n_estimators=10),
                trainable_processor_label=["Scaler"],
                classifier_label="RF",
            )
        # Correct
        chain_transformers = [StandardScaler(), PCA(n_components=5)]
        trans_labels = ["Scaler", "PCA8"]
        test_classifier = RandomForestClassifier(n_estimators=10)
        est_label = "RF"
        combined_model = combine_classifier(
            trainable_processor=chain_transformers,
            classifier=test_classifier,
            trainable_processor_label=trans_labels,
            classifier_label=est_label,
        )

        # Assert modeling
        assert hasattr(combined_model, '_is_combined_classifier')
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'predict_proba')
        assert hasattr(combined_model, 'score')
        combined_model.fit(X_train, y_train)
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (X_test.shape[0], 5)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == y_test.shape
        y_est_proba = combined_model.predict_proba(X_test)
        assert isinstance(y_est_proba, np.ndarray)
        assert y_est_proba.shape == (y_test.shape[0], 3)
        assert list(combined_model.classes_) == list(combined_model.classifier_.classes_)
        test_score = combined_model.score(X_test, y_test)
        assert test_score > 0

        # Assert dynamic name and labels
        combined_name = 'Scaler_PCA8_RF'
        assert combined_model.__class__.__name__ == combined_name
        assert combined_model._preprocessor_labels == trans_labels
        assert combined_model._classifier_label == est_label

    @staticmethod
    def test_param_routing() -> None:
        """Ensures set_params reaches both Sklearn and Custom models."""
        scaler = StandardScaler()
        est = MockUserModel()

        # Test classifier
        model = combine_classifier(
            trainable_processor=[scaler],
            classifier=est,
        )

        # Test routing to Sklearn component
        model.set_params(processor_0__with_mean=False)
        assert scaler.with_mean is False, "Failed to route to transformer"

        # Test routing to Custom component (should not crash)
        model.set_params(classifier__learning_rate=0.1)

        scaler = StandardScaler()
        est = MockUserModel()

    @staticmethod
    def test_grid_search_compatibility() -> None:
        """Tests if GridSearchCV can drive the CombinedClassifier."""
        # Data
        test_data = create_model_test_data(nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Test classifier
        model = combine_classifier(
            trainable_processor=[StandardScaler()],
            classifier=RandomForestClassifier(n_estimators=10),
        )

        # Grid targeting the transformer and the RF classifier
        param_grid = {
            'processor_0__with_std': [True, False],
            'classifier__n_estimators': [5, 10],
            'classifier__max_depth': [None, 5],
            'preserve_train_state': [False],
        }

        grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train, y_train)

        # Test best_params_ exists and contains our keys
        assert 'processor_0__with_std' in grid.best_params_
        assert 'classifier__n_estimators' in grid.best_params_

        # Test the best_estimator_ is actually a fitted CombinedClassifier
        best_model = grid.best_estimator_
        assert isinstance(best_model, CombinedClassifier)
        assert hasattr(best_model, "is_fitted_")
        assert best_model.is_fitted_ is True

        # Test inference works with the best model
        y_pred = best_model.predict(X_test)
        assert len(y_pred) == len(X_test)

        # Test the probability output (crucial for classifiers)
        y_proba = best_model.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), len(np.unique(y_train)))

        # Check performance
        score = accuracy_score(y_test, y_pred)
        assert score >= 0


# %% test functions: combine_regressor


class TestCombineTransformerRegressor:
    """Test combine_regressor functionalities."""

    @staticmethod
    def test_supervised_transformer() -> None:
        """Test supervised transformer."""
        # Data
        test_data = create_model_test_data(is_classification=False, nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Create model
        test_transformer = SelectKBest(f_regression, k=5)
        test_regressor = RandomForestRegressor(n_estimators=10)
        combined_model = combine_regressor(test_transformer, test_regressor)

        # Assert modeling
        assert hasattr(combined_model, '_is_combined_regressor')
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'score')
        combined_model.fit(X_train, y_train)
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (X_test.shape[0], 5)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == y_test.shape
        test_score = combined_model.score(X_test, y_test)
        assert test_score > 0

        # Assert dynamic name
        assert combined_model.__class__.__name__ == "SelectKBest_RandomForestRegressor"
        assert combined_model._preprocessor_labels == ['SelectKBest']
        assert combined_model._regressor_label == 'RandomForestRegressor'

    @staticmethod
    def test_unsupervised_transformer() -> None:
        """Test unsupervised transformer."""
        # Data
        test_data = create_model_test_data(is_classification=False, nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Create model
        test_transformer = PCA(n_components=5)
        test_regressor = RandomForestRegressor(n_estimators=10)
        combined_model = combine_regressor(test_transformer, test_regressor)

        # Assert modeling
        assert hasattr(combined_model, '_is_combined_regressor')
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'score')
        combined_model.fit(X_train, y_train)
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (X_test.shape[0], 5)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == y_test.shape
        test_score = combined_model.score(X_test, y_test)
        assert test_score > 0

        # Assert dynamic name
        assert combined_model.__class__.__name__ == "PCA_RandomForestRegressor"
        assert combined_model._preprocessor_labels == ['PCA']
        assert combined_model._regressor_label == 'RandomForestRegressor'

    @staticmethod
    def test_resampler_regressor() -> None:
        """Test imblearn resampler within CombinedRegressor."""
        # Data
        test_data = create_model_test_data(is_classification=False, nsample=100)
        X_train, y_train, X_test, y_test = test_data  # noqa: N806

        # Create model
        test_resampler = IdentityResampler()
        test_regressor = RandomForestRegressor(n_estimators=10)
        combined_model = combine_regressor(test_resampler, test_regressor)

        # Assert modeling
        assert hasattr(combined_model, '_is_combined_regressor')
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        combined_model.fit(X_train, y_train)
        # Assert identity of transform behavior if no transformer is applied
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == X_test.shape
        assert np.array_equal(X_trans, X_test)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == (X_test.shape[0],)

        # Assert dynamic name
        assert combined_model.__class__.__name__ == "IdentityResampler_RandomForestRegressor"
        assert combined_model._preprocessor_labels == ['IdentityResampler']
        assert combined_model._regressor_label == 'RandomForestRegressor'

    @staticmethod
    def test_multi_preprocessor_chain() -> None:
        """Test a chain of multiple transformers with mixed transformer types."""
        # Data
        test_data = create_model_test_data(is_classification=False, nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Create model
        chain_transformers = [IdentityResampler(), PCA(n_components=8), SelectKBest(f_regression, k=5)]
        test_regressor = RandomForestRegressor(n_estimators=10)
        combined_model = combine_regressor(chain_transformers, test_regressor)

        # Assert modeling
        assert hasattr(combined_model, '_is_combined_regressor')
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'score')
        combined_model.fit(X_train, y_train)
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (X_test.shape[0], 5)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == y_test.shape
        test_score = combined_model.score(X_test, y_test)
        assert test_score > 0

        # Assert dynamic name and labels
        combined_name = 'IdentityResampler_PCA_SelectKBest_RandomForestRegressor'
        assert combined_model.__class__.__name__ == combined_name
        assert combined_model._preprocessor_labels == ['IdentityResampler', 'PCA', 'SelectKBest']
        assert combined_model._regressor_label == 'RandomForestRegressor'

    @staticmethod
    def test_custom_labels() -> None:
        """Test setting custom labels for transformers and the estimator."""
        # Data
        test_data = create_model_test_data(is_classification=False, nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Create model
        # Label mismatch
        with pytest.raises(ValueError, match="Got 2 data transformers, but got 1 label"):
            combined_model = combine_regressor(
                trainable_processor=[StandardScaler(), PCA(n_components=5)],
                regressor=RandomForestRegressor(n_estimators=10),
                trainable_processor_label=["Scaler"],
                regressor_label="RF",
            )
        # Correct
        chain_transformers = [StandardScaler(), PCA(n_components=5)]
        trans_labels = ["Scaler", "PCA8"]
        test_regressor = RandomForestRegressor(n_estimators=10)
        est_label = "RF"
        combined_model = combine_regressor(
            trainable_processor=chain_transformers,
            regressor=test_regressor,
            trainable_processor_label=trans_labels,
            regressor_label=est_label,
        )

        # Assert modeling
        assert hasattr(combined_model, '_is_combined_regressor')
        assert hasattr(combined_model, 'fit')
        assert hasattr(combined_model, 'transform')
        assert hasattr(combined_model, 'predict')
        assert hasattr(combined_model, 'score')
        combined_model.fit(X_train, y_train)
        X_trans = combined_model.transform(X_test)  # noqa: N806
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape == (X_test.shape[0], 5)
        y_est = combined_model.predict(X_test)
        assert isinstance(y_est, np.ndarray)
        assert y_est.shape == y_test.shape
        test_score = combined_model.score(X_test, y_test)
        assert test_score > 0

        # Assert dynamic name and labels
        combined_name = 'Scaler_PCA8_RF'
        assert combined_model.__class__.__name__ == combined_name
        assert combined_model._preprocessor_labels == trans_labels
        assert combined_model._regressor_label == est_label

    @staticmethod
    def test_param_routing() -> None:
        """Ensures set_params reaches both Sklearn and Custom models."""
        scaler = StandardScaler()
        est = MockUserModel()

        # Test regressor
        model = combine_regressor(
            trainable_processor=[scaler],
            regressor=est,
        )

        # Test routing to Sklearn component
        model.set_params(processor_0__with_mean=False)
        assert scaler.with_mean is False, "Failed to route to transformer"

        # Test routing to Custom component (should not crash)
        model.set_params(regressor__learning_rate=0.1)

    @staticmethod
    def test_grid_search_compatibility() -> None:
        """Tests if GridSearchCV can drive the CombinedRegressor."""
        # Data
        test_data = create_model_test_data(is_classification=False, nsample=100)
        X_train = test_data[0]  # noqa: N806
        y_train = test_data[1]
        X_test = test_data[2]  # noqa: N806
        y_test = test_data[3]

        # Test regressor
        model = combine_regressor(
            trainable_processor=[StandardScaler()],
            regressor=RandomForestRegressor(n_estimators=10),
        )

        # Grid targeting the transformer and the RF regressor
        param_grid = {
            'processor_0__with_std': [True, False],
            'regressor__n_estimators': [5, 10],
            'regressor__max_depth': [None, 5],
            'preserve_train_state': [False],
        }

        grid = GridSearchCV(model, param_grid, cv=3, scoring='r2')
        grid.fit(X_train, y_train)

        # Test best_params_ exists and contains our keys
        assert 'processor_0__with_std' in grid.best_params_
        assert 'regressor__n_estimators' in grid.best_params_

        # Test the best_estimator_ is actually a fitted CombinedRegressor
        best_model = grid.best_estimator_
        assert isinstance(best_model, CombinedRegressor)
        assert hasattr(best_model, "is_fitted_")
        assert best_model.is_fitted_ is True

        # Test inference works with the best model
        y_pred = best_model.predict(X_test)
        assert len(y_pred) == len(X_test)

        # Check performance
        r2 = r2_score(y_test, y_pred)
        assert -1.0 <= r2 <= 1.0


# %% test functions: factorial_model_chains


class TestFactorialTransformerChains:
    """Test cases for factorial_model_chains function."""

    @staticmethod
    def test_basic_functionality_one_step_list() -> None:
        """Test basic functionality with one step of transformers (list input)."""
        # Arrange
        transformers_step1 = [PCA(n_components=5), SelectKBest(f_regression)]
        estimators = [KNeighborsRegressor(n_neighbors=3)]

        # Act
        models = factorial_model_chains(transformers_step1, estimators=estimators, is_regression=True)

        # Assert
        assert len(models) == 2

        # Check model attr
        for model in models:
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')

        # Test combined model functionality
        X_train, y_train, X_test, y_test = create_model_test_data(  # noqa: N806
            is_classification=False, nsample=20, nfeature=10, seed=42
        )
        for model in models:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            assert len(predictions) == len(y_test)

    @staticmethod
    def test_construct_multi_steps_list() -> None:
        """Test with multiple steps of transformers."""
        # Arrange
        transformers_step1 = [StandardScaler(), MinMaxScaler()]
        transformers_step2 = [PCA(n_components=10)]
        transformers_step3 = {
            'feat8': SelectKBest(f_regression, k=8),
            'feat5': SelectKBest(f_regression, k=5),
            'feat3': SelectKBest(f_regression, k=3),
        }
        estimators = [RandomForestRegressor(n_estimators=10), KNeighborsRegressor(n_neighbors=3)]

        # Act
        models = factorial_model_chains(
            transformers_step1, transformers_step2, transformers_step3, estimators=estimators, is_regression=True
        )

        # Assert
        assert len(models) == 12

        # Check model attr
        for model in models:
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')

    @staticmethod
    def test_basic_functionality_one_step_dict() -> None:
        """Test basic functionality with one step of transformers (dict input)."""
        # Arrange
        transformers_step1 = {'scaler1': StandardScaler()}
        estimators = {'rf': RandomForestRegressor(n_estimators=10)}

        # Act
        models = factorial_model_chains(transformers_step1, estimators=estimators, is_regression=True)

        # Assert
        assert len(models) == 1

        # Check model attr
        for model in models:
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')

    @staticmethod
    def test_classification() -> None:
        """Test combined models for classification."""
        # Arrange
        transformers_step1 = [StandardScaler()]
        estimators = [RandomForestClassifier(n_estimators=10)]

        # Act
        models = factorial_model_chains(transformers_step1, estimators=estimators, is_regression=False)

        # Assert
        assert len(models) == 1

        # Check model attr
        for model in models:
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
            assert hasattr(model, 'predict_proba')

        # Test combined model functionality
        X_train, y_train, X_test, y_test = create_model_test_data(  # noqa: N806
            is_classification=True, nsample=20, nfeature=10, seed=42
        )
        for model in models:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            assert hasattr(model, 'classes_')
            assert len(predictions) == len(y_test)

    @staticmethod
    def test_empty_step_trainable_processors() -> None:
        """Test error when step_trainable_processors is empty."""
        with pytest.raises(ValueError, match="step_trainable_processors is missing"):
            factorial_model_chains(estimators=[KNeighborsRegressor(n_neighbors=3)], is_regression=True)

    @staticmethod
    def test_none_step_trainable_processors() -> None:
        """Test error when a step in step_trainable_processors is None."""
        with pytest.raises(ValueError, match="step_trainable_processors cannot be None"):
            factorial_model_chains(None, estimators=[KNeighborsRegressor(n_neighbors=3)], is_regression=True)

    @staticmethod
    def test_empty_list_in_step_trainable_processors() -> None:
        """Test error when a step in step_trainable_processors is an empty list."""
        with pytest.raises(ValueError, match="step_trainable_processors cannot be empty"):
            factorial_model_chains([], estimators=[KNeighborsRegressor(n_neighbors=3)], is_regression=True)

    @staticmethod
    def test_duplicate_labels_across_steps() -> None:
        """Test error when there are duplicate labels across transformer steps."""
        transformers_step1 = {'scaler': StandardScaler(), 'pca': PCA(n_components=5)}
        transformers_step2 = {
            'scaler': MinMaxScaler(),  # Duplicate label with step1
            'kbest': SelectKBest(f_regression, k=5),
        }
        with pytest.raises(ValueError, match="Duplicate label"):
            factorial_model_chains(
                transformers_step1,
                transformers_step2,
                estimators={'lr': KNeighborsRegressor(n_neighbors=3)},
                is_regression=True,
            )

    @staticmethod
    def test_duplicate_default_labels() -> None:
        """Test error when default labels conflict."""
        # Act & Assert
        with pytest.raises(ValueError, match="Duplicate label"):
            factorial_model_chains(
                [PCA(n_components=5), PCA(n_components=3)],
                estimators=[KNeighborsRegressor(n_neighbors=3)],
                is_regression=True,
            )

    @staticmethod
    def test_transformer_label_conflicts_with_estimator() -> None:
        """Test error when transformer label conflicts with estimator label."""
        # Act & Assert
        with pytest.raises(ValueError, match="already used as an estimator label"):
            factorial_model_chains(
                {'model': StandardScaler()},
                estimators={'model': KNeighborsRegressor(n_neighbors=3)},
                is_regression=True,
            )


# %% test functions: combined_model_marginal_stats


class TestCombinedModelMarginalStats:
    """Test for combined_model_marginal_stats function."""

    @staticmethod
    @silent
    def test_combined_model_chain_regression() -> None:
        """Test for combined_model_marginal_stats regression functionality."""
        test_dir = tempfile.mkdtemp()

        # Create test files and test exp
        test_exp = create_test_spec_exp(dir_path=test_dir, sample_n=20, n_bands=10, is_regression=True)
        # Compose combined models
        models = factorial_model_chains(
            {'scale1': StandardScaler()},
            {'resample1': IdentityResampler()},
            {'feat5': SelectKBest(f_regression, k=5), 'feat3': SelectKBest(f_regression, k=3)},
            estimators=[KNeighborsRegressor(n_neighbors=3)],
        )
        # Create SpecPipe
        pipe = SpecPipe(test_exp)
        pipe.add_process(5, 7, 0, roi_mean)
        for model in models:
            pipe.add_model(model, validation_method="2-fold", influence_analysis_config=None)

        # Run pipeline
        pipe.run()

        # Assert model step reports
        path = test_dir + "/Modeling/Model_evaluation_reports/"
        reports = [
            entry.name
            for entry in os.scandir(path)
            if (not entry.is_dir()) and "_stats_model_step_" in entry.name and ".csv" in entry.name
        ]
        assert len(reports) == 1

        # Clear dir after assert
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    @staticmethod
    @silent
    def test_combined_model_chain_classification() -> None:
        """Test for combined_model_marginal_stats classification functionality."""
        test_dir = tempfile.mkdtemp()

        # Create test files and test exp
        test_exp = create_test_spec_exp(dir_path=test_dir, sample_n=20, n_bands=10, is_regression=False)
        # Compose combined models
        models = factorial_model_chains(
            {'scale1': StandardScaler()},
            {'resample1': IdentityResampler()},
            {'feat5': SelectKBest(f_classif, k=5), 'feat3': SelectKBest(f_classif, k=3)},
            estimators=[KNeighborsClassifier(n_neighbors=3)],
            is_regression=False,
        )
        # Create SpecPipe
        pipe = SpecPipe(test_exp)
        pipe.add_process(5, 7, 0, roi_mean)
        for model in models:
            pipe.add_model(model, validation_method="2-fold", influence_analysis_config=None)

        # Run pipeline
        pipe.run()

        # Assert model step reports
        path = test_dir + "/Modeling/Model_evaluation_reports/"
        reports = [
            entry.name
            for entry in os.scandir(path)
            if (not entry.is_dir()) and "_stats_model_step_" in entry.name and ".csv" in entry.name
        ]
        assert len(reports) == 2

        # Clear dir after assert
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


# %% test functions: IdentityTransformer for passthrough


def test_identity_transformer() -> None:
    """Test IdentityTransformer functionality"""
    model = IdentityTransformer()

    assert hasattr(model, "fit")
    assert hasattr(model, "transform")
    assert hasattr(model, "fit_transform")

    data = np.array([[1, 2, 3], [3, 4, 5]])
    model.fit(data)
    assert np.all(data == model.transform(data))
    assert np.all(data == model.fit_transform(data))


# %% Test main


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
