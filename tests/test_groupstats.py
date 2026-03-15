# -*- coding: utf-8 -*-
"""
Tests for group statistics tools

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# ruff: noqa: I001
# ruff: noqa: N806
# OS
import os  # noqa: E402
import sys  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

# Test
import unittest
import pytest

# Basic data
import numpy as np
import pandas as pd

# Models
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore

# Local
from swectral.specio import RealNumber, silent
from swectral.example_data import create_test_spec_exp
from swectral.specexp import SpecExp
from swectral.pipeline import SpecPipe
from swectral.roistats import roi_mean, roi_median, roi_std

# Functions to test
from swectral.groupstats import (
    chain_sample_group_stats,
    sample_group_stats,
    process_id_label_lookup_dict,
    process_id_to_label,
    process_label_to_id,
    performance_metrics_summary,
    regression_performance_marginal_stats,
    classification_performance_marginal_stats,
    performance_marginal_stats,
)


# %% Helpers for groupstats input data and test pipeline


# Helper: Create test data for SpecExp group stats
def create_test_spec_exp_two_groups(
    dir_path: str, sample_n: int = 10, n_bands: int = 4, is_regression: bool = True
) -> SpecExp:
    """Create a test SpecExp with two experiment groups"""
    specexp_1 = create_test_spec_exp(dir_path, sample_n, n_bands, is_regression)
    # Add a new group
    specexp_1.add_groups(['test_group1'])
    # Copy image info to a new group
    imgt1 = specexp_1.images[0]
    imgt2_list = list(imgt1)
    imgt2_list[1] = 'test_group1'
    imgt2 = tuple(imgt2_list)
    specexp_1._images.append(imgt2)
    # Change half rois to new group
    rois_1 = specexp_1._rois_from_file
    rois_2 = rois_1[:]
    for i in range(sample_n // 2, sample_n):
        roit = rois_2[i]
        roit_list = list(roit)
        roit_list[1] = 'test_group1'
        roit = tuple(roit_list)
        rois_2[i] = roit
    specexp_1._rois_from_file = rois_2
    # Update image mask and image sample
    specexp_1._update_image_rev()
    # Update ROIs
    specexp_1._update_roi()
    # Update sample labels & targets
    specexp_1._update_sample_labels_targets()

    assert isinstance(specexp_1, SpecExp)
    return specexp_1


# Helpers: processing functions


def replace_nan(v: np.ndarray) -> np.ndarray:  # type: ignore
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore


# Helper: Create test data for pipeline marginal performance stats


def create_test_spec_pipe_multi_factor(spec_exp: SpecExp) -> SpecPipe:
    """Create a test SpecPipe with two steps with two factors."""
    pipe = SpecPipe(spec_exp)
    # Step number fixe to 3 for the following test - cannot be modified independently
    # Step 0 - ROI stats
    pipe.add_process(5, 7, 0, roi_mean)
    pipe.add_process(5, 7, 0, roi_std)
    pipe.add_process(5, 7, 0, roi_median)
    # Step 1 - Spec1d process
    pipe.add_process("spec1d", "spec1d", 0, replace_nan)
    # Step 2 - Models
    is_regression = True
    for y in spec_exp.ls_targets()['Target_value']:
        if not isinstance(y, RealNumber):
            is_regression = False
    if is_regression:
        knn_regressor = KNeighborsRegressor(n_neighbors=3)
        rf_regressor = RandomForestRegressor(n_estimators=10)
        pipe.add_model(knn_regressor, validation_method="2-fold")
        pipe.add_model(rf_regressor, validation_method="2-fold")
    else:
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        rf_classifier = RandomForestClassifier(n_estimators=10)
        pipe.add_model(knn_classifier, validation_method="2-fold")
        pipe.add_model(rf_classifier, validation_method="2-fold")

    return pipe


# %% test functions : group statistics functions


class TestGroupStats(unittest.TestCase):
    """Test performance_marginal_stats functionality."""

    # For regression
    test_dir: str
    spec_exp_reg: SpecExp
    spec_pipe_reg: SpecPipe

    # For classification
    test_dir1: str
    spec_exp_cls: SpecExp
    spec_pipe_cls: SpecPipe

    @classmethod
    @silent
    def setUpClass(cls) -> None:
        """Setup method that runs before each test"""
        # Regression pipe
        cls.test_dir = (str(tempfile.mkdtemp()).replace("\\", "/") + "/").replace("//", "/")
        cls.spec_exp_reg = create_test_spec_exp_two_groups(cls.test_dir, is_regression=True)
        cls.spec_pipe_reg = create_test_spec_pipe_multi_factor(cls.spec_exp_reg)
        cls.spec_pipe_reg.run(summary=False)
        # Classification pipe
        cls.test_dir1 = (str(tempfile.mkdtemp()).replace("\\", "/") + "/").replace("//", "/")
        cls.spec_exp_cls = create_test_spec_exp_two_groups(cls.test_dir1, is_regression=False)
        cls.spec_pipe_cls = create_test_spec_pipe_multi_factor(cls.spec_exp_cls)
        cls.spec_pipe_cls.run(summary=False)

    @classmethod
    @silent
    def tearDownClass(cls) -> None:
        """Cleanup after each test method"""
        if hasattr(cls, "test_dir") and os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        if hasattr(cls, "test_dir") and os.path.exists(cls.test_dir1):
            shutil.rmtree(cls.test_dir1)

    @silent
    def test_chain_sample_group_stats(self) -> None:
        """Test chain_sample_group_stats functionality."""
        # Compute sample stats
        chain_sample_group_stats(
            preprocessing_chain_index=0,
            sample_data_path=f"{self.test_dir}Preprocessing/PreprocessingChainResult_chain_ind_0.csv",
            sample_target_path=f"{self.test_dir}Modeling/sample_targets.csv",
            output_directory=f"{self.test_dir}Preprocessing/",
        )

        # Resulting file paths
        path_X_mean = self.test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_mean.csv"
        path_X_std = self.test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_std.csv"
        path_X_skew = self.test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_skewness.csv"
        path_X_kurt = self.test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_kurtosis.csv"
        path_X_min = self.test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_min.csv"
        path_X_median = self.test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_median.csv"
        path_X_max = self.test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_max.csv"
        path_y_stats = self.test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_y_stats.csv"

        # Resulting files
        assert os.path.exists(path_X_mean)
        assert os.path.exists(path_X_std)
        assert os.path.exists(path_X_skew)
        assert os.path.exists(path_X_kurt)
        assert os.path.exists(path_X_min)
        assert os.path.exists(path_X_median)
        assert os.path.exists(path_X_max)
        assert os.path.exists(path_y_stats)

        # Resulting file contents
        X_mean = pd.read_csv(path_X_mean)
        assert list(X_mean["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
        assert not ((X_mean.iloc[:, 1:] == 0) | (X_mean.iloc[:, 1:].isna())).all().all()
        X_std = pd.read_csv(path_X_std)
        assert list(X_std["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
        assert not ((X_std.iloc[:, 1:] == 0) | (X_std.iloc[:, 1:].isna())).all().all()
        X_skew = pd.read_csv(path_X_skew)
        assert list(X_skew["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
        assert not ((X_skew.iloc[:, 1:] == 0) | (X_skew.iloc[:, 1:].isna())).all().all()
        X_kurt = pd.read_csv(path_X_kurt)
        assert list(X_kurt["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
        assert not ((X_kurt.iloc[:, 1:] == 0) | (X_kurt.iloc[:, 1:].isna())).all().all()
        X_min = pd.read_csv(path_X_min)
        assert list(X_min["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
        assert not ((X_min.iloc[:, 1:] == 0) | (X_min.iloc[:, 1:].isna())).all().all()
        X_median = pd.read_csv(path_X_median)
        assert list(X_median["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
        assert not ((X_median.iloc[:, 1:] == 0) | (X_median.iloc[:, 1:].isna())).all().all()
        X_max = pd.read_csv(path_X_max)
        assert list(X_max["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
        assert not ((X_max.iloc[:, 1:] == 0) | (X_max.iloc[:, 1:].isna())).all().all()
        y_stats = pd.read_csv(path_y_stats)
        assert list(y_stats["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
        assert list(y_stats.columns) == ["Group", "mean", "std", "skewness", "kurtosis", "min", "median", "max"]
        assert not ((y_stats.iloc[:, 1:] == 0) | (y_stats.iloc[:, 1:].isna())).all().all()

    @silent
    def test_sample_group_stats(self) -> None:
        """Test sample_group_stats functionality."""
        sample_group_stats(self.test_dir1)
        # For every preprocessing chains
        for i in range(3):
            # Resulting file paths
            path_X_mean = self.test_dir1 + "Preprocessing/" + f"PreprocessingChainResult_chain_ind_{i}_X_mean.csv"
            path_X_std = self.test_dir1 + "Preprocessing/" + f"PreprocessingChainResult_chain_ind_{i}_X_std.csv"
            path_X_skew = self.test_dir1 + "Preprocessing/" + f"PreprocessingChainResult_chain_ind_{i}_X_skewness.csv"
            path_X_kurt = self.test_dir1 + "Preprocessing/" + f"PreprocessingChainResult_chain_ind_{i}_X_kurtosis.csv"
            path_X_min = self.test_dir1 + "Preprocessing/" + f"PreprocessingChainResult_chain_ind_{i}_X_min.csv"
            path_X_median = self.test_dir1 + "Preprocessing/" + f"PreprocessingChainResult_chain_ind_{i}_X_median.csv"
            path_X_max = self.test_dir1 + "Preprocessing/" + f"PreprocessingChainResult_chain_ind_{i}_X_max.csv"
            path_y_stats = self.test_dir1 + "Preprocessing/" + f"PreprocessingChainResult_chain_ind_{i}_y_stats.csv"
            path_y1_stats = self.test_dir1 + "Modeling/" + "sample_targets_stats.csv"

            # Resulting files
            assert os.path.exists(path_X_mean)
            assert os.path.exists(path_X_std)
            assert os.path.exists(path_X_skew)
            assert os.path.exists(path_X_kurt)
            assert os.path.exists(path_X_min)
            assert os.path.exists(path_X_median)
            assert os.path.exists(path_X_max)
            assert os.path.exists(path_y_stats)
            assert os.path.exists(path_y1_stats)

            # Resulting file contents
            X_mean = pd.read_csv(path_X_mean)
            assert list(X_mean["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
            assert not ((X_mean.iloc[:, 1:] == 0) | (X_mean.iloc[:, 1:].isna())).all().all()
            X_std = pd.read_csv(path_X_std)
            assert list(X_std["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
            assert not ((X_std.iloc[:, 1:] == 0) | (X_std.iloc[:, 1:].isna())).all().all()
            X_skew = pd.read_csv(path_X_skew)
            assert list(X_skew["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
            assert not ((X_skew.iloc[:, 1:] == 0) | (X_skew.iloc[:, 1:].isna())).all().all()
            X_kurt = pd.read_csv(path_X_kurt)
            assert list(X_kurt["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
            assert not ((X_kurt.iloc[:, 1:] == 0) | (X_kurt.iloc[:, 1:].isna())).all().all()
            X_min = pd.read_csv(path_X_min)
            assert list(X_min["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
            assert not ((X_min.iloc[:, 1:] == 0) | (X_min.iloc[:, 1:].isna())).all().all()
            X_median = pd.read_csv(path_X_median)
            assert list(X_median["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
            assert not ((X_median.iloc[:, 1:] == 0) | (X_median.iloc[:, 1:].isna())).all().all()
            X_max = pd.read_csv(path_X_max)
            assert list(X_max["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
            assert not ((X_max.iloc[:, 1:] == 0) | (X_max.iloc[:, 1:].isna())).all().all()
            y_stats = pd.read_csv(path_y_stats)
            assert list(y_stats["Group"]) == ["OVERALL"] + self.spec_exp_reg.groups
            assert set(y_stats.columns) == {"Group", "a", "b"}
            assert not ((y_stats.iloc[:, 1:] == 0) | (y_stats.iloc[:, 1:].isna())).all().all()
            y1_stats = pd.read_csv(path_y1_stats)
            assert (y_stats.fillna(0) == y1_stats.fillna(0)).all().all()

    @silent
    def test_chain_process_id_label_conversion(self) -> None:
        """Test 'process_id_to_label' and 'process_label_to_id' functionality."""
        config_dir = self.test_dir + "SpecPipe_configuration/"
        process_config_df = pd.read_csv(config_dir + "SpecPipe_added_process.csv")
        proc_id_to_label, proc_label_to_id = process_id_label_lookup_dict(process_config_df)
        chain_proc_id, chain_proc_lab = self.spec_pipe_reg.ls_chains(print_label=False, return_label=True)
        for r in range(chain_proc_id.shape[0]):
            for c in range(chain_proc_id.shape[1]):
                cpid = chain_proc_id.iloc[r, c]
                cplab = chain_proc_lab.iloc[r, c]
                # process_id_to_label
                assert process_id_to_label(cpid, proc_id_to_label) == cplab
                # process_label_to_id
                assert process_label_to_id(cplab, proc_label_to_id) == cpid

        # No item matched
        # process_id_to_label
        with pytest.raises(ValueError, match="No process label"):
            process_id_to_label("non_existed_id", proc_id_to_label)
        # process_label_to_id
        with pytest.raises(ValueError, match="No process ID found"):
            process_label_to_id("non_existed_label", proc_label_to_id)

    @silent
    def test_performance_metrics_summary(self) -> None:
        """Test performance_metrics_summary functionality."""
        # Regression - pipeline has 3 steps
        config_dir = self.test_dir + "SpecPipe_configuration/"
        report_dir = self.test_dir + "Modeling/Model_evaluation_reports/"
        metrics_dict = performance_metrics_summary(
            pipeline_config_dir=config_dir,
            model_evaluation_report_dir=report_dir,
        )
        assert metrics_dict["is_regression"] is True
        assert metrics_dict["chains_in_ID"].shape[0] == len(self.spec_pipe_reg.process_chains)
        assert metrics_dict["chains_in_ID"].shape[1] == 3
        assert metrics_dict["regression_metrics"].shape[0] == len(self.spec_pipe_reg.process_chains)
        assert metrics_dict["regression_metrics"].shape[1] > 3
        assert (
            not (
                (metrics_dict["regression_metrics"].iloc[:, 6:] == 0)
                | (metrics_dict["regression_metrics"].iloc[:, 6:].isna())
            )
            .all()
            .all()
        )

        # Classification - pipeline has 3 steps
        config_dir1 = self.test_dir1 + "SpecPipe_configuration/"
        report_dir1 = self.test_dir1 + "Modeling/Model_evaluation_reports/"
        metrics_dict1 = performance_metrics_summary(
            pipeline_config_dir=config_dir1,
            model_evaluation_report_dir=report_dir1,
        )
        assert metrics_dict1["is_regression"] is False
        assert metrics_dict1["chains_in_ID"].shape[0] == len(self.spec_pipe_cls.process_chains)
        assert metrics_dict1["chains_in_ID"].shape[1] == 3
        assert metrics_dict1["macro_metrics"].shape[0] == len(self.spec_pipe_cls.process_chains)
        assert metrics_dict1["macro_metrics"].shape[1] == 3 * 2 + 5
        assert metrics_dict1["micro_metrics"].shape[0] == len(self.spec_pipe_cls.process_chains)
        assert metrics_dict1["micro_metrics"].shape[1] == 3 * 2 + 5
        assert (
            not (
                (metrics_dict1["macro_metrics"].iloc[:, 6:] == 0) | (metrics_dict1["macro_metrics"].iloc[:, 6:].isna())
            )
            .all()
            .all()
        )
        assert (
            not (
                (metrics_dict1["micro_metrics"].iloc[:, 6:] == 0) | (metrics_dict1["micro_metrics"].iloc[:, 6:].isna())
            )
            .all()
            .all()
        )

    @silent
    def test_performance_marginal_stats_for_regression(self) -> None:  # noqa: C901
        """
        Test functionality of regression_performance_marginal_stats and performance_marginal_stats for regression.
        """
        # Regression - file test
        config_dir = self.test_dir + "SpecPipe_configuration/"
        report_dir = self.test_dir + "Modeling/Model_evaluation_reports/"
        metrics_dict = performance_metrics_summary(
            pipeline_config_dir=config_dir,
            model_evaluation_report_dir=report_dir,
        )
        # Inner func - need 'metrics_dict' explicitly
        result_reg = regression_performance_marginal_stats(
            metrics_dict=metrics_dict,
            pipeline_config_dir=config_dir,
            model_evaluation_report_dir=report_dir,
        )
        # Performance summary
        step_perf_sum_path = f"{report_dir}Performance_summary.csv"
        assert os.path.exists(step_perf_sum_path)
        # Marginal performance
        df_chains = self.spec_pipe_reg.ls_chains(print_label=False)
        for step in df_chains.columns:
            step_perf_path = report_dir + f"Marginal_R2_stats_{str(step).lower()}.csv"
            if len(list(df_chains[step].unique())) > 1:
                assert os.path.exists(step_perf_path)
                step_perf_stats = pd.read_csv(step_perf_path)
                assert not ((step_perf_stats.iloc[1:, 1:] == 0) | (step_perf_stats.iloc[1:, 1:].isna())).all().all()
            else:
                assert not os.path.exists(step_perf_path)

        # Return test and performance_marginal_stats
        # Outer func - auto apply performance_metrics_summary to get 'metrics_dict'
        result_auto = performance_marginal_stats(self.test_dir)
        # The returns of performance_marginal_stats and regression_performance_marginal_stats should be equal
        for key in result_reg.keys():
            if isinstance(result_reg[key], dict):
                for key_in in result_reg[key].keys():
                    if isinstance(result_reg[key][key_in], pd.DataFrame):
                        try:
                            df_reg = result_reg[key][key_in].fillna(0)
                            df_auto = result_auto[key][key_in].fillna(0)
                            assert (df_reg == df_auto).all().all()
                        except Exception as e:
                            raise AssertionError(
                                f"\n\n key: {key} \n\n key_in: {key_in} \n\n\
                                \n\n result_reg[key][key_in]: {result_reg[key][key_in]}\n\n\
                                \n\n result_auto[key][key_in]: {result_auto[key][key_in]}\n\n"
                            ) from e
                    else:
                        try:
                            assert result_reg[key][key_in] == result_auto[key][key_in]
                        except Exception as e:
                            raise AssertionError(
                                f"\n\n key: {key} \n\n key_in: {key_in} \n\n\
                                \n\n result_reg[key][key_in]: {result_reg[key][key_in]}\n\n\
                                \n\n result_auto[key][key_in]: {result_auto[key][key_in]}\n\n"
                            ) from e
            elif isinstance(result_reg[key], pd.DataFrame):
                try:
                    df_reg1 = result_reg[key].fillna(0)
                    df_auto1 = result_auto[key].fillna(0)
                    assert (df_reg1 == df_auto1).all().all()
                except Exception as e:
                    raise AssertionError(
                        f"\n\n key: {key} \n\n key_in: {key_in} \n\n\
                        \n\n result_reg[key][key_in]: {result_reg[key][key_in]}\n\n\
                        \n\n result_auto[key][key_in]: {result_auto[key][key_in]}\n\n"
                    ) from e
            else:
                try:
                    assert result_reg[key] == result_auto[key]
                except Exception as e:
                    raise AssertionError(
                        f"\n\n key: {key} \n\n key_in: {key_in} \n\n\
                        \n\n result_reg[key][key_in]: {result_reg[key][key_in]}\n\n\
                        \n\n result_auto[key][key_in]: {result_auto[key][key_in]}\n\n"
                    ) from e

    @silent
    def test_performance_marginal_stats_for_classification(self) -> None:  # noqa: C901
        """
        Test functionality of regression_performance_marginal_stats and performance_marginal_stats for classification.
        """
        # Classification - file test
        config_dir1 = self.test_dir1 + "SpecPipe_configuration/"
        report_dir1 = self.test_dir1 + "Modeling/Model_evaluation_reports/"
        metrics_dict = performance_metrics_summary(
            pipeline_config_dir=config_dir1,
            model_evaluation_report_dir=report_dir1,
        )
        result_cls = classification_performance_marginal_stats(
            metrics_dict=metrics_dict,
            pipeline_config_dir=config_dir1,
            model_evaluation_report_dir=report_dir1,
        )
        # Performance summary
        step_mac_sum_path = f"{report_dir1}Macro_avg_performance_summary.csv"
        step_mic_sum_path = f"{report_dir1}Micro_avg_performance_summary.csv"
        assert os.path.exists(step_mac_sum_path)
        assert os.path.exists(step_mic_sum_path)
        # Marginal performance
        df_chains1 = self.spec_pipe_cls.ls_chains(print_label=False)
        for step in df_chains1.columns:
            step_macro_path = report_dir1 + f"Marginal_macro_avg_AUC_stats_{str(step).lower()}.csv"
            step_micro_path = report_dir1 + f"Marginal_micro_avg_AUC_stats_{str(step).lower()}.csv"
            if len(list(df_chains1[step].unique())) > 1:
                assert os.path.exists(step_macro_path)
                assert os.path.exists(step_micro_path)
                step_macro_stats = pd.read_csv(step_macro_path)
                step_micro_stats = pd.read_csv(step_micro_path)
                assert not ((step_macro_stats.iloc[1:, 1:] == 0) | (step_macro_stats.iloc[1:, 1:].isna())).all().all()
                assert not ((step_micro_stats.iloc[1:, 1:] == 0) | (step_micro_stats.iloc[1:, 1:].isna())).all().all()
            else:
                assert not os.path.exists(step_macro_path)
                assert not os.path.exists(step_micro_path)

        # Return test and performance_marginal_stats
        # Outer func - auto apply performance_metrics_summary to get 'metrics_dict'
        result_auto = performance_marginal_stats(self.test_dir1)
        # The returns of performance_marginal_stats and regression_performance_marginal_stats should be equal
        for key in result_cls.keys():
            if isinstance(result_cls[key], dict):
                for key_in in result_cls[key].keys():
                    if isinstance(result_cls[key][key_in], pd.DataFrame):
                        try:
                            df_reg = result_cls[key][key_in].fillna(0)
                            df_auto = result_auto[key][key_in].fillna(0)
                            assert (df_reg == df_auto).all().all()
                        except Exception as e:
                            raise AssertionError(
                                f"\n\n key: {key} \n\n key_in: {key_in} \n\n\
                                \n\n result_cls[key][key_in]: {result_cls[key][key_in]}\n\n\
                                \n\n result_auto[key][key_in]: {result_auto[key][key_in]}\n\n"
                            ) from e
                    else:
                        try:
                            assert result_cls[key][key_in] == result_auto[key][key_in]
                        except Exception as e:
                            raise AssertionError(
                                f"\n\n key: {key} \n\n key_in: {key_in} \n\n\
                                \n\n result_cls[key][key_in]: {result_cls[key][key_in]}\n\n\
                                \n\n result_auto[key][key_in]: {result_auto[key][key_in]}\n\n"
                            ) from e
            elif isinstance(result_cls[key], pd.DataFrame):
                try:
                    df_reg1 = result_cls[key].fillna(0)
                    df_auto1 = result_auto[key].fillna(0)
                    assert (df_reg1 == df_auto1).all().all()
                except Exception as e:
                    raise AssertionError(
                        f"\n\n key: {key} \n\n key_in: {key_in} \n\n\
                        \n\n result_cls[key][key_in]: {result_cls[key][key_in]}\n\n\
                        \n\n result_auto[key][key_in]: {result_auto[key][key_in]}\n\n"
                    ) from e
            else:
                try:
                    assert result_cls[key] == result_auto[key]
                except Exception as e:
                    raise AssertionError(
                        f"\n\n key: {key} \n\n key_in: {key_in} \n\n\
                        \n\n result_cls[key][key_in]: {result_cls[key][key_in]}\n\n\
                        \n\n result_auto[key][key_in]: {result_auto[key][key_in]}\n\n"
                    ) from e


# %% Test - group statistics functions

# test_groupstats = TestGroupStats()

# test_groupstats.setUpClass()

# test_groupstats.test_chain_sample_group_stats()
# test_groupstats.test_sample_group_stats()
# test_groupstats.test_chain_process_id_label_conversion()
# test_groupstats.test_performance_metrics_summary()
# test_groupstats.test_performance_marginal_stats_for_regression()
# test_groupstats.test_performance_marginal_stats_for_classification()

# test_groupstats.tearDownClass()


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
