# -*- coding: utf-8 -*-
"""
Tests for spectral image processing and modeling pipeline (SpecPipe)

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Temporarily ignore of dependency F401 for development
# ruff: noqa: F401

# OS
import os
import sys
import warnings

# from copy import deepcopy

# Initialize LOKY_MAX_CPU_COUNT if it does not exist before imports to prevent corresponding warning
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

# OS Files
import shutil  # noqa: E402

# Test
import tempfile  # noqa: E402
import unittest  # noqa: E402

# Time
import time

# Basic data
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

# Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # noqa: E402

# Visualization
import matplotlib.pyplot as plt  # noqa: E402

# Multiprocessing
from pathos.helpers import cpu_count  # noqa: E402

# Local
from swectral.example_data import create_test_raster, create_test_roi_xml, create_test_spec_exp  # noqa: E402
from swectral.roistats import Stats2d, roi_mean, roispec  # noqa: E402
from swectral.specexp import SpecExp  # noqa: E402
from swectral.specio import silent, lsdir_robust  # noqa: E402
from swectral.assembly import identity_assembly  # noqa: E402

# Functions to test
from swectral.pipeline_tensor import SpecPipeTensor  # noqa: E402

# Confirm proper LOKY_MAX_CPU_COUNT
loky_max_cpu_count = str(cpu_count())
os.environ.setdefault('LOKY_MAX_CPU_COUNT', loky_max_cpu_count)

# Check if cuda is available
try:
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

# Skip the execution of all tests in this file when no GPU
if not HAS_CUDA:
    pytest.skip("GPU not available", allow_module_level=True)


# %% Test process methods


# Image to image
def original_img(input_path: str, output_path: str) -> str:
    shutil.copyfile(input_path, output_path)
    return output_path


# Image to image
def original_img1(input_path: str, output_path: str) -> str:
    shutil.copyfile(input_path, output_path)
    return output_path


# Image to spec1d
def img_const(input_path: str, output_path: str) -> tuple:
    return (0, 0, 0, 0)


# Original pixel apply
def arr_ori(spec: np.ndarray) -> np.ndarray:
    return spec


# Array apply
def arr_simple_half(spec: np.ndarray) -> np.ndarray:
    half: np.ndarray = spec / 2
    return half


# Pixel apply
def snv(spec: np.ndarray) -> np.ndarray:
    spec = np.array(spec)
    snv = (spec - np.nanmean(spec)) / (np.nanstd(spec) + 1e-15)
    assert isinstance(snv, np.ndarray)
    return snv


# Array apply
def arr_snv(spec: np.ndarray) -> np.ndarray:
    vmean = np.mean(spec, axis=1, keepdims=True)
    vstd = np.std(spec, axis=1, keepdims=True) + 1e-15
    snv = (spec - vmean) / vstd
    assert isinstance(snv, np.ndarray)
    return snv


# Tensor apply
def tensor_snv(spectra_tensor: torch.Tensor) -> torch.Tensor:
    spectra_tensor = spectra_tensor.float()
    mean = torch.mean(spectra_tensor, dim=0, keepdim=True)
    std = torch.std(spectra_tensor, dim=0, keepdim=True, unbiased=False)
    snv = (spectra_tensor - mean) / (std + 1e-15)
    return snv


# Hyper-tensor apply
def hypert_snv(spectra_tensor: torch.Tensor) -> torch.Tensor:
    spectra_tensor = spectra_tensor.float()
    mean = torch.mean(spectra_tensor, dim=1, keepdim=True)
    std = torch.std(spectra_tensor, dim=1, keepdim=True, unbiased=False)
    snv = (spectra_tensor - mean) / (std + 1e-15)
    return snv


# ROI specs - roispec
# ROI specs to spec1d - Stats2d().mean, Stats2d().median


# Spec1d to Spec1d
def spec_double(spec: np.ndarray) -> np.ndarray:
    doubled: np.ndarray = np.array(spec) * 2
    return doubled


def replace_nan(spec: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.nan_to_num(spec, nan=0, posinf=0, neginf=0)
    return result


# %% test helper functions : create_test_spec_pipe


# Test helper functions : create_test_spec_pipe
def create_test_spec_pipe(
    dir_path: str,
    sample_n: int = 10,
    n_bands: int = 8,
    is_regression: bool = True,
    use_val_group: bool = False,
    validation_method: str = "2-fold",
) -> SpecPipeTensor:
    """Create a standard test SpecPipeTensor instance."""
    # Create test spec exp
    test_exp = create_test_spec_exp(
        dir_path=dir_path, sample_n=sample_n, n_bands=n_bands, is_regression=is_regression, use_val_group=use_val_group
    )
    pipe = SpecPipeTensor(test_exp)

    # Add process
    pipe.add_process(0, 0, 0, original_img)
    pipe.add_process(2, 2, 1, arr_ori)
    pipe.add_process(2, 2, 1, arr_simple_half)
    pipe.add_process(5, 6, 0, roispec)
    pipe.add_process(6, 7, 0, Stats2d().mean)
    pipe.add_process(7, 8, 0, identity_assembly, process_label="exp_assem1")
    pipe.add_process(8, 8, 1, identity_assembly, process_label="exp_assem2")
    if is_regression:
        pipe.add_process(8, 9, 0, RandomForestRegressor(n_estimators=6), validation_method=validation_method)
        pipe.add_process(8, 9, 0, KNeighborsRegressor(n_neighbors=3), validation_method=validation_method)
    else:
        pipe.add_process(8, 9, 0, RandomForestClassifier(n_estimators=6), validation_method=validation_method)
        pipe.add_process(8, 9, 0, KNeighborsClassifier(n_neighbors=3), validation_method=validation_method)

    return pipe


# %% Test modules


class TestSpecPipeTensor(unittest.TestCase):
    """Test class for SpecPipeTensor functionality."""

    test_dir = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    @classmethod
    def _init_test_dir(cls) -> None:
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
            os.makedirs(cls.test_dir)
        else:
            os.makedirs(cls.test_dir)

    @staticmethod
    @silent
    def test_initialization_image_exp() -> None:
        """Test SpecPipe instance initialization with spectral imaging experiment"""
        # Initialize test_dir
        TestSpecPipeTensor._init_test_dir()
        test_dir = TestSpecPipeTensor.test_dir
        assert os.path.exists(test_dir)

    @staticmethod
    @silent
    def criteria_preprocessing_result(pipe: SpecPipeTensor) -> str:  # noqa: C901
        """Test criteria for preprocessing"""

        test_dir = pipe.report_directory
        assert os.path.exists(test_dir)

        # Finished status
        assert os.path.exists(f"{test_dir}/SpecPipe_configuration/.__preprocessing_complete.s")

        # Assert results
        assert len(pipe._sample_data) == len(pipe.spec_exp.rois)

        # Assert resulting files
        preprocessed_img_path = f"{test_dir}/Preprocessing/Preprocessed_images/"
        assert os.path.exists(preprocessed_img_path)
        preproc_img_names = [
            name
            for name in lsdir_robust(preprocessed_img_path)
            if "test_img" in name and ".tif" in name and name != "test_img.tif"
        ]
        img_sum = (
            len(pipe.ls_process(input_data_level=0, print_result=False, return_result=True))
            + len(pipe.ls_process(input_data_level=1, print_result=False, return_result=True))
            + len(pipe.ls_process(input_data_level=2, print_result=False, return_result=True))
            + len(pipe.ls_process(input_data_level=3, print_result=False, return_result=True))
            + len(pipe.ls_process(input_data_level=4, print_result=False, return_result=True))
        )
        assert len(preproc_img_names) == img_sum
        result_dir = f"{test_dir}/Preprocessing/"
        assert os.path.exists(result_dir)

        # Result files
        preprocs = pipe.process_chains_to_df().iloc[:, :-1].drop_duplicates(ignore_index=True)
        preproc_csv_names = [f"PreprocessingChainResult_chain_ind_{i}.csv" for i in range(len(preprocs))]
        preproc_dill_names = [f"PreprocessingChainResult_chain_ind_{i}.dill" for i in range(len(preprocs))]
        assert set(preproc_csv_names).issubset(set(lsdir_robust(result_dir)))
        assert set(preproc_dill_names).issubset(set(lsdir_robust(result_dir)))

        # Step result files
        preproc_step_names = [
            name
            for name in lsdir_robust(f"{result_dir}/Step_results/")
            if "PreprocessingResult_sample_" in name and ".dill" in name
        ]
        assert len(preproc_step_names) == len(pipe._sample_data)

        # Summary files
        path_X_mean = test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_mean.csv"  # noqa: N806
        path_X_std = test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_std.csv"  # noqa: N806
        path_X_skew = test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_skewness.csv"  # noqa: N806
        path_X_kurt = test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_kurtosis.csv"  # noqa: N806
        path_X_min = test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_min.csv"  # noqa: N806
        path_X_median = test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_median.csv"  # noqa: N806
        path_X_max = test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_X_max.csv"  # noqa: N806
        path_y_stats = test_dir + "Preprocessing/" + "PreprocessingChainResult_chain_ind_0_y_stats.csv"
        path_y1_stats = test_dir + "Modeling/" + "sample_targets_stats.csv"

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

        return "finished"

    @staticmethod
    @silent
    def criteria_regression_model_report(pipe: SpecPipeTensor) -> str:  # noqa: C901
        """Test criteria for regression model reports"""
        test_dir = pipe.report_directory
        assert os.path.exists(test_dir)

        # Finished status
        assert os.path.exists(f"{test_dir}/SpecPipe_configuration/.__modeling_complete.s")

        # Assert reports
        model_report_dir = f"{test_dir}/Modeling/Model_evaluation_reports/"
        assert os.path.exists(model_report_dir)

        # Report contents
        model_reports = lsdir_robust(model_report_dir, 4, retry=10, time_wait_max=30)
        preprocs_in_modeling = [n for n in model_reports if ".txt" in n]
        model_reports = [n for n in model_reports if "Data_chain_" in n and "_Model_" in n]
        preprocs = pipe.process_chains_to_df().iloc[:, :-1].drop_duplicates(ignore_index=True)

        # Assert resulting files with path tracking
        crit_1 = len(preprocs_in_modeling) == len(preprocs)
        crit_2 = len(model_reports) == len(pipe.process_chains)
        if not (crit_1 and crit_2):
            all_existed_files = [
                os.path.join(root, name) for root, dirs, files in os.walk(test_dir) for name in dirs + files
            ]
            raise AssertionError(
                f"Incomplete output result files from model evaluation, \
                    \nlen(preprocs_in_modeling) == len(preprocs) result: {crit_1}, \
                    \nlen(model_reports) == len(pipe.process_chains) result: {crit_2}, \
                    found in the pipeline report dir: {all_existed_files}"
            )

        # Assert model evaluation reports of each chain
        for dirname in model_reports:
            # Reports
            reports = lsdir_robust(model_report_dir + dirname)
            assert len(reports) == 8
            # Output model dirs
            assert "Model_for_application" in reports
            assert "Model_in_validation" in reports
            # Check report files
            match_performance: int = 0
            match_influence: int = 0
            match_residual: int = 0
            match_validation: int = 0
            match_scatter: int = 0
            match_res_plot: int = 0
            for report in reports:
                if "Validation_results" in report:
                    match_validation = 1
                if "Regression_performance" in report:
                    match_performance = 1
                if "Residual_analysis" in report:
                    match_residual = 1
                if "Influence_analysis" in report:
                    match_influence = 1
                if "Scatter_plot" in report:
                    match_scatter = 1
                if "Residual_plot" in report:
                    match_res_plot = 1
            assert match_validation == 1
            assert match_performance == 1
            assert match_residual == 1
            assert match_influence == 1
            assert match_scatter == 1
            assert match_res_plot == 1

            # Models for application
            app_model_path = model_report_dir + dirname + "/Model_for_application/"
            model_files = [n for n in lsdir_robust(app_model_path) if "app_model_" in n and ".dill" in n]
            assert len(model_files) > 0

            # Models in validation
            val_model_path = model_report_dir + dirname + "/Model_in_validation/"
            model_files = [n for n in lsdir_robust(val_model_path) if "val_model_" in n and ".dill" in n]
            assert len(model_files) > 0
            n_fold = len(model_files)

            # Data in validation
            val_X_train_files = [  # noqa: N806
                n for n in lsdir_robust(val_model_path) if "val_X-train_" in n and ".csv" in n
            ]
            val_X_test_files = [  # noqa: N806
                n for n in lsdir_robust(val_model_path) if "val_X-test_" in n and ".csv" in n
            ]
            val_y_files = [n for n in lsdir_robust(val_model_path) if "val_y_" in n and ".csv" in n]
            assert len(val_X_train_files) == n_fold
            assert len(val_X_test_files) == n_fold
            assert len(val_y_files) == n_fold

        # Summary files
        rdir = f"{test_dir}Modeling/Model_evaluation_reports/"
        step_perf_sum_path = f"{rdir}Performance_summary.csv"
        assert os.path.exists(step_perf_sum_path)
        # Validate summary values
        df_summary = pd.read_csv(step_perf_sum_path)
        assert not df_summary.isnull().any().any(), "Performance_summary.csv contains NaN"
        report_subdir_names = df_summary["Result_subdirectory"].tolist()
        for subdir in report_subdir_names:
            full_path = os.path.join(rdir, subdir)
            assert os.path.isdir(full_path), f"Invalid model evaluation report subdirectory: {full_path}"
        # Step marginal performance
        df_chains = pipe.ls_chains(print_label=False)
        for step in df_chains.columns:
            step_perf_path = f"{test_dir}Modeling/Model_evaluation_reports/Marginal_R2_stats_{str(step).lower()}.csv"
            if len(list(df_chains[step].unique())) > 1:
                assert os.path.exists(step_perf_path)
                step_perf_stats = pd.read_csv(step_perf_path)
                assert not ((step_perf_stats.iloc[1:, 1:] == 0) | (step_perf_stats.iloc[1:, 1:].isna())).all().all()
            else:
                assert not os.path.exists(step_perf_path)

        return "finished"

    @staticmethod
    @silent
    def criteria_classification_model_report(pipe: SpecPipeTensor) -> str:  # noqa: C901
        """Test criteria for classification model reports"""
        test_dir = pipe.report_directory
        assert os.path.exists(test_dir)

        # Finished status
        assert os.path.exists(f"{test_dir}/SpecPipe_configuration/.__modeling_complete.s")

        # Assert reports
        model_report_dir = f"{test_dir}/Modeling/Model_evaluation_reports/"
        assert os.path.exists(model_report_dir)

        # Report contents
        model_reports = lsdir_robust(model_report_dir, 4, retry=10, time_wait_max=30)
        preprocs_in_modeling = [n for n in model_reports if ".txt" in n]
        model_reports = [n for n in model_reports if "Data_chain_" in n and "_Model_" in n]
        preprocs = pipe.process_chains_to_df().iloc[:, :-1].drop_duplicates(ignore_index=True)

        # Assert resulting files with path tracking
        crit_1 = len(preprocs_in_modeling) == len(preprocs)
        crit_2 = len(model_reports) == len(pipe.process_chains)
        if not (crit_1 and crit_2):
            all_existed_files = [
                os.path.join(root, name) for root, dirs, files in os.walk(test_dir) for name in dirs + files
            ]
            raise AssertionError(
                f"Incomplete output result files from model evaluation, \
                    \nlen(preprocs_in_modeling) == len(preprocs) result: {crit_1}, \
                    \nlen(model_reports) == len(pipe.process_chains) result: {crit_2}, \
                    found in the pipeline report dir: {all_existed_files}"
            )

        # Assert model evaluation reports of each chain
        for dirname in model_reports:
            # Reports
            reports = lsdir_robust(model_report_dir + dirname)
            assert len(reports) == 7
            # Output model dirs
            assert "Model_for_application" in reports
            assert "Model_in_validation" in reports
            # Check report files
            match_performance: int = 0
            match_influence: int = 0
            match_residual: int = 0
            match_validation: int = 0
            match_roc: int = 0
            for report in reports:
                if "Validation_results" in report:
                    match_validation = 1
                if "Classification_performance" in report:
                    match_performance = 1
                if "Residual_analysis" in report:
                    match_residual = 1
                if "Influence_analysis" in report:
                    match_influence = 1
                if "ROC_curve" in report:
                    match_roc = 1
            assert match_validation == 1
            assert match_performance == 1
            assert match_residual == 1
            assert match_influence == 1
            assert match_roc == 1

            # Models for application
            app_model_path = model_report_dir + dirname + "/Model_for_application/"
            model_files = [n for n in lsdir_robust(app_model_path) if "app_model_" in n and ".dill" in n]
            assert len(model_files) > 0

            # Models in validation
            val_model_path = model_report_dir + dirname + "/Model_in_validation/"
            model_files = [n for n in lsdir_robust(val_model_path) if "val_model_" in n and ".dill" in n]
            assert len(model_files) > 0
            n_fold = len(model_files)

            # Data in validation
            val_X_train_files = [  # noqa: N806
                n for n in lsdir_robust(val_model_path) if "val_X-train_" in n and ".csv" in n
            ]
            val_X_test_files = [  # noqa: N806
                n for n in lsdir_robust(val_model_path) if "val_X-test_" in n and ".csv" in n
            ]
            val_y_files = [n for n in lsdir_robust(val_model_path) if "val_y_" in n and ".csv" in n]
            assert len(val_X_train_files) == n_fold
            assert len(val_X_test_files) == n_fold
            assert len(val_y_files) == n_fold

        # Summary files
        rdir = f"{test_dir}Modeling/Model_evaluation_reports/"
        step_mac_sum_path = f"{rdir}Macro_avg_performance_summary.csv"
        step_mic_sum_path = f"{rdir}Micro_avg_performance_summary.csv"
        assert os.path.exists(step_mac_sum_path)
        assert os.path.exists(step_mic_sum_path)
        # Validate summary values
        # Macro
        df_summary = pd.read_csv(step_mac_sum_path)
        assert not df_summary.isnull().any().any(), "Macro_avg_performance_summary.csv contains NaN"
        report_subdir_names = df_summary["Result_subdirectory"].tolist()
        for subdir in report_subdir_names:
            full_path = os.path.join(rdir, subdir)
            assert os.path.isdir(
                full_path
            ), f"Invalid model evaluation report subdirectory from Macro_avg_performance_summary: {full_path}"
        # Micro
        df_summary = pd.read_csv(step_mac_sum_path)
        assert not df_summary.isnull().any().any(), "Micro_avg_performance_summary.csv contains NaN"
        report_subdir_names = df_summary["Result_subdirectory"].tolist()
        for subdir in report_subdir_names:
            full_path = os.path.join(rdir, subdir)
            assert os.path.isdir(
                full_path
            ), f"Invalid model evaluation report subdirectory from Micro_avg_performance_summary: {full_path}"
        # Step marginal performance
        df_chains1 = pipe.ls_chains(print_label=False)
        for step in df_chains1.columns:
            step_macro_path = f"{rdir}Marginal_macro_avg_AUC_stats_{str(step).lower()}.csv"
            step_micro_path = f"{rdir}Marginal_micro_avg_AUC_stats_{str(step).lower()}.csv"
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

        return "finished"


# %% Tests - SpecPipeTensor

# TestSpecPipeTensor.setUpClass()

# # New tests

# TestSpecPipeTensor.tearDownClass()


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
