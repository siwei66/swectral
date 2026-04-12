# -*- coding: utf-8 -*-
"""
Tests for file results reporters for console interface

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# ruff: noqa: I001
# OS
import sys  # noqa: E402

# OS Files
import shutil  # noqa: E402

# Test
import tempfile  # noqa: E402
import pytest  # noqa: E402
import unittest  # noqa: E402

# Data
import pandas as pd  # noqa: E402

# Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # noqa: E402

# Local
from swectral.example_data import create_test_spec_exp  # noqa: E402
from swectral.roistats import Stats2d, roispec  # noqa: E402
from swectral.specio import silent  # noqa: E402
from swectral.pipeline import SpecPipe  # noqa: E402

# Test
from swectral.resultcli import group_stats_report, core_chain_report  # noqa: E402


# %% Test helper functions : create_test_spec_pipe


# Test helper functions : create_test_spec_pipe
def create_test_spec_pipe(dir_path: str, sample_n: int = 10, n_bands: int = 8, is_regression: bool = True) -> SpecPipe:
    """Create a standard test SpecPipe instance."""
    # Create test spec exp
    test_exp = create_test_spec_exp(dir_path=dir_path, sample_n=sample_n, n_bands=n_bands, is_regression=is_regression)
    pipe = SpecPipe(test_exp)
    # Add process
    # Step 0
    pipe.add_process(5, 6, 0, roispec)
    # Step 1
    pipe.add_process(6, 7, 0, Stats2d().mean)
    pipe.add_process(6, 7, 0, Stats2d().median)
    # Step 2 - modeling
    if is_regression:
        pipe.add_process(7, 9, 0, RandomForestRegressor(n_estimators=6))
        pipe.add_process(7, 9, 0, KNeighborsRegressor(n_neighbors=3))
    else:
        pipe.add_process(7, 9, 0, RandomForestClassifier(n_estimators=6))
        pipe.add_process(7, 9, 0, KNeighborsClassifier(n_neighbors=3))
    return pipe


# %% Test reading SpecPipe reports in console


class TestReadReport(unittest.TestCase):
    """Test class for SpecPipe functionality."""

    test_dir_reg: str = ""
    test_dir_cls: str = ""
    pipe_reg: SpecPipe
    pipe_cls: SpecPipe

    @classmethod
    def setUpClass(cls) -> None:
        """Create a temporary directory structure for testing"""
        # Setup regression
        cls.test_dir_reg = tempfile.mkdtemp()
        pipe_reg = create_test_spec_pipe(cls.test_dir_reg, is_regression=True)
        pipe_reg.run(n_processor=1)
        cls.pipe_reg = pipe_reg
        # Setup classification
        cls.test_dir_cls = tempfile.mkdtemp()
        pipe_cls = create_test_spec_pipe(cls.test_dir_cls, is_regression=False)
        pipe_cls.run(n_processor=1)
        cls.pipe_cls = pipe_cls

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary directory"""
        shutil.rmtree(cls.test_dir_reg)
        shutil.rmtree(cls.test_dir_cls)

    @staticmethod
    @silent
    def test_group_stats_report_regression() -> None:
        """test run preprocessing and modeling functionality using SpecPipe.run()"""
        report = group_stats_report(TestReadReport.test_dir_reg)
        expected_report_keys = [
            'Marginal_R2_stats_step_1',
            'Marginal_R2_stats_step_2',
            'Performance_summary',
            'sample_targets_stats',
            'Marginal_R2_stats_model_step_1',
        ]
        assert set(report.keys()) == set(expected_report_keys)
        assert isinstance(report[expected_report_keys[0]], pd.DataFrame)
        assert isinstance(report[expected_report_keys[1]], pd.DataFrame)
        assert isinstance(report[expected_report_keys[2]], pd.DataFrame)
        assert isinstance(report[expected_report_keys[3]], pd.DataFrame)
        # SpecPipe method
        assert report.keys() == TestReadReport.pipe_reg.report_summary().keys()

    @staticmethod
    @silent
    def test_group_stats_report_classification() -> None:
        """test run preprocessing and modeling functionality using SpecPipe.run()"""
        report = group_stats_report(TestReadReport.test_dir_cls)
        expected_report_keys = [
            'Macro_avg_performance_summary',
            'Marginal_macro_avg_AUC_stats_step_1',
            'Marginal_macro_avg_AUC_stats_step_2',
            'Marginal_micro_avg_AUC_stats_step_1',
            'Marginal_micro_avg_AUC_stats_step_2',
            'Marginal_macro_avg_AUC_stats_model_step_1',
            'Marginal_micro_avg_AUC_stats_model_step_1',
            'Micro_avg_performance_summary',
            'sample_targets_stats',
        ]
        assert set(report.keys()) == set(expected_report_keys)
        assert isinstance(report[expected_report_keys[0]], pd.DataFrame)
        assert isinstance(report[expected_report_keys[1]], pd.DataFrame)
        assert isinstance(report[expected_report_keys[2]], pd.DataFrame)
        assert isinstance(report[expected_report_keys[3]], pd.DataFrame)
        assert isinstance(report[expected_report_keys[4]], pd.DataFrame)
        assert isinstance(report[expected_report_keys[5]], pd.DataFrame)
        assert isinstance(report[expected_report_keys[6]], pd.DataFrame)
        # SpecPipe method
        assert report.keys() == TestReadReport.pipe_cls.report_summary().keys()

    @staticmethod
    @silent
    def test_core_chain_report_regression() -> None:
        """test run preprocessing and modeling functionality using SpecPipe.run()"""
        report = core_chain_report(TestReadReport.test_dir_reg)
        assert isinstance(report, list)
        assert len(report) == 4
        expected_chain_report_keys = [
            'Chain_processes',
            'Influence_analysis',
            'Regression_performance',
            'Residual_analysis',
            'Residual_plot',
            'Scatter_plot',
            'Validation_results',
        ]
        assert set(report[0].keys()) == set(expected_chain_report_keys)
        for key in expected_chain_report_keys:
            assert report[0][key] is not None
        assert set(report[1].keys()) == set(expected_chain_report_keys)
        for key in expected_chain_report_keys:
            assert report[1][key] is not None
        assert set(report[2].keys()) == set(expected_chain_report_keys)
        for key in expected_chain_report_keys:
            assert report[2][key] is not None
        assert set(report[3].keys()) == set(expected_chain_report_keys)
        for key in expected_chain_report_keys:
            assert report[3][key] is not None
        # SpecPipe method
        for r1, r2 in zip(report, TestReadReport.pipe_reg.report_chains()):
            if isinstance(r1, dict):
                assert r1.keys() == r2.keys()
            else:
                assert r1.keys() == r2.keys()

    @staticmethod
    @silent
    def test_core_chain_report_classification() -> None:
        """test run preprocessing and modeling functionality using SpecPipe.run()"""
        report = core_chain_report(TestReadReport.test_dir_cls)
        assert isinstance(report, list)
        assert len(report) == 4
        expected_chain_report_keys = [
            'Chain_processes',
            'Classification_performance',
            'Influence_analysis',
            'Residual_analysis',
            'ROC_curve',
            'Validation_results',
        ]
        assert set(report[0].keys()) == set(expected_chain_report_keys)
        for key in expected_chain_report_keys:
            assert report[0][key] is not None
        assert set(report[1].keys()) == set(expected_chain_report_keys)
        for key in expected_chain_report_keys:
            assert report[1][key] is not None
        assert set(report[2].keys()) == set(expected_chain_report_keys)
        for key in expected_chain_report_keys:
            assert report[2][key] is not None
        assert set(report[3].keys()) == set(expected_chain_report_keys)
        for key in expected_chain_report_keys:
            assert report[3][key] is not None
        # SpecPipe method
        for r1, r2 in zip(report, TestReadReport.pipe_cls.report_chains()):
            if isinstance(r1, dict):
                assert r1.keys() == r2.keys()
            else:
                assert r1.keys() == r2.keys()


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
