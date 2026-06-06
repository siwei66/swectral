# -*- coding: utf-8 -*-
"""
Tests for spectral image processing and modeling pipeline (SpecPipe)

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Temporarily ignore of dependency F401 for development
# ruff: noqa: F401

# OS
import os  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

# Typing
from typing import Any  # noqa: E402

# from copy import deepcopy

# Initialize LOKY_MAX_CPU_COUNT if it does not exist before imports to prevent corresponding warning
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

# OS Files
import shutil  # noqa: E402

# Test
import tempfile  # noqa: E402
import pytest  # noqa: E402
import unittest  # noqa: E402

# Time
import time  # noqa: E402

# Basic data
from copy import deepcopy  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# Visualization
import matplotlib.pyplot as plt  # noqa: E402

# Local
from swectral.example_data import create_test_raster_shaped, create_test_roi_xml, create_test_spec_exp  # noqa: E402
from swectral.roistats import Stats2d, roi_mean, roispec  # noqa: E402
from swectral.specexp import SpecExp  # noqa: E402
from swectral.specio import silent, lsdir_robust, unc_path  # noqa: E402
from swectral.assembly import identity_assembly  # noqa: E402

# Functions to test
from swectral.pipeline_tensor import SpecPipeTensor  # noqa: E402

# Check if cuda is available
try:
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

# Skip the execution of all tests in this file when no GPU
if not HAS_CUDA:
    pytest.skip("GPU not available", allow_module_level=True)


# %% Test process methods


def dummy_deterministic_transform(x: object) -> object:
    """Mock deterministic function for 0->0 or 0->1 data levels."""
    return x


class DummyConv1x1(nn.Module):
    """Mock fittable transformation module for 1->1 data level."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(8, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.conv(x)
        return y


class DummyClassifier(nn.Module):
    """Mock classification model for 1->2 data level."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.fc(x)
        return y


class DummyRegressor(nn.Module):
    """Mock regression model for 1->2 data level."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.fc(x)
        return y


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
    pipe.add_process(0, 0, 0, dummy_deterministic_transform, process_label="trans_0_0_0_0")
    pipe.add_process(0, 1, 0, dummy_deterministic_transform, process_label="trans_0_1_0_0")
    pipe.add_process(1, 1, 0, DummyConv1x1, process_label="trans_1_1_0_0")
    pipe.add_process(1, 1, 0, DummyConv1x1, process_label="trans_1_1_0_1")
    if is_regression:
        pipe.add_process(1, 2, 0, DummyRegressor, process_label="trans_1_2_0_0")
        pipe.add_process(1, 2, 0, DummyRegressor, process_label="trans_1_2_0_1")
    else:
        pipe.add_process(1, 2, 0, DummyClassifier, process_label="trans_1_2_0_0")
        pipe.add_process(1, 2, 0, DummyClassifier, process_label="trans_1_2_0_1")

    return pipe


# %% Test modules


class TestSpecPipeTensor(unittest.TestCase):
    """Test class for SpecPipeTensor functionality."""

    test_dir = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_dir = unc_path(tempfile.mkdtemp() + "/")

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
    def test_initialization() -> None:
        """Test SpecPipe instance initialization"""
        # Create test spec exp for classification
        test_exp_class = create_test_spec_exp(
            dir_path=TestSpecPipeTensor.test_dir, sample_n=10, n_bands=8, is_regression=False, use_val_group=False
        )
        pipe_class = SpecPipeTensor(test_exp_class)
        assert unc_path(pipe_class.report_directory) == unc_path(TestSpecPipeTensor.test_dir)

        # Create test spec exp for classification
        test_exp_reg = create_test_spec_exp(
            dir_path=TestSpecPipeTensor.test_dir, sample_n=10, n_bands=8, is_regression=True, use_val_group=False
        )
        pipe_reg = SpecPipeTensor(test_exp_reg)
        assert unc_path(pipe_reg.report_directory) == unc_path(TestSpecPipeTensor.test_dir)

    @staticmethod
    @silent
    def test_add_process() -> None:
        """Test adding processing methods and models to the tensor pipeline."""
        TestSpecPipeTensor._init_test_dir()
        test_dir = TestSpecPipeTensor.test_dir

        test_exp_class = create_test_spec_exp(
            dir_path=test_dir, sample_n=10, n_bands=8, is_regression=False, use_val_group=False
        )
        pipe = SpecPipeTensor(test_exp_class)

        # 1. Deterministic transform (0 -> 0)
        assert len(pipe._process) == 0
        pipe.add_process(0, 0, 0, dummy_deterministic_transform, process_label="trans_0_0")
        assert len(pipe._process) == 1

        # 2. Deterministic transform for fittable output (0 -> 1)
        pipe.add_process(0, 1, 1, dummy_deterministic_transform, process_label="trans_0_1")
        assert len(pipe._process) == 2

        # 3. Trainable nn.Module (1 -> 1)
        conv_module = DummyConv1x1()
        optimizer_conv = torch.optim.Adam(conv_module.parameters())
        pipe.add_process(
            1,
            1,
            0,
            conv_module,
            process_label="conv_1_1",
            batch_size=32,
            shuffle=True,
            criterion=nn.MSELoss(),
            optimizer=optimizer_conv,
        )
        assert len(pipe._process) == 3

        # Verify that the nn.Module was physically serialized to disk
        model_data_conv = pipe._process[-1][5]
        assert isinstance(model_data_conv, dict)
        conv_file_path = model_data_conv['module']
        assert os.path.exists(conv_file_path), "The nn.Module file was not saved to disk!"

        # 4. Classification model final step (1 -> 2)
        cls_module = DummyClassifier()
        optimizer_cls = torch.optim.Adam(cls_module.parameters())
        pipe.add_process(
            1,
            2,
            1,
            cls_module,
            process_label="classifier_1_2",
            batch_size=32,
            shuffle=True,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer_cls,
            is_regression=False,
        )
        assert len(pipe._process) == 4

        # Validate chains and steps updating correctly
        assert len(pipe._process_steps) == 4
        assert len(pipe._process_chains) == 1
        assert len(pipe._process_chains[0]) == 4  # Pipeline depth of 4

        # 5. Alternative Regression model testing (1 -> 2) on a new pipeline
        test_exp_reg = create_test_spec_exp(
            dir_path=test_dir, sample_n=10, n_bands=8, is_regression=True, use_val_group=False
        )
        pipe_reg = SpecPipeTensor(test_exp_reg)

        reg_module = DummyRegressor()
        optimizer_reg = torch.optim.Adam(reg_module.parameters())
        pipe_reg.add_process(
            1,
            2,
            0,
            reg_module,
            process_label="regressor_1_2",
            batch_size=16,
            shuffle=False,
            criterion=nn.MSELoss(),
            optimizer=optimizer_reg,
            is_regression=True,
        )
        assert len(pipe_reg._process) == 1
        model_data_reg = pipe_reg._process[0][5]
        assert isinstance(model_data_reg, dict)
        assert model_data_reg['use'] == 'regression'

        # Edge Case: Missing required training parameters for nn.Module
        with pytest.raises(ValueError, match="Missing 'batch_size'"):
            pipe.add_process(1, 1, 2, DummyConv1x1(), process_label="fail_conv")

        # 6. Multiple deterministic methods (0 -> 0)
        pipe.add_process(
            0,
            0,
            2,
            [dummy_deterministic_transform, dummy_deterministic_transform],
            process_label=["multi_trans_0", "multi_trans_1"],
        )
        assert len(pipe._process) == 6

        # 7. Multiple nn.Modules (1 -> 1)
        conv1, conv2 = DummyConv1x1(), DummyConv1x1()
        opt1 = torch.optim.Adam(conv1.parameters())
        opt2 = torch.optim.Adam(conv2.parameters())
        sch1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=1)
        sch2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1)

        pipe.add_process(
            1,
            1,
            2,
            [conv1, conv2],
            process_label=["multi_conv_1", "multi_conv_2"],
            batch_size=16,
            shuffle=False,
            criterion=nn.MSELoss(),
            optimizer=[opt1, opt2],
            scheduler=[sch1, sch2],
        )
        assert len(pipe._process) == 8

        # 8. Test length mismatch errors
        with pytest.raises(ValueError, match="Length of process_label list must match"):
            pipe.add_process(
                0,
                0,
                3,
                [dummy_deterministic_transform, dummy_deterministic_transform],
                process_label=["only_one_label"],
            )

        with pytest.raises(ValueError, match="Length of optimizer list must match"):
            pipe.add_process(
                1,
                1,
                3,
                [conv1, conv2],
                process_label=["c1", "c2"],
                batch_size=16,
                shuffle=False,
                criterion=nn.MSELoss(),
                optimizer=[opt1],
                scheduler=[sch1, sch2],
            )

        with pytest.raises(ValueError, match="Length of scheduler list must match"):
            pipe.add_process(
                1,
                1,
                3,
                [conv1, conv2],
                process_label=["c1", "c2"],
                batch_size=16,
                shuffle=False,
                criterion=nn.MSELoss(),
                optimizer=[opt1, opt2],
                scheduler=[sch1],
            )

        # 9. Test multiple batch sizes and criteria
        pipe.add_process(
            1,
            2,
            1,
            [cls_module, cls_module],
            process_label=["cls_1", "cls_2"],
            batch_size=[16, 32],
            shuffle=True,
            criterion=[nn.MSELoss(), nn.CrossEntropyLoss()],
            optimizer=[optimizer_cls, deepcopy(optimizer_cls)],
            is_regression=False,
        )
        assert len(pipe._process) == 10
        assert pipe._process[-1][1] == "cls_2"
        assert pipe._process[-2][1] == "cls_1"
        model_data_mult1 = pipe._process[-2][5]
        model_data_mult2 = pipe._process[-1][5]
        assert isinstance(model_data_mult1, dict)
        assert isinstance(model_data_mult2, dict)
        assert model_data_mult1['batch_size'] == 16
        assert model_data_mult2['batch_size'] == 32
        assert model_data_mult1['criterion'].__class__.__name__ is nn.MSELoss().__class__.__name__
        assert model_data_mult2['criterion'].__class__.__name__ is nn.CrossEntropyLoss().__class__.__name__

    @staticmethod
    @silent
    def test_ls_process() -> None:
        """Test listing and filtering added processes in the tensor pipeline."""
        TestSpecPipeTensor._init_test_dir()
        pipe = SpecPipeTensor(create_test_spec_exp(TestSpecPipeTensor.test_dir))

        # Add a mix of processes
        pipe.add_process(0, 0, 0, dummy_deterministic_transform, process_label="trans_1")
        pipe.add_process(0, 0, 0, dummy_deterministic_transform, process_label="trans_2")

        conv = DummyConv1x1()
        pipe.add_process(
            1,
            1,
            0,
            conv,
            process_label="conv_1",
            batch_size=16,
            shuffle=True,
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam(conv.parameters()),
        )

        # List all
        df_all = pipe.ls_process(print_result=False, return_result=True)
        assert isinstance(df_all, pd.DataFrame)
        assert df_all.shape == (3, 8)

        # Filter by input data level (integer)
        df_lvl0 = pipe.ls_process(input_data_level=0, print_result=False, return_result=True)
        assert df_lvl0.shape == (2, 8)

        # Filter by string names
        df_fittable = pipe.ls_process(input_data_level='fittable', print_result=False, return_result=True)
        assert df_fittable.shape == (1, 8)

        # Filter by partial method name
        df_partial = pipe.ls_process(method='DummyConv', exact_match=False, print_result=False, return_result=True)
        assert df_partial.shape == (1, 8)
        # Verify the nn.Module serialization formatting is safe
        assert "<nn.Module:" in str(df_partial.iloc[0]['Method'])

        # No match conditions
        df_empty = pipe.ls_process(output_data_level='model', print_result=False, return_result=True)
        assert df_empty.shape == (0, 8)

    @staticmethod
    @silent
    def test_rm_process() -> None:
        """Test removing processes and verifying file cleanup."""
        TestSpecPipeTensor._init_test_dir()
        pipe = SpecPipeTensor(create_test_spec_exp(TestSpecPipeTensor.test_dir))

        # Build pipeline
        pipe.add_process(0, 0, 0, dummy_deterministic_transform, process_label="trans_1")

        conv = DummyConv1x1()
        pipe.add_process(
            1,
            1,
            0,
            conv,
            process_label="target_conv",
            batch_size=16,
            shuffle=True,
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam(conv.parameters()),
        )

        cls_mod = DummyClassifier()
        pipe.add_process(
            1,
            2,
            1,
            cls_mod,
            process_label="classifier",
            batch_size=32,
            shuffle=True,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(cls_mod.parameters()),
        )

        assert len(pipe._process) == 3
        assert len(pipe._process_chains) == 1

        # Extract file path of the saved conv module to verify disk cleanup
        model_data_conv = pipe._process[1][5]
        assert isinstance(model_data_conv, dict)
        conv_file_path = model_data_conv['module']
        assert os.path.exists(conv_file_path)

        # Remove specific nn.Module
        pipe.rm_process(process_label="target_conv")

        assert len(pipe._process) == 2
        assert len(pipe._process_steps) == 2
        # Verify disk file was deleted
        assert not os.path.exists(conv_file_path)

        # Remove by data level
        pipe.rm_process(input_data_level='function')
        assert len(pipe._process) == 1

        # Remove all remaining
        pipe.rm_process()
        assert len(pipe._process) == 0
        assert len(pipe._process_steps) == 0
        assert len(pipe._process_chains) == 0

    @staticmethod
    @silent
    def test_process_sorting() -> None:
        """Test if processes and processing chains are correctly sorted after add_process and rm_process."""
        TestSpecPipeTensor._init_test_dir()
        pipe = SpecPipeTensor(create_test_spec_exp(TestSpecPipeTensor.test_dir))

        # Add processes intentionally OUT OF ORDER

        # 1. Add final step first (1->2, seq=2)
        cls = DummyClassifier()
        pipe.add_process(
            1,
            2,
            2,
            cls,
            process_label="step_C",
            batch_size=32,
            shuffle=True,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(cls.parameters()),
        )

        # 2. Add starting step (0->0, seq=0)
        pipe.add_process(0, 0, 0, dummy_deterministic_transform, process_label="step_A")

        # 3. Add middle step with 2 alternatives (0->1, seq=1)
        pipe.add_process(
            0,
            1,
            1,
            [dummy_deterministic_transform, dummy_deterministic_transform],
            process_label=["step_B1", "step_B2"],
        )

        # --- Validate sorting AFTER add_process ---
        fapp_seqs = [p[6] for p in pipe._process]
        assert fapp_seqs == sorted(fapp_seqs), "Processes are NOT correctly sorted by _Full_app_seq after add_process!"

        # Validate exact order by label
        labels = [p[1] for p in pipe._process]
        assert labels == ["step_A", "step_B1", "step_B2", "step_C"], f"Expected sorted labels, got {labels}"

        # Extract dynamically generated IDs to validate internal routing structures
        id_A = next(p[0] for p in pipe._process if p[1] == "step_A")  # noqa: N806
        id_B1 = next(p[0] for p in pipe._process if p[1] == "step_B1")  # noqa: N806
        id_B2 = next(p[0] for p in pipe._process if p[1] == "step_B2")  # noqa: N806
        id_C = next(p[0] for p in pipe._process if p[1] == "step_C")  # noqa: N806

        # Validate generated _process_steps sequence
        expected_steps = [[id_A], [id_B1, id_B2], [id_C]]
        assert (
            pipe._process_steps == expected_steps
        ), f"Expected process_steps {expected_steps}, got {pipe._process_steps}"

        # Validate generated _process_chains sequence and combinations
        expected_chains = [(id_A, id_B1, id_C), (id_A, id_B2, id_C)]
        assert (
            pipe._process_chains == expected_chains
        ), f"Expected process_chains {expected_chains}, got {pipe._process_chains}"

        # --- Remove an item from the middle ---
        pipe.rm_process(process_label="step_B1")

        # --- Validate sorting AFTER rm_process ---
        fapp_seqs_after_rm = [p[6] for p in pipe._process]
        assert fapp_seqs_after_rm == sorted(fapp_seqs_after_rm), "Processes lost their sorted order after rm_process!"

        labels_after_rm = [p[1] for p in pipe._process]
        assert labels_after_rm == [
            "step_A",
            "step_B2",
            "step_C",
        ], f"Expected sorted labels after removal, got {labels_after_rm}"

        # Validate generated _process_steps updated correctly
        expected_steps_after_rm = [[id_A], [id_B2], [id_C]]
        assert (
            pipe._process_steps == expected_steps_after_rm
        ), f"Expected process_steps {expected_steps_after_rm}, got {pipe._process_steps}"

        # Validate generated _process_chains updated correctly
        expected_chains_after_rm = [(id_A, id_B2, id_C)]
        assert (
            pipe._process_chains == expected_chains_after_rm
        ), f"Expected process_chains {expected_chains_after_rm}, got {pipe._process_chains}"

    @staticmethod
    @silent
    def test_process_chain_generation() -> None:
        """Test if processing steps and full-factorial chains are correctly generated in complex scenarios."""
        TestSpecPipeTensor._init_test_dir()
        pipe = SpecPipeTensor(create_test_spec_exp(TestSpecPipeTensor.test_dir))

        # Step A (0->0, seq=0): 2 alternatives
        pipe.add_process(
            0,
            0,
            0,
            [dummy_deterministic_transform, dummy_deterministic_transform],
            process_label=["trans_A1", "trans_A2"],
        )

        # Step B (0->1, seq=1): 1 alternative
        pipe.add_process(0, 1, 1, dummy_deterministic_transform, process_label="trans_B1")

        # Step C (1->1, seq=0): 3 alternatives (trainable)
        conv1, conv2, conv3 = DummyConv1x1(), DummyConv1x1(), DummyConv1x1()
        opt = [
            torch.optim.Adam(conv1.parameters()),
            torch.optim.Adam(conv2.parameters()),
            torch.optim.Adam(conv3.parameters()),
        ]
        pipe.add_process(
            1,
            1,
            0,
            [conv1, conv2, conv3],
            process_label=["conv_C1", "conv_C2", "conv_C3"],
            batch_size=16,
            shuffle=False,
            criterion=nn.MSELoss(),
            optimizer=opt,
        )

        # Step D (1->2, seq=1): 2 alternatives (classifier)
        cls1, cls2 = DummyClassifier(), DummyClassifier()
        opt_cls = [torch.optim.Adam(cls1.parameters()), torch.optim.Adam(cls2.parameters())]
        pipe.add_process(
            1,
            2,
            1,
            [cls1, cls2],
            process_label=["cls_D1", "cls_D2"],
            batch_size=32,
            shuffle=True,
            criterion=nn.CrossEntropyLoss(),
            optimizer=opt_cls,
        )

        # 1. Validate processing steps grouping
        assert len(pipe._process_steps) == 4, f"Expected 4 distinct steps, got {len(pipe._process_steps)}"
        assert len(pipe._process_steps[0]) == 2  # A1, A2
        assert len(pipe._process_steps[1]) == 1  # B1
        assert len(pipe._process_steps[2]) == 3  # C1, C2, C3
        assert len(pipe._process_steps[3]) == 2  # D1, D2

        # 2. Validate full-factorial calculation (2 * 1 * 3 * 2 = 12 combinations)
        assert len(pipe._process_chains) == 12, f"Expected exactly 12 factorial chains, got {len(pipe._process_chains)}"

        # 3. Validate uniqueness
        assert len(set(pipe._process_chains)) == 12, "Duplicate chains detected in full-factorial generation!"

        # 4. Validate after process removal
        pipe.rm_process(process_label="conv_C3")

        assert len(pipe._process_steps) == 4, f"Expected 4 distinct steps, got {len(pipe._process_steps)}"
        assert len(pipe._process_steps[0]) == 2  # A1, A2
        assert len(pipe._process_steps[1]) == 1  # B1
        assert len(pipe._process_steps[2]) == 2  # C1, C2 - C3 removed
        assert len(pipe._process_steps[3]) == 2  # D1, D2

        # 2. Validate full-factorial calculation (2 * 1 * 2 * 2 = 8 combinations)
        assert len(pipe._process_chains) == 8, f"Expected exactly 12 factorial chains, got {len(pipe._process_chains)}"

    @staticmethod
    @silent
    def test_ls_process_chains() -> None:
        """Test ls_process_chains and ls_chains functionalities and combinations."""
        TestSpecPipeTensor._init_test_dir()
        test_dir = TestSpecPipeTensor.test_dir

        pipe = SpecPipeTensor(create_test_spec_exp(test_dir))

        # Build a pipeline with multiple alternatives to test combinatorial chain generation
        # Step 1 (0 -> 0): 2 alternatives
        pipe.add_process(
            0,
            0,
            0,
            [dummy_deterministic_transform, dummy_deterministic_transform],
            process_label=["step1_A", "step1_B"],
        )

        # Step 2 (0 -> 1): 1 alternative (Trainable)
        conv = DummyConv1x1()
        pipe.add_process(
            0,
            1,
            1,
            conv,
            process_label="step2_A",
            batch_size=16,
            shuffle=True,
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam(conv.parameters()),
        )

        # Step 3 (1 -> 2): 3 alternatives (Classifiers)
        cls1, cls2, cls3 = DummyClassifier(), DummyClassifier(), DummyClassifier()
        pipe.add_process(
            1,
            2,
            0,
            [cls1, cls2, cls3],
            process_label=["step3_A", "step3_B", "step3_C"],
            batch_size=32,
            shuffle=True,
            criterion=nn.CrossEntropyLoss(),
            optimizer=[
                torch.optim.Adam(cls1.parameters()),
                torch.optim.Adam(cls2.parameters()),
                torch.optim.Adam(cls3.parameters()),
            ],
        )

        # 2 (step1) * 1 (step2) * 3 (step3) = 6 unique chains. Each chain has 3 steps.
        chains_df = pipe.ls_chains(print_label=False)
        process_chains_df = pipe.ls_process_chains(print_label=False)

        # Validate exact match between generic and specific list methods when no custom chains exist
        assert np.all(chains_df == process_chains_df)
        assert chains_df.shape == (6, 3), f"Expected shape (6, 3), got {chains_df.shape}"

        # Check edge case: Empty pipeline
        pipe_empty = SpecPipeTensor(create_test_spec_exp(test_dir))
        assert pipe_empty.ls_chains() is None
        assert pipe_empty.ls_process_chains() is None

    @staticmethod
    @silent
    def test_process_chains_to_df() -> None:
        """Test returning dataframes of generated process chains and label mappings."""
        TestSpecPipeTensor._init_test_dir()
        test_dir = TestSpecPipeTensor.test_dir
        import numpy as np

        pipe = SpecPipeTensor(create_test_spec_exp(test_dir))

        # Add processes (1 * 1 * 2 = 2 chains)
        pipe.add_process(0, 0, 0, dummy_deterministic_transform, process_label="trans_1")
        conv = DummyConv1x1()
        pipe.add_process(
            0,
            1,
            1,
            conv,
            process_label="conv_1",
            batch_size=16,
            shuffle=True,
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam(conv.parameters()),
        )
        cls_mod = DummyClassifier()
        pipe.add_process(
            1,
            2,
            0,
            [cls_mod, cls_mod],
            process_label=["cls_1", "cls_2"],
            batch_size=32,
            shuffle=False,
            criterion=nn.CrossEntropyLoss(),
            optimizer=[
                torch.optim.Adam(cls_mod.parameters()),
                torch.optim.Adam(cls_mod.parameters()),
            ],
        )

        # Test process chains to df alias
        pcs_df = pipe.process_chains_to_df(print_label=False)
        assert pcs_df.shape == (2, 3)

        # Validate that the DataFrame perfectly matches the internal tuple list
        internal_chains_array = np.array(pipe._process_chains)
        assert (pcs_df.to_numpy() == internal_chains_array).all()
        assert np.all(pipe.process_chains_to_df(print_label=False) == pipe.process_chains_to_df(print_label=True))

        # Test return_label = True
        pcs_dfs = pipe.process_chains_to_df(print_label=False, return_label=True)
        assert isinstance(pcs_dfs, tuple)
        assert len(pcs_dfs) == 2

        # Index 0 is IDs, Index 1 is Labels
        assert np.all(pcs_dfs[0] == pcs_df)
        assert pcs_dfs[0].shape == pcs_dfs[1].shape

        # Verify alias mapping worked properly (labels instead of IDs in the second dataframe)
        labels_df = pcs_dfs[1]
        assert "trans_1" in labels_df.values
        assert "conv_1" in labels_df.values
        assert "cls_1" in labels_df.values

        # Test interaction with ls_custom_chains and ls_chains
        df_chains = pipe.ls_chains(print_label=False)
        assert df_chains.shape[0] == 2
        assert (df_chains.to_numpy() == pipe.ls_process_chains(print_label=False).to_numpy()).all()

        df_custom_chains = pipe.ls_custom_chains(print_label=False)
        assert df_custom_chains is None

        # Create a custom chain by slicing the DataFrame to take only the first row
        pcs_dfs_custom = pcs_dfs[0].iloc[:1, :]
        pipe.custom_chains_from_df(pcs_dfs_custom)

        # Validate custom chains override default behavior
        assert pipe.ls_custom_chains(print_label=False).shape[0] == 1
        assert (
            pipe.ls_chains(print_label=False).to_numpy() == pipe.ls_custom_chains(print_label=False).to_numpy()
        ).all()

    @staticmethod
    @silent
    def test_custom_chains_from_df() -> None:
        """Test loading and validating custom process chains from a dataframe."""
        TestSpecPipeTensor._init_test_dir()
        test_dir = TestSpecPipeTensor.test_dir

        pipe = SpecPipeTensor(create_test_spec_exp(test_dir))

        # Add basic linear processes
        pipe.add_process(0, 0, 0, dummy_deterministic_transform, process_label="step_1")
        conv = DummyConv1x1()
        pipe.add_process(
            0,
            1,
            1,
            conv,
            process_label="step_2",
            batch_size=16,
            shuffle=True,
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam(conv.parameters()),
        )
        cls_mod = DummyClassifier()
        pipe.add_process(
            1,
            2,
            0,
            cls_mod,
            process_label="step_3",
            batch_size=32,
            shuffle=False,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(cls_mod.parameters()),
        )

        # Retrieve default chains
        pcs_df = pipe.process_chains_to_df(print_label=False)
        assert pipe._custom_chains == []

        # Load the dataframe directly back into custom chains
        pipe.custom_chains_from_df(pcs_df)

        # Assert internal state was updated correctly
        assert (np.array(pipe._custom_chains) == pcs_df.to_numpy()).all()

        # Test Error Handling: Loading an invalid process ID should raise a ValueError
        invalid_df = pcs_df.copy()
        invalid_df.iloc[0, 0] = "invalid_process_id_123"

        with pytest.raises(ValueError, match="Invalid process chain in given chains"):
            pipe.custom_chains_from_df(invalid_df)

    @staticmethod
    @silent
    def test_compose_pipeline() -> None:
        """Test compose_pipeline functionality, internal state validation, and structural equivalence."""
        TestSpecPipeTensor._init_test_dir()
        test_dir = TestSpecPipeTensor.test_dir

        # Shared mock objects
        conv = DummyConv1x1()
        cls1 = DummyClassifier()
        cls2 = DummyClassifier()
        opt_conv = torch.optim.Adam(conv.parameters())
        opt_cls1 = torch.optim.Adam(cls1.parameters())
        opt_cls2 = torch.optim.Adam(cls2.parameters())
        crit = nn.CrossEntropyLoss()

        # =====================================================================
        # PART 1: Equivalence Testing (Recreated from old test method)
        # =====================================================================
        test_exp = create_test_spec_exp(dir_path=test_dir)

        # 1. Create SpecPipeTensor by individual add_process
        pipe1 = SpecPipeTensor(test_exp)
        pipe1.add_process(0, 0, 0, dummy_deterministic_transform)
        pipe1.add_process(0, 1, 1, dummy_deterministic_transform, process_label="deterministic_transform_1")
        pipe1.add_process(0, 1, 1, dummy_deterministic_transform, process_label="deterministic_transform_2")
        pipe1.add_process(
            1, 2, 0, cls1, process_label="Model_A", batch_size=16, shuffle=True, criterion=crit, optimizer=opt_cls1
        )
        pipe1.add_process(
            1, 2, 0, cls2, process_label="Model_B", batch_size=16, shuffle=True, criterion=crit, optimizer=opt_cls2
        )

        # 2. Create SpecPipeTensor by add_process with lists (multiple adding)
        pipe2 = SpecPipeTensor(test_exp)
        pipe2.add_process(0, 0, 0, dummy_deterministic_transform)
        pipe2.add_process(
            0,
            1,
            1,
            [dummy_deterministic_transform, dummy_deterministic_transform],
            process_label=["deterministic_transform_1", "deterministic_transform_2"],
        )
        pipe2.add_process(
            1,
            2,
            0,
            [cls1, cls2],
            process_label=["Model_A", "Model_B"],
            batch_size=16,
            shuffle=True,
            criterion=crit,
            optimizer=[opt_cls1, opt_cls2],
        )
        # Assert structural equivalence
        assert pipe2.ls_process(return_result=True, print_result=False).equals(
            pipe1.ls_process(return_result=True, print_result=False)
        )
        assert pipe2.ls_chains(print_label=False).equals(pipe1.ls_chains(print_label=False))

        # 3. Create SpecPipeTensor by compose_pipeline (direct structure)
        pipe3 = SpecPipeTensor(test_exp)
        pipe3.compose_pipeline(
            [
                ((0, 0), dummy_deterministic_transform),
                (
                    (0, 1),
                    {
                        "deterministic_transform_1": dummy_deterministic_transform,
                        "deterministic_transform_2": dummy_deterministic_transform,
                    },
                ),
                (
                    (1, 2),
                    {'Model_A': cls1, 'Model_B': cls2},
                    {'batch_size': 16, 'shuffle': True, 'criterion': crit, 'optimizer': [opt_cls1, opt_cls2]},
                ),
            ]
        )
        # Assert structural equivalence
        assert pipe3.ls_process(return_result=True, print_result=False).equals(
            pipe1.ls_process(return_result=True, print_result=False)
        )
        assert pipe3.ls_chains(print_label=False).equals(pipe1.ls_chains(print_label=False))

        # 4. Create SpecPipeTensor by compose_pipeline (testing additional param overrides)
        pipe4 = SpecPipeTensor(test_exp)
        pipe4.compose_pipeline(
            [
                ((0, 0), dummy_deterministic_transform),
                (
                    (0, 1),
                    {
                        "deterministic_transform_1": dummy_deterministic_transform,
                        "deterministic_transform_2": dummy_deterministic_transform,
                    },
                ),
                (
                    (1, 2),
                    {'Model_A': cls1, 'Model_B': cls2},
                    {'batch_size': 64, 'shuffle': False, 'criterion': crit, 'optimizer': [opt_cls1, opt_cls2]},
                ),
            ]
        )
        # The output df of ls_process formats the method as `<nn.Module: file_name>`
        # and hides internal dict configurations (like batch_size), so it should still equate structurally.
        assert pipe4.ls_process(return_result=True, print_result=False).equals(
            pipe1.ls_process(return_result=True, print_result=False)
        )
        assert pipe4.ls_chains(print_label=False).equals(pipe1.ls_chains(print_label=False))

        # =====================================================================
        # PART 2: Validate Internal State (Parameter Injection)
        # =====================================================================

        # Build a complex pipeline
        pipe_complex = SpecPipeTensor(test_exp)
        pipe_complex.compose_pipeline(
            [
                (
                    (0, 0),
                    {
                        "deterministic_transform_1": dummy_deterministic_transform,
                        "deterministic_transform_2": dummy_deterministic_transform,
                    },
                ),
                (('function', 'fittable'), dummy_deterministic_transform),
                ((1, 1), conv, {'batch_size': 8, 'shuffle': True, 'criterion': nn.MSELoss(), 'optimizer': opt_conv}),
                (
                    (1, 2),
                    {'Cls_A': cls1, 'Cls_B': cls2},
                    {
                        'batch_size': 32,
                        'shuffle': True,
                        'criterion': crit,
                        'optimizer': [opt_cls1, opt_cls2],
                        'is_regression': False,
                    },
                ),
            ]
        )

        # 1. Total number of added discrete processes (2 + 1 + 1 + 2 = 6)
        assert len(pipe_complex._process) == 6, f"Expected 6 processes, got {len(pipe_complex._process)}"

        # 2. Sequence Counter tracking (appseq_tracker verification)
        level_0_seqs = [p[4] for p in pipe_complex._process if p[2] == 'function']
        assert level_0_seqs == [0, 0, 1], f"Expected sequence [0, 0, 1] for level 0, got {level_0_seqs}"

        level_1_seqs = [p[4] for p in pipe_complex._process if p[2] == 'fittable']
        assert level_1_seqs == [0, 1, 1], f"Expected sequence [0, 1, 1] for level 1, got {level_1_seqs}"

        # 3. Label Extraction Verification from Dict
        model_labels = [p[1] for p in pipe_complex._process if p[3] == 'model']
        assert "Cls_A" in model_labels and "Cls_B" in model_labels, "Dictionary keys were not mapped to process_labels!"

        # 4. Parameter Injection Verification
        conv_process = next(p for p in pipe_complex._process if p[1] == 'DummyConv1x1')
        method_dict = conv_process[5]
        assert isinstance(method_dict, dict)
        assert method_dict['batch_size'] == 8, "batch_size was not passed correctly"
        assert isinstance(method_dict['criterion'], nn.MSELoss), "criterion was not passed correctly"

        # 5. Full Pipeline Chain Generation Verification
        # Step 1 (2 options) * Step 2 (1 option) * Step 3 (1 option) * Step 4 (2 options) = 4 full chains
        assert (
            len(pipe_complex._process_steps) == 4
        ), f"Expected 4 distinct structural steps, got {len(pipe_complex._process_steps)}"
        assert (
            len(pipe_complex._process_chains) == 4
        ), f"Expected 4 generated chains, got {len(pipe_complex._process_chains)}"

        # Every generated chain must have exactly 4 steps
        for chain in pipe_complex._process_chains:
            assert len(chain) == 4, "A generated chain did not contain exactly 4 steps!"

    @staticmethod
    @silent
    def test_save_load_config() -> None:  # noqa: C901
        """Test save and load pipeline configurations"""
        # Initialize test_dir
        TestSpecPipeTensor._init_test_dir()
        test_dir = TestSpecPipeTensor.test_dir

        conv = DummyConv1x1()
        cls1 = DummyClassifier()
        opt_cls = torch.optim.Adam(cls1.parameters())
        opt_conv = torch.optim.Adam(conv.parameters())
        crit = nn.CrossEntropyLoss()

        # Create test spec exp
        test_exp = create_test_spec_exp(test_dir)
        pipe = SpecPipeTensor(test_exp)

        # Add process
        pipe.add_process(0, 1, 0, dummy_deterministic_transform)
        pipe.add_process(1, 1, 0, conv, batch_size=8, shuffle=True, criterion=nn.MSELoss(), optimizer=opt_conv)
        pipe.add_process(
            1, 2, 0, cls1, process_label="Model_A", batch_size=16, shuffle=True, criterion=crit, optimizer=opt_cls
        )

        assert len(pipe.process_chains) == 1

        pipe.save_pipe_config()

        pipe.add_process(
            1, 2, 0, cls1, process_label="Model_B", batch_size=16, shuffle=True, criterion=crit, optimizer=opt_cls
        )
        assert len(pipe.process_chains) == 2

        pipe.load_pipe_config()
        assert len(pipe.process_chains) == 1

    # %% Tests for running pipelines

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
