# -*- coding: utf-8 -*-
"""
Tests for spectral image processing and modeling pipeline (SpecPipe)

Copyright (c) 2025 Siwei Luo. MIT License.
"""

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
from swectral.pipeline import SpecPipe  # noqa: E402

# Confirm proper LOKY_MAX_CPU_COUNT
loky_max_cpu_count = str(cpu_count())
os.environ.setdefault('LOKY_MAX_CPU_COUNT', loky_max_cpu_count)

# Check if cuda is available
try:
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


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
) -> SpecPipe:
    """Create a standard test SpecPipe instance."""
    # Create test spec exp
    test_exp = create_test_spec_exp(
        dir_path=dir_path, sample_n=sample_n, n_bands=n_bands, is_regression=is_regression, use_val_group=use_val_group
    )
    pipe = SpecPipe(test_exp)

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


class TestSpecPipe(unittest.TestCase):
    """Test class for SpecPipe functionality."""

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
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Error test: No group
        test_exp = SpecExp(test_dir)
        with pytest.raises(ValueError, match="No group is found"):
            SpecPipe(test_exp)

        # Add group - no img
        test_exp.add_groups(["test_group"])
        with pytest.raises(ValueError, match="Neither image path nor standalone spectrum is found"):
            SpecPipe(test_exp)

        # Add img - no ROI
        img_path = test_dir + "/test_img.tif"
        create_test_raster(raster_path=img_path, width=50, height=50, bands=4)
        test_exp.add_images_by_name(image_name=["test_img.tif"], image_directory=test_dir, group="test_group")
        with pytest.raises(ValueError, match="Neither sample ROI nor standalone spectrum is found"):
            SpecPipe(test_exp)

        # Add ROI - no targets
        roi_path = test_dir + "/test_roi.xml"
        create_test_roi_xml(roi_path, roi_count=10)
        test_exp.add_rois_by_file(group="test_group", path=[roi_path], image_name="test_img.tif")
        with pytest.raises(ValueError, match="No sample target value is found"):
            SpecPipe(test_exp)

        # Add targets - valid img spec exp
        # Sample labels
        dflb = test_exp.ls_sample_labels()
        dflb.iloc[:, 1] = [f"sample_{str(i + 1)}" for i in range(len(dflb))]
        test_exp.sample_labels = dflb
        # Target values
        dft = test_exp.ls_sample_targets()
        dft["Target_value"] = list(np.random.rand(len(dft)))
        test_exp.sample_targets = dft
        # Create spec pipe
        pipe_img_spec = SpecPipe(test_exp)
        assert type(pipe_img_spec.spec_exp) is SpecExp
        assert pipe_img_spec._sample_targets == test_exp.sample_targets
        assert pipe_img_spec.report_directory == test_exp.report_directory
        assert type(pipe_img_spec.create_time) is str
        assert len(pipe_img_spec.create_time) > 0

        # Error test: Add blank group - incomplete single group without img
        test_exp.add_groups(["test_group2"])
        with pytest.raises(
            ValueError,
            match="Neither image nor standalone spectrum is found in group: 'test_group2'",
        ):
            SpecPipe(test_exp)

        # Add img - incomplete single group without ROI
        test_exp.add_images_by_name(image_name=["test_img.tif"], image_directory=test_dir, group="test_group2")
        with pytest.raises(
            ValueError,
            match="Neither image sample ROI nor standalone spectrum is found in group: 'test_group2'",
        ):
            SpecPipe(test_exp)

        # Add standalone spec - hybrid exp error
        test_exp.add_standalone_specs(group="test_group", spec_data=[[1, 2, 3, 4], [5, 6, 7, 8]])
        with pytest.raises(
            ValueError,
            match="Hybrid samples",
        ):
            SpecPipe(test_exp)
        with pytest.raises(
            ValueError,
            match="not allowed",
        ):
            SpecPipe(test_exp)

    @staticmethod
    @silent
    def test_initialization_standalone_spec_exp() -> None:
        """Test SpecPipe instance initialization with standalone spectra experiment"""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        test_exp = SpecExp(test_dir)
        test_exp.add_groups(["test_group"])
        test_exp.add_standalone_specs(group="test_group", spec_data=[[1, 2, 3, 4], [5, 6, 7, 8]])
        test_exp.add_groups(["test_group2"])
        with pytest.raises(ValueError, match="No spectrum is found in group: 'test_group2'"):
            SpecPipe(test_exp)

        ## Correct 1d spec exp
        test_exp.rm_group("test_group2")
        # Sample labels
        dflb = test_exp.ls_sample_labels()
        dflb.iloc[:, 1] = ["sample1", "sample2"]
        test_exp.sample_labels = dflb
        # Target values
        dft = test_exp.ls_sample_targets()
        dft["Target_value"] = list(np.random.rand(2))
        test_exp.sample_targets = dft
        # Create pipe
        pipe_standalone_spec = SpecPipe(test_exp)
        assert type(pipe_standalone_spec.spec_exp) is SpecExp
        assert pipe_standalone_spec._sample_targets == test_exp.sample_targets
        assert pipe_standalone_spec.report_directory == test_exp.report_directory
        assert type(pipe_standalone_spec.create_time) is str
        assert len(pipe_standalone_spec.create_time) > 0

    @staticmethod
    @silent
    def test_add_process() -> None:
        """Test adding preprocessing processes to the pipeline"""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create test spec exp
        test_exp = create_test_spec_exp(test_dir)

        # Basic
        pipe = SpecPipe(test_exp)
        assert len(pipe.process) == 0
        pipe.add_process(0, 0, 0, original_img)
        assert len(pipe.process) == 1
        pipe.add_process(1, 1, 1, snv)
        # Correct process updating
        assert len(pipe.process) == 2
        for proc in pipe.process:
            assert len(proc) == 8
        # Correct chain updating
        assert pipe.process_steps == [["0_0_%#1"], ["1_1_%#1"]]
        # Correct chains updating
        assert pipe.process_chains == [("0_0_%#1", "1_1_%#1")]

        # Multiple process options
        pipe.add_process(1, 1, 1, snv, process_label="snv1")
        # Correct process updating
        assert len(pipe.process) == 3
        for proc in pipe.process:
            assert len(proc) == 8
        # Correct chain updating
        assert pipe.process_steps == [["0_0_%#1"], ["1_1_%#1", "1_1_%#2"]]
        # Correct chains updating
        assert pipe.process_chains == [("0_0_%#1", "1_1_%#1"), ("0_0_%#1", "1_1_%#2")]

        # Sequential processes
        pipe = SpecPipe(test_exp)
        pipe.add_process(1, 1, 0, snv, process_label="snv1")
        pipe.add_process(1, 1, 1, snv, process_label="snv2")
        pipe.add_process(1, 1, 2, snv, process_label="snv3")
        assert len(pipe.process) == 3
        assert pipe.process_steps == [["1_0_%#1"], ["1_1_%#1"], ["1_2_%#1"]]
        assert pipe.process_chains == [("1_0_%#1", "1_1_%#1", "1_2_%#1")]

        # Method of other preprocessing data levels
        pipe.add_process(2, 2, 3, arr_snv)
        assert len(pipe.process) == 4
        if HAS_CUDA:
            pipe.add_process(3, 3, 4, tensor_snv)
            assert len(pipe.process) == 5
            pipe.add_process(4, 4, 5, hypert_snv)
            assert len(pipe.process) == 6
            pipe.add_process(5, 6, 0, roispec)
            assert len(pipe.process) == 7
            pipe.add_process(6, 7, 0, Stats2d().median)
            assert len(pipe.process) == 8
            pipe.add_process(7, 7, 0, snv)
            assert len(pipe.process) == 9
            n_added = 9
        else:
            pipe.add_process(5, 6, 0, roispec)
            assert len(pipe.process) == 5
            pipe.add_process(6, 7, 0, Stats2d().median)
            assert len(pipe.process) == 6
            pipe.add_process(7, 7, 0, snv)
            assert len(pipe.process) == 7
            n_added = 7

        # Method of assembly data levels
        pipe.add_process(7, 8, 0, identity_assembly, process_label='identity_assembly1')
        assert len(pipe.process) == n_added + 1
        pipe.add_process(8, 8, 0, identity_assembly, process_label='identity_assembly2')
        assert len(pipe.process) == n_added + 2

        # Image levels cross-level parallel processes
        pipe = SpecPipe(test_exp)
        pipe.add_process(0, 0, 0, original_img)
        pipe.add_process(1, 1, 0, snv)
        pipe.add_process(1, 1, 0, snv, process_label="snv1")
        assert len(pipe.process) == 3
        assert pipe.process_steps == [["0_0_%#1", "1_0_%#2", "1_0_%#3"]]
        assert pipe.process_chains == [("0_0_%#1",), ("1_0_%#2",), ("1_0_%#3",)]

    @staticmethod
    @silent
    def test_add_process_validation() -> None:
        """Test data validation in SpecPipe.add_process"""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create spec pipe
        test_exp = create_test_spec_exp(test_dir)
        pipe = SpecPipe(test_exp)

        # Heterogenous data levels at same sequence
        pipe.add_process(0, 2, 0, original_img)
        with pytest.raises(ValueError, match="must have identical output data levels"):
            pipe.add_process(0, 7, 0, img_const)

        # Invalid combination of input and output data levels
        with pytest.raises(ValueError, match="Output_data_level cannot precede the input_data_level"):
            pipe.add_process(7, 0, 0, Stats2d().mean)

        # Invalid previous and subsequent data levels
        with pytest.raises(ValueError, match="inconsistent with the output data level"):
            pipe.add_process(6, 7, 0, Stats2d().mean)

        # Invalid methods
        with pytest.raises(TypeError, match="must be a model or Callable or list"):
            pipe.add_process(0, 7, 1, "Invalid process")
        with pytest.raises(ValueError, match="Method validation fails"):
            pipe.add_process(0, 7, 1, print)

    @staticmethod
    @silent
    def test_add_multiple_processes() -> None:
        """Test add multiple processes"""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create spec pipe
        test_exp = create_test_spec_exp(test_dir)
        pipe = SpecPipe(test_exp)

        # Add list of processes
        pipe.add_process(0, 0, 0, [original_img, original_img1])

        # Add list of processes with labels
        pipe.add_process(0, 0, 1, [original_img, original_img], process_label=['process_label_a', 'process_label_b'])

        # Label type mismatch
        with pytest.raises(TypeError, match="must be a list of str with a length equal to the number of methods"):
            pipe.add_process(0, 0, 2, [original_img, original_img1], process_label='process_label_c')
        # Label length mismatch
        with pytest.raises(ValueError, match="do not match"):
            pipe.add_process(0, 0, 2, [original_img, original_img1], process_label=['process_label_d'])

        # Single element list
        pipe.add_process(0, 0, 2, [original_img], process_label='original_img_2')
        pipe.add_process(0, 0, 2, [original_img], process_label=['original_img_3'])
        pipe.add_process(0, 0, 2, original_img, process_label=['original_img_4'])
        pipe.add_process(0, 0, 2, original_img, process_label='original_img_5')

        # Add multiple models
        pipe.add_process(5, 7, 0, roi_mean)
        knn1 = KNeighborsRegressor(n_neighbors=3)
        knn2 = KNeighborsRegressor(n_neighbors=5)
        pipe.add_model([knn1, knn2])
        # Add with labels
        pipe.add_model([knn1, knn2], model_label=["knn1", "knn2"])
        with pytest.raises(TypeError, match="must be a list of str with a length equal to the number of methods"):
            pipe.add_model([knn1, knn2], model_label="knn1")
        with pytest.raises(ValueError, match="do not match"):
            pipe.add_model([knn1, knn2], model_label=["knn1"])

    @staticmethod
    @silent
    def test_ls_process() -> None:
        """Test listing added processes in the pipeline"""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create test spec exp
        test_exp = create_test_spec_exp(test_dir)
        pipe = SpecPipe(test_exp)
        pipe.add_process(0, 0, 0, original_img, process_label="original_img1")
        pipe.add_process(0, 0, 0, original_img, process_label="original_img2")
        pipe.add_process(0, 0, 1, original_img, process_label="original_img3")
        pipe.add_process(1, 1, 2, snv)
        pipe.add_process(2, 2, 3, arr_snv, process_label="arr_snv1")
        pipe.add_process(2, 2, 3, arr_snv, process_label="arr_snv2")
        pipe.add_process(2, 2, 3, arr_snv, process_label="arr_snv3")

        # List all
        procs = pipe.ls_process(print_result=False, return_result=True)
        assert procs.shape == (7, 8)

        # Filter conditions
        procs = pipe.ls_process(input_data_level=0, print_result=False, return_result=True)
        assert procs.shape == (3, 8)
        procs = pipe.ls_process(input_data_level=0, application_sequence=0, print_result=False, return_result=True)
        assert procs.shape == (2, 8)
        procs = pipe.ls_process(output_data_level=1, print_result=False, return_result=True)
        assert procs.shape == (1, 8)
        procs = pipe.ls_process(method="snv", print_result=False, return_result=True)
        assert procs.shape == (1, 8)
        # No match
        procs = pipe.ls_process(input_data_level=0, output_data_level=1, print_result=False, return_result=True)
        assert procs.shape == (0, 8)

        # Not exact match
        procs = pipe.ls_process(method="snv", exact_match=False, print_result=False, return_result=True)
        assert procs.shape == (4, 8)

        # Not return result
        procs = pipe.ls_process(method="snv", print_result=True, return_result=False)
        assert procs is None

    @staticmethod
    @silent
    def test_rm_process() -> None:
        """Test removing added process in the pipeline"""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create test spec exp
        test_exp = create_test_spec_exp(test_dir)
        pipe = SpecPipe(test_exp)
        pipe.add_process(0, 0, 0, original_img, process_label="original_img1")
        pipe.add_process(0, 0, 0, original_img, process_label="original_img2")
        pipe.add_process(0, 0, 1, original_img, process_label="original_img3")
        pipe.add_process(1, 1, 2, snv)
        pipe.add_process(2, 2, 3, arr_snv)

        assert len(pipe.process) == 5
        assert len(pipe.process_steps) == 4
        assert len(pipe.process_chains) == 2
        for chain in pipe.process_chains:
            assert len(chain) == 4

        # Remove procs
        pipe.rm_process(method="snv")
        assert len(pipe.process) == 4
        assert len(pipe.process_steps) == 3
        assert len(pipe.process_chains) == 2
        for chain in pipe.process_chains:
            assert len(chain) == 3

        pipe.rm_process(process_id=pipe.process_steps[0][0])
        assert len(pipe.process) == 3
        assert len(pipe.process_steps) == 3
        assert len(pipe.process_chains) == 1
        for chain in pipe.process_chains:
            assert len(chain) == 3

        pipe.rm_process(input_data_level=2)
        assert len(pipe.process) == 2
        assert len(pipe.process_steps) == 2
        assert len(pipe.process_chains) == 1
        for chain in pipe.process_chains:
            assert len(chain) == 2

        pipe.rm_process(output_data_level=1)
        assert len(pipe.process) == 2
        assert len(pipe.process_steps) == 2
        assert len(pipe.process_chains) == 1
        for chain in pipe.process_chains:
            assert len(chain) == 2

    @staticmethod
    @silent
    def test_add_model() -> None:
        """Test adding model process to processing pipeline"""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        regressor = RandomForestRegressor(n_estimators=10)

        # Create test spec exp
        test_exp = create_test_spec_exp(test_dir)
        pipe = SpecPipe(test_exp)

        # Add process
        pipe.add_process(0, 0, 0, original_img)
        pipe.add_process(5, 7, 0, roi_mean)
        assert len(pipe.ls_process(return_result=True, print_result=False)) == 2
        assert len(pipe.ls_process(output_data_level=9, return_result=True, print_result=False)) == 0
        assert len(pipe.process_steps) == 2
        assert len(pipe.process_chains) == 1

        # Add models
        pipe.add_process(7, 9, 0, regressor)
        assert len(pipe.ls_process(return_result=True, print_result=False)) == 3
        assert len(pipe.ls_process(output_data_level=9, return_result=True, print_result=False)) == 1
        assert len(pipe.process_steps) == 3
        assert len(pipe.process_chains) == 1
        pipe.add_process(7, 9, 0, regressor)
        assert len(pipe.ls_process(return_result=True, print_result=False)) == 4
        assert len(pipe.ls_process(output_data_level=9, return_result=True, print_result=False)) == 2
        assert len(pipe.process_steps) == 3
        assert len(pipe.process_chains) == 2

    @staticmethod
    @silent
    def test_ls_model() -> None:
        """Test listing added models in the pipeline"""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        regressor = RandomForestRegressor(n_estimators=10)

        # Create test spec exp
        test_exp = create_test_spec_exp(test_dir)
        pipe = SpecPipe(test_exp)

        # Add process
        pipe.add_process(0, 0, 0, original_img)
        pipe.add_process(5, 7, 0, roi_mean)
        assert len(pipe.ls_model(return_result=True, print_result=False)) == 0

        # Add and list models
        pipe.add_process(7, 9, 0, regressor)
        assert len(pipe.ls_model(return_result=True, print_result=False)) == 1
        pipe.add_process(7, 9, 0, regressor)
        assert len(pipe.ls_model(return_result=True, print_result=False)) == 2
        assert len(pipe.ls_model(model_id=pipe.process_chains[0][-1], return_result=True, print_result=False)) == 1
        assert pipe.ls_model(return_result=False, print_result=True) is None

    @staticmethod
    @silent
    def test_rm_model() -> None:
        """Test removing added models in the pipeline"""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        regressor_1 = KNeighborsRegressor(n_neighbors=3)
        regressor_2 = RandomForestRegressor(n_estimators=10)

        # Create test spec exp
        test_exp = create_test_spec_exp(test_dir)
        pipe = SpecPipe(test_exp)
        pipe.add_process(0, 0, 0, original_img)
        pipe.add_process(5, 7, 0, roi_mean)

        # Add models
        def add_models() -> None:
            pipe.add_process(7, 9, 0, regressor_1)
            pipe.add_process(7, 9, 0, regressor_1, process_label="test_regressor")
            pipe.add_process(7, 9, 0, regressor_2)

        add_models()
        assert len(pipe.ls_process(return_result=True, print_result=False)) == 5
        assert len(pipe.ls_model(return_result=True, print_result=False)) == 3
        assert len(pipe.process_steps) == 3
        assert len(pipe.process_chains) == 3

        # Remove model by method name
        pipe.rm_model(model_method="RandomForestRegressor")
        assert len(pipe.ls_process(return_result=True, print_result=False)) == 4
        assert len(pipe.ls_model(return_result=True, print_result=False)) == 2
        assert len(pipe.process_steps) == 3
        assert len(pipe.process_chains) == 2

        # Remove all models
        pipe.rm_model()
        assert len(pipe.ls_process(return_result=True, print_result=False)) == 2
        assert len(pipe.ls_model(return_result=True, print_result=False)) == 0
        assert len(pipe.process_steps) == 2
        assert len(pipe.process_chains) == 1

        # Remove model by custom label
        add_models()
        pipe.rm_model(model_label="test_regressor")
        assert len(pipe.ls_process(return_result=True, print_result=False)) == 4
        assert len(pipe.ls_model(return_result=True, print_result=False)) == 2
        assert len(pipe.process_steps) == 3
        assert len(pipe.process_chains) == 2

        # Remove model by id
        pipe.rm_model(model_id=pipe.process_chains[0][-1])
        assert len(pipe.ls_process(return_result=True, print_result=False)) == 3
        assert len(pipe.ls_model(return_result=True, print_result=False)) == 1
        assert len(pipe.process_steps) == 3
        assert len(pipe.process_chains) == 1

        # Invalid removal
        pipe.rm_model(model_id=pipe.process_chains[0][-1], model_label="non_existed")
        assert len(pipe.ls_process(return_result=True, print_result=False)) == 3
        assert len(pipe.ls_model(return_result=True, print_result=False)) == 1
        assert len(pipe.process_steps) == 3
        assert len(pipe.process_chains) == 1

    @staticmethod
    @silent
    def test_build_pipeline() -> None:
        """Test build_pipeline functionality."""
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create test spec exp
        test_exp = create_test_spec_exp(
            dir_path=test_dir, sample_n=10, n_bands=8, is_regression=True, use_val_group=False
        )

        # Create SpecPipe by add_process
        pipe1 = SpecPipe(test_exp)
        # Add processes
        pipe1.add_process(0, 0, 0, original_img)
        pipe1.add_process(2, 2, 1, arr_ori)
        pipe1.add_process(2, 2, 1, arr_simple_half)
        pipe1.add_process(5, 6, 0, roispec)
        pipe1.add_process(6, 7, 0, Stats2d().mean)
        pipe1.add_process(7, 9, 0, RandomForestRegressor(n_estimators=6), validation_method="2-fold")
        pipe1.add_process(7, 9, 0, KNeighborsRegressor(n_neighbors=3), validation_method="2-fold")

        # Create SpecPipe by add_process - multiple adding
        pipe2 = SpecPipe(test_exp)
        # Add processes
        pipe2.add_process(0, 0, 0, original_img)
        pipe2.add_process(2, 2, 1, [arr_ori, arr_simple_half])
        pipe2.add_process(5, 6, 0, roispec)
        pipe2.add_process(6, 7, 0, Stats2d().mean)
        pipe2.add_process(
            7,
            9,
            0,
            [RandomForestRegressor(n_estimators=6), KNeighborsRegressor(n_neighbors=3)],
            validation_method="2-fold",
        )
        assert pipe2.ls_process(return_result=True).equals(pipe1.ls_process(return_result=True))
        assert pipe2.ls_chains().equals(pipe1.ls_chains())

        # Create SpecPipe by build_pipeline
        pipe3 = SpecPipe(test_exp)
        # Add processes
        pipe3.build_pipeline(
            [
                ((0, 0), original_img),
                ((2, 2), [arr_ori, arr_simple_half]),
                ((5, 6), roispec),
                ((6, 7), Stats2d().mean),
                ((7, 9), [RandomForestRegressor(n_estimators=6), KNeighborsRegressor(n_neighbors=3)]),
            ]
        )
        assert pipe3.ls_process(return_result=True).equals(pipe1.ls_process(return_result=True))
        assert pipe3.ls_chains().equals(pipe1.ls_chains())

        # Test additional params of build_pipeline
        pipe4 = SpecPipe(test_exp)
        # Add processes
        pipe4.build_pipeline(
            [
                ((0, 0), original_img),
                ((2, 2), [arr_ori, arr_simple_half]),
                ((5, 6), roispec),
                ((6, 7), Stats2d().mean),
                (
                    (7, 9),
                    [RandomForestRegressor(n_estimators=6), KNeighborsRegressor(n_neighbors=3)],
                    {'validation_method': 'loo'},
                ),
            ]
        )
        assert pipe4.ls_process(return_result=True).equals(pipe1.ls_process(return_result=True))
        assert pipe4.ls_chains().equals(pipe1.ls_chains())

    @staticmethod
    @silent
    def test_ls_process_chains() -> None:
        """Test ls_process_chains and ls_chains functionalities and edge cases."""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        regressor = RandomForestRegressor(n_estimators=10)

        # Basic functionality
        pipe = create_test_spec_pipe(test_dir)
        assert np.all(pipe.ls_chains() == pipe.ls_process_chains())
        assert pipe.ls_chains().shape[1] == 7
        assert pipe.ls_chains(stage="preprocessing").shape[1] == 4
        assert pipe.ls_chains(stage="assembly").shape[1] == 2
        assert pipe.ls_chains(stage="modeling").shape[1] == 1

        # Edge cases
        # Create test spec exp
        test_exp = create_test_spec_exp(test_dir)
        # Preproc:Assembly:Model = T:0:0
        pipe = SpecPipe(test_exp)
        pipe.add_process(5, 7, 0, roi_mean)
        assert pipe.ls_chains().shape[1] == 1
        assert pipe.ls_chains(stage="preprocessing").shape[1] == 1
        assert pipe.ls_chains(stage="assembly").shape[1] == 0
        assert pipe.ls_chains(stage="modeling").shape[1] == 0
        # Preproc:Assembly:Model = T:1:0
        pipe = SpecPipe(test_exp)
        pipe.add_process(5, 7, 0, roi_mean)
        pipe.add_process(7, 8, 0, identity_assembly)
        assert pipe.ls_chains().shape[1] == 2
        assert pipe.ls_chains(stage="preprocessing").shape[1] == 1
        assert pipe.ls_chains(stage="assembly").shape[1] == 1
        assert pipe.ls_chains(stage="modeling").shape[1] == 0
        # Preproc:Assembly:Model = T:0:1
        pipe = SpecPipe(test_exp)
        pipe.add_process(5, 7, 0, roi_mean)
        pipe.add_process(7, 9, 0, regressor)
        assert pipe.ls_chains().shape[1] == 2
        assert pipe.ls_chains(stage="preprocessing").shape[1] == 1
        assert pipe.ls_chains(stage="assembly").shape[1] == 0
        assert pipe.ls_chains(stage="modeling").shape[1] == 1

    @staticmethod
    @silent
    def test_process_chains_to_df() -> None:
        """Test return dataframe of generated process chains."""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        regressor = RandomForestRegressor(n_estimators=10)

        # Create test spec exp
        test_exp = create_test_spec_exp(test_dir)
        pipe = SpecPipe(test_exp)

        # Add process
        pipe.add_process(0, 0, 0, original_img)
        pipe.add_process(5, 7, 0, roi_mean)
        pipe.add_process(7, 9, 0, regressor)
        pipe.add_process(7, 9, 0, regressor)

        # Test procs to df
        pcs_df = pipe.process_chains_to_df(print_label=False)
        assert pcs_df.shape == (2, 3)
        assert (pcs_df.to_numpy() == np.array(pipe.process_chains)).all()
        assert np.all(pipe.process_chains_to_df(print_label=False) == pipe.process_chains_to_df(print_label=True))

        # Test return_label = True
        pcs_dfs = pipe.process_chains_to_df(print_label=False, return_label=True)
        assert isinstance(pcs_dfs, tuple)
        assert np.all(pcs_dfs[0] == pcs_df)
        assert pcs_dfs[0].shape == pcs_dfs[1].shape
        pcs_dfs1 = pipe.process_chains_to_df(print_label=True, return_label=True)
        assert np.all(pcs_dfs[0] == pcs_dfs1[0])
        assert np.all(pcs_dfs[1] == pcs_dfs1[1])

        # ls_custom_chains and ls_chains
        df_chains = pipe.ls_chains(print_label=False)
        assert df_chains.shape[0] == 2
        assert (df_chains.to_numpy() == pipe.ls_process_chains(print_label=False).to_numpy()).all()
        df_custom_chains = pipe.ls_custom_chains(print_label=False)
        assert df_custom_chains is None
        pcs_dfs2 = pcs_dfs[0].iloc[:-1, :]
        pipe.custom_chains_from_df(pcs_dfs2)
        assert pipe.ls_custom_chains(print_label=False).shape[0] == 1
        assert (pipe.ls_chains().to_numpy() == pipe.ls_custom_chains().to_numpy()).all()

    @staticmethod
    @silent
    def test_custom_chains_from_df() -> None:
        """Test load custom process chains from dataframe."""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        regressor = RandomForestRegressor(n_estimators=10)

        # Create test spec exp
        test_exp = create_test_spec_exp(test_dir)
        pipe = SpecPipe(test_exp)

        # Add process
        pipe.add_process(0, 0, 0, original_img)
        pipe.add_process(5, 7, 0, roi_mean)
        pipe.add_process(7, 9, 0, regressor)
        pipe.add_process(7, 9, 0, regressor)

        # Test procs to df
        pcs_df = pipe.process_chains_to_df()
        assert pipe.custom_chains == []
        pipe.custom_chains_from_df(pcs_df)
        assert (np.array(pipe.custom_chains) == pcs_df.to_numpy()).all()

    @staticmethod
    @silent
    def test_save_load_config() -> None:  # noqa: C901
        """Test save and load pipeline configurations"""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        regressor = RandomForestRegressor(n_estimators=10)

        # Create test spec exp
        test_exp = create_test_spec_exp(test_dir)
        pipe = SpecPipe(test_exp)

        # Add process
        pipe.add_process(0, 0, 0, original_img)
        pipe.add_process(5, 7, 0, roi_mean)
        pipe.add_process(7, 9, 0, regressor)

        assert len(pipe.process_chains) == 1

        pipe.save_pipe_config()

        pipe.add_process(7, 9, 0, regressor)
        assert len(pipe.process_chains) == 2

        pipe.load_pipe_config()
        assert len(pipe.process_chains) == 1

    @staticmethod
    @silent
    def test_test_run_regression() -> None:  # noqa: C901
        """Test test_run for regression tasks"""

        if os.getenv("SPECPIPE_PREPROCESS_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"]
        if os.getenv("SPECPIPE_MODEL_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"]

        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create pipe
        pipe = create_test_spec_pipe(dir_path=test_dir, sample_n=12)

        # Fast test without saving result
        assert pipe._tested is False
        assert pipe._sample_data == []

        plt.close("all")
        pipe.test_run(test_modeling=False, dump_result=False, save_preprocessed_images=True)
        time.sleep(0.1)

        assert pipe._tested is False
        preprocessed_img_path = f"{test_dir}/test_run/Preprocessed_images/"
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

        # Full test without saving result
        plt.close("all")
        pipe.test_run(dump_result=False)
        time.sleep(0.1)

        # Assert modeling test status
        assert pipe._tested is True
        # Assert report files
        model_report_dir = f"{test_dir}/test_run/Model_evaluation_reports/"
        assert os.path.exists(model_report_dir)
        procs_result_dirs = lsdir_robust(model_report_dir)
        assert len(procs_result_dirs) == len(pipe.process_chains)
        for dirname in procs_result_dirs:
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

        # Partial test
        plt.close("all")
        pipe.test_run(model_test_coverage=0.5)
        time.sleep(0.1)

        assert pipe._tested is True
        assert pipe._sample_data == []
        assert os.path.exists(f"{test_dir}/test_run/Step_results/PreprocessingTestingResult.dill")

        with pytest.raises(ValueError, match="test_coverage must be in"):
            pipe.test_run(model_test_coverage=-0.1)
        with pytest.raises(ValueError, match="test_coverage must be in"):
            pipe.test_run(model_test_coverage=0.0)
        with pytest.raises(ValueError, match="test_coverage must be in"):
            pipe.test_run(model_test_coverage=1.1)

        # Full test
        pipe.__tested = False
        os.remove(f"{test_dir}/test_run/Step_results/PreprocessingTestingResult.dill")

        plt.close("all")
        pipe.test_run()
        time.sleep(0.1)

        assert pipe._tested is True
        assert pipe._sample_data == []
        assert os.path.exists(f"{test_dir}/test_run/Step_results/PreprocessingTestingResult.dill")

        # Backup result
        plt.close("all")
        pipe.test_run(dump_backup=True)
        time.sleep(0.1)

        result_dills = [
            name
            for name in lsdir_robust(f"{test_dir}/test_run/Step_results/")
            if "PreprocessingTestingResult" in name and name != "PreprocessingTestingResult.dill"
        ]
        assert len(result_dills) > 0

    @staticmethod
    @silent
    def test_test_run_classification() -> None:  # noqa: C901
        """Test test_run for classification tasks"""

        if os.getenv("SPECPIPE_PREPROCESS_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"]
        if os.getenv("SPECPIPE_MODEL_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"]

        # Test classification
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create pipe
        pipe = create_test_spec_pipe(test_dir, is_regression=False)

        # Fast test without saving result
        assert pipe._tested is False
        assert pipe._sample_data == []

        plt.close("all")
        pipe.test_run(test_modeling=False, dump_result=False, save_preprocessed_images=True)
        time.sleep(0.1)

        assert pipe._tested is False
        preprocessed_img_path = f"{test_dir}/test_run/Preprocessed_images/"
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

        # Full test without saving result
        plt.close("all")
        pipe.test_run(dump_result=False)
        time.sleep(0.1)

        # Assert modeling test status
        assert pipe._tested is True
        # Assert report files
        model_report_dir = f"{test_dir}/test_run/Model_evaluation_reports/"
        assert os.path.exists(model_report_dir)
        procs_result_dirs = lsdir_robust(model_report_dir)
        assert len(procs_result_dirs) == len(pipe.process_chains)
        for dirname in procs_result_dirs:
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

        # Partial test
        plt.close("all")
        pipe.test_run(model_test_coverage=0.5)
        time.sleep(0.1)

        assert pipe._tested is True
        assert pipe._sample_data == []
        assert os.path.exists(f"{test_dir}/test_run/Step_results/PreprocessingTestingResult.dill")

        # Full test
        pipe.__tested = False
        os.remove(f"{test_dir}/test_run/Step_results/PreprocessingTestingResult.dill")

        plt.close("all")
        pipe.test_run()
        time.sleep(0.1)

        assert pipe._tested is True
        assert pipe._sample_data == []
        assert os.path.exists(f"{test_dir}/test_run/Step_results/PreprocessingTestingResult.dill")

        # Backup result
        plt.close("all")
        pipe.test_run(dump_backup=True)
        time.sleep(0.1)

        result_dills = [
            name
            for name in lsdir_robust(f"{test_dir}/test_run/Step_results/")
            if "PreprocessingTestingResult" in name and name != "PreprocessingTestingResult.dill"
        ]
        assert len(result_dills) > 0

    @staticmethod
    @silent
    def criteria_preprocessing_result(pipe: SpecPipe) -> str:  # noqa: C901
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
    def criteria_regression_model_report(pipe: SpecPipe) -> str:  # noqa: C901
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
    def criteria_classification_model_report(pipe: SpecPipe) -> str:  # noqa: C901
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

    @staticmethod
    @silent
    def test_preprocessing_modeling_regression() -> None:
        """test preprocessing and modeling functionality for regression"""

        if os.getenv("SPECPIPE_PREPROCESS_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"]
        if os.getenv("SPECPIPE_MODEL_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"]

        plt.close("all")

        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create regression exp data and pipe
        pipe = create_test_spec_pipe(test_dir, is_regression=True)

        # Preprocessing
        assert pipe._sample_data == []
        pipe.preprocessing()
        time.sleep(0.1)

        finished_1 = TestSpecPipe.criteria_preprocessing_result(pipe)

        # Modeling
        pipe.assembly()
        pipe.model_evaluation()
        time.sleep(0.1)

        finished_2 = TestSpecPipe.criteria_regression_model_report(pipe)

        # Clear test report dir
        if os.path.exists(test_dir) and finished_1 == "finished" and finished_2 == "finished":
            shutil.rmtree(test_dir)

    @staticmethod
    @silent
    def test_preprocessing_modeling_classification() -> None:
        """test preprocessing and modeling functionality for classification"""

        if os.getenv("SPECPIPE_PREPROCESS_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"]
        if os.getenv("SPECPIPE_MODEL_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"]

        plt.close("all")

        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create regression exp data and pipe
        pipe = create_test_spec_pipe(test_dir, is_regression=False)

        # Preprocessing
        assert pipe._sample_data == []
        pipe.preprocessing()
        time.sleep(0.1)

        finished_1 = TestSpecPipe.criteria_preprocessing_result(pipe)

        # Modeling
        pipe.assembly()
        pipe.model_evaluation()
        time.sleep(0.1)

        finished_2 = TestSpecPipe.criteria_classification_model_report(pipe)

        # Clear test report dir
        if os.path.exists(test_dir) and finished_1 == "finished" and finished_2 == "finished":
            shutil.rmtree(test_dir)

    @staticmethod
    @silent
    def test_run_pipe() -> None:
        """test run preprocessing and modeling functionality using SpecPipe.run()"""

        if os.getenv("SPECPIPE_PREPROCESS_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"]
        if os.getenv("SPECPIPE_MODEL_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"]

        plt.close("all")

        # Regression
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        pipe = create_test_spec_pipe(test_dir, is_regression=True)

        pipe.run(n_processor=1)
        time.sleep(0.1)

        finished_1 = TestSpecPipe.criteria_preprocessing_result(pipe)
        finished_2 = TestSpecPipe.criteria_regression_model_report(pipe)
        finished_3 = "not_started"
        finished_4 = "not_started"

        # Clear test report dir
        crit_1 = finished_1 == "finished" and finished_2 == "finished"
        crit_2 = finished_3 == "not_started" and finished_4 == "not_started"
        if os.path.exists(test_dir) and crit_1 and crit_2:
            shutil.rmtree(test_dir)
            run_1_cleared: bool = True

        plt.close("all")

        # Classification
        assert crit_1
        assert run_1_cleared

        pipe = create_test_spec_pipe(test_dir, is_regression=False)

        pipe.run(n_processor=1)
        time.sleep(0.1)

        finished_3 = TestSpecPipe.criteria_preprocessing_result(pipe)
        finished_4 = TestSpecPipe.criteria_classification_model_report(pipe)

        plt.close("all")

        # Clear test report dir
        if os.path.exists(test_dir) and finished_3 == "finished" and finished_4 == "finished" and crit_1:
            shutil.rmtree(test_dir)

    @staticmethod
    @silent
    def test_run_pipe_no_inter_results() -> None:
        """test run preprocessing and modeling functionality using SpecPipe.run() with step_result=False."""

        if os.getenv("SPECPIPE_PREPROCESS_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"]
        if os.getenv("SPECPIPE_MODEL_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"]

        plt.close("all")

        # Regression
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        pipe = create_test_spec_pipe(test_dir, is_regression=True)

        pipe.run(n_processor=1, step_result=False)
        time.sleep(0.1)

        finished_1 = TestSpecPipe.criteria_preprocessing_result(pipe)
        finished_2 = TestSpecPipe.criteria_regression_model_report(pipe)
        finished_3 = "not_started"
        finished_4 = "not_started"

        # Clear test report dir
        crit_1 = finished_1 == "finished" and finished_2 == "finished"
        crit_2 = finished_3 == "not_started" and finished_4 == "not_started"
        if os.path.exists(test_dir) and crit_1 and crit_2:
            shutil.rmtree(test_dir)
            run_1_cleared: bool = True

        plt.close("all")

        # Classification
        assert crit_1
        assert run_1_cleared

        pipe = create_test_spec_pipe(test_dir, is_regression=False)

        pipe.run(n_processor=1, step_result=False)
        time.sleep(0.1)

        finished_3 = TestSpecPipe.criteria_preprocessing_result(pipe)
        finished_4 = TestSpecPipe.criteria_classification_model_report(pipe)

        plt.close("all")

        # Clear test report dir
        if os.path.exists(test_dir) and finished_3 == "finished" and finished_4 == "finished" and crit_1:
            shutil.rmtree(test_dir)

    @staticmethod
    @silent
    def test_run_pipe_with_validation_group() -> None:
        """test running preprocessing and modeling functionalities using SpecPipe.run() with validation groups."""

        if os.getenv("SPECPIPE_PREPROCESS_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"]
        if os.getenv("SPECPIPE_MODEL_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"]

        plt.close("all")

        # Run with validation group enabled
        # Regression - "2-fold" validation
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        pipe = create_test_spec_pipe(test_dir, sample_n=12, is_regression=True, use_val_group=True)

        with warnings.catch_warnings():
            # Ignore Scipy Wilcoxon warnings in this edge case
            warnings.filterwarnings(
                "ignore", message=".*Sample size too small for normal approximation.*", category=UserWarning
            )
            warnings.filterwarnings(
                "ignore", message=".*Exact p-value calculation does not work if there are zeros.*", category=UserWarning
            )
            pipe.run(n_processor=1)
        time.sleep(0.1)

        finished_1 = TestSpecPipe.criteria_preprocessing_result(pipe)
        finished_2 = TestSpecPipe.criteria_regression_model_report(pipe)
        finished_3 = "not_started"
        finished_4 = "not_started"

        # Clear test report dir
        crit_1 = finished_1 == "finished" and finished_2 == "finished"
        crit_2 = finished_3 == "not_started" and finished_4 == "not_started"
        if os.path.exists(test_dir) and crit_1 and crit_2:
            shutil.rmtree(test_dir)
            run_1_cleared: bool = True

        plt.close("all")

        # Classification - "2-fold" validation
        assert crit_1
        assert run_1_cleared

        pipe = create_test_spec_pipe(test_dir, sample_n=12, is_regression=False, use_val_group=True)

        with warnings.catch_warnings():
            # Ignore Scipy Wilcoxon warnings in this edge case
            warnings.filterwarnings(
                "ignore", message=".*Sample size too small for normal approximation.*", category=UserWarning
            )
            warnings.filterwarnings(
                "ignore", message=".*Exact p-value calculation does not work if there are zeros.*", category=UserWarning
            )
            pipe.run(n_processor=1)
        time.sleep(0.1)

        finished_3 = TestSpecPipe.criteria_preprocessing_result(pipe)
        finished_4 = TestSpecPipe.criteria_classification_model_report(pipe)

        plt.close("all")

        # Regression - "loo" validation - logo
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        pipe = create_test_spec_pipe(
            test_dir, sample_n=12, is_regression=True, use_val_group=True, validation_method="loo"
        )

        with warnings.catch_warnings():
            # Ignore Scipy Wilcoxon warnings in this edge case
            warnings.filterwarnings(
                "ignore", message=".*Sample size too small for normal approximation.*", category=UserWarning
            )
            warnings.filterwarnings(
                "ignore", message=".*Exact p-value calculation does not work if there are zeros.*", category=UserWarning
            )
            pipe.run(n_processor=1)
        time.sleep(0.1)

        finished_1 = TestSpecPipe.criteria_preprocessing_result(pipe)
        finished_2 = TestSpecPipe.criteria_regression_model_report(pipe)
        finished_3 = "not_started"
        finished_4 = "not_started"

        # Clear test report dir
        crit_1 = finished_1 == "finished" and finished_2 == "finished"
        crit_2 = finished_3 == "not_started" and finished_4 == "not_started"
        if os.path.exists(test_dir) and crit_1 and crit_2:
            shutil.rmtree(test_dir)
            run_1_cleared = True

        plt.close("all")

        # Classification - "loo" validation - logo
        assert crit_1
        assert run_1_cleared

        pipe = create_test_spec_pipe(
            test_dir, sample_n=12, is_regression=False, use_val_group=True, validation_method="loo"
        )

        with warnings.catch_warnings():
            # Ignore Scipy Wilcoxon warnings in this edge case
            warnings.filterwarnings(
                "ignore", message=".*Sample size too small for normal approximation.*", category=UserWarning
            )
            warnings.filterwarnings(
                "ignore", message=".*Exact p-value calculation does not work if there are zeros.*", category=UserWarning
            )
            pipe.run(n_processor=1)
        time.sleep(0.1)

        finished_3 = TestSpecPipe.criteria_preprocessing_result(pipe)
        finished_4 = TestSpecPipe.criteria_classification_model_report(pipe)

        plt.close("all")

        # Clear test report dir
        if os.path.exists(test_dir) and finished_3 == "finished" and finished_4 == "finished" and crit_1:
            shutil.rmtree(test_dir)

    @staticmethod
    @silent
    def test_property_access() -> None:
        """Test read-only property access."""

        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        pipe = create_test_spec_pipe(test_dir, is_regression=True)
        with pytest.raises(ValueError, match="cannot be modified"):
            pipe._sample_targets = pipe._sample_targets
        with pytest.raises(ValueError, match="cannot be modified"):
            pipe.process = pipe.process
        with pytest.raises(ValueError, match="cannot be modified"):
            pipe.process_steps = pipe.process_steps
        with pytest.raises(ValueError, match="cannot be modified"):
            pipe.process_chains = pipe.process_chains
        with pytest.raises(ValueError, match="cannot be modified"):
            pipe.custom_chains = pipe.custom_chains
        with pytest.raises(ValueError, match="cannot be modified"):
            pipe._sample_data = pipe._sample_data
        with pytest.raises(ValueError, match="cannot be modified"):
            pipe._pretest_data = pipe._pretest_data
        with pytest.raises(ValueError, match="cannot be modified"):
            pipe._preprocess_result_path = pipe._preprocess_result_path
        with pytest.raises(ValueError, match="cannot be modified"):
            pipe._tested = pipe._tested
        with pytest.raises(ValueError, match="cannot be modified"):
            pipe.create_time = pipe.create_time

    @staticmethod
    @silent
    def test_resume_preprocessing() -> None:
        """Test resume functionality of 'preprocessing'."""

        if os.getenv("SPECPIPE_PREPROCESS_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"]
        if os.getenv("SPECPIPE_MODEL_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"]

        plt.close("all")

        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create regression exp data and pipe
        pipe = create_test_spec_pipe(test_dir, is_regression=True)

        # Test preprocessing with error break
        os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"] = "1"
        pipe.__tested = True  # skip test_run
        with pytest.raises(ValueError) as excinfo:
            pipe.preprocessing(resume=True)
            time.sleep(0.1)
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "Preprocessing resume test raise" in str(excinfo.value.__cause__)

        # Assert step result files with break
        test_dir = pipe.report_directory
        result_dir = f"{test_dir}/Preprocessing/"
        step_dir_content = lsdir_robust(f"{result_dir}/Step_results/")
        preproc_step_names = [
            name for name in step_dir_content if "PreprocessingResult_sample_" in name and ".dill" in name
        ]
        assert len(preproc_step_names) > 0
        assert len(preproc_step_names) < len(pipe._sample_data)
        assert "Error_logs" in step_dir_content
        assert "Preprocess_progress_logs" in step_dir_content
        assert len(lsdir_robust(f"{result_dir}/Step_results/Error_logs")) == 1
        assert len(lsdir_robust(f"{result_dir}/Step_results/Preprocess_progress_logs")) == 1
        result_fn = preproc_step_names[0]
        result_ctime = os.stat(f"{result_dir}/Step_results/{result_fn}").st_ctime_ns

        # Run rest
        del os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"]
        pipe.preprocessing(resume=True)
        time.sleep(0.1)

        # Assert resume results
        TestSpecPipe.criteria_preprocessing_result(pipe)
        step_dir_content = lsdir_robust(f"{result_dir}/Step_results/")
        assert "Error_logs" in step_dir_content
        assert "Preprocess_progress_logs" in step_dir_content

        # Assert no secondary creation of results
        result_ctime1 = os.stat(f"{result_dir}/Step_results/{result_fn}").st_ctime_ns
        assert result_ctime1 == result_ctime

    @staticmethod
    @silent
    def test_resume_modeling_regression() -> None:
        """Test resume functionality of 'model_evaluation' of regression models."""

        if os.getenv("SPECPIPE_PREPROCESS_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"]
        if os.getenv("SPECPIPE_MODEL_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"]

        plt.close("all")

        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create regression exp data and pipe
        pipe = create_test_spec_pipe(test_dir, is_regression=True)

        pipe.preprocessing()
        pipe.assembly()
        time.sleep(0.1)

        # Test modeling with error break
        os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"] = "1"
        pipe.__tested = True  # skip test_run
        with pytest.raises(ValueError) as excinfo:
            pipe.model_evaluation(resume=True)
            time.sleep(0.1)
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "Modeling resume test raise" in str(excinfo.value.__cause__)

        # Result dir paths
        test_dir = pipe.report_directory
        result_dir = f"{test_dir}/Modeling/"
        model_report_dir = f"{test_dir}/Modeling/Model_evaluation_reports/"

        # Test error log
        assert os.path.exists(result_dir)
        assert "Error_logs" in lsdir_robust(result_dir)

        # Test running progress log
        assert os.path.exists(f"{model_report_dir}/modeling_progress_log.dill")

        # Test finished results
        assert os.path.exists(model_report_dir)
        # Test modeling break after one iteration, should have 2 chain txts and 1 result dir (dir item = 3)
        model_reports = lsdir_robust(model_report_dir, 2)
        preprocs_in_modeling = [n for n in model_reports if ".txt" in n]
        model_reports = [n for n in model_reports if "Data_chain_" in n and "_Model_" in n]
        preprocs = pipe.process_chains_to_df().iloc[:, :-1].drop_duplicates(ignore_index=True)
        assert len(preprocs_in_modeling) < len(preprocs)
        assert len(model_reports) < len(pipe.process_chains)

        # Creation time (For testing no secondary creation of reports)
        result_fn = model_reports[0]
        result_ctime = os.stat(f"{model_report_dir}/{result_fn}").st_ctime_ns

        # Run rest
        del os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"]
        pipe.model_evaluation(resume=True)
        time.sleep(0.1)

        # Assert result
        TestSpecPipe.criteria_regression_model_report(pipe)

        # Assert no secondary creation of results
        result_ctime1 = os.stat(f"{model_report_dir}/{result_fn}").st_ctime_ns
        assert result_ctime1 == result_ctime

    @staticmethod
    @silent
    def test_resume_modeling_classification() -> None:
        """Test resume functionality of 'model_evaluation' of classification models."""

        if os.getenv("SPECPIPE_PREPROCESS_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"]
        if os.getenv("SPECPIPE_MODEL_RESUME_TEST_NUM") is not None:
            del os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"]

        plt.close("all")

        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        # Create classification exp data and pipe
        pipe = create_test_spec_pipe(test_dir, is_regression=False)

        pipe.preprocessing()
        pipe.assembly()
        time.sleep(0.1)

        # Test modeling with error break
        os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"] = "1"
        pipe.__tested = True  # skip test_run
        with pytest.raises(ValueError) as excinfo:
            pipe.model_evaluation(resume=True)
            time.sleep(0.1)
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "Modeling resume test raise" in str(excinfo.value.__cause__)

        # Result dir paths
        test_dir = pipe.report_directory
        result_dir = f"{test_dir}/Modeling/"
        model_report_dir = f"{test_dir}/Modeling/Model_evaluation_reports/"

        # Test error log
        assert os.path.exists(result_dir)
        assert "Error_logs" in lsdir_robust(result_dir)

        # Test running progress log
        assert os.path.exists(f"{model_report_dir}/modeling_progress_log.dill")

        # Test finished results
        assert os.path.exists(model_report_dir)
        # Test modeling break after one iteration, should have 2 chain txts and 1 result dir (dir item = 3)
        model_reports = lsdir_robust(model_report_dir, 2)
        preprocs_in_modeling = [n for n in model_reports if ".txt" in n]
        model_reports = [n for n in model_reports if "Data_chain_" in n and "_Model_" in n]
        preprocs = pipe.process_chains_to_df().iloc[:, :-1].drop_duplicates(ignore_index=True)
        assert len(preprocs_in_modeling) < len(preprocs)
        assert len(model_reports) < len(pipe.process_chains)

        # Creation time (For testing no secondary creation of reports)
        result_fn = model_reports[0]
        result_ctime = os.stat(f"{model_report_dir}/{result_fn}").st_ctime_ns

        # Run rest
        del os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"]
        pipe.model_evaluation(resume=True)
        time.sleep(0.1)

        # Assert result
        TestSpecPipe.criteria_classification_model_report(pipe)

        # Assert no secondary creation of results
        result_ctime1 = os.stat(f"{model_report_dir}/{result_fn}").st_ctime_ns
        assert result_ctime1 == result_ctime

    @staticmethod
    def test_method_alias() -> None:
        """Test method alias"""
        # Initialize test_dir
        TestSpecPipe._init_test_dir()
        test_dir = TestSpecPipe.test_dir

        pipe = create_test_spec_pipe(test_dir)

        # Assert alias
        assert pipe.ls_process_chains == pipe.process_chains_to_df
        assert pipe.ls_custom_chains == pipe.custom_chains_to_df


# %% Tests - SpecPipe

# TestSpecPipe.setUpClass()

# TestSpecPipe.test_initialization_image_exp()
# TestSpecPipe.test_initialization_standalone_spec_exp()

# TestSpecPipe.test_add_process()
# TestSpecPipe.test_add_process_validation()
# TestSpecPipe.test_ls_process()
# TestSpecPipe.test_rm_process()

# TestSpecPipe.test_add_model()
# TestSpecPipe.test_ls_model()
# TestSpecPipe.test_rm_model()

# TestSpecPipe.test_build_pipeline()

# TestSpecPipe.test_process_chains_to_df()
# TestSpecPipe.test_custom_chains_from_df()

# TestSpecPipe.test_save_load_config()

# TestSpecPipe.test_test_run_regression()
# TestSpecPipe.test_test_run_classification()

# TestSpecPipe.test_preprocessing_modeling_regression()
# TestSpecPipe.test_preprocessing_modeling_classification()
# TestSpecPipe.test_run_pipe()

# TestSpecPipe.test_property_access()

# TestSpecPipe.test_resume_preprocessing()
# TestSpecPipe.test_resume_modeling_regression()
# TestSpecPipe.test_resume_modeling_classification()

# TestSpecPipe.test_method_alias()

# TestSpecPipe.tearDownClass()


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
