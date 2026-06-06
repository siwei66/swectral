# -*- coding: utf-8 -*-
"""
Tests for Swectral MinMax (MinMax Normalization) functions

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# ruff: noqa: I001
# OS
import os  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

# Initialize LOKY_MAX_CPU_COUNT if it does not exist before imports to prevent corresponding warning
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')  # noqa: E402

# OS Files
import shutil  # noqa: E402
import tempfile  # noqa: E402

# Test
import pytest  # noqa: E402
import unittest  # noqa: E402


# Basic data
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Raster
import rasterio  # noqa: E402

# Local
from swectral.example_data import create_test_raster  # noqa: E402

# Function to test
from swectral.rasterop import pixel_apply  # noqa: E402
from swectral.functions.minmax import minmax  # noqa: E402
from swectral.functions.minmax_hyper import minmax_hyper  # noqa: E402

# Check if cuda is available
try:
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


# %% Test


class TestMinMax(unittest.TestCase):
    """Test snv and snv_hyper functionalities."""

    test_dir: str = ""
    img_path: str = ""
    dst_path: str = ""
    dst_path_hyper: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a temporary directory and a test image."""
        cls.test_dir = tempfile.mkdtemp()
        cls.img_path = cls.test_dir + "/test_img.tif"
        cls.dst_path = cls.test_dir + "/processed.tif"
        cls.dst_path_hyper = cls.test_dir + "/processed_hyper.tif"
        create_test_raster(raster_path=cls.img_path, width=50, height=50, bands=4)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary directory"""
        shutil.rmtree(cls.test_dir)

    @staticmethod
    def test_minmax() -> None:
        """Test minmax basic functionality."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
            pixel_apply(
                image_path=TestMinMax.img_path,
                output_path=TestMinMax.dst_path,
                spectral_function=minmax,
                tile_size=1,
                progress=False,
                function_type='array',
            )
            assert os.path.exists(TestMinMax.dst_path)

    @staticmethod
    def test_minmax_hyper() -> None:
        """Test minmax_hyper basic functionality."""
        if HAS_CUDA:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
                pixel_apply(
                    image_path=TestMinMax.img_path,
                    output_path=TestMinMax.dst_path_hyper,
                    spectral_function=minmax_hyper,
                    tile_size=1,
                    progress=False,
                    function_type='tensor_hyper',
                )
                assert os.path.exists(TestMinMax.dst_path)
                with rasterio.open(TestMinMax.dst_path) as src1, rasterio.open(TestMinMax.dst_path_hyper) as src2:
                    data1 = src1.read()
                    data2 = src2.read()
                    np.testing.assert_allclose(
                        data1, data2, rtol=1e-5, atol=1e-5, err_msg="Inconsistent resulting rasters."
                    )
        else:
            return

    @staticmethod
    def test_minmax_tensor_dimensions() -> None:
        """Test minmax with 1D, 2D, 3D, and 4D PyTorch tensors."""
        try:
            import torch
        except ImportError:
            return

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            # 1D: (C,) - Spectral dim is 0
            t1 = torch.rand(10)
            res1 = minmax(t1)
            assert res1.shape == (10,)

            # 2D: (N, C) - Spectral dim is 1
            t2 = torch.rand(5, 10)
            res2 = minmax(t2)
            assert res2.shape == (5, 10)

            # 3D: (C, H, W) - Spectral dim is 0
            t3 = torch.rand(10, 4, 4)
            res3 = minmax(t3)
            assert res3.shape == (10, 4, 4)

            # 4D: (N, C, H, W) - Spectral dim is 1
            t4 = torch.rand(5, 10, 4, 4)
            res4 = minmax(t4)
            assert res4.shape == (5, 10, 4, 4)

            # Verify computation logic (min ~0, max ~1) on a specific dimension
            # For 3D, operations run on dim=0
            assert torch.allclose(torch.min(res3, dim=0).values, torch.zeros(4, 4), atol=1e-5)
            assert torch.allclose(torch.max(res3, dim=0).values, torch.ones(4, 4), atol=1e-5)

            # For 4D, operations run on dim=1
            assert torch.allclose(torch.min(res4, dim=1).values, torch.zeros(5, 4, 4), atol=1e-5)
            assert torch.allclose(torch.max(res4, dim=1).values, torch.ones(5, 4, 4), atol=1e-5)

    @staticmethod
    def test_minmax_tensor_edge_cases() -> None:
        """Test minmax tensor functionality with edge cases like NaNs and constant values."""
        try:
            import torch
        except ImportError:
            return

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            # NaN handling
            t_nan = torch.tensor([[1.0, float("nan"), 3.0], [2.0, 2.0, 2.0]])
            res_nan = minmax(t_nan)

            # Ensure the computation gracefully handles NaNs without turning valid pixels into NaNs
            assert not torch.isnan(res_nan[0, 0])
            assert not torch.isnan(res_nan[0, 2])
            assert torch.isnan(res_nan[0, 1])

            # Validate that the min value maps to 0 and the max maps to 1
            assert torch.allclose(res_nan[0, 0], torch.tensor(0.0), atol=1e-5)
            assert torch.allclose(res_nan[0, 2], torch.tensor(1.0), atol=1e-5)

            # Zero variance (all elements identical along the spectral dimension)
            t_zero_var = torch.ones(2, 5)
            res_zero = minmax(t_zero_var)

            # Should not contain NaNs due to the 1e-15 epsilon addition
            assert not torch.isnan(res_zero).any()
            # The resulting tensor should be approximately zero since (data - min) is 0
            assert torch.allclose(res_zero, torch.zeros_like(res_zero), atol=1e-5)

    @staticmethod
    def test_minmax_tensor_errors() -> None:
        """Test minmax tensor error raises for unsupported dimensions."""
        try:
            import torch
        except ImportError:
            return

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            # 5D tensor is unsupported and should raise a ValueError
            t5 = torch.rand(2, 3, 4, 5, 6)

            with pytest.raises(ValueError, match="got dimension"):
                minmax(t5)


# %% Test main


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
