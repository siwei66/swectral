# -*- coding: utf-8 -*-
"""
Tests for basic ROI statistics functions (ROIStats)

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os
import sys  # noqa: E402
import tempfile

# Test
import pytest
import unittest

# Typing
from typing import Any, Union

# Basic data
import numpy as np
import pandas as pd
import torch

# Math
import math

# Raster
from rasterio import RasterioIOError

# Local
from swectral.example_data import create_test_raster

# Functions to test
from swectral.roistats import (
    make_img_func,
    make_roi_func,
    make_array_func,
    roispec,
    Stats2d,
    axisconv,
    bandquant,
    cmval,
    minbbox,
    moment2d,
    nderiv,
    np_sig_digit,
    num_sig_digit,
    pixcount,
    pixcounts,
    roi_mean,
    roi_median,
    roi_std,
    round_digit,
    smopt,
    spectral_angle,
    spectral_angle_arr,
)


# %% test functions : make_img_func


class TestMakeImgOnly(unittest.TestCase):

    temp_file: tempfile._TemporaryFileWrapper
    temp_file_path: str

    @classmethod
    def setUpClass(cls) -> None:
        # Create a temporary image file once for all tests
        cls.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        cls.temp_file.write(b"Test image content")
        cls.temp_file.close()
        cls.temp_file_path = cls.temp_file.name

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up the temporary file
        if os.path.exists(cls.temp_file_path):
            os.remove(cls.temp_file_path)

    @staticmethod
    def test_basic_functionality() -> None:
        def sample_func(image_path: str, param1: int = 0, param2: int = 0) -> str:
            return f"{image_path}-{param1}-{param2}"

        img_only_func = make_img_func(sample_func, param1=5, param2=10)
        result = img_only_func(TestMakeImgOnly.temp_file_path)
        assert (
            result == f"{TestMakeImgOnly.temp_file_path}-5-10"
        ), f"Expected '{TestMakeImgOnly.temp_file_path}-5-10', got {result}"

    @staticmethod
    def test_function_name_suffix() -> None:
        def sample_func(image_path: str) -> str:
            return image_path

        img_only_func = make_img_func(sample_func, name_suffix="custom_suffix")
        assert (
            img_only_func.__name__ == "sample_func_custom_suffix"
        ), f"Expected function name 'sample_func_custom_suffix', got {img_only_func.__name__}"

    @staticmethod
    def test_fixed_args_applied() -> None:
        def sample_func(image_path: str, a: int, b: int) -> str:
            return f"{image_path}-{a}-{b}"

        img_only_func = make_img_func(sample_func, "", 1, 2)
        result = img_only_func(TestMakeImgOnly.temp_file_path)
        assert (
            result == f"{TestMakeImgOnly.temp_file_path}-1-2"
        ), f"Expected '{TestMakeImgOnly.temp_file_path}-1-2', got {result}"

    @staticmethod
    def test_insufficient_parameters_raises_typeerror() -> None:
        def sample_func() -> str:  # No parameters
            return "ok"

        with pytest.raises(TypeError, match="Function must accept at least 1 data parameter"):
            make_img_func(sample_func)

    @staticmethod
    def test_non_positional_data_parameter_raises_typeerror() -> None:
        def sample_func(*, image_path: str) -> str:  # Keyword-only first parameter
            return image_path

        with pytest.raises(TypeError, match="Data parameter 1 must be positional"):
            make_img_func(sample_func)


# %% test functions : make_roi_func


class TestMakeRoiOnly(unittest.TestCase):

    temp_file: tempfile._TemporaryFileWrapper
    temp_file_path: str
    example_roi: list[list]

    @classmethod
    def setUpClass(cls) -> None:
        # Create a temporary image file once for all tests
        cls.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        cls.temp_file.write(b"Test image content")
        cls.temp_file.close()
        cls.temp_file_path = cls.temp_file.name

        # Example ROI coordinates for testing
        cls.example_roi = [[(0, 0), (0, 10), (10, 0), (0, 0)]]

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up the temporary file
        if os.path.exists(cls.temp_file_path):
            os.remove(cls.temp_file_path)

    @staticmethod
    def test_basic_functionality() -> None:
        def sample_func(image_path: str, roi_coordinates: list[list], param1: int = 0, param2: int = 0) -> str:
            return f"{image_path}-{roi_coordinates}-{param1}-{param2}"

        roi_only_func = make_roi_func(sample_func, param1=5, param2=10)
        result = roi_only_func(TestMakeRoiOnly.temp_file_path, TestMakeRoiOnly.example_roi)
        expected = f"{TestMakeRoiOnly.temp_file_path}-{TestMakeRoiOnly.example_roi}-5-10"
        assert result == expected, f"Expected '{expected}', got {result}"

    @staticmethod
    def test_function_name_suffix() -> None:
        def sample_func(image_path: str, roi_coordinates: list[list]) -> str:
            return image_path

        roi_only_func = make_roi_func(sample_func, name_suffix="custom_suffix")
        assert (
            roi_only_func.__name__ == "sample_func_custom_suffix"
        ), f"Expected function name 'sample_func_custom_suffix', got {roi_only_func.__name__}"

    @staticmethod
    def test_fixed_args_applied() -> None:
        def sample_func(image_path: str, roi_coordinates: list[list], a: int, b: int) -> str:
            return f"{image_path}-{roi_coordinates}-{a}-{b}"

        roi_only_func = make_roi_func(sample_func, "", 1, 2)
        result = roi_only_func(TestMakeRoiOnly.temp_file_path, TestMakeRoiOnly.example_roi)
        expected = f"{TestMakeRoiOnly.temp_file_path}-{TestMakeRoiOnly.example_roi}-1-2"
        assert result == expected, f"Expected '{expected}', got {result}"

    @staticmethod
    def test_insufficient_parameters_raises_typeerror() -> None:
        def sample_func(image_path: str) -> str:  # Only 1 parameter
            return image_path

        with pytest.raises(TypeError, match="Function must accept at least 2 data parameters"):
            make_roi_func(sample_func)

    @staticmethod
    def test_non_positional_data_parameter_raises_typeerror() -> None:
        def sample_func(*, image_path: str, roi_coordinates: list[list]) -> str:  # Keyword-only parameters
            return image_path

        with pytest.raises(TypeError, match="Data parameter 1 must be positional"):
            make_roi_func(sample_func)


# %% test functions : make_array_func


class TestMakeArrayOnly(unittest.TestCase):

    test_array: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        # Example 2D array for testing
        cls.test_array = np.array([[1, 2, 3], [4, 5, 6]])

    @staticmethod
    def test_basic_functionality() -> None:
        def sample_func(data_array: np.ndarray, param1: int = 0, param2: int = 0) -> int:
            result = data_array.sum() + param1 + param2
            return int(result)

        arr_only_func = make_array_func(sample_func, param1=5, param2=10)
        result = arr_only_func(TestMakeArrayOnly.test_array)
        expected = TestMakeArrayOnly.test_array.sum() + 5 + 10
        assert result == expected, f"Expected {expected}, got {result}"

    @staticmethod
    def test_function_name_suffix() -> None:
        def sample_func(data_array: np.ndarray) -> np.ndarray:
            return data_array

        arr_only_func = make_array_func(sample_func, name_suffix="custom_suffix")
        assert (
            arr_only_func.__name__ == "sample_func_custom_suffix"
        ), f"Expected function name 'sample_func_custom_suffix', got {arr_only_func.__name__}"

    @staticmethod
    def test_fixed_args_applied() -> None:
        def sample_func(data_array: np.ndarray, a: int, b: int) -> int:
            result = data_array.sum() + a + b
            return int(result)

        arr_only_func = make_array_func(sample_func, "", 1, 2)
        result = arr_only_func(TestMakeArrayOnly.test_array)
        expected = TestMakeArrayOnly.test_array.sum() + 1 + 2
        assert result == expected, f"Expected {expected}, got {result}"

    @staticmethod
    def test_insufficient_parameters_raises_typeerror() -> None:
        def sample_func() -> int:  # No parameters
            return 0

        with pytest.raises(TypeError, match="Function must accept at least 1 data parameter"):
            make_array_func(sample_func)

    @staticmethod
    def test_non_positional_data_parameter_raises_typeerror() -> None:
        def sample_func(*, data_array: np.ndarray) -> np.ndarray:  # Keyword-only parameter
            return data_array

        with pytest.raises(TypeError, match="Data parameter 1 must be positional"):
            make_array_func(sample_func)


# %% test functions : roispec


class TestROISpec(unittest.TestCase):
    """Test cases for roispec class"""

    @staticmethod
    def create_simple_polygon() -> list[list[tuple[float, float]]]:
        """
        Create a simple polygon within raster bounds.

        Returns
        -------
        list[list[tuple[float, float]]]
            Polygon coordinates
        """
        return [[(2.0, 2.0), (2.0, 8.0), (8.0, 8.0), (8.0, 2.0), (2.0, 2.0)]]

    @staticmethod
    def create_multipolygon() -> list[list[tuple[float, float]]]:
        """
        Create multiple polygons for testing.

        Returns
        -------
        list[list[tuple[float, float]]]
            Multipolygon coordinates
        """
        return [
            [(1.0, 1.0), (1.0, 4.0), (4.0, 4.0), (4.0, 1.0), (1.0, 1.0)],
            [(6.0, 6.0), (6.0, 9.0), (9.0, 9.0), (9.0, 6.0), (6.0, 6.0)],
        ]

    @staticmethod
    def create_out_of_bounds_polygon() -> list[list[tuple[float, float]]]:
        """
        Create a polygon outside raster bounds.

        Returns
        -------
        list[list[tuple[float, float]]]
            Out-of-bounds polygon coordinates
        """
        return [[(15.0, 15.0), (15.0, 20.0), (20.0, 20.0), (20.0, 15.0), (15.0, 15.0)]]

    @staticmethod
    def test_roispec_single_polygon() -> None:
        """Test ROI extraction with single polygon"""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            test_raster = tmp.name

            # Create test raster
            _ = create_test_raster(test_raster)

            # Extract spectra
            polygon = TestROISpec.create_simple_polygon()
            result = roispec(test_raster, polygon)

            # Verify results
            assert isinstance(result, np.ndarray)
            assert result.ndim == 2
            assert result.dtype == np.float32
            assert result.shape[0] > 0  # Should have some pixels
            assert result.shape[1] == 4  # 5 spectral bands

        if os.path.exists(test_raster):
            os.remove(test_raster)

    @staticmethod
    def test_roispec_multipolygon() -> None:
        """Test ROI extraction with multiple polygons"""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            test_raster = tmp.name

            # Create test raster
            _ = create_test_raster(test_raster)

            # Extract spectra from multiple polygons
            polygons = TestROISpec.create_multipolygon()
            result = roispec(test_raster, polygons, as_type="float64")

            # Verify results
            assert isinstance(result, np.ndarray)
            assert result.ndim == 2
            assert result.dtype == np.float64
            assert result.shape[0] > 0
            assert result.shape[1] == 4

        if os.path.exists(test_raster):
            os.remove(test_raster)

    @staticmethod
    def test_roispec_different_dtype() -> None:
        """Test ROI extraction with different output data types"""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            test_raster = tmp.name

            # Create test raster
            _ = create_test_raster(test_raster)

            # Test float32
            polygon = TestROISpec.create_simple_polygon()
            result_float32 = roispec(test_raster, polygon, as_type="float32")
            assert result_float32.dtype == np.float32

            # Test float64
            result_float64 = roispec(test_raster, polygon, as_type="float64")
            assert result_float64.dtype == np.float64

            # Test int16
            result_int16 = roispec(test_raster, polygon, as_type="int16")
            assert result_int16.dtype == np.int16

        if os.path.exists(test_raster):
            os.remove(test_raster)

    @staticmethod
    def test_roispec_empty_polygon() -> None:
        """Test ROI extraction with empty polygon list"""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            test_raster = tmp.name

            # Create test raster
            _ = create_test_raster(test_raster)

            # Attempt extraction with empty polygon list
            with pytest.raises(ValueError, match="No valid polygon found in the given roi_coordinates"):
                roispec(test_raster, [])

        if os.path.exists(test_raster):
            os.remove(test_raster)

    @staticmethod
    def test_roispec_nonexistent_file() -> None:
        """Test ROI extraction with non-existent file raises error"""
        polygon = TestROISpec.create_simple_polygon()

        with pytest.raises(RasterioIOError):
            roispec("nonexistent_file.tif", polygon)

    @staticmethod
    def test_output_shape_consistency() -> None:
        """
        Test that output shape is consistent across multiple runs.
        """
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            test_raster = tmp.name

            # Create test raster
            _ = create_test_raster(test_raster)

            # Extract spectra multiple times
            polygon = TestROISpec.create_simple_polygon()
            result1 = roispec(test_raster, polygon)
            result2 = roispec(test_raster, polygon)

            # Check consistency
            assert result1.shape == result2.shape
            assert result1.dtype == result2.dtype

        if os.path.exists(test_raster):
            os.remove(test_raster)

    @staticmethod
    def test_no_nodata_values() -> None:
        """
        Test that output contains no nodata values.
        """
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            test_raster = tmp.name

            # Create test raster with nodata
            _ = create_test_raster(test_raster, incl_nodata=-9999.0, dtype="float32")

            polygon = TestROISpec.create_simple_polygon()
            result = roispec(test_raster, polygon)

            # Check for nodata values
            assert not np.any(result == -9999.0)

        if os.path.exists(test_raster):
            os.remove(test_raster)


# %% Test - roispec

# TestROISpec.test_roispec_single_polygon()
# TestROISpec.test_roispec_multipolygon()

# TestROISpec.test_roispec_different_dtype()
# TestROISpec.test_no_nodata_values()

# TestROISpec.test_output_shape_consistency()
# TestROISpec.test_roispec_empty_polygon()

# TestROISpec.test_roispec_nonexistent_file()


# %% test functions : pixcount


class TestPixCount(unittest.TestCase):

    tmp_path: str

    @classmethod
    def setUpClass(cls) -> None:
        """Create a single temporary raster file for all tests."""
        tmp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp_file.close()
        cls.tmp_path = tmp_file.name

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up the temporary raster file."""
        if os.path.exists(cls.tmp_path):
            os.remove(cls.tmp_path)

    @staticmethod
    def test_basic_pixel_count() -> None:
        """Test basic pixel counting without threshold."""
        tmp_path = TestPixCount.tmp_path

        _ = create_test_raster(tmp_path, width=4, height=4, bands=1, dtype="float32")

        roi_coordinates = [[(0.5, 0.5), (0.5, 3.5), (3.5, 3.5), (3.5, 0.5)]]

        result = pixcount(tmp_path, roi_coordinates, band=1, threshold=None)

        assert result == 16

    @staticmethod
    def test_pixel_count_with_threshold() -> None:
        """Test pixel counting with threshold filtering."""
        tmp_path = TestPixCount.tmp_path

        data = np.arange(1, 17, dtype=np.float32).reshape((1, 4, 4))

        _ = create_test_raster(
            tmp_path,
            width=4,
            height=4,
            bands=1,
            dtype="float32",
            data=data,
        )

        roi_coordinates = [[(0.5, 0.5), (0.5, 3.5), (3.5, 3.5), (3.5, 0.5)]]

        result = pixcount(tmp_path, roi_coordinates, band=1, threshold=(5, 10))

        assert result == 6

    @staticmethod
    def test_nodata_handling() -> None:
        """Test that nodata values are properly excluded."""
        tmp_path = TestPixCount.tmp_path

        data = np.array(
            [[[1, -9999, 3, 4], [5, 6, -9999, 8], [9, 10, 11, 12], [-9999, 14, 15, 16]]],
            dtype=np.float32,
        )

        _ = create_test_raster(
            tmp_path,
            width=4,
            height=4,
            bands=1,
            nodata_value=-9999,
            dtype="float32",
            data=data,
        )

        roi_coordinates = [[(0.5, 0.5), (0.5, 3.5), (3.5, 3.5), (3.5, 0.5)]]

        result = pixcount(tmp_path, roi_coordinates, band=1)

        assert result == 13

    @staticmethod
    def test_multiple_polygons() -> None:
        """Test counting across multiple polygons."""
        tmp_path = TestPixCount.tmp_path

        _ = create_test_raster(tmp_path, width=4, height=4, bands=1, dtype="float32")

        roi_coordinates = [
            [(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)],
            [(2.5, 2.5), (2.5, 3.5), (3.5, 3.5), (3.5, 2.5)],
        ]

        result = pixcount(tmp_path, roi_coordinates, band=1, threshold=None)

        assert result == 8

    @staticmethod
    def test_out_of_bounds_polygon() -> None:
        """Test polygon that doesn't intersect with the raster."""
        tmp_path = TestPixCount.tmp_path

        _ = create_test_raster(tmp_path, width=2, height=2, bands=1, dtype="float32")

        roi_coordinates = [[(10.0, 10.0), (10.0, 12.0), (12.0, 12.0), (12.0, 10.0)]]

        result = pixcount(tmp_path, roi_coordinates, band=1, threshold=None)

        assert result == 0

    @staticmethod
    def test_invalid_threshold() -> None:
        """Test that invalid threshold raises ValueError."""
        tmp_path = TestPixCount.tmp_path

        _ = create_test_raster(tmp_path, width=2, height=2, bands=1, dtype="float32")

        roi_coordinates = [[(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]]

        with pytest.raises(ValueError, match="Invalid threshold"):
            pixcount(tmp_path, roi_coordinates, band=1, threshold=(10, 5))

    @staticmethod
    def test_different_band() -> None:
        """Test counting from a specific band."""
        tmp_path = TestPixCount.tmp_path

        data = np.array(
            [[[1, 2], [3, 4]], [[10, 20], [30, 40]]],
            dtype=np.float32,
        )

        _ = create_test_raster(
            tmp_path,
            width=2,
            height=2,
            bands=2,
            dtype="float32",
            data=data,
        )

        roi_coordinates = [[(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]]

        result = pixcount(tmp_path, roi_coordinates, band=2, threshold=(15, 35))

        assert result == 2

    @staticmethod
    def test_empty_polygon_list() -> None:
        """Test with empty polygon list."""
        tmp_path = TestPixCount.tmp_path

        _ = create_test_raster(tmp_path, width=2, height=2, bands=1, dtype="float32")

        roi_coordinates: list[list[tuple[float, float]]] = []

        result = pixcount(tmp_path, roi_coordinates, band=1, threshold=None)

        assert result == 0

    @staticmethod
    def test_partial_intersection() -> None:
        """Test polygon that partially intersects with the raster."""
        tmp_path = TestPixCount.tmp_path

        _ = create_test_raster(tmp_path, width=4, height=4, bands=1, dtype="float32")

        roi_coordinates = [[(2.0, 2.0), (2.0, 5.0), (5.0, 5.0), (5.0, 2.0)]]

        result = pixcount(tmp_path, roi_coordinates, band=1, threshold=None)

        assert result == 4

    @staticmethod
    def test_pixcounts_basic() -> None:
        """Test pixcounts basic functionalities."""
        tmp_path = TestPixCount.tmp_path

        _ = create_test_raster(tmp_path, width=4, height=4, bands=4, dtype="float32")

        roi_coordinates = [[(0.5, 0.5), (0.5, 3.5), (3.5, 3.5), (3.5, 0.5)]]

        # Multiple bands
        df_counts = pixcounts(tmp_path, roi_coordinates, band=[1, 2, 3, 4], threshold=None)

        assert isinstance(df_counts, pd.DataFrame)
        assert df_counts.shape == (1, 4)
        assert list(df_counts.columns) == [f"Band_{i + 1}" for i in range(4)]
        assert np.sum(np.array(df_counts)) > 0

        # Single band
        df_counts = pixcounts(tmp_path, roi_coordinates, band=1, threshold=None)

        assert isinstance(df_counts, pd.DataFrame)
        assert df_counts.shape == (1, 1)
        assert list(df_counts.columns) == ["Band_1"]
        assert np.sum(np.array(df_counts)) > 0

    @staticmethod
    def test_pixcounts_thresholds() -> None:
        """Test pixcounts functionality with thresholds."""
        tmp_path = TestPixCount.tmp_path

        _ = create_test_raster(tmp_path, width=4, height=4, bands=4, dtype="float32")

        roi_coordinates = [[(0.5, 0.5), (0.5, 3.5), (3.5, 3.5), (3.5, 0.5)]]

        # Multiple bands
        df_counts = pixcounts(
            tmp_path,
            roi_coordinates,
            band=[1, 2, 3, 4],
            threshold=[(10000, 10001), (-1, 10001), (10000, 10001), (-1, 10001)],
        )

        assert isinstance(df_counts, pd.DataFrame)
        assert df_counts.shape == (1, 4)
        assert list(df_counts.columns) == [f"Band_{i + 1}" for i in range(4)]
        assert df_counts.iloc[0, 0] == 0
        assert df_counts.iloc[0, 1] > 0
        assert df_counts.iloc[0, 2] == 0
        assert df_counts.iloc[0, 3] > 0

        # Single bands
        df_counts = pixcounts(
            tmp_path,
            roi_coordinates,
            band=2,
            threshold=(-1, 10001),
        )

        assert isinstance(df_counts, pd.DataFrame)
        assert df_counts.shape == (1, 1)
        assert list(df_counts.columns) == ["Band_2"]
        assert df_counts.iloc[0, 0] > 0


# %% Test - pixcount

# test_pixcount = TestPixCount()
# test_pixcount.setUpClass()

# test_pixcount.test_basic_pixel_count()
# test_pixcount.test_pixel_count_with_threshold()
# test_pixcount.test_nodata_handling()
# test_pixcount.test_multiple_polygons()
# test_pixcount.test_out_of_bounds_polygon()
# test_pixcount.test_invalid_threshold()
# test_pixcount.test_different_band()
# test_pixcount.test_empty_polygon_list()
# test_pixcount.test_partial_intersection()
# test_pixcount.test_pixcounts_basic()
# test_pixcount.test_pixcounts_thresholds()

# test_pixcount.tearDownClass()


# %% test functions : minbbox


class TestMinBbox:
    """Test class for minbbox function"""

    @staticmethod
    def test_minbbox_sufficient_pixels() -> None:
        """Test minbbox when ROI already has sufficient valid pixels"""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            tmp_path = tmp_file.name

            # Create a raster with all valid pixels
            data = np.ones((1, 100, 100), dtype=np.float32)

            # Create test raster
            _ = create_test_raster(
                tmp_path,
                width=data.shape[2],
                height=data.shape[1],
                bands=data.shape[0],
                data=data,
                dtype="float32",
            )

            roi_coordinates = [[(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0), (10.0, 10.0)]]
            minimum_valid_pixel = 100

            result = minbbox(tmp_path, roi_coordinates, minimum_valid_pixel)

            # Should return the original bounds since it already has sufficient pixels
            assert len(result) == 1
            assert len(result[0]) == 5  # Closed polygon
            assert result[0][0] == (10.0, 10.0)  # Should be the original ROI

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    @staticmethod
    def test_minbbox_insufficient_pixels() -> None:
        """Test minbbox when ROI doesn't have enough valid pixels"""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            tmp_path = tmp_file.name

            # Create a raster
            data = np.zeros((1, 100, 100), dtype=np.float32)
            data[40:60, 40:60] = 1.0  # Only center 20x20 region is valid

            # Create test raster
            _ = create_test_raster(
                tmp_path,
                width=data.shape[2],
                height=data.shape[1],
                bands=data.shape[0],
                data=data,
                dtype="float32",
            )

            roi_coordinates = [[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)]]
            minimum_valid_pixel = 2000

            result = minbbox(tmp_path, roi_coordinates, minimum_valid_pixel, valid_threshold=(1, 10000))

            # Should return a smaller bounding box
            assert len(result) == 1
            assert len(result[0]) == 5
            assert result == roi_coordinates

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    @staticmethod
    def test_minbbox_with_valid_threshold() -> None:
        """Test minbbox with valid threshold parameter"""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            tmp_path = tmp_file.name

            # Create a raster
            data = np.full((1, 100, 100), 50.0, dtype=np.float32)
            data[20:40, 20:40] = 150.0  # Region with values above threshold

            # Create test raster
            _ = create_test_raster(
                tmp_path,
                width=data.shape[2],
                height=data.shape[1],
                bands=data.shape[0],
                data=data,
                dtype="float32",
            )

            roi_coordinates = [[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)]]
            minimum_valid_pixel = 50
            valid_threshold = (100.0, 200.0)  # Only values between 100-200 are valid

            result = minbbox(tmp_path, roi_coordinates, minimum_valid_pixel, valid_threshold=valid_threshold)

            # Should find the region with values in threshold range
            assert len(result) == 1
            assert len(result[0]) == 5

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    @staticmethod
    def test_minbbox_different_band() -> None:
        """Test minbbox with different band parameter"""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            tmp_path = tmp_file.name

            # Create a raster
            data = np.full((3, 100, 100), 0.0, dtype=np.float32)
            data[0, :, :] = 1.0  # All valid
            data[2, :, :] = 1.0  # All valid
            data[1, 20:40, 20:40] = 1.0  # Center region valid

            # Create test raster
            _ = create_test_raster(
                tmp_path,
                width=data.shape[2],
                height=data.shape[1],
                bands=data.shape[0],
                data=data,
                dtype="float32",
            )

            roi_coordinates = [[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)]]
            minimum_valid_pixel = 100

            # Test band 1 (all valid)
            result_band1 = minbbox(tmp_path, roi_coordinates, minimum_valid_pixel, valid_threshold=(1, 100), band=1)

            # Test band 2 (only center valid)
            result_band2 = minbbox(tmp_path, roi_coordinates, minimum_valid_pixel, valid_threshold=(1, 100), band=2)

            # Band 1 should return original ROI, Band 2 should return larger bbox
            assert result_band1[0][2][0] <= result_band2[0][2][0]  # Original ROI
            assert result_band1[0][2][1] <= result_band2[0][2][1]  # Larger bbox

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    @staticmethod
    def test_minbbox_empty_roi() -> None:
        """Test minbbox with empty ROI coordinates"""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            tmp_path = tmp_file.name

            # Create a raster
            data = np.ones((1, 10, 10), dtype=np.float32)

            # Create test raster
            _ = create_test_raster(
                tmp_path,
                width=data.shape[2],
                height=data.shape[1],
                bands=data.shape[0],
                data=data,
                dtype="float32",
            )

            roi_coordinates: list[list] = [[]]
            minimum_valid_pixel = 10

            with pytest.raises((ValueError, IndexError)):
                minbbox(tmp_path, roi_coordinates, minimum_valid_pixel)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    @staticmethod
    def test_minbbox_no_valid_pixels() -> None:
        """Test minbbox when no pixels meet the criteria"""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            tmp_path = tmp_file.name

            # Create a raster
            data = np.zeros((1, 100, 100), dtype=np.float32)  # All zeros

            # Create test raster
            _ = create_test_raster(
                tmp_path,
                width=data.shape[2],
                height=data.shape[1],
                bands=data.shape[0],
                data=data,
                dtype="float32",
            )

            roi_coordinates = [[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)]]
            minimum_valid_pixel = 1
            valid_threshold = (1.0, 2.0)  # No pixels in this range

            result = minbbox(tmp_path, roi_coordinates, minimum_valid_pixel, valid_threshold=valid_threshold)

            # Should return the original bounds since no valid pixels found
            assert result == roi_coordinates

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    @staticmethod
    def test_minbbox_return_type() -> None:
        """Test that minbbox returns correct type annotations"""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            tmp_path = tmp_file.name

            data = np.ones((1, 50, 50), dtype=np.float32)

            # Create test raster
            _ = create_test_raster(
                tmp_path,
                width=data.shape[2],
                height=data.shape[1],
                bands=data.shape[0],
                data=data,
                dtype="float32",
            )

            roi_coordinates = [[(10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0), (10.0, 10.0)]]
            minimum_valid_pixel = 10

            result = minbbox(tmp_path, roi_coordinates, minimum_valid_pixel)

            # Check return type structure
            assert isinstance(result, list)
            assert isinstance(result[0], list)
            assert isinstance(result[0][0], tuple)
            assert isinstance(result[0][0][0], float)
            assert isinstance(result[0][0][1], float)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# %% Test - minbbox

# TestMinBbox.test_minbbox_sufficient_pixels()
# TestMinBbox.test_minbbox_insufficient_pixels()
# TestMinBbox.test_minbbox_with_valid_threshold()
# TestMinBbox.test_minbbox_different_band()
# TestMinBbox.test_minbbox_empty_roi()
# TestMinBbox.test_minbbox_no_valid_pixels()
# TestMinBbox.test_minbbox_return_type()


# %% test functions : nderiv


class TestNDeriv:
    """Test suite for the nderiv function."""

    @staticmethod
    def test_first_derivative_column_wise() -> None:
        """Test first derivative calculation along axis=1 (column-wise)."""
        # Create test data: y = x^2
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        data = np.array([x**2])  # Shape: (1, 5)

        # Expected first derivative: dy/dx = 2x
        expected = np.array([[2.0, 4.0, 6.0, 8.0, 10.0]])

        result = nderiv(data, n=1, axis=1)

        # Check central values (edges will be NaN)
        assert np.allclose(result[:, 1:-1], expected[:, 1:-1], equal_nan=True)
        assert np.isnan(result[:, 0])
        assert np.isnan(result[:, -1])

    @staticmethod
    def test_first_derivative_row_wise() -> None:
        """Test first derivative calculation along axis=0 (row-wise)."""
        # Create test data: y = x^2
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        data = np.array([x**2]).T  # Shape: (5, 1)

        # Expected first derivative: dy/dx = 2x
        expected = np.array([[2.0], [4.0], [6.0], [8.0], [10.0]])

        result = nderiv(data, n=1, axis=0)

        # Check central values (edges will be NaN)
        assert np.allclose(result[1:-1, :], expected[1:-1, :], equal_nan=True)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[-1, 0])

    @staticmethod
    def test_second_derivative() -> None:
        """Test second derivative calculation."""
        # Create test data: y = x^3
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        data = np.array([x**3])  # Shape: (1, 5)

        # Expected second derivative: d²y/dx² = 6x
        expected = np.array([[np.nan, np.nan, 18.0, np.nan, np.nan]])

        result = nderiv(data, n=2, axis=1)

        # Check central values (more edges will be NaN for 2nd derivative)
        assert np.allclose(result[:, 2:-2], expected[:, 2:-2], equal_nan=True)
        assert np.isnan(result[:, 0:2]).all()
        assert np.isnan(result[:, -2:]).all()

    @staticmethod
    def test_padding() -> None:
        """Test derivative with different padding methods."""
        data = np.array([[1, 4, 9, 16, 25]], dtype=float)  # y = x^2

        # Padding with custom value
        result = nderiv(data, n=1, axis=1, padding=999.0)
        # Check that edges have the custom value
        assert result.shape == data.shape
        assert result[0, 0] == 999.0
        assert result[0, -1] == 999.0
        # Check central values are calculated correctly
        assert np.allclose(result[:, 1:-1], [[4.0, 6.0, 8.0]])

        # Padding method names
        result = nderiv(data, n=1, axis=1, padding='nan')
        assert result.shape == data.shape
        # Check central values are calculated correctly
        assert np.allclose(result[:, 1:-1], [[4.0, 6.0, 8.0]])

        result = nderiv(data, n=1, axis=1, padding='edge')
        assert result.shape == data.shape
        # Check central values are calculated correctly
        assert np.allclose(result[:, 1:-1], [[4.0, 6.0, 8.0]])

        # No padding
        result = nderiv(data, n=1, axis=1, padding=None)
        assert result.shape == (data.shape[0], data.shape[1] - 2)
        # Check central values are calculated correctly
        assert np.allclose(result, [[4.0, 6.0, 8.0]])

        data = np.array([[1, 4, 9, 16, 25, 36, 49]], dtype=float)  # y = x^2
        result = nderiv(data, n=2, axis=1, padding=None)
        assert result.shape == (data.shape[0], data.shape[1] - 4)
        assert np.allclose(result, [[2.0, 2.0, 2.0]])

    @staticmethod
    def test_zero_derivative() -> None:
        """Test that 0th derivative returns original data."""
        data = np.array([[1, 2, 3, 4, 5]], dtype=float)

        result = nderiv(data, n=0, axis=1)

        assert np.array_equal(result, data)

    @staticmethod
    def test_multiple_rows() -> None:
        """Test derivative calculation with multiple data rows."""
        data = np.array([[1, 2, 3, 4, 5], [1, 4, 9, 16, 25]], dtype=float)  # y = x  # y = x^2

        result = nderiv(data, n=1, axis=1)

        # First row derivative should be ~1 (constant)
        assert np.allclose(result[0, 1:-1], [1.0, 1.0, 1.0], equal_nan=True)
        # Second row derivative should be ~2x
        assert np.allclose(result[1, 1:-1], [4.0, 6.0, 8.0], equal_nan=True)

    @staticmethod
    def test_insufficient_data_length() -> None:
        """Test error handling for insufficient data length."""
        data = np.array([[1, 2, 3]], dtype=float)  # Only 3 elements

        with pytest.raises(ValueError, match="Insufficient data length"):
            nderiv(data, n=2, axis=1)

    @staticmethod
    def test_preserves_input_data() -> None:
        """Test that input data is not modified."""
        original_data = np.array([[1, 2, 3, 4, 5]], dtype=float)
        data_copy = original_data.copy()

        result = nderiv(original_data, n=1, axis=1)

        # Input data should remain unchanged
        assert np.array_equal(original_data, data_copy)
        # Result should be different from input
        assert not np.array_equal(result, original_data)

    @staticmethod
    def test_output_shape() -> None:
        """Test that output shape matches input shape."""
        data = np.random.rand(3, 10)  # 3 rows, 10 columns

        result = nderiv(data, n=1, axis=1)

        assert result.shape == data.shape

    @staticmethod
    def test_with_different_data_types() -> None:
        """Test that function handles different numeric input types."""
        # Test with integer input
        int_data = np.array([[1, 2, 3, 4, 5]], dtype=int)
        result_int = nderiv(int_data, n=1, axis=1)
        assert result_int.dtype == np.float32

        # Test with float input
        float_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=float)
        result_float = nderiv(float_data, n=1, axis=1)
        assert result_float.dtype == np.float32

    @staticmethod
    def test_edge_case_single_row() -> None:
        """Test with single row of data."""
        data = np.array([[1, 2, 3, 4, 5]], dtype=float)

        result = nderiv(data, n=1, axis=1)

        assert result.shape == (1, 5)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, -1])

    @staticmethod
    def test_edge_case_single_column() -> None:
        """Test with single column of data."""
        data = np.array([[1], [2], [3], [4], [5]], dtype=float)

        result = nderiv(data, n=1, axis=0)

        assert result.shape == (5, 1)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[-1, 0])

    @staticmethod
    def test_accuracy_linear_function() -> None:
        """Test derivative accuracy for linear function (should be constant)."""
        # For y = 2x + 3, derivative should be exactly 2
        x = np.linspace(0, 9, 10)
        y = 2 * x + 3
        data = y.reshape(1, -1)

        result = nderiv(data, n=1, axis=1)

        # Check central values (ignore edges)
        central_values = result[0, 1:-1]
        assert np.allclose(central_values, 2.0)

    @staticmethod
    def test_accuracy_quadratic_function() -> None:
        """Test second derivative accuracy for quadratic function."""
        # For y = 3x² + 2x + 1, second derivative should be exactly 6
        x = np.linspace(0, 99, 100)
        y = 3 * x**2 + 2 * x + 1
        data = y.reshape(1, -1)

        result = nderiv(data, n=2, axis=1)

        # Check central values (ignore more edges for 2nd derivative)
        central_values = result[0, 2:-2]
        assert np.allclose(central_values, 6.0)


# %% Test - nderiv

# TestNDeriv.test_first_derivative_column_wise()
# TestNDeriv.test_first_derivative_row_wise()
# TestNDeriv.test_second_derivative()
# TestNDeriv.test_custom_edge_value()
# TestNDeriv.test_zero_derivative()
# TestNDeriv.test_multiple_rows()
# TestNDeriv.test_insufficient_data_length()
# TestNDeriv.test_preserves_input_data()
# TestNDeriv.test_output_shape()
# TestNDeriv.test_with_different_data_types()
# TestNDeriv.test_edge_case_single_row()
# TestNDeriv.test_edge_case_single_column()
# TestNDeriv.test_accuracy_linear_function()
# TestNDeriv.test_accuracy_quadratic_function()


# %% test functions : axisconv


def test_axisconv() -> None:
    # Test input array
    input_arr = [[1, 2], [3, 4]]

    # Test axis=0
    result_axis0 = axisconv(input_arr, axis=0, astype=float)
    expected_axis0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(result_axis0, expected_axis0)

    # Test axis=1 (should transpose)
    result_axis1 = axisconv(input_arr, axis=1, astype=float)
    expected_axis1 = np.array([[1.0, 3.0], [2.0, 4.0]])
    assert np.array_equal(result_axis1, expected_axis1)

    # Test invalid axis using pytest.raises
    with pytest.raises(ValueError, match=r"Axis must be 0 or 1, got: 2"):
        axisconv(input_arr, axis=2, astype=float)

    # Test type conversion
    result_type = axisconv(input_arr, axis=0, astype=int)
    assert result_type.dtype == int


# %% Test - axisconv

# test_axisconv()


# %% test functions : moment2d


class TestMoment2D:
    """Test suite for moment2d function"""

    @staticmethod
    def test_first_moment_without_reference() -> None:
        """Test first moment calculation without reference point"""
        # Arrange
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        expected = (4.0, 5.0, 6.0)  # Mean of each column

        # Act
        result = moment2d(data, n=1)

        # Assert
        assert np.allclose(expected, result)

    @staticmethod
    def test_second_moment_without_reference() -> None:
        """Test second moment calculation without reference point"""
        # Arrange
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        expected = (6.0, 6.0, 6.0)  # Population variance

        # Act
        result = moment2d(data, n=2)

        # Assert
        assert np.allclose(expected, result)

    @staticmethod
    def test_second_moment_with_custom_reference() -> None:
        """Test second moment calculation with custom reference point"""
        # Arrange
        data = np.array([[1.0, 1.0], [4.0, 4.0], [7.0, 16.0]])
        reference = (2.0, 3.0)
        expected = (10.0, 58.0)  # MSE from reference

        # Act
        result = moment2d(data, n=2, reference=reference)

        # Assert
        assert np.allclose(expected, result)

    @staticmethod
    def test_standardized_second_moment() -> None:
        """Test standardized second moment (variance divided by variance)"""
        # Arrange
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected = (1.0, 1.0)  # Should be 1 for standardized variance

        # Act
        result = moment2d(data, n=2, standardized=True)

        # Assert
        assert np.allclose(result, expected, rtol=1e-10)

    @staticmethod
    def test_axis_parameter() -> None:
        """Test different axis parameter values"""
        # Arrange
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Act & Assert - axis=0 (default)
        result_axis0 = moment2d(data, n=1, axis=0)
        expected_axis0 = (2.5, 3.5, 4.5)
        assert np.allclose(result_axis0, expected_axis0, rtol=1e-10)

        # Act & Assert - axis=1
        result_axis1 = moment2d(data, n=1, axis=1)
        expected_axis1 = (2.0, 5.0)
        assert np.allclose(result_axis1, expected_axis1, rtol=1e-10)

    @staticmethod
    def test_zero_division_policies() -> None:
        """Test different zero division handling policies"""
        # Arrange - data with zero variance
        data = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

        # Test 'add' policy
        result_add = moment2d(data, n=2, standardized=True, zero_division="add")
        assert all(not np.isnan(x) and not np.isinf(x) for x in result_add)

        # Test 'replace' policy
        result_replace = moment2d(data, n=2, standardized=True, zero_division="replace")
        assert all(not np.isnan(x) and not np.isinf(x) for x in result_replace)

        # Test 'nan' policy
        result_nan = moment2d(data, n=2, standardized=True, zero_division="nan")
        assert all(np.isnan(x) for x in result_nan)

        # Test 'numpy' policy
        with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
            result_numpy = moment2d(data, n=2, standardized=True, zero_division="numpy")
        assert all(np.isnan(x) for x in result_numpy)

    @staticmethod
    def test_nan_handling() -> None:
        """Test that NaN values are properly handled"""
        # Arrange
        data = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        expected_mean = (4.0, 5.0, 7.5)  # Mean ignoring NaN

        # Act
        result = moment2d(data, n=1)

        # Assert
        assert np.allclose(result, expected_mean, rtol=1e-10)

    @staticmethod
    def test_insufficient_sample_size() -> None:
        """Test error handling for insufficient sample size"""
        # Arrange
        data = np.array([[1.0, 2.0], [3.0, 4.0]])  # Only 2 samples

        # Act & Assert
        with pytest.raises(ValueError, match="Sample size must be at least 3"):
            moment2d(data, n=3)

    @staticmethod
    def test_invalid_reference_length() -> None:
        """Test error handling for invalid reference length"""
        # Arrange
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        invalid_reference = (1.0, 2.0)  # Only 2 values for 3 variables

        # Act & Assert
        with pytest.raises(ValueError, match="same number of variables"):
            moment2d(data, n=2, reference=invalid_reference)

    @staticmethod
    def test_raise_zero_division_policy() -> None:
        """Test 'raise' policy for zero division"""
        # Arrange
        data = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])  # Zero variance

        # Act & Assert
        with pytest.raises(ValueError, match="Zero value found in moment denominator"):
            moment2d(data, n=2, standardized=True, zero_division="raise")

    @staticmethod
    def test_higher_order_moments() -> None:
        """Test higher order moments (skewness, kurtosis equivalents)"""
        # Arrange - normally distributed data should have specific moments
        np.random.seed(42)
        data = np.random.normal(0, 1, (1000, 2))

        # Test skewness (should be close to 0)
        skewness = moment2d(data, n=3, standardized=True)
        assert all(abs(x) < 0.2 for x in skewness)  # Allow some tolerance

        # Test kurtosis (should be close to 3 for excess=False)
        kurtosis = moment2d(data, n=4, standardized=True)
        assert all(2.5 < x < 3.5 for x in kurtosis)  # Allow some tolerance

    @staticmethod
    def test_edge_case_single_sample() -> None:
        """Test edge case with single sample (only for first moment)"""
        # Arrange
        data = np.array([[5.0, 10.0]])

        # First moment should work
        result = moment2d(data, n=1)
        assert result == (5.0, 10.0)

        # Second moment should fail
        with pytest.raises(ValueError, match="Sample size must be at least 2"):
            moment2d(data, n=2)

    @staticmethod
    def test_invalid_zero_division_policy() -> None:
        """Test error handling for invalid zero division policy"""
        # Arrange
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid zero_division policy"):
            moment2d(data, n=2, standardized=True, zero_division="invalid")

    @staticmethod
    def test_mixed_data_types_conversion() -> None:
        """Test that mixed data types are properly converted to float"""
        # Arrange
        data = np.array([[1, 2.5], [3, 4]])  # Mix of int and float

        # Act
        result = moment2d(data, n=1)

        # Assert - should work without errors and return floats
        assert all(isinstance(x, float) for x in result)
        assert np.allclose(result, (2.0, 3.25), rtol=1e-10)

    @staticmethod
    def test_return_type() -> None:
        """Test that return type is always a tuple of floats"""
        test_cases = [
            np.array([[1.0]]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ]

        for data in test_cases:
            result = moment2d(data, n=1)
            assert isinstance(result, tuple)
            assert all(isinstance(x, float) for x in result)


# %% Test - moment2d

# TestMoment2D.test_first_moment_without_reference()
# TestMoment2D.test_second_moment_without_reference()
# TestMoment2D.test_second_moment_with_custom_reference()
# TestMoment2D.test_standardized_second_moment()
# TestMoment2D.test_axis_parameter()
# TestMoment2D.test_zero_division_policies()
# TestMoment2D.test_nan_handling()
# TestMoment2D.test_insufficient_sample_size()
# TestMoment2D.test_invalid_reference_length()
# TestMoment2D.test_raise_zero_division_policy()
# TestMoment2D.test_higher_order_moments()
# TestMoment2D.test_edge_case_single_sample()
# TestMoment2D.test_invalid_zero_division_policy()
# TestMoment2D.test_mixed_data_types_conversion()
# TestMoment2D.test_return_type()


# %% test functions : bandquant


class TestBandQuant:
    """Test class for bandquant function"""

    @staticmethod
    def test_basic_functionality() -> None:
        """Test basic functionality with integer bins"""
        # Create test data
        spec_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

        # Test with band 0 and 3 bins
        result = bandquant(spec_array, band=0, bins=3)

        # Expected quantiles: 0%, 50%, 100%
        expected = (1.0, 5.5, 10.0)

        assert len(result) == 3
        assert all(np.isclose(result[i], expected[i]) for i in range(3))

    @staticmethod
    def test_custom_quantiles() -> None:
        """Test with custom quantile list"""
        spec_array = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])

        # Test with custom quantiles
        quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = bandquant(spec_array, band=1, bins=quantiles)

        expected = (10.0, 20.0, 30.0, 40.0, 50.0)

        assert len(result) == 5
        assert all(np.isclose(result[i], expected[i]) for i in range(5))

    @staticmethod
    def test_negative_band_index() -> None:
        """Test with negative band index"""
        spec_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Test with negative index (should access last band)
        result_neg = bandquant(spec_array, band=-1, bins=2)
        result_pos = bandquant(spec_array, band=2, bins=2)

        assert result_neg == result_pos

    @staticmethod
    def test_different_axis() -> None:
        """Test with different axis parameter"""
        spec_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        result1 = bandquant(spec_array, band=0, bins=2, axis=1)
        result2 = bandquant(spec_array.T, band=0, bins=2, axis=0)

        assert result1 == result2

    @staticmethod
    def test_with_nan_values() -> None:
        """Test handling of NaN values"""
        spec_array = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0], [np.nan, 8.0, 9.0], [10.0, 11.0, 12.0]])

        # Should handle NaN values gracefully
        result = bandquant(spec_array, band=1, bins=3)

        # NaN values should be ignored in quantile calculation
        assert not any(np.isnan(val) for val in result)

    @staticmethod
    def test_edge_cases() -> None:
        """Test edge cases"""
        # Single row array
        single_row = np.array([[1.0, 2.0, 3.0]])
        result = bandquant(single_row, band=0, bins=2)
        assert result == (1.0, 1.0)  # All quantiles same for single value

        # Two identical values
        identical = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        result = bandquant(identical, band=0, bins=3)
        assert all(val == 5.0 for val in result)

    @staticmethod
    def test_error_cases() -> None:
        """Test error cases"""
        spec_array = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Band index out of bounds
        with pytest.raises(ValueError, match="band index exceeds"):
            bandquant(spec_array, band=5, bins=2)

        with pytest.raises(ValueError, match="band index exceeds"):
            bandquant(spec_array, band=-5, bins=2)

        # Invalid bin count
        with pytest.raises(ValueError, match="number of bins must be greater than 1"):
            bandquant(spec_array, band=0, bins=1)

        # Invalid quantile range
        with pytest.raises(ValueError, match="Bin value must be in range"):
            bandquant(spec_array, band=0, bins=[-0.1, 0.5, 1.1])

        with pytest.raises(ValueError, match="number of bins must be greater than 1"):
            bandquant(spec_array, band=0, bins=[0.5])

    @staticmethod
    def test_return_type() -> None:
        """Test that return type is correct"""
        spec_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = bandquant(spec_array, band=0, bins=3)

        assert isinstance(result, tuple)
        assert all(isinstance(val, float) for val in result)

    @staticmethod
    def test_multiple_bands() -> None:
        """Test with multiple bands (though function only handles single band)"""
        spec_array = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]])

        # Test different bands give different results
        result_band0 = bandquant(spec_array, band=0, bins=2)
        result_band1 = bandquant(spec_array, band=1, bins=2)
        result_band2 = bandquant(spec_array, band=2, bins=2)

        assert result_band0 != result_band1 != result_band2

    @staticmethod
    def test_with_list_input() -> None:
        """Test that function works with list input (not just numpy arrays)"""
        spec_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

        result = bandquant(spec_list, band=1, bins=3)

        expected = (2.0, 5.0, 8.0)
        assert all(np.isclose(result[i], expected[i]) for i in range(3))

    @staticmethod
    def test_large_dataset() -> None:
        """Test with larger dataset to ensure performance and correctness"""
        np.random.seed(42)  # For reproducible results
        large_array = np.random.rand(1000, 5)

        result = bandquant(large_array, band=2, bins=5)

        # Should return 5 values
        assert len(result) == 5
        # Values should be in increasing order
        assert all(result[i] <= result[i + 1] for i in range(4))


# %% Test - bandquant

# TestBandQuant.test_basic_functionality()
# TestBandQuant.test_custom_quantiles()
# TestBandQuant.test_negative_band_index()
# TestBandQuant.test_different_axis()
# TestBandQuant.test_with_nan_values()
# TestBandQuant.test_edge_cases()
# TestBandQuant.test_error_cases()
# TestBandQuant.test_return_type()
# TestBandQuant.test_multiple_bands()
# TestBandQuant.test_with_list_input()
# TestBandQuant.test_large_dataset()


# %% test functions : smopt


class TestSmopt:
    """Test suite for smopt function"""

    @staticmethod
    def test_mean() -> None:
        """Test smopt with mean measure"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        measure_func = smopt("mean")
        result = measure_func(arr)

        expected = tuple(np.nanmean(arr, axis=0))
        assert np.allclose(result, expected)
        assert isinstance(result, np.ndarray)

    @staticmethod
    def test_median() -> None:
        """Test smopt with median measure"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        measure_func = smopt("median")
        result = measure_func(arr)

        expected = tuple(np.nanmedian(arr, axis=0))
        assert np.allclose(result, expected)

    @staticmethod
    def test_min() -> None:
        """Test smopt with min measure"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        measure_func = smopt("min")
        result = measure_func(arr)

        expected = tuple(np.nanmin(arr, axis=0))
        assert np.allclose(result, expected)

    @staticmethod
    def test_max() -> None:
        """Test smopt with max measure"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        measure_func = smopt("max")
        result = measure_func(arr)

        expected = tuple(np.nanmax(arr, axis=0))
        assert np.allclose(result, expected)

    @staticmethod
    def test_var() -> None:
        """Test smopt with variance measure"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        measure_func = smopt("var")
        result = measure_func(arr)

        expected = tuple(np.nanvar(arr, axis=0))
        assert np.allclose(result, expected)

    @staticmethod
    def test_std() -> None:
        """Test smopt with standard deviation measure"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        measure_func = smopt("std")
        result = measure_func(arr)

        expected = tuple(np.nanstd(arr, axis=0))
        assert np.allclose(result, expected)

    @staticmethod
    def test_stdev() -> None:
        """Test smopt with stdev alias"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        measure_func = smopt("stdev")
        result = measure_func(arr)

        expected = tuple(np.nanstd(arr, axis=0))
        assert np.allclose(result, expected)

    @staticmethod
    def test_case_insensitive() -> None:
        """Test smopt with different case inputs"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        for measure_name in ["MEAN", "Mean", "mEaN"]:
            measure_func = smopt(measure_name)
            result = measure_func(arr)
            expected = tuple(np.nanmean(arr, axis=0))
            assert np.allclose(result, expected)

    @staticmethod
    def test_with_nan_values() -> None:
        """Test smopt handles NaN values correctly"""
        arr = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        measure_func = smopt("mean")
        result = measure_func(arr)

        expected = tuple(np.nanmean(arr, axis=0))
        assert np.allclose(result, expected)

    @staticmethod
    def test_skewness() -> None:
        """Test smopt with skewness measure"""
        arr = np.array([[1.0, 4.0, 7.0], [2.0, 6.0, 8.0], [4.0, 6.0, 9.0]])

        measure_func = smopt("skewness")
        result1 = measure_func(arr)

        from scipy.stats import skew

        result2 = skew(arr)

        assert np.allclose(result1, result2)

    @staticmethod
    def test_skewness_with_zero_var() -> None:
        """Test smopt with skewness measure on data with zero variance"""
        arr = np.array([[1.0, 4.0, 7.0], [1.0, 4.0, 7.0], [1.0, 4.0, 7.0]])

        measure_func = smopt("skewness")
        result = measure_func(arr)

        assert np.allclose(result, (0, 0, 0))

    @staticmethod
    def test_skew_alias() -> None:
        """Test smopt with skew alias"""
        arr = np.array([[1.0, 4.0, 7.0], [2.0, 6.0, 8.0], [4.0, 6.0, 9.0]])

        measure_func1 = smopt("skewness")
        result1 = measure_func1(arr)

        measure_func2 = smopt("skew")
        result2 = measure_func2(arr)

        assert np.allclose(result1, result2)

    @staticmethod
    def test_kurtosis() -> None:
        """Test smopt with kurtosis measure"""
        arr = np.array([[1.0, 4.0, 7.0], [2.0, 6.0, 8.0], [4.0, 6.0, 9.0], [8.0, 6.0, 10.0]])

        measure_func = smopt("kurtosis")
        result1 = measure_func(arr)

        from scipy.stats import kurtosis  # Excess kurtosis

        result2 = kurtosis(arr) + 3.0

        assert np.allclose(result1, result2)

    @staticmethod
    def test_kurtosis_with_zero_var() -> None:
        """Test smopt with kurtosis measure on data with zero variance"""
        arr = np.array([[1.0, 4.0, 7.0], [1.0, 4.0, 7.0], [1.0, 4.0, 7.0], [1.0, 4.0, 7.0]])

        measure_func = smopt("kurtosis")
        result = measure_func(arr)

        assert np.allclose(result, (0, 0, 0))

    @staticmethod
    def test_kurt_alias() -> None:
        """Test smopt with kurt alias"""
        arr = np.array([[1.0, 4.0, 7.0], [2.0, 6.0, 8.0], [4.0, 6.0, 9.0], [8.0, 6.0, 10.0]])

        measure_func1 = smopt("kurtosis")
        result1 = measure_func1(arr)

        measure_func2 = smopt("kurt")
        result2 = measure_func2(arr)

        assert np.allclose(result1, result2)

    @staticmethod
    def test_invalid_measure() -> None:
        """Test smopt raises ValueError for invalid measure name"""
        with pytest.raises(ValueError, match="Undefined measure name"):
            smopt("invalid_measure")

    @staticmethod
    def test_return_type() -> None:
        """Test that smopt returns a callable function"""
        measure_func = smopt("mean")
        assert callable(measure_func)

        # Test the returned function signature
        arr = np.array([[1.0, 2.0, 3.0]])
        result = measure_func(arr)
        assert isinstance(result, np.ndarray)

    @staticmethod
    def test_single_column() -> None:
        """Test smopt with single column array"""
        arr = np.array([[1.0], [2.0], [3.0]])
        measure_func = smopt("mean")
        result = measure_func(arr)

        expected = np.nanmean(arr, axis=0)
        assert np.allclose(result, expected)
        assert len(result) == 1

    @staticmethod
    def test_empty_array() -> None:
        """Test smopt with empty array"""
        arr = np.array([[]], dtype=float)
        measure_func = smopt("mean")
        result = measure_func(arr)

        expected = np.nanmean(arr, axis=0)
        assert np.allclose(result, expected)


# %% Test - smopt

# TestSmopt.test_mean()
# TestSmopt.test_median()
# TestSmopt.test_min()
# TestSmopt.test_max()
# TestSmopt.test_var()
# TestSmopt.test_std()
# TestSmopt.test_stdev()
# TestSmopt.test_case_insensitive()
# TestSmopt.test_with_nan_values()
# TestSmopt.test_skewness()
# TestSmopt.test_skew_alias()
# TestSmopt.test_kurtosis()
# TestSmopt.test_kurt_alias()
# TestSmopt.test_invalid_measure()
# TestSmopt.test_return_type()
# TestSmopt.test_single_column()
# TestSmopt.test_empty_array()


# %% test functions : cmval


class TestCmval:
    """Test class for cmval function"""

    @staticmethod
    def test_valid_function() -> None:
        """Test with a valid custom function"""
        # Create test data
        test_arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Define a valid custom function
        def valid_func(arr: np.ndarray) -> float:
            return float(np.mean(arr))

        # Should return the same function without errors
        result_func = cmval(test_arr, valid_func)

        assert result_func is valid_func
        assert result_func.__name__ == valid_func.__name__

    @staticmethod
    def test_valid_function_with_list_input() -> None:
        """Test with list input instead of numpy array"""
        test_list = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        def mean_func(arr: np.ndarray) -> float:
            return float(np.mean(arr))

        # Should work with list input
        result_func = cmval(test_list, mean_func)
        assert result_func is mean_func

    @staticmethod
    def test_function_returning_scalar() -> None:
        """Test with function returning scalar value"""
        test_arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        def scalar_func(arr: np.ndarray) -> Any:
            return arr[0, 0]  # Return single scalar

        result_func = cmval(test_arr, scalar_func)
        assert result_func is scalar_func

    @staticmethod
    def test_function_returning_1d_array() -> None:
        """Test with function returning 1D array"""
        test_arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        def array_func(arr: np.ndarray) -> np.ndarray:
            result: np.ndarray = np.array([np.mean(arr)])
            return result  # Return 1D array with single element

        result_func = cmval(test_arr, array_func)
        assert result_func is array_func

    @staticmethod
    def test_function_with_different_dtypes() -> None:
        """Test with function returning different numeric dtypes"""
        test_arr = np.array([[1, 2], [3, 4]], dtype=int)

        # Test different numeric return types
        def int_func(arr: np.ndarray) -> int:
            return int(np.mean(arr))

        def float_func(arr: np.ndarray) -> float:
            return float(np.mean(arr))

        # Both should work
        result_int = cmval(test_arr, int_func)
        result_float = cmval(test_arr, float_func)

        assert result_int is int_func
        assert result_float is float_func

    @staticmethod
    def test_error_multidimensional_return() -> None:
        """Test error when function returns multi-dimensional array"""
        test_arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        def invalid_func(arr: np.ndarray) -> np.ndarray:
            return arr  # Return 2D array instead of scalar/1D

        with pytest.raises(ValueError, match="returned result shape"):
            cmval(test_arr, invalid_func)

    @staticmethod
    def test_error_non_numeric_return() -> None:
        """Test error when function returns non-numeric type"""
        test_arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        def string_func(arr: np.ndarray) -> str:
            return "invalid"  # Return string instead of number

        with pytest.raises(TypeError, match="must return numbers"):
            cmval(test_arr, string_func)

    @staticmethod
    def test_error_function_raises_exception() -> None:
        """Test error when function raises exception during testing"""
        test_arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        def failing_func(arr: np.ndarray) -> float:
            raise ValueError("Test error in function")

        with pytest.raises(ValueError, match="Error in the testing"):
            cmval(test_arr, failing_func)

    @staticmethod
    def test_error_function_name_in_message() -> None:
        """Test that function name appears in error messages"""
        test_arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        def invalid_func(arr: np.ndarray) -> np.ndarray:
            return arr  # This will cause shape error

        # Check that function name appears in error message
        try:
            cmval(test_arr, invalid_func)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            assert "invalid_func" in str(e)

    @staticmethod
    def test_preserves_function_identity() -> None:
        """Test that the exact same function object is returned"""
        test_arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        def test_func(arr: np.ndarray) -> float:
            return float(np.sum(arr))

        result_func = cmval(test_arr, test_func)

        # Should return the exact same function object, not a copy
        assert result_func is test_func
        assert id(result_func) == id(test_func)

    @staticmethod
    def test_with_lambda_function() -> None:
        """Test with lambda function"""
        test_arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        lambda_func = lambda arr: np.median(arr)  # noqa: E731

        result_func = cmval(test_arr, lambda_func)
        assert result_func is lambda_func

    @staticmethod
    def test_with_numpy_function() -> None:
        """Test with numpy function"""
        test_arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Test with numpy function (should work if it has the right signature)
        result_func = cmval(test_arr, np.mean)
        assert result_func is np.mean

    @staticmethod
    def test_function_returning_zero() -> None:
        """Test with function that returns zero"""
        test_arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        def zero_func(arr: np.ndarray) -> float:
            return 0.0

        result_func = cmval(test_arr, zero_func)
        assert result_func is zero_func

    @staticmethod
    def test_function_with_side_effects() -> None:
        """Test that function side effects are preserved"""
        test_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        call_count = 0

        def counting_func(arr: np.ndarray) -> float:
            nonlocal call_count
            call_count += 1
            return float(np.mean(arr))

        # First call (during validation)
        result_func = cmval(test_arr, counting_func)
        assert call_count == 1

        # Second call (after validation)
        result = result_func(test_arr)
        assert call_count == 2
        assert isinstance(result, float)

    @staticmethod
    def test_with_empty_array() -> None:
        """Test with empty array (edge case)"""
        test_arr = np.array([[], []])  # Empty 2D array

        def empty_func(arr: np.ndarray) -> float:
            if arr.size == 0:
                result: float = 0.0
            else:
                result = float(np.mean(arr))
            return result

        # Should work with empty array
        result_func = cmval(test_arr, empty_func)
        assert result_func is empty_func


# %% Test - cmval

# TestCmval.test_valid_function()
# TestCmval.test_valid_function_with_list_input()

# TestCmval.test_function_returning_scalar()
# TestCmval.test_function_returning_1d_array()

# TestCmval.test_function_with_different_dtypes()

# TestCmval.test_error_multidimensional_return()
# TestCmval.test_error_non_numeric_return()
# TestCmval.test_error_function_raises_exception()
# TestCmval.test_error_function_name_in_message()

# TestCmval.test_preserves_function_identity()

# TestCmval.test_with_lambda_function()
# TestCmval.test_with_numpy_function()
# TestCmval.test_function_returning_zero()
# TestCmval.test_function_with_side_effects()
# TestCmval.test_with_empty_array()


# %% test functions : Stats2d


class TestStats2d:
    """Test suite for Stats2d class"""

    @staticmethod
    def test_static_mean() -> None:
        """Test Stats2d.mean static method"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = Stats2d(axis=0).mean(arr)
        expected = np.nanmean(arr, axis=0)

        assert np.allclose(result, expected)

    @staticmethod
    def test_static_std() -> None:
        """Test Stats2d.std static method"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = Stats2d(axis=0).std(arr)
        expected = np.nanstd(arr, axis=0)

        assert np.allclose(result, expected)

    @staticmethod
    def test_static_var() -> None:
        """Test Stats2d.var static method"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = Stats2d(axis=0).var(arr)
        expected = np.nanvar(arr, axis=0)

        assert np.allclose(result, expected)

    @staticmethod
    def test_static_skew() -> None:
        """Test Stats2d.skew static method"""
        arr = np.array([[1.0, 4.0, 7.0], [2.0, 6.0, 8.0], [4.0, 6.0, 9.0]])

        result1 = Stats2d(axis=0).skew(arr)

        from scipy.stats import skew

        result2 = skew(arr)

        assert np.allclose(result1, result2)

    @staticmethod
    def test_static_kurt() -> None:
        """Test Stats2d.kurt static method"""
        arr = np.array([[1.0, 4.0, 7.0], [2.0, 6.0, 8.0], [4.0, 6.0, 9.0], [8.0, 6.0, 10.0]])

        result1 = Stats2d(axis=0).kurt(arr)

        from scipy.stats import kurtosis  # Excess kurtosis

        result2 = kurtosis(arr) + 3.0

        assert np.allclose(result1, result2)

    @staticmethod
    def test_static_minimum() -> None:
        """Test Stats2d.minimum static method"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = Stats2d(axis=0).minimum(arr)
        expected = np.nanmin(arr, axis=0)

        assert np.allclose(result, expected)

    @staticmethod
    def test_static_median() -> None:
        """Test Stats2d.median static method"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = Stats2d(axis=0).median(arr)
        expected = np.nanmedian(arr, axis=0)

        assert np.allclose(result, expected)

    @staticmethod
    def test_static_maximum() -> None:
        """Test Stats2d.maximum static method"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = Stats2d(axis=0).maximum(arr)
        expected = np.nanmax(arr, axis=0)

        assert np.allclose(result, expected)

    @staticmethod
    def test_init_default() -> None:
        """Test Stats2d initialization with default parameters"""
        stats = Stats2d()
        assert stats.measure is None
        assert stats.axis == 0

    @staticmethod
    def test_init_with_measure_and_axis() -> None:
        """Test Stats2d initialization with custom parameters"""
        stats = Stats2d(measure="mean", axis=1)
        assert stats.measure == "mean"
        assert stats.axis == 1

    @staticmethod
    def test_values_property_callable_measure() -> None:
        """Test Stats2d.values property with callable measure"""
        arr: np.ndarray = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [10.0, 11.0, 12.0],
            ]
        )

        def custom_measure(data: np.ndarray) -> np.ndarray:
            result: np.ndarray = np.array([1.0, 2.0, 3.0])
            return result

        stats = Stats2d(measure=custom_measure)
        result = stats.values(arr)

        assert np.allclose(result, np.array([1.0, 2.0, 3.0]))
        assert custom_measure.__name__ in stats.values.__name__

        with pytest.raises(ValueError, match="accepts single measure only"):
            Stats2d(measure=["mean", "std"]).values(arr)

    @staticmethod
    def test_single_string_measure() -> None:
        """Test Stats2d.summary with single string measure"""
        arr = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [10.0, 11.0, 12.0],
            ]
        )

        result = Stats2d(measure="mean", axis=0).summary(arr)
        expected = np.nanmean(arr, axis=0)

        assert isinstance(result, dict)
        assert "mean" in result.keys()
        assert np.allclose(result["mean"], expected)

    @staticmethod
    def test_multiple_measures() -> None:
        """Test Stats2d.summary with multiple measures"""
        arr = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [10.0, 11.0, 12.0],
            ]
        )

        result = Stats2d(measure=["mean", "std"], axis=0).summary(arr)
        expected = [np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)]

        assert isinstance(result, dict)
        assert "mean" in result.keys()
        assert "std" in result.keys()
        assert np.allclose(result["mean"], expected[0])
        assert np.allclose(result["std"], expected[1])

    @staticmethod
    def test_default_measures() -> None:
        """Test Stats2d.summary with default measures (None)"""
        arr = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [10.0, 11.0, 12.0],
            ]
        )

        result = Stats2d().summary(arr)

        expected_keys = ["mean", "std", "skewness", "kurtosis", "min", "median", "max"]
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], np.ndarray)

    @staticmethod
    def test_insufficient_samples() -> None:
        """Test Stats2d.summary with insufficient samples"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])  # Only 4 samples

        with pytest.warns(UserWarning, match="Sample size"):
            Stats2d().summary(arr)

    @staticmethod
    def test_callable_measure() -> None:
        """Test Stats2d.summary with callable measure"""
        arr = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [10.0, 11.0, 12.0],
            ]
        )

        def custom_measure(data: np.ndarray) -> np.ndarray:
            result = np.nanmean(data, axis=0)
            assert isinstance(result, np.ndarray)
            return result

        result = Stats2d(measure=custom_measure, axis=0).summary(arr)
        expected = np.nanmean(arr, axis=0)

        assert isinstance(result, dict)
        assert "custom_measure" in result.keys()
        assert np.allclose(result["custom_measure"], expected)

    @staticmethod
    def test_invalid_measure_type() -> None:
        """Test Stats2d.summary with invalid measure type"""
        with pytest.raises(TypeError):
            Stats2d(measure=123)


# %% Test - Stats2d

# TestStats2d.test_static_mean()
# TestStats2d.test_static_std()
# TestStats2d.test_static_var()
# TestStats2d.test_static_skew()
# TestStats2d.test_static_kurt()
# TestStats2d.test_static_minimum()
# TestStats2d.test_static_median()
# TestStats2d.test_static_maximum()
# TestStats2d.test_init_default()
# TestStats2d.test_init_with_measure_and_axis()
# TestStats2d.test_values_property_callable_measure()
# TestStats2d.test_single_string_measure()
# TestStats2d.test_multiple_measures()
# TestStats2d.test_default_measures()
# TestStats2d.test_insufficient_samples()
# TestStats2d.test_callable_measure()
# TestStats2d.test_invalid_measure_type()


# %% test functions : roi_mean, roi_std, roi_median


def test_roi_mean() -> None:
    """Test roi_mean"""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

        data = np.arange(1, 17, dtype=np.float32).reshape((1, 4, 4))

        # Create test raster
        _ = create_test_raster(
            tmp_path,
            width=data.shape[2],
            height=data.shape[1],
            bands=data.shape[0],
            data=data,
            dtype="float32",
        )

        roi_coordinates = [[(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0), (0.0, 0.0)]]

        result = roi_mean(tmp_path, roi_coordinates)
        expected = np.mean(data[:, 2:4, 0:2], axis=(1, 2))

    assert np.allclose(result, expected)


def test_roi_std() -> None:
    """Test roi_std"""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

        data = np.arange(1, 17, dtype=np.float32).reshape((1, 4, 4))

        # Create test raster
        _ = create_test_raster(
            tmp_path,
            width=data.shape[2],
            height=data.shape[1],
            bands=data.shape[0],
            data=data,
            dtype="float32",
        )

        roi_coordinates = [[(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0), (0.0, 0.0)]]

        result = roi_std(tmp_path, roi_coordinates)
        expected = np.std(data[:, 2:4, 0:2], axis=(1, 2))

    assert np.allclose(result, expected)


def test_roi_median() -> None:
    """Test roi_median"""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

        data = np.arange(1, 17, dtype=np.float32).reshape((1, 4, 4))

        # Create test raster
        _ = create_test_raster(
            tmp_path,
            width=data.shape[2],
            height=data.shape[1],
            bands=data.shape[0],
            data=data,
            dtype="float32",
        )

        roi_coordinates = [[(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0), (0.0, 0.0)]]

        result = roi_median(tmp_path, roi_coordinates)
        expected = np.median(data[:, 2:4, 0:2], axis=(1, 2))

    assert np.allclose(result, expected)


# %% Test - roi_mean, roi_std, roi_median

# test_roi_mean()
# test_roi_std()
# test_roi_median()


# %% test functions : spectral_angle


def test_spectral_angle() -> None:
    """Test spectral_angle"""
    v1 = [0, 0, 1]
    v2 = [0, 0, 1]
    assert spectral_angle(v1, v2) == 0.0

    v3 = [0, 1, 0]
    assert spectral_angle(v1, v3) - 0.5 * np.pi < 1e-5

    v4 = [0, 1, 1]
    assert spectral_angle(v1, v4) - 0.25 * np.pi < 1e-5

    v5 = [0, 0, 0]
    with pytest.raises(ValueError, match="Undefined spectral angle for given spectral vector"):
        spectral_angle(v1, v5, invalid_raise=True)


# %% Test - spectral_angle

# test_spectral_angle()


# %% test functions : spectral_angle_arr


class TestSpectralAngleArr:
    """Test class for spectral_angle_arr function"""

    @staticmethod
    def create_test_data() -> tuple[np.ndarray, np.ndarray]:
        """Create test spectra array and reference spectrum"""
        # Create sample spectra array (3 spectra, 4 bands)
        spec_array_2d = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],  # Spectrum 1
                [2.0, 4.0, 6.0, 8.0],  # Spectrum 2 (scaled version of ref)
                [0.5, 1.0, 1.5, 2.0],  # Spectrum 3 (another scaled version)
            ],
            dtype=np.float64,
        )

        # Create reference spectrum
        reference_spectrum = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        return spec_array_2d, reference_spectrum

    @staticmethod
    def test_basic_functionality() -> None:
        """Test basic functionality with valid inputs"""
        spec_array_2d, reference_spectrum = TestSpectralAngleArr.create_test_data()

        # Calculate spectral angles
        result = spectral_angle_arr(spec_array_2d, reference_spectrum, axis=0)

        # Expected results
        expected = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        assert result.shape == (3,)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    @staticmethod
    def test_axis_1_functionality() -> None:
        """Test functionality with axis=1"""
        spec_array_2d, reference_spectrum = TestSpectralAngleArr.create_test_data()

        # Transpose the array to test axis=1
        spec_array_2d_transposed = spec_array_2d.T

        result = spectral_angle_arr(spec_array_2d_transposed, reference_spectrum, axis=1)

        expected = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        assert result.shape == (3,)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    @staticmethod
    def test_non_zero_angles() -> None:
        """Test with spectra that should have non-zero angles"""
        spec_array_2d = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],  # Different shape from reference
                [4.0, 3.0, 2.0, 1.0],  # Reversed order
            ],
            dtype=np.float64,
        )

        reference_spectrum = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        result = spectral_angle_arr(spec_array_2d, reference_spectrum, axis=0)

        # Angles should be positive and non-zero
        assert result.shape == (2,)
        assert np.all(result > 0)
        assert not np.any(np.isnan(result))

    @staticmethod
    def test_undefined_angles() -> None:
        """Test that zeros are handled properly"""
        spec_array_2d = np.array(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]],
            dtype=np.float64,  # All zeros
        )

        reference_spectrum = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        with pytest.raises(ValueError):
            spectral_angle_arr(spec_array_2d, reference_spectrum, axis=0, invalid_raise=True)

        result = spectral_angle_arr(spec_array_2d, reference_spectrum, axis=0, invalid_raise=False)
        result1 = spectral_angle_arr(spec_array_2d, reference_spectrum, axis=0)

        # Should not raise error and should return valid angles
        assert result.shape == (2,)
        assert result[1] == 0.0
        assert np.isnan(result[0])
        assert np.allclose(result, result1, equal_nan=True)

    @staticmethod
    def test_invalid_axis() -> None:
        """Test that invalid axis raises ValueError"""
        spec_array_2d, reference_spectrum = TestSpectralAngleArr.create_test_data()

        with pytest.raises(ValueError, match="axis can only be 0 or 1"):
            spectral_angle_arr(spec_array_2d, reference_spectrum, axis=2)

    @staticmethod
    def test_non_2d_array() -> None:
        """Test that 1D array raises ValueError"""
        reference_spectrum = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        with pytest.raises(ValueError, match="input spec_array_2d must be 2d array like"):
            spectral_angle_arr([1, 2, 3, 4], reference_spectrum)

    @staticmethod
    def test_nan_values() -> None:
        """Test that NaN values raise ValueError"""
        spec_array_2d = np.array([[1.0, 2.0, np.nan, 4.0], [2.0, 4.0, 6.0, 8.0]], dtype=np.float64)

        reference_spectrum = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        with pytest.raises(ValueError, match="input spec_array_2d must not contain nan values"):
            spectral_angle_arr(spec_array_2d, reference_spectrum)

    @staticmethod
    def test_dimension_mismatch() -> None:
        """Test that dimension mismatch raises ValueError"""
        spec_array_2d = np.array(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
            dtype=np.float64,  # Only 3 bands
        )

        reference_spectrum = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)  # 4 bands

        with pytest.raises(ValueError, match="input spec_array_2d does not match with reference_spectrum"):
            spectral_angle_arr(spec_array_2d, reference_spectrum)

    @staticmethod
    def test_empty_array() -> None:
        """Test with empty spectra array"""
        spec_array_2d = np.array([[]], dtype=np.float64)  # Empty 2D array
        reference_spectrum = np.array([], dtype=np.float64)  # Empty reference

        result = spectral_angle_arr(spec_array_2d, reference_spectrum, axis=0)

        assert result.shape == (1,)

    @staticmethod
    def test_single_spectrum() -> None:
        """Test with single spectrum in array"""
        spec_array_2d = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
        reference_spectrum = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        result = spectral_angle_arr(spec_array_2d, reference_spectrum, axis=0)

        assert result.shape == (1,)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    @staticmethod
    def test_list_input() -> None:
        """Test that function works with list inputs"""
        spec_array_2d = [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]

        reference_spectrum = [1.0, 2.0, 3.0, 4.0]

        result = spectral_angle_arr(spec_array_2d, reference_spectrum, axis=0)

        assert result.shape == (2,)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-6)


# %% Test - spectral_angle

# TestSpectralAngleArr.test_basic_functionality()
# TestSpectralAngleArr.test_axis_1_functionality()
# TestSpectralAngleArr.test_non_zero_angles()
# TestSpectralAngleArr.test_undefined_angles()
# TestSpectralAngleArr.test_invalid_axis()
# TestSpectralAngleArr.test_non_2d_array()
# TestSpectralAngleArr.test_nan_values()
# TestSpectralAngleArr.test_dimension_mismatch()
# TestSpectralAngleArr.test_empty_array()
# TestSpectralAngleArr.test_single_spectrum()
# TestSpectralAngleArr.test_list_input()


# %% test functions : num_sig_digit


class TestNumSigDigit:
    """Test cases for num_sig_digit function"""

    @staticmethod
    def test_round_mode_basic() -> None:
        """Test basic rounding functionality"""
        # Test with integers
        assert num_sig_digit(123456, 3, "round") == 123000.0
        assert num_sig_digit(123456, 4, "round") == 123500.0
        assert num_sig_digit(999, 2, "round") == 1000.0

        # Test with floats
        assert num_sig_digit(123.456, 3, "round") == 123.0
        assert num_sig_digit(123.456, 4, "round") == 123.5
        assert num_sig_digit(0.00123456, 3, "round") == 0.00123

    @staticmethod
    def test_ceil_mode_basic() -> None:
        """Test ceiling functionality"""
        assert num_sig_digit(123.456, 3, "ceil") == 124.0
        assert num_sig_digit(123.001, 3, "ceil") == 124.0
        assert num_sig_digit(0.00123456, 3, "ceil") == 0.00124

    @staticmethod
    def test_floor_mode_basic() -> None:
        """Test floor functionality"""
        assert num_sig_digit(123.999, 3, "floor") == 123.0
        assert num_sig_digit(123.456, 3, "floor") == 123.0
        assert num_sig_digit(0.00123456, 3, "floor") == 0.00123

    @staticmethod
    def test_zero_value() -> None:
        """Test handling of zero value"""
        assert num_sig_digit(0, 3, "round") == 0.0
        assert num_sig_digit(0.0, 5, "ceil") == 0.0
        assert num_sig_digit(0, 2, "floor") == 0.0

    @staticmethod
    def test_negative_numbers() -> None:
        """Test with negative numbers"""
        assert num_sig_digit(-123.456, 3, "round") == -123.0
        assert num_sig_digit(-123.456, 3, "ceil") == -123.0
        assert num_sig_digit(-123.456, 3, "floor") == -124.0

    @staticmethod
    def test_single_significant_digit() -> None:
        """Test with single significant digit"""
        assert num_sig_digit(123.456, 1, "round") == 100.0
        assert num_sig_digit(567.89, 1, "ceil") == 600.0
        assert num_sig_digit(567.89, 1, "floor") == 500.0

    @staticmethod
    def test_large_numbers() -> None:
        """Test with very large numbers"""
        assert math.isclose(num_sig_digit(123456789, 4, "round"), 123500000.0, rel_tol=1e-10)
        assert math.isclose(num_sig_digit(999999999, 3, "round"), 1000000000.0, rel_tol=1e-10)

    @staticmethod
    def test_small_numbers() -> None:
        """Test with very small numbers"""
        assert math.isclose(num_sig_digit(0.00000123456, 3, "round"), 0.00000123, rel_tol=1e-10)
        assert math.isclose(num_sig_digit(1.23456e-6, 4, "round"), 0.000001235, rel_tol=1e-10)

    @staticmethod
    def test_invalid_mode() -> None:
        """Test invalid mode handling"""
        with pytest.raises(ValueError, match="mode must one of 'round', 'ceil' and 'floor'"):
            num_sig_digit(123.456, 3, "invalid")

    @staticmethod
    def test_edge_cases() -> None:
        """Test various edge cases"""
        # Exactly at rounding boundary
        assert num_sig_digit(123.5, 3, "round") == 124.0
        assert num_sig_digit(124.5, 3, "round") == 124.0  # Banker's rounding

        # Very precise numbers
        assert num_sig_digit(math.pi, 5, "round") == 3.1416
        assert num_sig_digit(math.e, 4, "round") == 2.718

    @staticmethod
    def test_consistency_across_modes() -> None:
        """Test that different modes produce expected relationships"""
        value = 123.456
        sig_digits = 3

        rounded = num_sig_digit(value, sig_digits, "round")
        ceiled = num_sig_digit(value, sig_digits, "ceil")
        floored = num_sig_digit(value, sig_digits, "floor")

        assert floored <= rounded <= ceiled

    @staticmethod
    def test_comprehensive_test_cases() -> None:
        """Comprehensive test cases covering various scenarios"""
        test_cases: list[tuple[Union[int, float], int, str, float]] = [
            # (value, sig_digits, mode, expected)
            (123.456, 3, "round", 123.0),
            (123.456, 4, "round", 123.5),
            (123.456, 5, "round", 123.46),
            (999.999, 3, "round", 1000.0),
            (0.00123456, 3, "round", 0.00123),
            (-123.456, 3, "round", -123.0),
            (123.456, 3, "ceil", 124.0),
            (123.001, 3, "ceil", 124.0),
            (123.456, 3, "floor", 123.0),
            (123.999, 3, "floor", 123.0),
        ]

        for value, sig_digits, mode, expected in test_cases:
            result = num_sig_digit(value, sig_digits, mode)
            assert math.isclose(
                result, expected, rel_tol=1e-10
            ), f"Failed for value={value}, sig_digits={sig_digits}, mode={mode}"


# %% Test - num_sig_digit

# TestNumSigDigit.test_round_mode_basic()
# TestNumSigDigit.test_ceil_mode_basic()
# TestNumSigDigit.test_floor_mode_basic()
# TestNumSigDigit.test_zero_value()
# TestNumSigDigit.test_negative_numbers()
# TestNumSigDigit.test_single_significant_digit()
# TestNumSigDigit.test_large_numbers()
# TestNumSigDigit.test_small_numbers()
# TestNumSigDigit.test_invalid_mode()
# TestNumSigDigit.test_edge_cases()
# TestNumSigDigit.test_consistency_across_modes()
# TestNumSigDigit.test_comprehensive_test_cases()


# %% test functions : np_sig_digit


class TestNPSigDigit:
    """Test class for np_sig_digit function"""

    @staticmethod
    def test_round_mode_basic() -> None:
        """Test basic functionality with round mode"""
        input_arr = [123.456, 0.012346, 9876.54]
        expected = np.array([123.5, 0.01235, 9877.0])
        result = np_sig_digit(input_arr, 4, "round")
        assert np.allclose(result, expected)

        input_arr_neg = [-123.456, -0.012346, -9876.54]
        expected_neg = np.array([-123.5, -0.01235, -9877.0])
        result_neg = np_sig_digit(input_arr_neg, 4, "round")
        assert np.allclose(result_neg, expected_neg)

    @staticmethod
    def test_ceil_mode() -> None:
        """Test ceil mode functionality"""
        input_arr = [123.456, 0.012345, 9876.54]
        expected = np.array([123.5, 0.01235, 9877.0])
        result = np_sig_digit(input_arr, 4, "ceil")
        assert np.allclose(result, expected)

    @staticmethod
    def test_floor_mode() -> None:
        """Test floor mode functionality"""
        input_arr = [123.456, 0.012345, 9876.54]
        expected = np.array([123.4, 0.01234, 9876.0])
        result = np_sig_digit(input_arr, 4, "floor")
        assert np.allclose(result, expected)

    @staticmethod
    def test_zero_values() -> None:
        """Test handling of zero values"""
        input_arr = [0.0, 123.456, 0.0, -0.0]
        expected = np.array([0.0, 120.0, 0.0, -0.0])
        result = np_sig_digit(input_arr, 2, "round")
        assert np.allclose(result, expected)

    @staticmethod
    def test_different_input_types() -> None:
        """Test function with different input types"""
        list_input = [123.456, 0.012345]
        result_list = np_sig_digit(list_input, 3, "round")

        array_input = np.array([123.456, 0.012345])
        result_array = np_sig_digit(array_input, 3, "round")

        tuple_input = (123.456, 0.012345)
        result_tuple = np_sig_digit(tuple_input, 3, "round")

        assert np.allclose(result_list, result_array)
        assert np.allclose(result_array, result_tuple)

    @staticmethod
    def test_different_sig_digits() -> None:
        """Test with different significant digit values"""
        input_arr = [123.456789]

        result_1 = np_sig_digit(input_arr, 1, "round")
        expected_1 = np.array([100.0])
        assert np.allclose(result_1, expected_1)

        result_3 = np_sig_digit(input_arr, 3, "round")
        expected_3 = np.array([123.0])
        assert np.allclose(result_3, expected_3)

        result_6 = np_sig_digit(input_arr, 6, "round")
        expected_6 = np.array([123.457])
        assert np.allclose(result_6, expected_6)

    @staticmethod
    def test_edge_cases() -> None:
        """Test edge cases and boundary values"""
        small_input = [1.23456e-10, 9.87654e-15]
        expected_small = np.array([1.23e-10, 9.88e-15])
        result_small = np_sig_digit(small_input, 3, "round")
        assert np.allclose(result_small, expected_small)

        large_input = [1.23456e10, 9.87654e15]
        expected_large = np.array([1.23e10, 9.88e15])
        result_large = np_sig_digit(large_input, 3, "round")
        assert np.allclose(result_large, expected_large)

    @staticmethod
    def test_invalid_mode() -> None:
        """Test that invalid mode raises ValueError"""
        input_arr = [123.456]

        with pytest.raises(ValueError, match="mode must one of 'round', 'ceil' and 'floor'"):
            np_sig_digit(input_arr, 3, "invalid_mode")

    @staticmethod
    def test_empty_array() -> None:
        """Test with empty array"""
        input_arr: list = []
        expected = np.array([])
        result = np_sig_digit(input_arr, 3, "round")
        assert np.allclose(result, expected)

    @staticmethod
    def test_single_value() -> None:
        """Test with single value input"""
        input_val = 123.456
        expected = np.array([123.0])
        result = np_sig_digit(input_val, 3, "round")
        assert np.allclose(result, expected)

    @staticmethod
    def test_2d_array() -> None:
        """Test with 2D array input"""
        input_2d = [[123.456, 0.012346], [9876.54, 0.000123]]
        expected = np.array([[123.5, 0.01235], [9877.0, 0.000123]])
        result = np_sig_digit(input_2d, 4, "round")
        assert np.allclose(result, expected)


# %% Test - np_sig_digit

# TestNPSigDigit.test_round_mode_basic()
# TestNPSigDigit.test_ceil_mode()
# TestNPSigDigit.test_floor_mode()
# TestNPSigDigit.test_zero_values()
# TestNPSigDigit.test_different_input_types()
# TestNPSigDigit.test_different_sig_digits()
# TestNPSigDigit.test_edge_cases()
# TestNPSigDigit.test_invalid_mode()
# TestNPSigDigit.test_empty_array()
# TestNPSigDigit.test_single_value()
# TestNPSigDigit.test_2d_array()


# %% test functions : round_digit


class TestRoundDigit:
    """Test cases for num_sig_digit function"""

    @staticmethod
    def test_number() -> None:
        assert round_digit(123.5, 3, "round") == num_sig_digit(123.5, 3, "round")
        assert round_digit(123.5, 3, "floor") == num_sig_digit(123.5, 3, "floor")
        assert round_digit(123.5, 2, "round") == num_sig_digit(123.5, 2, "round")

    @staticmethod
    def test_list() -> None:
        input_2d = [[123.456, 0.012346], [9876.54, 0.000123]]
        result = round_digit(input_2d, 4, "round")
        assert type(result) is list
        assert np.allclose(np.array(result), np_sig_digit(input_2d, 4, "round"))

    @staticmethod
    def test_tuple() -> None:
        input_2d = ((123.456, 0.012346), (9876.54, 0.000123))
        result = round_digit(input_2d, 4, "round")
        assert type(result) is tuple
        assert np.allclose(np.array(result), np_sig_digit(input_2d, 4, "round"))

    @staticmethod
    def test_array() -> None:
        input_2d = np.array(((123.456, 0.012346), (9876.54, 0.000123)))
        result = round_digit(input_2d, 4, "round")
        assert type(result) is np.ndarray
        assert np.allclose(result, np_sig_digit(input_2d, 4, "round"))

    @staticmethod
    def test_dataframe() -> None:
        input_2d = pd.DataFrame({"a": (123.456, 0.012346), "b": (9876.54, 0.000123)})
        result = round_digit(input_2d, 4, "round")
        assert type(result) is pd.DataFrame
        assert np.all(result.columns == input_2d.columns)
        assert np.allclose(result, np_sig_digit(input_2d, 4, "round"))

    @staticmethod
    def test_torch() -> None:
        input_2d = torch.tensor(((123.456, 0.012346), (9876.54, 0.000123)))
        result = round_digit(input_2d, 4, "round")
        assert type(result) is torch.Tensor
        assert np.allclose(result, np_sig_digit(input_2d, 4, "round"))


# %% Test - round_digit

# TestRoundDigit.test_number()
# TestRoundDigit.test_list()
# TestRoundDigit.test_tuple()
# TestRoundDigit.test_array()
# TestRoundDigit.test_dataframe()
# TestRoundDigit.test_torch()


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
