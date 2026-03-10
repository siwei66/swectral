# -*- coding: utf-8 -*-
"""
Tests for sample augmentation helpers

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import sys  # noqa: E402

# Typing
from typing import Any

# Test
import pytest

# Basic data
import numpy as np

# Raster
from shapely.geometry import Polygon, MultiPolygon

# Functions to test
from swectral.sample_aug import resample_roi, _blend_samples, blend_samples


# %% test functions : resample_roi


class TestResampleROI:

    @staticmethod
    def test_closed_shape_output() -> None:
        """Verify returned polygon coordinate pairs are closed."""
        roi_coords = [[(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)]]
        resolution = 1
        coverage = 0.4

        result_coords = resample_roi(roi_coords, resolution, coverage)

        for part in result_coords:
            assert part[0] == part[-1]

    @staticmethod
    def test_float_coords() -> None:
        """Verify all coordinates are in float."""
        roi_coords = [[(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)]]
        resolution = 1
        coverage = 0.4

        result_coords = resample_roi(roi_coords, resolution, coverage)

        for part in result_coords:
            for pair in part:
                assert isinstance(pair[0], float)
                assert isinstance(pair[1], float)

    @staticmethod
    def test_reproducibility() -> None:
        """Test reproducibility using random state."""
        roi_coords = [[(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)]]
        resolution = 1
        coverage = 0.4
        seed = 42

        output1 = resample_roi(roi_coords, resolution, coverage, random_state=seed)
        output2 = resample_roi(roi_coords, resolution, coverage, random_state=seed)

        assert output1 == output2

    @staticmethod
    def test_strict_containment() -> None:
        """Verify that all resampled polygons are strictly inside the original ROI."""
        roi_coords = [[(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)]]
        resolution = 1
        coverage = 0.4

        result_coords = resample_roi(roi_coords, resolution, coverage)

        original_poly = Polygon(roi_coords[0])
        for part in result_coords:
            resampled_poly = Polygon(part)
            assert original_poly.contains(resampled_poly)

    @staticmethod
    def test_coverage_ratio_error() -> None:
        """Test for invalid coverage ratios."""
        roi_coords = [[(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)]]

        with pytest.raises(ValueError, match="coverage_ratio must be between 0 and 1"):
            resample_roi(roi_coords, 1, 1.5)

        with pytest.raises(ValueError, match="coverage_ratio must be between 0 and 1"):
            resample_roi(roi_coords, 1, -0.1)

    @staticmethod
    def test_resolution_too_large() -> None:
        """Test edge cases when resolution exceeds the smallest dimension of the ROI."""
        roi_coords = [[(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]]

        with pytest.raises(ValueError, match="Resolution 3 is too large"):
            resample_roi(roi_coords, 3, 0.5)

    @staticmethod
    def test_merged_output_structure() -> None:
        """Test the output type."""
        roi_coords = [[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]]
        result = resample_roi(roi_coords, 2, 0.2)

        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], tuple)
        assert len(result[0][0]) == 2

    @staticmethod
    def test_multipart_roi_input() -> None:
        """Test functionality for multi-part multipolygon ROIs."""
        roi_coords = [[(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)], [(10, 10), (14, 10), (14, 14), (10, 14), (10, 10)]]
        result = resample_roi(roi_coords, 1, 0.1)

        master = MultiPolygon([Polygon(p) for p in roi_coords])
        for part in result:
            assert master.contains(Polygon(part))


# %% Test - resample_roi


# TestResampleROI.test_reproducibility()
# TestResampleROI.test_strict_containment()
# TestResampleROI.test_coverage_ratio_error()
# TestResampleROI.test_resolution_too_large()
# TestResampleROI.test_merged_output_structure()
# TestResampleROI.test_multipart_roi_input()


# %% test functions : _blend_samples


class TestRemixSamples:

    @staticmethod
    def mock_data_regression() -> list[tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, np.ndarray]]:
        """Create a dataset with spectral-like predictors."""
        return [
            # Group A: Two samples close in target
            ("S1", "L1", "A", np.int8(1), np.int8(1), (3,), 10.0, np.array([1.0, 1.0, 1.0])),
            ("S2", "L1", "A", np.int8(1), np.int8(1), (3,), 10.2, np.array([1.1, 1.1, 1.1])),
            ("S3", "L1", "A", np.int8(1), np.int8(1), (3,), 15.0, np.array([2.1, 3.1, 2.1])),
            # Group B: Two samples close in target
            ("S4", "L2", "B", np.int8(1), np.int8(1), (3,), 20.0, np.array([5.0, 5.0, 5.0])),
            ("S5", "L2", "B", np.int8(1), np.int8(1), (3,), 20.1, np.array([5.2, 5.2, 5.2])),
            # Group C: Lonely sample (no neighbors)
            ("S6", "L3", "C", np.int8(1), np.int8(1), (3,), 30.0, np.array([10.0, 10.0, 10.0])),
        ]

    @staticmethod
    def mock_data_classification() -> list[tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, np.ndarray]]:
        """Helper to create a dataset with string targets (labels)."""
        return [
            # Group A: Three samples, two share 'Class_X', one is 'Class_Y'
            ("S1", "L1", "A", np.int8(1), np.int8(1), (3,), "Class_X", np.array([1.0, 1.0, 1.0])),
            ("S2", "L1", "A", np.int8(1), np.int8(1), (3,), "Class_X", np.array([1.2, 1.2, 1.2])),
            ("S3", "L1", "A", np.int8(1), np.int8(1), (3,), "Class_Y", np.array([2.1, 3.1, 2.1])),
            # Group B: Two samples share 'Class_Z'
            ("S4", "L2", "B", np.int8(1), np.int8(1), (3,), "Class_Z", np.array([5.0, 5.0, 5.0])),
            ("S5", "L2", "B", np.int8(1), np.int8(1), (3,), "Class_Z", np.array([5.2, 5.2, 5.2])),
            # Group C: Lonely sample (no neighbors)
            ("S6", "L3", "C", np.int8(1), np.int8(1), (3,), "Class_Z", np.array([10.0, 10.0, 10.0])),
        ]

    @staticmethod
    def test_name_and_metadata() -> None:
        """Verifies function exists under the new name and returns correct tuple structure."""
        data = TestRemixSamples.mock_data_regression()
        result = _blend_samples(data, n_samples=1, is_regression=True, abs_tol=0.5)

        assert len(result) == len(data) + 2
        # Check metadata: Index 3 should be int8(0), Index 4 should be int8(1)
        for sample in result[len(data) :]:
            assert isinstance(sample[3], np.int8) and sample[3] == 0
            assert isinstance(sample[4], np.int8) and sample[4] == 1

    @staticmethod
    def test_lonely_group_exclusion() -> None:
        """Ensures groups with no neighbors produce no samples."""
        data = TestRemixSamples.mock_data_regression()
        # Filter for only Group C (S5 is lonely)
        group_c_only = [s for s in data if s[2] == "C"]
        result = _blend_samples(group_c_only, n_samples=5, is_regression=True, abs_tol=1.0)

        assert len(result) == 0 + len(group_c_only)

    @staticmethod
    def test_regression_blending_math() -> None:
        """Verifies that the target and predictors are a weighted average (remix)."""
        # Two points: [1,1,1] with target 10, and [2,2,2] with target 20
        data = [
            ("A1", "L", "G", np.int8(0), np.int8(1), (3,), 10.0, np.array([1.0, 1.0, 1.0])),
            ("A2", "L", "G", np.int8(0), np.int8(1), (3,), 20.0, np.array([2.0, 2.0, 2.0])),
        ]
        # Check target values fall between 10 and 20
        result = _blend_samples(data, n_samples=10, is_regression=True, abs_tol=15.0, random_state=42)

        for sample in result[: -len(data)]:
            target = sample[-2]
            predictors = sample[-1]
            # Target must be between the original values
            assert 10.0 <= target <= 20.0
            # Predictor values must be between 1.0 and 2.0
            assert np.all((predictors >= 1.0) & (predictors <= 2.0))

    @staticmethod
    def test_global_vs_grouped_count() -> None:
        """Validates your n_per_group logic for total sample satisfaction."""
        data = TestRemixSamples.mock_data_regression()
        n_request = 10

        # Test Grouped - not limited
        _ = _blend_samples(data, n_samples=n_request, use_validation_group=True, is_regression=True, abs_tol=1.0)

        # Test Global
        res_global = _blend_samples(
            data, n_samples=n_request, use_validation_group=False, is_regression=True, abs_tol=1.0
        )
        # Global should hit the target exactly because of your +int(not use_validation_group) logic
        assert len(res_global) == n_request + len(data)

    @staticmethod
    def test_id_uniqueness() -> None:
        """Checks that generated IDs do not collide within the same call."""
        data = TestRemixSamples.mock_data_regression()
        n_request = 20
        result = _blend_samples(data, n_samples=n_request, is_regression=True, abs_tol=1.0, use_validation_group=False)

        ids = [s[0] for s in result]
        assert len(ids) == len(set(ids)), "Duplicate IDs"

    @staticmethod
    def test_random_state_determinism() -> None:
        """
        Validates that providing the same random_state results in identical synthetic samples.
        """
        data = TestRemixSamples.mock_data_regression()
        seed = 666

        # First Run
        res_a = _blend_samples(sample_data=data, n_samples=5, is_regression=True, abs_tol=1.0, random_state=seed)

        # Second Run
        res_b = _blend_samples(sample_data=data, n_samples=5, is_regression=True, abs_tol=1.0, random_state=seed)

        assert len(res_a) == len(res_b)

        for s_a, s_b in zip(res_a, res_b):
            assert s_a[0] == s_b[0]
            assert s_a[-2] == s_b[-2]
            assert np.allclose(s_a[-1], s_b[-1])

    @staticmethod
    def test_classification_neighbor_filtering() -> None:
        """
        Verifies that in classification mode, only samples with the exact same string target are remixed together.
        """
        data = TestRemixSamples.mock_data_classification()

        # We request samples. S1 and S2 should blend. S3 has no neighbors in Group A.
        # S4 and S5 should blend in Group B.
        result = _blend_samples(
            sample_data=data, n_samples=10, is_regression=False, use_validation_group=True, random_state=42
        )

        # Ensure we actually generated samples
        assert len(result) > 0 + len(data)

        for sample in result[: -len(data)]:
            target = sample[-2]
            # The target must remain a string
            assert isinstance(target, str)
            # A 'Class_X' remix should never be influenced by 'Class_Y' predictors
            if "S1" in sample[0] or "S2" in sample[0]:
                assert target == "Class_X"
                # Predictors should be a blend of [1.0] and [1.2], so ~1.1
                assert 1.0 <= sample[-1][0] <= 1.2
            elif "S4" in sample[0] or "S5" in sample[0]:
                assert target == "Class_Z"

    @staticmethod
    def test_classification_global_mode() -> None:
        """
        Verifies classification works across groups when use_validation_group=False.
        """
        data = [
            ("S1", "L", "Group_1", np.int8(0), np.int8(1), (3,), "Oak", np.array([1, 1, 1])),
            ("S2", "L", "Group_2", np.int8(0), np.int8(1), (3,), "Oak", np.array([2, 2, 2])),
        ]

        # With validation groups ON, S1 and S2 are lonely (different groups).
        res_grouped = _blend_samples(data, n_samples=1, is_regression=False, use_validation_group=True)
        assert len(res_grouped) == 0 + len(data)

        # With validation groups OFF, they find each other because they are both 'Oak'.
        res_global = _blend_samples(data, n_samples=1, is_regression=False, use_validation_group=False)
        assert len(res_global) == 1 + len(data)
        assert res_global[0][-2] == "Oak"

    @staticmethod
    def test_determinism_with_strings() -> None:
        """Checks if random_state works consistently for classification data."""
        data = TestRemixSamples.mock_data_classification()
        seed = 99

        res_1 = _blend_samples(data, 5, is_regression=False, random_state=seed)
        res_2 = _blend_samples(data, 5, is_regression=False, random_state=seed)

        for s1, s2 in zip(res_1, res_2):
            assert s1[0] == s2[0]
            assert s1[-2] == s2[-2]
            np.testing.assert_allclose(s1[-1], s2[-1])

    @staticmethod
    def test_blend_samples_generator() -> None:
        """Test functionality of _blend_samples generator"""

        data1 = TestRemixSamples.mock_data_regression()

        blend1 = blend_samples(n_samples=10, is_regression=True)
        res_1 = blend1(data1)
        assert len(res_1) == 12

        data2 = TestRemixSamples.mock_data_classification()

        blend2 = blend_samples(n_samples=10, is_regression=False)
        res_2 = blend2(data2)
        assert len(res_2) == 12


# %% Test - _blend_samples


# TestRemixSamples.test_name_and_metadata()
# TestRemixSamples.test_lonely_group_exclusion()
# TestRemixSamples.test_regression_blending_math()
# TestRemixSamples.test_global_vs_grouped_count()
# TestRemixSamples.test_id_uniqueness()
# TestRemixSamples.test_random_state_determinism()
# TestRemixSamples.test_classification_neighbor_filtering()
# TestRemixSamples.test_classification_global_mode()
# TestRemixSamples.test_determinism_with_strings()
# TestRemixSamples.test_blend_samples_generator()


# %% Test main


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
