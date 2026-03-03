# -*- coding: utf-8 -*-
"""
Tests for spectral image processing and modeling pipeline (SpecPipe)

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# ruff: noqa: I001
# Test
import pytest
import sys  # noqa: E402

# Functions to test
from swectral.pipeline_processor import _dl_val  # noqa: E402


# %% test functions : SpecPipe


def test_dl_val() -> None:
    """Test Data_level validator."""

    assert _dl_val(0) == (0, "image")
    assert _dl_val(1) == (1, "pixel_spec")
    assert _dl_val(2) == (2, "pixel_specs_array")
    assert _dl_val(3) == (3, "pixel_specs_tensor")
    assert _dl_val(4) == (4, "pixel_hyperspecs_tensor")
    assert _dl_val(5) == (5, "image_roi")
    assert _dl_val(6) == (6, "roi_specs")
    assert _dl_val(7) == (7, "spec1d")
    # TODO: assert _dl_val(8) == (8, "model")
    assert _dl_val(9) == (9, "model")
    assert _dl_val("image") == (0, "image")
    assert _dl_val("pixel_spec") == (1, "pixel_spec")
    assert _dl_val("pixel_specs_array") == (2, "pixel_specs_array")
    assert _dl_val("pixel_specs_tensor") == (3, "pixel_specs_tensor")
    assert _dl_val("pixel_hyperspecs_tensor") == (4, "pixel_hyperspecs_tensor")
    assert _dl_val("image_roi") == (5, "image_roi")
    assert _dl_val("roi_specs") == (6, "roi_specs")
    assert _dl_val("spec1d") == (7, "spec1d")
    # TODO: assert _dl_val("model") == (8, "model")
    assert _dl_val("model") == (9, "model")


# %% Tests - SpecPipe

# test_dl_val()


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
