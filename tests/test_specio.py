# -*- coding: utf-8 -*-
"""
Tests for spectral data input, output and validation functions (SpecIO)

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os
import sys  # noqa: E402
import shutil
import tempfile
from pathlib import Path
import dill

# Warning
import warnings

# Test
import pytest
import unittest
from unittest.mock import patch

# Typing
from typing import Annotated, Any, Callable, Optional, Union, Literal
from pydantic import BaseModel

# Basic data
import numpy as np
import pandas as pd
import torch

# Geo
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

# Functions to test
from swectral.specio import (
    _pd_dtype_cond,
    arraylike_validator,
    dataframe_validator,
    df_from_csv,
    df_to_csv,
    dict_value_validator,
    dump_vars,
    envi_roi_coords,
    load_vars,
    names_filter,
    roi_to_envi,
    roi_to_shp,
    search_file,
    shp_roi_coords,
    simple_type_validator,
    lsdir_robust,
    unc_path,
)


# %% test functions : simple_type_validator


# Test the simple_type_validator decorator
class TestSimpleTypeValidator:
    """
    Test functions for simple_type_validator.
    """

    @staticmethod
    def test_basic_types() -> None:
        """Test basic type validation"""

        @simple_type_validator
        def test_func(a: int, b: str, c: float) -> str:
            return f"{a} {b} {c}"

        # Valid calls
        assert test_func(1, "hello", 3.14) == "1 hello 3.14"

        # Invalid calls
        with pytest.raises(TypeError):
            test_func("not_int", "hello", 3.14)

        with pytest.raises(TypeError):
            test_func(1, 123, 3.14)  # b should be str

    @staticmethod
    def test_simple_non_native_types() -> None:
        """Test non-native type validation"""

        @simple_type_validator
        def test_func(a: np.ndarray, b: pd.DataFrame, c: torch.Tensor) -> str:
            return f"{a} {b} {c}"

        arr = np.array([1, 2])
        df = pd.DataFrame([1, 2])
        ten = torch.tensor([1, 2])

        # Valid calls
        assert test_func(arr, df, ten) == "[1 2]    0\n0  1\n1  2 tensor([1, 2])"

        # Invalid calls
        with pytest.raises(TypeError):
            test_func([1, 2], df, ten)

        with pytest.raises(TypeError):
            test_func(arr, arr, ten)

        with pytest.raises(TypeError):
            test_func(arr, df, df)

    @staticmethod
    def test_optional_none() -> None:
        """Test Optional types and None values"""

        @simple_type_validator
        def test_func(a: Optional[int], b: Optional[str] = None) -> str:  # type: ignore[assignment]
            return f"{a} {b}"

        # Valid calls with None
        assert test_func(None, None) == "None None"
        assert test_func(42, None) == "42 None"

        # Invalid calls
        with pytest.raises(TypeError):
            test_func("not_int", None)  # a should be Optional[int]

    @staticmethod
    def test_union_types() -> None:
        """Test Union types"""

        @simple_type_validator
        def test_func(a: Union[int, str], b: Union[float, None]) -> str:
            return f"{a} {b}"

        # Valid calls
        assert test_func(42, 3.14) == "42 3.14"
        assert test_func("hello", None) == "hello None"
        assert test_func(42, None) == "42 None"

        # Invalid calls
        with pytest.raises(TypeError):
            test_func([1, 2, 3], 3.14)  # a should be Union[int, str]

    @staticmethod
    def test_any_type() -> None:
        """Test Any type"""

        @simple_type_validator
        def test_func(a: Any, b: int) -> str:
            return f"{a} {b}"

        # Any type should accept anything except None
        assert test_func("string", 42) == "string 42"
        assert test_func([1, 2, 3], 42) == "[1, 2, 3] 42"
        assert test_func({"key": "value"}, 42) == "{'key': 'value'} 42"

    @staticmethod
    def test_callable_type() -> None:
        """Test Callable type"""

        @simple_type_validator
        def test_func(a: Callable, b: int) -> str:
            return f"{a.__name__} {b}"

        def dummy_func() -> None:
            pass

        # Valid call with callable
        assert test_func(dummy_func, 42) == "dummy_func 42"

        # Invalid call with non-callable
        with pytest.raises(TypeError):
            test_func("not_callable", 42)

    @staticmethod
    def test_class_type() -> None:
        """Test class type"""

        @simple_type_validator
        def test_func(a: type) -> str:
            return f"{a.__name__}"

        class DummyClass:
            def __init__(self) -> None:
                pass

        dummy_instance = DummyClass()

        # Valid call with class / type
        assert test_func(DummyClass) == "DummyClass"

        # Invalid call with non-class / non-type
        with pytest.raises(TypeError):
            test_func("not_type")

        with pytest.raises(TypeError):
            test_func(dummy_instance)

    @staticmethod
    def test_class_object_type() -> None:
        """Test class type"""

        class DummyClass:
            def __init__(self) -> None:
                pass

        @simple_type_validator
        def test_func(a: DummyClass) -> str:
            return f"{a.__class__.__name__}"

        dummy_instance = DummyClass()

        # Valid call with class / type
        assert test_func(dummy_instance) == "DummyClass"

        # Invalid call with non-class / non-type
        with pytest.raises(TypeError):
            test_func("not_specified_class_instance")

        with pytest.raises(TypeError):
            test_func(DummyClass)

    @staticmethod
    def test_list_types() -> None:
        """Test list types"""

        @simple_type_validator
        def test_func(a: list[int], b: list) -> str:  # type: ignore[type-arg]
            return f"{sum(a)} {len(b)}"

        # Valid calls
        assert test_func([1, 2, 3], [4, 5, 6]) == "6 3"
        assert test_func([1, 2, 3], ["a", "b"]) == "6 2"

        # Invalid calls
        with pytest.raises(TypeError):
            test_func([1, "not_int", 3], [4, 5, 6])  # a should be list[int]

    @staticmethod
    def test_tuple_types() -> None:
        """Test tuple types"""

        @simple_type_validator
        def test_func(a: tuple[int, str], b: tuple[float, ...], c: tuple) -> str:  # type: ignore[type-arg]
            return f"{a[0]} {len(b)} {len(c)}"

        # Valid calls
        assert test_func((1, "hello"), (1.0, 2.0, 3.0), (4, 5)) == "1 3 2"
        assert test_func((1, "hello"), (), ()) == "1 0 0"

        # Invalid calls
        with pytest.raises(TypeError):
            test_func((1, 2), (1.0, 2.0), (4, 5))  # a[1] should be str

        with pytest.raises(TypeError):
            test_func((1, "hello"), ("not_float",), (4, 5))  # b should be tuple[float, ...]

    @staticmethod
    def test_set_types() -> None:
        """Test set types"""

        @simple_type_validator
        def test_func(a: set[int], b: set) -> str:  # type: ignore[type-arg]
            return f"{len(a)} {len(b)}"

        # Valid calls
        assert test_func({1, 2, 3}, {4, 5, 6}) == "3 3"
        assert test_func({1, 2, 3}, {"a", "b"}) == "3 2"

        # Invalid calls
        with pytest.raises(TypeError):
            test_func({1, "not_int", 3}, {4, 5, 6})  # a should be set[int]

    @staticmethod
    def test_annotated_types() -> None:
        """Test Annotated types with validators"""

        def positive_validator(x: Union[int, float]) -> None:
            if x <= 0:
                raise ValueError("Must be positive")

        def length_validator(x: str) -> None:
            if len(x) < 3:
                raise ValueError("Too short")

        @simple_type_validator
        def test_func(a: Annotated[int, positive_validator], b: Annotated[str, length_validator]) -> str:
            return f"{a} {b}"

        # Valid calls
        assert test_func(42, "hello") == "42 hello"

        # Invalid calls - type errors
        with pytest.raises(TypeError):
            test_func("not_int", "hello")

        # Invalid calls - validator errors
        with pytest.raises(TypeError, match="Must be positive"):
            test_func(-1, "hello")

        with pytest.raises(TypeError, match="Too short"):
            test_func(42, "a")

    @staticmethod
    def test_literal_types() -> None:
        """Test Literal types with validators"""

        @simple_type_validator
        def test_func(v: Literal["a", "b", "c"]) -> str:  # type: ignore[type-arg]
            return f"{v}"

        # Valid calls
        assert test_func("a") == "a"
        assert test_func("c") == "c"

        # Invalid calls
        with pytest.raises(TypeError):
            test_func("d")

    @staticmethod
    def test_return_type_not_validated() -> None:
        """Test that return type is not validated (only parameters are validated)"""

        @simple_type_validator
        def test_func(a: int) -> str:
            return a  # type: ignore[return-value] # This should return int but annotation says str

        # This should not raise an error because return types aren't validated
        result = test_func(42)
        assert result == 42

    @staticmethod
    def test_no_type_hints() -> None:
        """Test function without type hints"""

        @simple_type_validator
        def test_func(a, b):  # type: ignore[no-untyped-def]
            return f"{a} {b}"

        # Should work fine without type hints
        assert test_func(1, "hello") == "1 hello"
        assert test_func("anything", "goes") == "anything goes"

    @staticmethod
    def test_nested_containers() -> None:
        """Test nested container types"""

        @simple_type_validator
        def test_func(a: list[list[int]], b: tuple[set[str], ...]) -> str:
            return f"{len(a)} {len(b)}"

        # Valid calls
        assert test_func([[1, 2], [3, 4]], ({"a", "b"}, {"c"})) == "2 2"

        # Invalid calls
        with pytest.raises(TypeError):
            test_func([[1, "not_int"], [3, 4]], [{"a", "b"}, {"c"}])  # a should be list[list[int]]

    @staticmethod
    def test_complex_nested_containers() -> None:
        """Test nested container types"""

        @simple_type_validator
        def test_func(
            a: list[tuple[Union[float, np.ndarray], Optional[int]]],
            b: tuple[str, list[int], object, type],
        ) -> str:
            return f"{len(a)} {b[0]} {len(b[1])} {b[2].__class__.__name__} {b[3].__name__}"

        class DumCl:
            def __init__(self) -> None:
                pass

        dum_inst = DumCl()

        # Valid calls
        assert test_func([(1.0, None), (np.array([1]), 1)], ("a", [1], dum_inst, DumCl)) == "2 a 1 DumCl DumCl"

        # Invalid calls
        with pytest.raises(TypeError):
            test_func([(1, None), (np.array([1]), 1)], ("a", [1], dum_inst, DumCl))

        with pytest.raises(TypeError):
            test_func([(1, None), ([1, 2], 1)], ("a", [1], dum_inst, DumCl))

        with pytest.raises(TypeError):
            test_func([(1.0, None), ([1, 2], 1)], ("a", [1.0], dum_inst, DumCl))

        with pytest.raises(TypeError):
            test_func([(1.0, None), ([1, 2], 1)], ("a", [1], 1, DumCl))

        with pytest.raises(TypeError):
            test_func([(1.0, None), ([1, 2], 1)], ("a", [1], DumCl, DumCl))

        with pytest.raises(TypeError):
            test_func([(1.0, None), ([1, 2], 1)], ("a", [1], dum_inst, dum_inst))

    @staticmethod
    def test_type_of_dill_serialized_values() -> None:
        """Test dill serialized values"""
        # Python native types
        a = True
        b = 324
        c = 32.4
        d = "abc"
        e = (True, 324, 32.4, "abc", [324], {"a": 324})

        # Test function
        @simple_type_validator
        def test_func(
            value_a: bool,
            value_b: int,
            value_c: float,
            value_d: str,
            value_e: tuple[bool, int, float, str, list[int], dict[str, int]],
        ) -> None:
            pass

        # Test before dill serialization
        test_func(a, b, c, d, e)

        # Test after dill serialization
        var_dict = {"a": a, "b": b, "c": c, "d": d, "e": e}

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/test_type_serialize.dill"

            with open(file_path, "wb") as f_write:
                dill.dump(var_dict, f_write)

            assert os.path.exists(file_path)
            assert os.path.isfile(file_path)

            with open(file_path, "rb") as f_read:
                data = dill.load(f_read)

            a, b, c, d, e = data["a"], data["b"], data["c"], data["d"], data["e"]

            test_func(a, b, c, d, e)

        # Clear after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @staticmethod
    def test_error_messages() -> None:
        """Test that error messages are informative"""

        @simple_type_validator
        def test_func(value_a: list[int], value_b: str) -> None:
            pass

        try:
            test_func(["not_int"], "abc")
            raise AssertionError("Should have raised TypeError")
        except TypeError as e:
            error_msg = str(e)
            # Should mention both parameter names, types, expected types and values
            assert "value_a" in error_msg
            assert "list[int]" in error_msg
            assert "['not_int']" in error_msg

        try:
            test_func([1, 2, 3], 123)
            raise AssertionError("Should have raised TypeError")
        except TypeError as e:
            error_msg = str(e)
            # Should mention both parameter names, types, expected types and values
            assert "value_b" in error_msg
            assert "str" in error_msg
            assert "123" in error_msg


# %% Test - simple_type_validator


# TestSimpleTypeValidator.test_basic_types()

# TestSimpleTypeValidator.test_simple_non_native_types()

# TestSimpleTypeValidator.test_optional_none()
# TestSimpleTypeValidator.test_union_types()
# TestSimpleTypeValidator.test_any_type()

# TestSimpleTypeValidator.test_callable_type()
# TestSimpleTypeValidator.test_class_type()
# TestSimpleTypeValidator.test_class_object_type()

# TestSimpleTypeValidator.test_list_types()
# TestSimpleTypeValidator.test_tuple_types()
# TestSimpleTypeValidator.test_set_types()

# TestSimpleTypeValidator.test_annotated_types()
# TestSimpleTypeValidator.test_literal_types()

# TestSimpleTypeValidator.test_nested_containers()
# TestSimpleTypeValidator.test_complex_nested_containers()

# TestSimpleTypeValidator.test_error_messages()

# TestSimpleTypeValidator.test_no_type_hints()

# TestSimpleTypeValidator.test_return_type_not_validated()

# TestSimpleTypeValidator.test_type_of_dill_serialized_values()


# %% test functions : arraylike_validator


class TestArraylikeValidator:
    """
    Test functions for arraylike_validator.
    """

    @staticmethod
    def test_basic_conversion() -> None:
        """Test basic conversion from various data types."""
        validator = arraylike_validator()

        # Test with numpy array
        np_array = np.array([1, 2, 3])
        result = validator(np_array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np_array)

        # Test with list
        list_data = [1, 2, 3]
        result = validator(list_data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(list_data))

        # Test with tuple
        tuple_data = (1, 2, 3)
        result = validator(tuple_data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(tuple_data))

        # Inhomogenous list and tuple
        with pytest.raises(ValueError, match="cannot be converted to numpy.ndarray"):
            validator([1, 2, [3, 4]])
        with pytest.raises(ValueError, match="cannot be converted to numpy.ndarray"):
            validator((1, 2, (3, 4)))

        # Test with pandas DataFrame
        df_data = pd.DataFrame([[1, 2], [3, 4]])
        result = validator(df_data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(df_data))

        # Test with torch tensor
        torch_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = validator(torch_tensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0]))

    @staticmethod
    def test_unsupported_type() -> None:
        """Test that unsupported types raise TypeError."""
        validator = arraylike_validator()

        with pytest.raises(TypeError, match="cannot be directly converted to numpy.ndarray"):
            validator({"a": 1, "b": 2, "c": 3})

        with pytest.raises(TypeError, match="cannot be directly converted to numpy.ndarray"):
            validator({1, 2, 3})

        with pytest.raises(TypeError, match="cannot be converted to numpy.ndarray"):
            validator(None)

        with pytest.raises(TypeError, match="cannot be converted to numpy.ndarray"):
            validator(sum)

    @staticmethod
    def test_ndim_validation() -> None:
        """Test ndim validation."""
        # Test 1D validation
        validator_1d = arraylike_validator(ndim=1)
        result = validator_1d([1, 2, 3])
        assert result.ndim == 1

        # Test 2D validation
        validator_2d = arraylike_validator(ndim=2)
        result = validator_2d([[1, 2], [3, 4]])
        assert result.ndim == 2

        # Test invalid ndim
        with pytest.raises(ValueError, match="incompatible ndim"):
            validator_2d([1, 2, 3])

    @staticmethod
    def test_negative_ndim() -> None:
        """Test that negative ndim raises ValueError."""
        with pytest.raises(ValueError, match="ndim cannot be negative"):
            arraylike_validator(ndim=-1)

    @staticmethod
    def test_shape_validation() -> None:
        """Test shape validation."""
        # Test exact shape match
        validator = arraylike_validator(shape=(2, 3))
        data = [[1, 2, 3], [4, 5, 6]]
        result = validator(data)
        assert result.shape == (2, 3)

        # Test variable length dimension (0)
        validator_var = arraylike_validator(shape=(2, 0))
        data1 = [[1, 2], [3, 4]]
        data2 = [[1, 2, 3], [4, 5, 6]]

        result1 = validator_var(data1)
        result2 = validator_var(data2)
        assert result1.shape == (2, 2)
        assert result2.shape == (2, 3)

        # Test shape mismatch
        with pytest.raises(ValueError, match="incompatible shape"):
            validator([[1, 2], [3, 4]])  # Should be (2, 3)

    @staticmethod
    def test_negative_shape_dimension() -> None:
        """Test that negative shape dimensions raise ValueError."""
        with pytest.raises(ValueError, match="shape dimension cannot be negative"):
            arraylike_validator(shape=(2, -1))

    @staticmethod
    def test_dtype_validation() -> None:
        """Test dtype validation."""
        # Native type d_type
        validator_float = arraylike_validator(d_type=np.float64)
        assert validator_float([1.0, 2.0, 3.0]).dtype == np.float64
        with pytest.raises(TypeError):
            validator_float([1, 2, 3])
        with pytest.raises(TypeError):
            validator_float(["1", "2", "3"])

        # Numpy dtype
        validator_float32 = arraylike_validator(d_type=np.float32)
        assert validator_float32(np.array([1.0, 2.0, 3.0], dtype="float32")).dtype == np.float32
        with pytest.raises(TypeError):
            validator_float32(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        with pytest.raises(TypeError):
            validator_float32(["1", "2", "3"])

        # String d_type
        validator_float = arraylike_validator(d_type="float64")
        assert np.all(validator_float([1.0, 2.0, 3.0]) == np.array([1.0, 2.0, 3.0]))
        with pytest.raises(TypeError):
            validator_float([1, 2, 3])

        # D_type of string
        validator_str = arraylike_validator(d_type=str)
        assert validator_str(["1", "2", "3"]).dtype == np.dtype("<U1")
        with pytest.raises(TypeError):
            validator_str([1.0, 2.0, 3.0])

        # Other d_type
        validator_int = arraylike_validator(d_type=int)
        assert validator_int([1, 2, 3]).dtype == np.array([1, 2, 3]).astype(int).dtype
        with pytest.raises(TypeError):
            validator_int([1.0, 2.0, 3.0])

    @staticmethod
    def test_dtype_conversion() -> None:
        """Test dtype conversion."""
        # Test float conversion
        validator_float = arraylike_validator(as_type=float)
        data = [1, 2, 3]
        result = validator_float(data)
        assert result.dtype == float

        # Test int conversion
        validator_int = arraylike_validator(as_type=int)
        data_float = [1.0, 2.0, 3.0]
        result = validator_int(data_float)
        assert result.dtype == int

        # Test string conversion
        validator_str = arraylike_validator(as_type=str)
        data = [1, 2, 3]
        result = validator_str(data)
        assert result.dtype.kind in ["U", "S"]

        # Test boolean conversion
        validator = arraylike_validator(as_type=bool)
        data = [1, 0, 1]
        result = validator(data)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, [True, False, True])

    @staticmethod
    def test_dtype_conversion_failure() -> None:
        """Test that invalid dtype conversion raises ValueError."""
        validator = arraylike_validator(as_type=float)

        # This should fail conversion to float
        with pytest.raises(ValueError, match="Failed to convert array data type"):
            validator(["not", "a", "number"])

    @staticmethod
    def test_combined_constraints() -> None:
        """Test combining multiple constraints."""
        # Test ndim + shape + dtype
        validator = arraylike_validator(ndim=2, shape=(2, 3), as_type=float)
        data = [[1, 2, 3], [4, 5, 6]]
        result = validator(data)

        assert result.ndim == 2
        assert result.shape == (2, 3)
        assert result.dtype == float

    @staticmethod
    def test_none_constraints() -> None:
        """Test that None constraints don't apply validation."""
        validator = arraylike_validator(ndim=None, shape=None, as_type=None)

        # Should accept any valid array-like without constraints
        data_1d = [1, 2, 3]
        data_2d = [[1, 2], [3, 4]]

        result_1d = validator(data_1d)
        result_2d = validator(data_2d)

        assert result_1d.ndim == 1
        assert result_2d.ndim == 2

    @staticmethod
    def test_empty_array() -> None:
        """Test with empty arrays."""
        validator = arraylike_validator(ndim=1, shape=(0,))
        result = validator([])
        assert result.shape == (0,)

        validator_2d = arraylike_validator(ndim=2, shape=(0, 0))
        result = validator_2d([[]])
        assert result.shape == (1, 0)  # Note: [[]] becomes shape (1, 0)

    @staticmethod
    def test_high_dim_array() -> None:
        """Test with complex nested structures."""
        validator = arraylike_validator(ndim=3, shape=(2, 2, 2))
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        result = validator(data)
        assert result.shape == (2, 2, 2)

    @staticmethod
    def test_variable_dimensions() -> None:
        """Test variable dimension with fixed dimensions."""
        validator_mixed = arraylike_validator(shape=(2, 0, 3))
        data = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        result = validator_mixed(data)
        assert result.shape == (2, 2, 3)


# %% Test - arraylike_validator

# TestArraylikeValidator.test_basic_conversion()
# TestArraylikeValidator.test_unsupported_type()

# TestArraylikeValidator.test_ndim_validation()
# TestArraylikeValidator.test_negative_ndim()

# TestArraylikeValidator.test_shape_validation()
# TestArraylikeValidator.test_negative_shape_dimension()

# TestArraylikeValidator.test_dtype_validation()
# TestArraylikeValidator.test_dtype_conversion()
# TestArraylikeValidator.test_dtype_conversion_failure()

# TestArraylikeValidator.test_combined_constraints()

# TestArraylikeValidator.test_empty_array()
# TestArraylikeValidator.test_high_dim_array()
# TestArraylikeValidator.test_variable_dimensions()


# %% test functions : _pd_dtype_cond


class TestPandasDtypeCondition:
    """Test class for _pd_dtype_cond helper function"""

    @staticmethod
    def test_basic_types() -> None:
        """Test _pd_dtype_cond with basic python types"""
        series_int = pd.Series([1, 2, 3])
        series_float = pd.Series([1.1, 2.2, 3.3])
        series_str = pd.Series(["a", "b", "c"])
        series_bool = pd.Series([True, False, True])

        assert _pd_dtype_cond(series_int.dtypes, int)
        assert _pd_dtype_cond(series_float.dtypes, float)
        assert _pd_dtype_cond(series_str.dtypes, str)
        assert _pd_dtype_cond(series_bool.dtypes, bool)

        assert not _pd_dtype_cond(series_int.dtypes, float)
        assert not _pd_dtype_cond(series_str.dtypes, int)

    @staticmethod
    def test_string_types() -> None:
        """Test _pd_dtype_cond with string type specifications"""
        series_int = pd.Series([1, 2, 3], dtype="int64")
        series_float = pd.Series([1.1, 2.2, 3.3], dtype="float64")
        series_str = pd.Series(["a", "b", "c"], dtype="string")
        series_bool = pd.Series([True, False, True], dtype="bool")

        assert _pd_dtype_cond(series_int.dtypes, "int64")
        assert _pd_dtype_cond(series_float.dtypes, "float64")
        assert _pd_dtype_cond(series_str.dtypes, "string")
        assert _pd_dtype_cond(series_bool.dtypes, "bool")

        assert _pd_dtype_cond(series_int.dtypes, "numeric")
        assert _pd_dtype_cond(series_float.dtypes, "numeric")

    @staticmethod
    def test_invalid_type() -> None:
        """Test _pd_dtype_cond with invalid type specifications"""
        series = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="not a valid data type"):
            _pd_dtype_cond(series.dtypes, "invalid_type")

        with pytest.raises(ValueError, match="must be bool, int, float or str"):
            _pd_dtype_cond(series.dtypes, list)


# %% Test - _pd_dtype_cond

# TestPandasDtypeCondition.test_basic_types()
# TestPandasDtypeCondition.test_string_types()
# TestPandasDtypeCondition.test_invalid_type()


# %% test functions : dataframe_validator


class TestDataframeValidator:
    """
    Test class for dataframe_validator function.
    """

    @staticmethod
    def test_basic_dataframe_validation() -> None:
        """Test basic dataframe validation without any constraints."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        # Should pass with no constraints
        validator = dataframe_validator()
        result = validator(df)
        assert result.equals(df)

    @staticmethod
    def test_colname_dtypes_dict_validation() -> None:
        """Test validation with column name and dtype dictionary."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        # Valid case
        validator = dataframe_validator(
            colname_dtypes_dict={
                "int_col": int,
                "float_col": float,
                "str_col": str,
                "bool_col": bool,
            }
        )
        result = validator(df)
        assert result.equals(df)

    @staticmethod
    def test_colname_dtypes_dict_wrong_columns() -> None:
        """Test validation fails with wrong column names."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        validator = dataframe_validator(colname_dtypes_dict={"x": int, "y": float})

        with pytest.raises(ValueError, match="incompatible column names"):
            validator(df)

    @staticmethod
    def test_colname_dtypes_dict_wrong_dtypes() -> None:
        """Test validation fails with wrong data types"""
        df = pd.DataFrame({"int_col": ["a", "b", "c"]})  # String instead of int

        validator = dataframe_validator(colname_dtypes_dict={"int_col": int})

        with pytest.raises(TypeError, match="incompatible column data type"):
            validator(df)

    @staticmethod
    def test_global_dtype_validation() -> None:
        """Test validation with global dtype constraint"""
        df_numeric = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_mixed = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        # Valid case - all numeric
        validator = dataframe_validator(dtype="numeric")
        result = validator(df_numeric)
        assert result.equals(df_numeric)

        # Invalid case - mixed types
        with pytest.raises(TypeError, match="incompatible data types"):
            validator(df_mixed)

    @staticmethod
    def test_shape_validation() -> None:
        """Test validation with shape constraint"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Valid case
        validator = dataframe_validator(shape=(3, 2))
        result = validator(df)
        assert result.equals(df)

        # Invalid case
        validator_wrong = dataframe_validator(shape=(2, 2))
        with pytest.raises(ValueError, match="incompatible shape"):
            validator_wrong(df)

    @staticmethod
    def test_nrow_validation() -> None:
        """Test validation with nrow constraint"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Valid case
        validator = dataframe_validator(nrow=3)
        result = validator(df)
        assert result.equals(df)

        # Invalid case
        validator_wrong = dataframe_validator(nrow=2)
        with pytest.raises(ValueError, match="incompatible number of rows"):
            validator_wrong(df)

    @staticmethod
    def test_ncol_validation() -> None:
        """Test validation with ncol constraint"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Valid case
        validator = dataframe_validator(ncol=2)
        result = validator(df)
        assert result.equals(df)

        # Invalid case
        validator_wrong = dataframe_validator(ncol=3)
        with pytest.raises(ValueError, match="incompatible number of columns"):
            validator_wrong(df)

    @staticmethod
    def test_index_validation() -> None:
        """Test validation with index constraint"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[0, 1, 2])
        df_custom_index = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "y", "z"])

        # Valid case with default index
        validator = dataframe_validator(index=[0, 1, 2])
        result = validator(df)
        assert result.equals(df)

        # Valid case with custom index
        validator_custom = dataframe_validator(index=["x", "y", "z"])
        result_custom = validator_custom(df_custom_index)
        assert result_custom.equals(df_custom_index)

        # Invalid case
        validator_wrong = dataframe_validator(index=[10, 20, 30])
        with pytest.raises(TypeError, match="incompatible index"):
            validator_wrong(df)

    @staticmethod
    def test_redundant_dtype_specification() -> None:
        """Test that redundant dtype specification raises error"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with pytest.raises(ValueError, match="Redundant dtype specification"):
            validator = dataframe_validator(colname_dtypes_dict={"a": int, "b": int}, dtype="numeric")
            validator(df)

    @staticmethod
    def test_negative_shape_dimension() -> None:
        """Test that negative shape dimensions raise error"""
        with pytest.raises(ValueError, match="shape dimension must be positive"):
            dataframe_validator(shape=(-1, 2))

    @staticmethod
    def test_negative_nrow() -> None:
        """Test that negative nrow raises error"""
        with pytest.raises(ValueError, match="nrow must be positive"):
            dataframe_validator(nrow=-5)

    @staticmethod
    def test_negative_ncol() -> None:
        """Test that negative ncol raises error"""
        with pytest.raises(ValueError, match="ncol must be positive"):
            dataframe_validator(ncol=-3)

    @staticmethod
    def test_nested_type_validation() -> None:
        """Test validation with nested types using pydantic models"""

        class NestedModel(BaseModel):
            value: int
            name: str

        df = pd.DataFrame(
            {
                "nested_col": [
                    NestedModel(value=1, name="a"),
                    NestedModel(value=2, name="b"),
                    NestedModel(value=3, name="c"),
                ]
            }
        )

        # Valid case with nested type
        validator = dataframe_validator(colname_dtypes_dict={"nested_col": NestedModel})
        result = validator(df)
        assert result.equals(df)

        # Invalid case with wrong nested type
        df_invalid = pd.DataFrame(
            {
                "nested_col": [
                    {"value": "not_int", "name": "a"},  # Wrong type for value
                    NestedModel(value=2, name="b"),
                ]
            }
        )

        with pytest.raises(ValueError, match="incompatible nested data type"):
            validator(df_invalid)

    @staticmethod
    def test_string_dtype_aliases() -> None:
        """Test validation with string dtype aliases"""
        df_int = pd.DataFrame({"col": [1, 2, 3]})
        df_float = pd.DataFrame({"col": [1.1, 2.2, 3.3]})
        df_str = pd.DataFrame({"col": ["a", "b", "c"]})
        df_bool = pd.DataFrame({"col": [True, False, True]})

        # Test various string representations
        validator_int = dataframe_validator(colname_dtypes_dict={"col": "int64"})
        validator_float = dataframe_validator(colname_dtypes_dict={"col": "float64"})
        validator_str = dataframe_validator(colname_dtypes_dict={"col": "string"})
        validator_bool = dataframe_validator(colname_dtypes_dict={"col": "bool"})

        assert validator_int(df_int).equals(df_int)
        assert validator_float(df_float).equals(df_float)
        assert validator_str(df_str).equals(df_str)
        assert validator_bool(df_bool).equals(df_bool)

    @staticmethod
    def test_combined_constraints() -> None:
        """Test validation with multiple combined constraints"""
        df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "score": [95.5, 88.0, 92.3]},
            index=[10, 20, 30],
        )

        # Combined validation
        validator = dataframe_validator(
            colname_dtypes_dict={"id": int, "name": str, "score": float},
            nrow=3,
            ncol=3,
            index=[10, 20, 30],
        )

        result = validator(df)
        assert result.equals(df)


# %% Test - dataframe_validator

# TestDataframeValidator.test_basic_dataframe_validation()

# TestDataframeValidator.test_colname_dtypes_dict_validation()
# TestDataframeValidator.test_colname_dtypes_dict_wrong_columns()
# TestDataframeValidator.test_colname_dtypes_dict_wrong_dtypes()
# TestDataframeValidator.test_global_dtype_validation()

# TestDataframeValidator.test_shape_validation()
# TestDataframeValidator.test_nrow_validation()
# TestDataframeValidator.test_ncol_validation()

# TestDataframeValidator.test_index_validation()

# TestDataframeValidator.test_redundant_dtype_specification()

# TestDataframeValidator.test_negative_shape_dimension()
# TestDataframeValidator.test_negative_nrow()
# TestDataframeValidator.test_negative_ncol()
# TestDataframeValidator.test_nested_type_validation()

# TestDataframeValidator.test_string_dtype_aliases()
# TestDataframeValidator.test_combined_constraints()


# %% test functions : dict_value_validator


class TestDictValueValidator:
    """Test cases for dict_value_validator function"""

    @staticmethod
    def test_basic_value_type_validation() -> None:
        """Test basic value type validation with value_type_list"""
        # Via value_type_list
        validator = dict_value_validator(value_type_list=[int, str, float])
        test_dict = {1: 100, 2: "hello", 3: 3.14}

        result = validator(test_dict)
        assert result == test_dict

        # Via value_type_dict
        validator = dict_value_validator(value_type_dict={2: str, 3: float, 1: int})
        test_dict = {1: 100, 2: "hello", 3: 3.14}

        result = validator(test_dict)
        assert result == test_dict

    @staticmethod
    def test_basic_key_type_validation() -> None:
        """Test basic key type validation with key_type_list"""
        # With value_type_list
        validator = dict_value_validator(value_type_list=[int, str, float], key_type_list=[int, int, str])
        test_dict = {1: 100, 2: "hello", "3": 3.14}

        result = validator(test_dict)
        assert result == test_dict

        # Compatible with value_type_dict
        validator = dict_value_validator(value_type_dict={1: int, "2": str, 3: float}, key_type_list=[int, str, int])
        test_dict = {1: 100, "2": "hello", 3: 3.14}

        result = validator(test_dict)
        assert result == test_dict

    @staticmethod
    def test_value_type_validation_wrong_types() -> None:
        """Test value type validation with incorrect types"""
        validator = dict_value_validator(value_type_list=[int, str, float])
        test_dict = {1: "wrong", 2: 100, 3: 3.14}  # First value should be int, not str

        with pytest.raises(TypeError):
            validator(test_dict)

    @staticmethod
    def test_value_type_validation_length_mismatch() -> None:
        """Test value type validation with length mismatch"""
        validator = dict_value_validator(value_type_list=[int, str])
        test_dict = {1: 100, 2: "hello", 3: 3.14}  # Extra key-value pair

        with pytest.raises(ValueError, match="Inconsistent length"):
            validator(test_dict)

    @staticmethod
    def test_key_type_validation() -> None:
        """Test key type validation with key_type_list"""
        validator = dict_value_validator(value_type_list=[int, str], key_type_list=[str, int])
        test_dict = {"a": 100, 2: "hello"}

        # Should not raise any exception
        result = validator(test_dict)
        assert result == test_dict

    @staticmethod
    def test_key_type_validation_wrong_types() -> None:
        """Test key type validation with incorrect key types"""
        validator = dict_value_validator(value_type_list=[int, str], key_type_list=[str, int])
        test_dict = {1: 100, 2: "hello"}  # First key should be str, not int

        with pytest.raises(TypeError):
            validator(test_dict)

    @staticmethod
    def test_key_type_validation_length_mismatch() -> None:
        """Test key type validation with length mismatch"""
        validator = dict_value_validator(value_type_list=[int, str], key_type_list=[str, int])
        test_dict = {"a": 100, 2: "hello", 3: 3.14}  # Extra key-value pair

        with pytest.raises(ValueError, match="Inconsistent length"):
            validator(test_dict)

    @staticmethod
    def test_value_type_dict_validation() -> None:
        """Test value type validation with value_type_dict"""
        validator = dict_value_validator(value_type_dict={"a": int, "b": str, "c": float})
        test_dict = {"a": 100, "b": "hello", "c": 3.14}

        # Should not raise any exception
        result = validator(test_dict)
        assert result == test_dict

    @staticmethod
    def test_value_type_dict_validation_wrong_types() -> None:
        """Test value type validation with incorrect types using value_type_dict"""
        validator = dict_value_validator(value_type_dict={"a": int, "b": str, "c": float})
        test_dict = {"a": "wrong", "b": "hello", "c": 3.14}  # 'a' value should be int, not str

        with pytest.raises(TypeError):
            validator(test_dict)

    @staticmethod
    def test_value_type_dict_validation_missing_keys() -> None:
        """Test value type validation with missing keys"""
        validator = dict_value_validator(value_type_dict={"a": int, "b": str, "c": float})
        test_dict = {"a": 100, "b": "hello"}  # Missing key 'c'

        with pytest.raises(ValueError, match="Inconsistent keys"):
            validator(test_dict)

    @staticmethod
    def test_value_type_dict_validation_extra_keys() -> None:
        """Test value type validation with extra keys"""
        validator = dict_value_validator(value_type_dict={"a": int, "b": str})
        test_dict = {"a": 100, "b": "hello", "c": 3.14}  # Extra key 'c'

        with pytest.raises(ValueError, match="Inconsistent keys"):
            validator(test_dict)

    @staticmethod
    def test_key_type_with_value_type_dict() -> None:
        """Test key type validation combined with value_type_dict"""
        validator = dict_value_validator(key_type_list=[str, int], value_type_dict={"a": int, 2: str})
        test_dict = {"a": 100, 2: "hello"}

        # Should not raise any exception
        result = validator(test_dict)
        assert result == test_dict

    @staticmethod
    def test_key_type_with_value_type_dict_length_mismatch() -> None:
        """Test key type validation with value_type_dict length mismatch"""
        with pytest.raises(ValueError, match="Inconsistent length"):
            _ = dict_value_validator(
                key_type_list=[str],
                value_type_dict={"a": int, "b": str},  # Different lengths
            )

    @staticmethod
    def test_conflicting_parameters_value_type_list_and_dict() -> None:
        """Test that specifying both value_type_list and value_type_dict raises error"""
        with pytest.raises(
            ValueError,
            match="Simultaneously specifying value_type_list and value_type_dict is not allowed",
        ):
            dict_value_validator(value_type_list=[int, str], value_type_dict={"a": int, "b": str})

    @staticmethod
    def test_empty_dictionary() -> None:
        """Test validation with empty dictionary"""
        validator = dict_value_validator(value_type_list=[])
        test_dict: dict = {}

        result = validator(test_dict)
        assert result == test_dict

    @staticmethod
    def test_none_values() -> None:
        """Test validation with None values when allowed by type"""
        validator = dict_value_validator(value_type_list=[Optional[int], int])
        test_dict = {1: None, 2: 100}

        result = validator(test_dict)
        assert result == test_dict

    @staticmethod
    def test_complex_types() -> None:
        """Test validation with complex types"""
        validator = dict_value_validator(
            value_type_list=[
                list[int],
                dict[str, float],
                tuple[int, str, tuple[int, list[int], str, float]],
            ]
        )

        test_dict = {1: [1, 2, 3], 2: {"a": 1.1, "b": 2.2}, 3: (1, "a", (1, [2, 3], "4", 5.0))}

        result = validator(test_dict)
        assert result == test_dict

    @staticmethod
    def test_custom_classes() -> None:
        """Test validation with custom classes"""

        class CustomClass:
            pass

        class AnotherClass:
            pass

        validator = dict_value_validator(value_type_list=[CustomClass, AnotherClass])
        test_dict = {1: CustomClass(), 2: AnotherClass()}

        result = validator(test_dict)
        assert result == test_dict

    @staticmethod
    def test_inheritance_types() -> None:
        """Test validation with inheritance (subclasses should be accepted)"""
        validator = dict_value_validator(value_type_list=[object, Exception])
        test_dict = {1: "string", 2: ValueError("test")}  # str is object, ValueError is Exception

        result = validator(test_dict)
        assert result == test_dict

    @staticmethod
    def test_no_parameters() -> None:
        """Test validator with no parameters (should accept any dict)"""
        validator = dict_value_validator()
        test_dict = {1: "anything", "key": 123, 3.14: [1, 2, 3]}

        result = validator(test_dict)
        assert result == test_dict


# %% Test - dict_value_validator

# TestDictValueValidator.test_basic_value_type_validation()
# TestDictValueValidator.test_basic_key_type_validation()
# TestDictValueValidator.test_value_type_validation_wrong_types()
# TestDictValueValidator.test_value_type_validation_length_mismatch()
# TestDictValueValidator.test_key_type_validation()
# TestDictValueValidator.test_key_type_validation_wrong_types()
# TestDictValueValidator.test_key_type_validation_length_mismatch()
# TestDictValueValidator.test_value_type_dict_validation()
# TestDictValueValidator.test_value_type_dict_validation_wrong_types()
# TestDictValueValidator.test_value_type_dict_validation_missing_keys()
# TestDictValueValidator.test_value_type_dict_validation_extra_keys()
# TestDictValueValidator.test_key_type_with_value_type_dict()
# TestDictValueValidator.test_key_type_with_value_type_dict_length_mismatch()
# TestDictValueValidator.test_conflicting_parameters_value_type_list_and_dict()
# TestDictValueValidator.test_empty_dictionary()
# TestDictValueValidator.test_none_values()
# TestDictValueValidator.test_complex_types()
# TestDictValueValidator.test_custom_classes()
# TestDictValueValidator.test_inheritance_types()
# TestDictValueValidator.test_no_parameters()


# %% test functions : search_file


class TestSearchFile(unittest.TestCase):
    test_dir: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a temporary directory structure for testing"""
        cls.test_dir = tempfile.mkdtemp()

        # Create test directory structure
        os.makedirs(os.path.join(cls.test_dir, "subdir1", "nested"), exist_ok=True)
        os.makedirs(os.path.join(cls.test_dir, "subdir2"), exist_ok=True)

        # Create test files
        test_files = [
            "file1.txt",
            "file2.txt",
            "file11.txt",
            "file3.csv",
            "image1.png",
            "image2.jpg",
            "config.ini",
            "readme.md",
            "backup.txt",
            "temp.txt",
            "extf.abc",
            "extf.abb",
            "extf.acc",
        ]

        for file_path in test_files:
            full_path = os.path.join(cls.test_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(f"Content of {file_path}")

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary directory"""
        shutil.rmtree(cls.test_dir)

    @staticmethod
    def test_basic_search_without_wildcards() -> None:
        """Test basic search without wildcards"""
        results = search_file(TestSearchFile.test_dir, "file1.")
        assert len(results) == 1
        assert any("file1.txt" in result for result in results)

    @staticmethod
    def test_search_with_wildcard_pattern() -> None:
        """Test search with wildcard pattern"""
        results = search_file(TestSearchFile.test_dir, "file*.txt")
        assert len(results) == 3

    @staticmethod
    def test_search_with_question_mark_wildcard() -> None:
        """Test search with question mark wildcard"""
        results = search_file(TestSearchFile.test_dir, "file?.txt")
        assert len(results) == 2

    @staticmethod
    def test_search_with_bracket_wildcards() -> None:
        """Test search with bracket wildcards"""
        results = search_file(TestSearchFile.test_dir, "file[12].*")
        assert len(results) == 2

    @staticmethod
    def test_search_with_negated_bracket_wildcards() -> None:
        """Test search with negated bracket wildcards"""
        results = search_file(TestSearchFile.test_dir, "file[!2-6].*")
        assert len(results) == 1

    @staticmethod
    def test_search_with_end_with_filter() -> None:
        """Test search with end_with filter"""
        results = search_file(TestSearchFile.test_dir, "file*", end_with=".csv")
        assert len(results) == 1
        assert all(result.endswith(".csv") for result in results)

    @staticmethod
    def test_search_with_specific_extension() -> None:
        """Test search for specific file extension"""
        results = search_file(TestSearchFile.test_dir, "*.png")
        assert len(results) == 1
        assert results[0].endswith(".png")

    @staticmethod
    def test_search_with_extension_pattern() -> None:
        """Test search with bracket wildcards"""
        results1 = search_file(TestSearchFile.test_dir, "*.a*")
        results2 = search_file(TestSearchFile.test_dir, "*.a*c")
        results3 = search_file(TestSearchFile.test_dir, "*.*b")
        assert len(results1) == 3
        assert len(results2) == 2
        assert len(results3) == 1

    @staticmethod
    def test_search_with_exclude_list() -> None:
        """Test search with exclude list"""
        results1 = search_file(TestSearchFile.test_dir, "*.txt")
        results2 = search_file(TestSearchFile.test_dir, "*.txt", exclude_list=[])
        results3 = search_file(TestSearchFile.test_dir, "*.txt", exclude_list=["temp", "backup"])
        assert len(results1) == 5
        assert len(results2) == 5
        assert len(results3) == 3
        assert not any("temp" in result or "backup" in result for result in results3)

    @staticmethod
    def test_empty_directory() -> None:
        """Test search in empty directory"""
        empty_dir = tempfile.mkdtemp()

        results = search_file(empty_dir, "*")
        assert len(results) == 0

        shutil.rmtree(empty_dir)

    @staticmethod
    def test_search_nonexistent_pattern() -> None:
        """Test search for non-existent pattern"""
        results = search_file(TestSearchFile.test_dir, "nonexistent*")
        assert len(results) == 0

    @staticmethod
    def test_path_format_consistency() -> None:
        """Test that paths are returned with consistent forward slash format"""
        results = search_file(TestSearchFile.test_dir, "file1.txt")
        assert len(results) == 1
        assert "\\" not in results[0]


# %% Test - search_file

# TestSearchFile.setUpClass()

# TestSearchFile.test_basic_search_without_wildcards()

# TestSearchFile.test_search_with_wildcard_pattern()
# TestSearchFile.test_search_with_question_mark_wildcard()
# TestSearchFile.test_search_with_bracket_wildcards()
# TestSearchFile.test_search_with_negated_bracket_wildcards()

# TestSearchFile.test_search_with_end_with_filter()
# TestSearchFile.test_search_with_specific_extension()
# TestSearchFile.test_search_with_extension_pattern()
# TestSearchFile.test_search_with_exclude_list()

# TestSearchFile.test_empty_directory()
# TestSearchFile.test_search_nonexistent_pattern()

# TestSearchFile.test_path_format_consistency()

# TestSearchFile.tearDownClass()


# %% test functions : names_filter


class TestNamesFilter(unittest.TestCase):
    @staticmethod
    def test_list_basic_filtering() -> None:
        """Test basic filtering with list input"""
        names = ["file1.txt", "file2.jpg", "data.csv", "image.png"]
        pattern = "*.txt"

        selected, removed = names_filter(names, pattern)

        assert selected == ["file1.txt"]
        assert removed == ["file2.jpg", "data.csv", "image.png"]

    @staticmethod
    def test_list_multiple_matches() -> None:
        """Test filtering with multiple matches"""
        names = ["test.py", "test.txt", "data.py", "config.ini"]
        pattern = "*.py"

        selected, removed = names_filter(names, pattern)

        assert selected == ["test.py", "data.py"]
        assert removed == ["test.txt", "config.ini"]

    @staticmethod
    def test_list_no_matches() -> None:
        """Test filtering when no names match pattern"""
        names = ["file1.txt", "file2.jpg", "data.csv"]
        pattern = "*.pdf"

        selected, removed = names_filter(names, pattern)

        assert selected == []
        assert removed == names

    @staticmethod
    def test_list_all_matches() -> None:
        """Test filtering when all names match pattern"""
        names = ["image1.png", "image2.png", "image3.png"]
        pattern = "*.png"

        selected, removed = names_filter(names, pattern)

        assert selected == names
        assert removed == []

    @staticmethod
    def test_list_return_ids() -> None:
        """Test filtering with return_ids=True"""
        names = ["file1.txt", "file2.jpg", "data.csv", "image.png"]
        pattern = "*.txt"

        selected_ids, removed_ids = names_filter(names, pattern, return_ids=True)

        assert selected_ids == [0]
        assert removed_ids == [1, 2, 3]

    @staticmethod
    def test_dict_key_filtering() -> None:
        """Test filtering with dictionary input (key-based filtering)"""
        names = {
            "file1.txt": "/path/to/file1.txt",
            "file2.jpg": "/path/to/file2.jpg",
            "data.csv": "/path/to/data.csv",
        }
        pattern = "*.txt"

        selected, removed = names_filter(names, pattern)

        assert selected == {"file1.txt": "/path/to/file1.txt"}
        assert removed == {"file2.jpg": "/path/to/file2.jpg", "data.csv": "/path/to/data.csv"}

    @staticmethod
    def test_dict_value_filtering() -> None:
        """Test filtering with dictionary input (value-based filtering)"""
        names = {"doc1": "document.txt", "img1": "image.jpg", "data1": "data.csv"}
        pattern = "*.txt"

        selected, removed = names_filter(names, pattern, dict_value_as_filename=True)

        assert selected == {"doc1": "document.txt"}
        assert removed == {"img1": "image.jpg", "data1": "data.csv"}

    @staticmethod
    def test_dict_complex_pattern() -> None:
        """Test filtering with complex patterns"""
        names = {
            "file1_2023.txt": "content1",
            "file2_2023.jpg": "content2",
            "data_2022.csv": "content3",
        }
        pattern = "*_2023.*"

        selected, removed = names_filter(names, pattern)

        assert selected == {"file1_2023.txt": "content1", "file2_2023.jpg": "content2"}
        assert removed == {"data_2022.csv": "content3"}

    @staticmethod
    def test_empty_list() -> None:
        """Test filtering with empty list"""
        names: list = []
        pattern = "*.txt"

        selected, removed = names_filter(names, pattern)

        assert selected == []
        assert removed == []

    @staticmethod
    def test_empty_dict() -> None:
        """Test filtering with empty dictionary"""
        names: dict = {}
        pattern = "*.txt"

        selected, removed = names_filter(names, pattern)

        assert selected == {}
        assert removed == {}

    @staticmethod
    def test_question_mark_pattern() -> None:
        """Test pattern with question mark wildcard"""
        names = ["file1.txt", "file2.txt", "file10.txt", "data.csv"]
        pattern = "file?.txt"

        selected, removed = names_filter(names, pattern)

        assert selected == ["file1.txt", "file2.txt"]
        assert removed == ["file10.txt", "data.csv"]

    @staticmethod
    def test_char_range_pattern() -> None:
        """Test pattern with character range"""
        names = ["file1.txt", "file2.txt", "filea.txt", "filez.txt"]
        pattern = "file[0-9].txt"

        selected, removed = names_filter(names, pattern)

        assert selected == ["file1.txt", "file2.txt"]
        assert removed == ["filea.txt", "filez.txt"]


# %% Test - names_filter

# TestNamesFilter.test_list_basic_filtering()
# TestNamesFilter.test_list_multiple_matches()
# TestNamesFilter.test_list_no_matches()
# TestNamesFilter.test_list_all_matches()
# TestNamesFilter.test_list_return_ids()
# TestNamesFilter.test_empty_list()

# TestNamesFilter.test_dict_key_filtering()
# TestNamesFilter.test_dict_value_filtering()
# TestNamesFilter.test_dict_complex_pattern()
# TestNamesFilter.test_empty_dict()

# TestNamesFilter.test_question_mark_pattern()
# TestNamesFilter.test_char_range_pattern()


# %% test functions : envi_roi_coords


class TestEnviROICoords:
    @staticmethod
    def create_test_roi_file(content: str) -> str:
        """Helper method to create temporary ROI files for testing"""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
        temp_file.write(content)
        temp_file.close()
        return temp_file.name

    @staticmethod
    def test_single_polygon_roi() -> None:
        """Test reading single polygon ROI"""
        # fmt: off
        roi_content = """<?xml version="1.0" encoding="UTF-8"?>
<RegionsOfInterest version="1.1">
  <Region name="TestROI" color="255,0,0">
    <GeometryDef>
      <CoordSysStr>none</CoordSysStr>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates>100.0 200.0 300.0 400.0 500.0 600.0 100.0 200.0</Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
    </GeometryDef>
  </Region>
</RegionsOfInterest>"""  # noqa
        # fmt: on

        roi_path = TestEnviROICoords.create_test_roi_file(roi_content)

        result = envi_roi_coords(roi_path)
        assert len(result) == 1
        assert result[0]["name"] == "TestROI"
        assert result[0]["type"] == "Polygon"
        assert len(result[0]["coordinates"]) == 1
        assert result[0]["coordinates"][0] == [
            (100.0, 200.0),
            (300.0, 400.0),
            (500.0, 600.0),
            (100.0, 200.0),
        ]

        if os.path.exists(roi_path):
            os.remove(roi_path)

    @staticmethod
    def test_multi_polygon_roi() -> None:
        """Test reading multi-polygon ROI"""
        # fmt: off
        roi_content = """<?xml version="1.0" encoding="UTF-8"?>
<RegionsOfInterest version="1.1">
  <Region name="MultiROI" color="255,0,0">
    <GeometryDef>
      <CoordSysStr>none</CoordSysStr>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates>10.0 20.0 30.0 40.0 50.0 60.0 10.0 20.0</Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates>50.0 60.0 70.0 80.0 90.0 100.0 50.0 60.0</Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
    </GeometryDef>
  </Region>
</RegionsOfInterest>"""  # noqa
        # fmt: on

        roi_path = TestEnviROICoords.create_test_roi_file(roi_content)

        result = envi_roi_coords(roi_path)
        assert len(result) == 1
        assert result[0]["name"] == "MultiROI"
        assert result[0]["type"] == "MultiPolygon"
        assert len(result[0]["coordinates"]) == 2
        assert result[0]["coordinates"][0] == [
            (10.0, 20.0),
            (30.0, 40.0),
            (50.0, 60.0),
            (10.0, 20.0),
        ]
        assert result[0]["coordinates"][1] == [
            (50.0, 60.0),
            (70.0, 80.0),
            (90.0, 100.0),
            (50.0, 60.0),
        ]

        if os.path.exists(roi_path):
            os.remove(roi_path)

    @staticmethod
    def test_multiple_regions() -> None:
        """Test reading multiple ROI regions"""
        # fmt: off
        roi_content = """<?xml version="1.0" encoding="UTF-8"?>
<RegionsOfInterest version="1.1">
  <Region name="ROI #1" color="255,0,0">
    <GeometryDef>
      <CoordSysStr>none</CoordSysStr>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates>10.0 20.0 30.0 40.0 50.0 60.0 10.0 20.0</Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
    </GeometryDef>
  </Region>
  <Region name="ROI #2" color="0,255,0">
    <GeometryDef>
      <CoordSysStr>none</CoordSysStr>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates>50.0 60.0 70.0 80.0 90.0 100.0 50.0 60.0</Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
    </GeometryDef>
  </Region>
</RegionsOfInterest>"""  # noqa
        # fmt: on

        roi_path = TestEnviROICoords.create_test_roi_file(roi_content)

        result = envi_roi_coords(roi_path)
        assert len(result) == 2
        assert result[0]["name"] == "ROI #1"
        assert result[1]["name"] == "ROI #2"
        assert result[0]["coordinates"][0] == [
            (10.0, 20.0),
            (30.0, 40.0),
            (50.0, 60.0),
            (10.0, 20.0),
        ]
        assert result[1]["coordinates"][0] == [
            (50.0, 60.0),
            (70.0, 80.0),
            (90.0, 100.0),
            (50.0, 60.0),
        ]

        if os.path.exists(roi_path):
            os.remove(roi_path)

    @staticmethod
    def test_scientific_notation_coordinates() -> None:
        """Test reading coordinates in scientific notation"""
        # fmt: off
        roi_content = """<?xml version="1.0" encoding="UTF-8"?>
<RegionsOfInterest version="1.1">
  <Region name="SciNotROI" color="255,0,0">
    <GeometryDef>
      <CoordSysStr>none</CoordSysStr>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates>1.5e2 2.5e-1 -3.5e+3 4.5E2 3.5E+2 7.9E-2 1.5e2 2.5e-1</Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
    </GeometryDef>
  </Region>
</RegionsOfInterest>"""  # noqa
        # fmt: on

        roi_path = TestEnviROICoords.create_test_roi_file(roi_content)

        result = envi_roi_coords(roi_path)
        assert result[0]["coordinates"][0] == [
            (150.0, 0.25),
            (-3500.0, 450.0),
            (350, 0.079),
            (150.0, 0.25),
        ]

        if os.path.exists(roi_path):
            os.remove(roi_path)

    @staticmethod
    def test_decimal_coordinates() -> None:
        """Test reading decimal coordinates"""
        # fmt: off
        roi_content = """<?xml version="1.0" encoding="UTF-8"?>
<RegionsOfInterest version="1.1">
  <Region name="DeciNotROI" color="255,0,0">
    <GeometryDef>
      <CoordSysStr>none</CoordSysStr>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates>123.456 -789.012 345.678 -901.234 345.678 901.234 123.456 -789.012</Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
    </GeometryDef>
  </Region>
</RegionsOfInterest>"""  # noqa
        # fmt: on

        roi_path = TestEnviROICoords.create_test_roi_file(roi_content)

        result = envi_roi_coords(roi_path)
        assert result[0]["coordinates"][0] == [
            (123.456, -789.012),
            (345.678, -901.234),
            (345.678, 901.234),
            (123.456, -789.012),
        ]

        if os.path.exists(roi_path):
            os.remove(roi_path)

    @staticmethod
    def test_no_polygon_error() -> None:
        """Test that ValueError is raised when no polygon is found"""
        roi_content = """<?xml version="1.0" ?>
<RegionsOfInterest version="1.1">
  <Region name="NoPolyROI" color="255,0,0">
    <!-- No Polygon elements -->
  </Region>
</RegionsOfInterest>"""

        roi_path = TestEnviROICoords.create_test_roi_file(roi_content)

        with pytest.raises(ValueError, match="no polygon for ROI NoPolyROI is found"):
            envi_roi_coords(roi_path)

        if os.path.exists(roi_path):
            os.remove(roi_path)

    @staticmethod
    def test_file_not_found() -> None:
        """Test handling of non-existent file"""
        with pytest.raises(FileNotFoundError):
            envi_roi_coords("non_existent_file.xml")

    @staticmethod
    def test_invalid_xml() -> None:
        """Test handling of invalid XML"""
        roi_content = """Invalid XML content"""

        roi_path = TestEnviROICoords.create_test_roi_file(roi_content)

        with pytest.raises(ValueError, match="No ROI is found"):
            envi_roi_coords(roi_path)

        if os.path.exists(roi_path):
            os.remove(roi_path)

    @staticmethod
    def test_empty_coordinates() -> None:
        """Test handling of empty coordinates"""
        roi_content = """<?xml version="1.0" ?>
<RegionsOfInterest version="1.1">
  <Region name="EmptyROI" color="255,0,0">
    <GeometryDef>
      <CoordSysStr>none</CoordSysStr>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates></Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
    </GeometryDef>
  </Region>
</RegionsOfInterest>"""  # noqa

        roi_path = TestEnviROICoords.create_test_roi_file(roi_content)

        result = envi_roi_coords(roi_path)
        assert result[0]["coordinates"][0] == []

        if os.path.exists(roi_path):
            os.remove(roi_path)

    @staticmethod
    def test_whitespace_handling() -> None:
        """Test handling of various whitespace in coordinates"""
        roi_content = """<?xml version="1.0" ?>
<RegionsOfInterest version="1.1">
  <Region name="ROI #1" color="255,0,0">
    <GeometryDef>
      <CoordSysStr>none</CoordSysStr>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates>   100.0   200.0   300.0    400.0     500.0  600.0   100.0   200.0   </Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
    </GeometryDef>
  </Region>
</RegionsOfInterest>"""  # noqa

        roi_path = TestEnviROICoords.create_test_roi_file(roi_content)

        result = envi_roi_coords(roi_path)

        assert result[0]["coordinates"][0] == [
            (100.0, 200.0),
            (300.0, 400.0),
            (500.0, 600.0),
            (100.0, 200.0),
        ]

        if os.path.exists(roi_path):
            os.remove(roi_path)


# %% Test - envi_roi_coords

# TestEnviROICoords.test_single_polygon_roi()
# TestEnviROICoords.test_multi_polygon_roi()
# TestEnviROICoords.test_multiple_regions()

# TestEnviROICoords.test_scientific_notation_coordinates()
# TestEnviROICoords.test_decimal_coordinates()

# TestEnviROICoords.test_file_not_found()
# TestEnviROICoords.test_invalid_xml()
# TestEnviROICoords.test_no_polygon_error()
# TestEnviROICoords.test_empty_coordinates()
# TestEnviROICoords.test_whitespace_handling()


# %% test functions : shp_roi_coords


class TestShpROICoords:
    @classmethod
    def setup_method(cls) -> None:
        """Setup method to filter warnings before each test"""
        warnings.filterwarnings(
            "ignore",
            message="'crs' was not provided",
            category=UserWarning,
            module="pyogrio.geopandas",
        )

    @classmethod
    def teardown_method(cls) -> None:
        """Reset warnings after each test"""
        warnings.resetwarnings()

    @staticmethod
    def test_single_polygon() -> None:
        """Test function with a simple Polygon shapefile"""
        # Create a temporary shapefile with a polygon
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, "test_polygon.shp")

            # Create a simple polygon
            polygon = Polygon([(0, 0), (0.8, 0), (1, 1), (0, 0.9), (0, 0)])
            gdf = gpd.GeoDataFrame({"name": ["test_polygon"], "geometry": [polygon]})
            gdf.to_file(shp_path)

            # Test the function
            result = shp_roi_coords(shp_path)

            # Assertions
            assert len(result) == 1
            assert result[0]["name"] == "test_polygon"
            assert result[0]["type"] == "Polygon"
            assert len(result[0]["coordinates"]) == 1
            assert result[0]["coordinates"][0] == [
                (0.0, 0.0),
                (0, 0.9),
                (1.0, 1.0),
                (0.8, 0),
                (0.0, 0.0),
            ]

    @staticmethod
    def test_multipolygon() -> None:
        """Test function with a MultiPolygon shapefile"""
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, "test_multipolygon.shp")

            # Create a multipolygon
            poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
            poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
            multipolygon = MultiPolygon([poly1, poly2])

            gdf = gpd.GeoDataFrame({"name": ["test_multipolygon"], "geometry": [multipolygon]})
            gdf.to_file(shp_path)

            # Test the function
            result = shp_roi_coords(shp_path)

            # Assertions
            assert len(result) == 1
            assert result[0]["name"] == "test_multipolygon"
            assert result[0]["type"] == "MultiPolygon"
            assert len(result[0]["coordinates"]) == 2
            assert result[0]["coordinates"][0] == [
                (0.0, 0.0),
                (0, 1.0),
                (1.0, 1.0),
                (1.0, 0),
                (0.0, 0.0),
            ]
            assert result[0]["coordinates"][1] == [
                (2.0, 2.0),
                (2.0, 3.0),
                (3.0, 3.0),
                (3.0, 2.0),
                (2.0, 2.0),
            ]

    @staticmethod
    def test_multiple_features() -> None:
        """Test function with multiple features in shapefile"""
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, "test_multiple.shp")

            # Create multiple polygons
            poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
            poly2_1 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
            poly2_2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6), (5, 5)])
            multipolygon = MultiPolygon([poly2_1, poly2_2])

            gdf = gpd.GeoDataFrame({"name": ["poly", "multipoly"], "geometry": [poly1, multipolygon]})
            gdf.to_file(shp_path)

            # Test the function
            result = shp_roi_coords(shp_path)

            # Assertions
            assert len(result) == 2
            assert result[0]["name"] == "poly"
            assert result[1]["name"] == "multipoly"
            assert result[0]["type"] == "Polygon"
            assert result[1]["type"] == "MultiPolygon"

    @staticmethod
    def test_missing_name_field() -> None:
        """Test function when 'name' field is missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, "test_no_name.shp")

            # Create polygon without name field
            polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
            gdf = gpd.GeoDataFrame({"geometry": [polygon]})
            gdf.to_file(shp_path)

            # Test the function
            result = shp_roi_coords(shp_path)

            # Should use default naming convention
            assert len(result) == 1
            assert result[0]["name"] == "ROI_0"
            assert result[0]["type"] == "Polygon"

    @staticmethod
    def test_unsupported_geometry() -> None:
        """Test function with unsupported geometry type"""
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, "test_unsupported.shp")

            # Create a Point geometry (unsupported)
            from shapely.geometry import Point

            point = Point(0, 0)

            gdf = gpd.GeoDataFrame({"name": ["test_point"], "geometry": [point]})
            gdf.to_file(shp_path)

            # Should raise ValueError
            with pytest.raises(ValueError, match="geom_type Point is not supported"):
                shp_roi_coords(shp_path)

    @staticmethod
    def test_empty_shapefile() -> None:
        """Test function with empty shapefile"""
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, "test_empty.shp")

            # Create empty GeoDataFrame
            gdf = gpd.GeoDataFrame(columns=["name", "geometry"])
            gdf.to_file(shp_path)

            # Test the function
            result = shp_roi_coords(shp_path)

            # Should return empty list
            assert result == []

    @staticmethod
    def test_invalid_path() -> None:
        """Test function with invalid file path"""
        try:
            shp_roi_coords("/non/existent/path.shp")
            raise AssertionError("Should raise error when file path is invalid.")
        except Exception:
            assert True


# %% Test - shp_roi_coords

# TestShpROICoords.setup_method()

# TestShpROICoords.test_single_polygon()
# TestShpROICoords.test_multipolygon()
# TestShpROICoords.test_multiple_features()

# TestShpROICoords.test_missing_name_field()
# TestShpROICoords.test_invalid_path()
# TestShpROICoords.test_empty_shapefile()
# TestShpROICoords.test_unsupported_geometry()

# TestShpROICoords.teardown_method()


# %% test functions : dump_vars


class TestDumpVars(unittest.TestCase):
    @staticmethod
    def test_with_backup() -> None:
        """Test dump_vars with backup=True"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file path
            test_file = Path(temp_dir) / "test_file.txt"
            test_vars = {"var1": "value1", "var2": 42, "var3": [1, 2, 3]}

            # Call the function
            dump_vars(str(test_file), test_vars, backup=True)

            # Check that main file was created
            main_file = Path(temp_dir) / "test_file.dill"
            assert main_file.exists(), "Main file should be created"

            # Check that backup file was created (should have timestamp pattern)
            backup_files = list(Path(temp_dir).glob("test_file_*.dill"))
            assert len(backup_files) == 1, "Backup file should be created"

            # Verify content of main file
            with open(main_file, "rb") as f:
                loaded_vars = dill.load(f)
            assert loaded_vars == test_vars, "Main file content should match original"

            # Verify content of backup file
            with open(backup_files[0], "rb") as f:
                backup_vars = dill.load(f)
            assert backup_vars == test_vars, "Backup file content should match original"

    @staticmethod
    def test_without_backup() -> None:
        """Test dump_vars with backup=False"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file path
            test_file = Path(temp_dir) / "test_file.txt"
            test_vars = {"test_var": "test_value", "number": 123}

            # Call the function
            dump_vars(str(test_file), test_vars, backup=False)

            # Check that main file was created
            main_file = Path(temp_dir) / "test_file.dill"
            assert main_file.exists(), "Main file should be created"

            # Check that no backup files were created
            backup_files = list(Path(temp_dir).glob("test_file_*.dill"))
            assert len(backup_files) == 0, "No backup files should be created when backup=False"

            # Verify content
            with open(main_file, "rb") as f:
                loaded_vars = dill.load(f)
            assert loaded_vars == test_vars, "File content should match original"

    @staticmethod
    def test_default_backup() -> None:
        """Test dump_vars with default backup parameter (should be True)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file path
            test_file = Path(temp_dir) / "test_file.txt"
            test_vars = {"default_test": True}

            # Call the function without specifying backup
            dump_vars(str(test_file), test_vars)

            # Check that both main and backup files were created (default backup=True)
            main_file = Path(temp_dir) / "test_file.dill"
            assert main_file.exists(), "Main file should be created"

            backup_files = list(Path(temp_dir).glob("test_file_*.dill"))
            assert len(backup_files) == 1, "Backup file should be created with default backup=True"

    @staticmethod
    def test_already_dill_extension() -> None:
        """Test dump_vars with .dill extension already in filename"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file path with .dill extension
            test_file = Path(temp_dir) / "test_file.dill"
            test_vars = {"extension_test": "dill"}

            # Call the function
            dump_vars(str(test_file), test_vars, backup=False)

            # Check that file was created with same name (no double .dill)
            assert test_file.exists(), "File should be created with original .dill extension"

            # Verify content
            with open(test_file, "rb") as f:
                loaded_vars = dill.load(f)
            assert loaded_vars == test_vars, "File content should match original"

    @staticmethod
    def test_no_extension() -> None:
        """Test dump_vars with filename without extension"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file path without extension
            test_file = Path(temp_dir) / "test_file"
            test_vars = {"no_extension": "test"}

            # Call the function
            dump_vars(str(test_file), test_vars, backup=False)

            # Check that .dill extension was added
            dill_file = Path(temp_dir) / "test_file.dill"
            assert dill_file.exists(), "File should be created with .dill extension"

            # Verify content
            with open(dill_file, "rb") as f:
                loaded_vars = dill.load(f)
            assert loaded_vars == test_vars, "File content should match original"

    @staticmethod
    def test_complex_data_structures() -> None:
        """Test dump_vars with complex data structures"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file path
            test_file = Path(temp_dir) / "complex_test.txt"

            # Complex test data
            test_vars = {
                "string": "hello world",
                "number": 42,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
                "set": {1, 2, 3},
                "tuple": (1, 2, 3),
                "none": None,
                "bool": True,
            }

            # Call the function
            dump_vars(str(test_file), test_vars, backup=False)

            # Check that file was created
            main_file = Path(temp_dir) / "complex_test.dill"
            assert main_file.exists(), "File should be created"

            # Verify content
            with open(main_file, "rb") as f:
                loaded_vars = dill.load(f)
            assert loaded_vars == test_vars, "Complex data structures should be preserved"

    @staticmethod
    def test_empty_dict() -> None:
        """Test dump_vars with empty dictionary"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file path
            test_file = Path(temp_dir) / "empty_test.txt"
            test_vars: dict = {}

            # Call the function
            dump_vars(str(test_file), test_vars, backup=False)

            # Check that file was created
            main_file = Path(temp_dir) / "empty_test.dill"
            assert main_file.exists(), "File should be created even with empty dict"

            # Verify content
            with open(main_file, "rb") as f:
                loaded_vars = dill.load(f)
            assert loaded_vars == test_vars, "Empty dict should be preserved"


# %% Test - dump_vars

# TestDumpVars.test_with_backup()
# TestDumpVars.test_without_backup()
# TestDumpVars.test_default_backup()

# TestDumpVars.test_already_dill_extension()
# TestDumpVars.test_no_extension()

# TestDumpVars.test_complex_data_structures()
# TestDumpVars.test_empty_dict()


# %% test functions : load_vars


class TestLoadVars(unittest.TestCase):
    @staticmethod
    def test_success() -> None:
        """Test successful loading of variables from dill file."""
        # Create test data
        test_data = {
            "string_var": "test_string",
            "int_var": 42,
            "list_var": [1, 2, 3],
            "dict_var": {"key": "value"},
        }

        # Create temporary dill file
        with tempfile.NamedTemporaryFile(suffix=".dill", delete=False) as temp_file:
            temp_path = temp_file.name
            with open(temp_path, "wb") as f:
                dill.dump(test_data, f)

            # Test the function
            result = load_vars(temp_path)

            # Verify the result
            assert result == test_data
            assert isinstance(result, dict)
            assert "string_var" in result.keys()
            assert result["int_var"] == 42

        if os.path.exists(temp_path):
            os.remove(temp_path)

    @staticmethod
    def test_invalid_extension() -> None:
        """Test that ValueError is raised for non-dill files."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test content")

            with pytest.raises(ValueError, match="must be a 'dill' file"):
                load_vars(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

    @staticmethod
    def test_file_not_exists() -> None:
        """Test that ValueError is raised for non-existent files."""
        non_existent_path = "/non/existent/path/file.dill"

        with pytest.raises(ValueError, match="path does not exist"):
            load_vars(non_existent_path)

    @staticmethod
    def test_corrupted_file() -> None:
        """Test handling of corrupted dill file."""
        # Create a file with invalid dill content
        with tempfile.NamedTemporaryFile(suffix=".dill", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"invalid dill content")

            with pytest.raises((dill.UnpicklingError, EOFError)):
                load_vars(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

    @staticmethod
    def test_empty_file() -> None:
        """Test loading from an empty dill file."""
        with tempfile.NamedTemporaryFile(suffix=".dill", delete=False) as temp_file:
            temp_path = temp_file.name
            # Leave file empty

            with pytest.raises((dill.UnpicklingError, EOFError)):
                load_vars(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

    @staticmethod
    def test_with_different_data_types() -> None:
        """Test loading various data types from dill file."""
        test_data = {
            "none_val": None,
            "bool_val": True,
            "float_val": 3.14,
            "tuple_val": (1, 2, 3),
            "set_val": {1, 2, 3},
        }

        with tempfile.NamedTemporaryFile(suffix=".dill", delete=False) as temp_file:
            temp_path = temp_file.name
            with open(temp_path, "wb") as f:
                dill.dump(test_data, f)

            result = load_vars(temp_path)

            assert result["none_val"] is None
            assert result["bool_val"] is True
            assert result["float_val"] == 3.14
            assert result["tuple_val"] == (1, 2, 3)
            assert isinstance(result["set_val"], set)

        if os.path.exists(temp_path):
            os.remove(temp_path)

    @staticmethod
    def test_complex_objects() -> None:
        """Test loading complex objects like functions and classes."""

        def test_function(x: Union[int, float]) -> Union[int, float]:
            return x * 2

        class TestClass:
            def __init__(self, value: Union[int, float]) -> None:
                self.value = value

            def get_value(self) -> Union[int, float]:
                return self.value

        test_data = {"function": test_function, "class_obj": TestClass, "instance": TestClass(42)}

        with tempfile.NamedTemporaryFile(suffix=".dill", delete=False) as temp_file:
            temp_path = temp_file.name
            with open(temp_path, "wb") as f:
                dill.dump(test_data, f)

            result = load_vars(temp_path)

            # Test the loaded function
            assert result["function"](5) == 10

            # Test the loaded class
            instance = result["class_obj"](100)
            assert instance.get_value() == 100

            # Test the loaded instance
            assert result["instance"].get_value() == 42

        if os.path.exists(temp_path):
            os.remove(temp_path)


# %% Test - load_vars

# TestLoadVars.test_success()

# TestLoadVars.test_invalid_extension()
# TestLoadVars.test_file_not_exists()

# TestLoadVars.test_corrupted_file()
# TestLoadVars.test_empty_file()

# TestLoadVars.test_with_different_data_types()
# TestLoadVars.test_complex_objects()


# %% test functions : df_to_csv


class TestDfToCsv(unittest.TestCase):
    @staticmethod
    def create_test_dataframe(rows: int = 100, cols: int = 5) -> pd.DataFrame:
        """Create a test DataFrame"""
        return pd.DataFrame({f"col_{i}": np.random.randn(rows) for i in range(cols)})

    @staticmethod
    def test_basic_csv_creation() -> None:
        """Test basic CSV creation without compression"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_df = TestDfToCsv.create_test_dataframe(10, 3)
            output_path = os.path.join(temp_dir, "test.csv")

            result = df_to_csv(test_df, output_path, compression_format="infer")

            assert unc_path(result) == unc_path(output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

            # Verify the file can be read back
            df_read = pd.read_csv(output_path)
            assert df_read.shape[0] == 10
            assert df_read.shape[1] == 3

    @staticmethod
    def test_compression_auto_disable_for_small_data() -> None:
        """Test that compression is automatically disabled for small datasets"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_df = TestDfToCsv.create_test_dataframe(100, 5)
            output_path = os.path.join(temp_dir, "test.csv")

            result = df_to_csv(test_df, output_path, compression_format="zstd")

            assert unc_path(result) == unc_path(output_path)
            assert os.path.exists(output_path)
            # Should not have compression extension for small data
            assert not output_path.endswith(".zst")

            # Verify file content
            df_read = pd.read_csv(output_path)
            assert df_read.shape[1] == 5  # 5 columns + index

    @staticmethod
    def test_compression_enabled_for_large_data() -> None:
        """Test that compression is enabled for large datasets"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_df = TestDfToCsv.create_test_dataframe(10000, 50)
            output_path = os.path.join(temp_dir, "test.csv")

            result = df_to_csv(test_df, output_path, compress_nvalue_threshold=1000, compression_format="zstd")

            expected_path = output_path + ".zst"
            assert unc_path(result) == unc_path(expected_path)
            assert os.path.exists(expected_path)
            assert os.path.getsize(expected_path) > 0

    @staticmethod
    def test_different_compression_formats() -> None:
        """Test different compression formats"""
        compression_formats = ["gzip", "bz2", "zstd"]
        extensions = [".gz", ".bz2", ".zst"]

        for fmt, ext in zip(compression_formats, extensions):
            with tempfile.TemporaryDirectory() as temp_dir:
                test_df = TestDfToCsv.create_test_dataframe(10000, 20)
                output_path = os.path.join(temp_dir, "test.csv")

                result = df_to_csv(test_df, output_path, compress_nvalue_threshold=1000, compression_format=fmt)

                expected_path = output_path + ext
                assert unc_path(result) == unc_path(expected_path)
                assert os.path.exists(expected_path)

    @staticmethod
    def test_overwrite_behavior() -> None:
        """Test overwrite parameter behavior"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_df = TestDfToCsv.create_test_dataframe(10, 3)
            output_path = os.path.join(temp_dir, "test.csv")

            # Create file first
            df_to_csv(test_df, output_path)

            # Should raise error when overwrite=False and file exists
            with pytest.raises(ValueError, match="already exists"):
                df_to_csv(test_df, output_path, overwrite=False)

            # Should not raise error when overwrite=True (default)
            result = df_to_csv(test_df, output_path, overwrite=True)
            assert unc_path(result) == unc_path(output_path)
            # File should be overwritten (size might be different)
            assert os.path.exists(output_path)

    @staticmethod
    def test_return_path_false() -> None:
        """Test return_path=False behavior"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_df = TestDfToCsv.create_test_dataframe(10, 3)
            output_path = os.path.join(temp_dir, "test.csv")

            result = df_to_csv(test_df, output_path, return_path=False)

            assert result is None
            assert os.path.exists(output_path)

    @staticmethod
    def test_invalid_compression_format() -> None:
        """Test invalid compression format raises error"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_df = TestDfToCsv.create_test_dataframe(10, 3)
            output_path = os.path.join(temp_dir, "test.csv")

            with pytest.raises(ValueError, match="Compression_format must be one of"):
                df_to_csv(test_df, output_path, compression_format="invalid_format")

    @staticmethod
    def test_invalid_file_extension() -> None:
        """Test invalid file extension raises error"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_df = TestDfToCsv.create_test_dataframe(10, 3)
            output_path = os.path.join(temp_dir, "test.txt")  # Wrong extension

            with pytest.raises(ValueError, match="Invalid csv file path"):
                df_to_csv(test_df, output_path)

    @staticmethod
    def test_kwargs_passed_to_to_csv() -> None:
        """Test that kwargs are properly passed to DataFrame.to_csv"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_df = TestDfToCsv.create_test_dataframe(10, 3)
            output_path = os.path.join(temp_dir, "test.csv")

            # Test with custom separator and no header
            result = df_to_csv(test_df, output_path, sep="|", header=False, index=True)

            assert unc_path(result) == unc_path(output_path)
            assert os.path.exists(output_path)

            # Verify the file was written with custom parameters
            with open(output_path, "r") as f:
                content = f.read()

            # Check if pipe separator is used
            assert "|" in content.split("\n")[0]
            # Check if header is missing (first line should start with index)
            first_line = content.split("\n")[0]
            assert first_line.startswith("0|") or first_line.startswith("0|")

    @staticmethod
    def test_precompressed_file_path() -> None:
        """Test behavior with pre-compressed file paths"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_df = TestDfToCsv.create_test_dataframe(10000, 20)

            # Test with .csv.gz path and matching compression format
            output_path = os.path.join(temp_dir, "test.csv.gz")
            result = df_to_csv(test_df, output_path, compress_nvalue_threshold=1000, compression_format="gzip")

            assert unc_path(result) == unc_path(output_path)
            assert os.path.exists(output_path)

    @staticmethod
    def test_custom_compression_thresholds() -> None:
        """Test custom compression thresholds"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_df = TestDfToCsv.create_test_dataframe(500, 10)

            # With low threshold, should compress
            output_path1 = os.path.join(temp_dir, "test1.csv")
            result1 = df_to_csv(test_df, output_path1, compress_nvalue_threshold=1000, compression_format="zstd")
            assert type(result1) is str
            assert result1.endswith(".zst")
            assert os.path.exists(result1)

            # With high threshold, should not compress
            output_path2 = os.path.join(temp_dir, "test2.csv")
            result2 = df_to_csv(test_df, output_path2, compress_nvalue_threshold=1000000, compression_format="zstd")
            assert unc_path(result2) == unc_path(output_path2)
            assert result2 is not None
            assert isinstance(result2, str)
            assert not result2.endswith(".zst")
            assert os.path.exists(result2)

    @staticmethod
    def test_index_parameter() -> None:
        """Test index parameter behavior"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_df = TestDfToCsv.create_test_dataframe(5, 2)

            # Test with index=True
            output_path1 = os.path.join(temp_dir, "test_with_index.csv")
            df_to_csv(test_df, output_path1, index=True, return_path=False)

            # Test with index=False
            output_path2 = os.path.join(temp_dir, "test_without_index.csv")
            df_to_csv(test_df, output_path2, index=False, return_path=False)

            # Read files and check if index is present
            df1 = pd.read_csv(output_path1)
            df2 = pd.read_csv(output_path2)

            # With index=True, should have an extra column for index
            assert df1.shape[1] == df2.shape[1] + 1

    @staticmethod
    def test_path_consistency_across_multiple_calls() -> None:
        """Test that paths are consistent across multiple function calls"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_df = TestDfToCsv.create_test_dataframe(10, 3)
            specified_path = os.path.join(temp_dir, "consistent_test.csv")

            # Call multiple times with same parameters
            result1 = df_to_csv(test_df, specified_path, compression_format="infer")
            result2 = df_to_csv(test_df, specified_path, compression_format="infer", overwrite=True)

            # Validate both calls return the same path
            assert unc_path(result1) == unc_path(specified_path)
            assert unc_path(result2) == unc_path(specified_path)
            assert unc_path(result1) == unc_path(result2)


# %% Test - df_to_csv

# TestDfToCsv.test_basic_csv_creation()

# TestDfToCsv.test_compression_auto_disable_for_small_data()
# TestDfToCsv.test_compression_enabled_for_large_data()
# TestDfToCsv.test_different_compression_formats()

# TestDfToCsv.test_overwrite_behavior()

# TestDfToCsv.test_return_path_false()

# TestDfToCsv.test_invalid_compression_format()
# TestDfToCsv.test_invalid_file_extension()

# TestDfToCsv.test_kwargs_passed_to_to_csv()

# TestDfToCsv.test_precompressed_file_path()
# TestDfToCsv.test_custom_compression_thresholds()
# TestDfToCsv.test_index_parameter()

# TestDfToCsv.test_path_consistency_across_multiple_calls()


# %% test functions : df_from_csv


class TestDfFromCsv:
    @staticmethod
    def create_test_csv(content: str, extension: str = "csv", compression: str = "") -> str:
        """Helper method to create test CSV files with optional compression"""
        with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as tmp:
            tmp_path = tmp.name

        if compression != "":
            import bz2
            import gzip
            import lzma
            import zipfile

            import zstandard as zstd

            if compression == "gzip":
                with gzip.open(tmp_path, "wt") as f:
                    f.write(content)
            elif compression == "bz2":
                with bz2.open(tmp_path, "wt") as f:
                    f.write(content)
            elif compression == "zip":
                with zipfile.ZipFile(tmp_path, "w") as zf:
                    zf.writestr("test.csv", content)
            elif compression == "xz":
                with lzma.open(tmp_path, "wt") as f:
                    f.write(content)
            elif compression == "zstd":
                with zstd.open(tmp_path, "wt") as f:
                    f.write(content)
        else:
            with open(tmp_path, "w") as f:
                f.write(content)

        return tmp_path

    @staticmethod
    def test_read_regular_csv() -> None:
        """Test reading a regular CSV file without compression"""
        content = "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9"
        file_path = TestDfFromCsv.create_test_csv(content, "csv")

        df = df_from_csv(file_path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert list(df.columns) == ["col1", "col2", "col3"]
        assert df.iloc[0, 0] == 1

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_read_gzipped_csv() -> None:
        """Test reading a gzip compressed CSV file"""
        content = "name,age,city\nAlice,25,NY\nBob,30,LA\nCharlie,35,SF"
        file_path = TestDfFromCsv.create_test_csv(content, "csv.gz", "gzip")

        df = df_from_csv(file_path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert list(df.columns) == ["name", "age", "city"]
        assert df.iloc[1, 1] == 30

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_read_bzipped_csv() -> None:
        """Test reading a bzip2 compressed CSV file"""
        content = "id,value,score\n1,100,0.8\n2,200,0.9\n3,300,0.7"
        file_path = TestDfFromCsv.create_test_csv(content, "csv.bz2", "bz2")

        df = df_from_csv(file_path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert list(df.columns) == ["id", "value", "score"]
        assert df.iloc[2, 2] == 0.7

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_read_zipped_csv() -> None:
        """Test reading a zip compressed CSV file"""
        content = "product,price,quantity\nA,10.5,100\nB,20.3,50\nC,15.7,75"
        file_path = TestDfFromCsv.create_test_csv(content, "csv.zip", "zip")

        df = df_from_csv(file_path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert list(df.columns) == ["product", "price", "quantity"]
        assert df.iloc[0, 1] == 10.5

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_read_with_kwargs() -> None:
        """Test reading CSV with additional pandas read_csv parameters"""
        content = "col1;col2;col3\n1;2;3\n4;5;6"
        file_path = TestDfFromCsv.create_test_csv(content, "csv")

        df = df_from_csv(file_path, sep=";")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert list(df.columns) == ["col1", "col2", "col3"]

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_invalid_file_path() -> None:
        """Test that invalid file path raises ValueError"""
        with pytest.raises(ValueError, match="File path 'nonexistent.csv' is invalid"):
            df_from_csv("nonexistent.csv")

    @staticmethod
    def test_invalid_extension() -> None:
        """Test that invalid file extension raises ValueError"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name

            with pytest.raises(ValueError, match="Invalid CSV file"):
                df_from_csv(tmp_path)

        os.unlink(tmp_path)

    @staticmethod
    def test_invalid_compression_format() -> None:
        """Test that unsupported compression format raises ValueError"""
        with tempfile.NamedTemporaryFile(suffix=".csv.rar", delete=False) as tmp:
            tmp_path = tmp.name

            with pytest.raises(ValueError, match="Invalid CSV file"):
                df_from_csv(tmp_path)

        os.unlink(tmp_path)

    @staticmethod
    def test_empty_csv() -> None:
        """Test reading an empty CSV file"""
        content = "col1,col2,col3"
        file_path = TestDfFromCsv.create_test_csv(content, "csv")

        df = df_from_csv(file_path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (0, 3)
        assert list(df.columns) == ["col1", "col2", "col3"]

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_single_row_csv() -> None:
        """Test reading a CSV file with only header row"""
        content = "col1,col2,col3\n1,2,3"
        file_path = TestDfFromCsv.create_test_csv(content, "csv")

        df = df_from_csv(file_path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 3)
        assert list(df.columns) == ["col1", "col2", "col3"]
        assert df.iloc[0, 0] == 1

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_multiple_kwargs() -> None:
        """Test reading CSV with multiple pandas parameters"""
        content = "col1\tcol2\tcol3\n1\t2\t3\n4\t5\t6"
        file_path = TestDfFromCsv.create_test_csv(content, "csv")

        df = df_from_csv(file_path, sep="\t", header=0, dtype={"col1": int})
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert df["col1"].dtype == int

        if os.path.exists(file_path):
            os.remove(file_path)


# %% Test - df_from_csv

# TestDfFromCsv.test_read_regular_csv()

# TestDfFromCsv.test_read_gzipped_csv()
# TestDfFromCsv.test_read_bzipped_csv()
# TestDfFromCsv.test_read_zipped_csv()

# TestDfFromCsv.test_read_with_kwargs()
# TestDfFromCsv.test_multiple_kwargs()

# TestDfFromCsv.test_single_row_csv()
# TestDfFromCsv.test_empty_csv()

# TestDfFromCsv.test_invalid_file_path()
# TestDfFromCsv.test_invalid_extension()
# TestDfFromCsv.test_invalid_compression_format()


# %% test functions : roi_to_envi


class TestRoiToEnvi:
    @staticmethod
    def create_test_coordinates() -> list[list[tuple[float, float]]]:
        """Create test polygon coordinates for testing."""
        return [
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
            [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)],
        ]

    @staticmethod
    def create_test_roi_list() -> list[dict]:
        """Create test ROI list for testing."""
        return [
            {
                "name": "test_roi_1",
                "crs": "EPSG:4326",
                "color": (255, 0, 0),
                "type": "polygon",
                "coordinates": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
            },
            {
                "name": "test_roi_2",
                "crs": "EPSG:3857",
                "color": (0, 255, 0),
                "type": "polygon",
                "coordinates": [[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]],
            },
        ]

    @staticmethod
    def read_file_content(file_path: str) -> str:
        """Read and return the content of a file."""
        with open(file_path, "r") as f:
            return f.read()

    @staticmethod
    def test_basic_functionality() -> None:
        """Test basic functionality with individual parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_roi")
            coordinates = TestRoiToEnvi.create_test_coordinates()

            result = roi_to_envi(
                file_path=file_path,
                name="test_roi",
                coordinates=coordinates,
                crs="EPSG:4326",
                color=(255, 0, 0),
            )

            assert result is not None
            assert os.path.exists(result)
            assert result.endswith(".xml")

            content = TestRoiToEnvi.read_file_content(result)
            assert "test_roi" in content
            assert "255,0,0" in content
            assert "EPSG:4326" in content

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_roi_list_functionality() -> None:
        """Test functionality with roi_list parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_roi_list")
            roi_list = TestRoiToEnvi.create_test_roi_list()

            result = roi_to_envi(file_path=file_path, roi_list=roi_list)

            assert result is not None
            assert os.path.exists(result)

            content = TestRoiToEnvi.read_file_content(result)
            assert "test_roi_1" in content
            assert "test_roi_2" in content
            assert "255,0,0" in content
            assert "0,255,0" in content

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_random_color_generation() -> None:
        """Test that random colors are generated when color is None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_random_color")
            coordinates = TestRoiToEnvi.create_test_coordinates()

            result = roi_to_envi(file_path=file_path, name="test_roi", coordinates=coordinates, color=None)

            content = TestRoiToEnvi.read_file_content(result)
            # Should contain color values in the format "r,g,b"
            import re

            assert re.search(r"\d{1,3},\d{1,3},\d{1,3}", content) is not None

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_auto_closing_polygons() -> None:
        """Test that polygons are automatically closed if not closed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_auto_close")
            # Open polygon (first and last points not the same)
            coordinates = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]

            result = roi_to_envi(file_path=file_path, name="test_roi", coordinates=coordinates)

            content = TestRoiToEnvi.read_file_content(result)
            # Should contain the closing coordinate
            assert "0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0 0.0 0.0" in content

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_directory_creation() -> None:
        """Test that non-existent directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "nonexistent_dir", "test_roi")
            coordinates = TestRoiToEnvi.create_test_coordinates()

            with pytest.warns(
                UserWarning,
                match="The specified path directory does not exist, the directory is created",
            ):
                result = roi_to_envi(file_path=file_path, name="test_roi", coordinates=coordinates)

            assert os.path.exists(os.path.dirname(result))
            assert os.path.exists(result)

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_return_path_false() -> None:
        """Test behavior when return_path is False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_no_return")
            coordinates = TestRoiToEnvi.create_test_coordinates()

            result = roi_to_envi(file_path=file_path, name="test_roi", coordinates=coordinates, return_path=False)

            assert result is None
            assert os.path.exists(file_path + ".xml")

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_invalid_color_range() -> None:
        """Test that invalid color values raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_invalid_color")
            coordinates = TestRoiToEnvi.create_test_coordinates()

            with pytest.raises(ValueError, match="RGB values must be in the range of 0 to 255"):
                roi_to_envi(
                    file_path=file_path,
                    name="test_roi",
                    coordinates=coordinates,
                    color=(256, 0, 0),  # Invalid RGB value
                )

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_missing_name_individual() -> None:
        """Test that missing name with individual parameters raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_missing_name")
            coordinates = TestRoiToEnvi.create_test_coordinates()

            with pytest.raises(ValueError, match="ROI name is not specified"):
                roi_to_envi(file_path=file_path, name="", coordinates=coordinates)  # Empty name

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_missing_name_roi_list() -> None:
        """Test that missing name in roi_list raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_missing_name_list")
            roi_list = [
                {
                    "name": "",
                    "crs": "EPSG:4326",
                    "color": None,
                    "type": "polygon",
                    "coordinates": [[(0, 0), (1, 0), (1, 1)]],
                }
            ]

            with pytest.raises(ValueError, match="ROI name is not specified"):
                roi_to_envi(file_path=file_path, roi_list=roi_list)

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_duplicate_names() -> None:
        """Test that duplicate ROI names raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_duplicate_names")
            roi_list = [
                {
                    "name": "duplicate",
                    "crs": "EPSG:4326",
                    "color": None,
                    "type": "polygon",
                    "coordinates": [[(0, 0), (1, 0), (1, 1)]],
                },
                {
                    "name": "duplicate",
                    "crs": "EPSG:4326",
                    "color": None,
                    "type": "polygon",
                    "coordinates": [[(2, 2), (3, 2), (3, 3)]],
                },
            ]

            with pytest.raises(ValueError, match="ROI name must be unique"):
                roi_to_envi(file_path=file_path, roi_list=roi_list)

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_insufficient_vertices() -> None:
        """Test that polygons with insufficient vertices raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_insufficient_vertices")
            coordinates = [[(0.0, 0.0), (1.0, 0.0)]]  # Only 2 vertices

            with pytest.raises(ValueError, match="At least 3 vertices must be defined"):
                roi_to_envi(file_path=file_path, name="test_roi", coordinates=coordinates)

        if os.path.exists(file_path):
            os.remove(file_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_insufficient_vertices")
            coordinates = [[(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]]  # Only 3 vertices including closing one

            with pytest.raises(ValueError, match="At least 3 vertices must be defined"):
                roi_to_envi(file_path=file_path, name="test_roi", coordinates=coordinates)

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_empty_coordinates() -> None:
        """Test that empty coordinates raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_empty_coords")

            with pytest.raises(ValueError, match="ROI polygon vertex coordinates is not provided"):
                roi_to_envi(
                    file_path=file_path,
                    name="test_roi",
                    coordinates=[],  # Empty coordinates
                )

        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def test_xml_structure() -> None:
        """Test that the generated XML has the correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_xml_structure")
            coordinates = TestRoiToEnvi.create_test_coordinates()

            result = roi_to_envi(
                file_path=file_path,
                name="test_roi",
                coordinates=coordinates,
                crs="EPSG:4326",
                color=(255, 0, 0),
            )

            content = TestRoiToEnvi.read_file_content(result)

            # Check XML declaration
            assert '<?xml version="1.0" encoding="UTF-8"?>' in content
            # Check root element
            assert '<RegionsOfInterest version="1.1">' in content
            assert "</RegionsOfInterest>" in content
            # Check region element
            assert '<Region name="test_roi" color="255,0,0">' in content
            assert "</Region>" in content
            # Check coordinate system
            assert "<CoordSysStr>EPSG:4326</CoordSysStr>" in content
            # Check polygon structure
            assert "<Polygon>" in content
            assert "</Polygon>" in content
            assert "<Exterior>" in content
            assert "</Exterior>" in content
            assert "<LinearRing>" in content
            assert "</LinearRing>" in content
            assert "<Coordinates>" in content
            assert "</Coordinates>" in content

        if os.path.exists(file_path):
            os.remove(file_path)


# %% Test - roi_to_envi

# TestRoiToEnvi.test_basic_functionality()
# TestRoiToEnvi.test_roi_list_functionality()
# TestRoiToEnvi.test_random_color_generation()
# TestRoiToEnvi.test_auto_closing_polygons()
# TestRoiToEnvi.test_directory_creation()
# TestRoiToEnvi.test_return_path_false()
# TestRoiToEnvi.test_invalid_color_range()
# TestRoiToEnvi.test_missing_name_individual()
# TestRoiToEnvi.test_missing_name_roi_list()
# TestRoiToEnvi.test_duplicate_names()
# TestRoiToEnvi.test_insufficient_vertices()
# TestRoiToEnvi.test_empty_coordinates()
# TestRoiToEnvi.test_xml_structure()


# %% test functions : roi_to_shp


class TestRoiToShp:
    @staticmethod
    def create_test_polygon() -> list[list[tuple[float, float]]]:
        """Create a simple test polygon coordinates."""
        return [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]]

    @staticmethod
    def create_test_multipolygon() -> list[list[tuple[float, float]]]:
        """Create test multipolygon coordinates."""
        return [
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
            [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0), (2.0, 2.0)],
        ]

    @staticmethod
    def create_test_roi_list() -> list[dict[str, Union[str, list[list[tuple[float, float]]]]]]:
        """Create test ROI list with multiple ROIs."""
        return [
            {
                "name": "test_roi_1",
                "type": "polygon",
                "coordinates": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]],
            },
            {
                "name": "test_roi_2",
                "type": "polygon",
                "coordinates": [[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0), (2.0, 2.0)]],
            },
        ]

    @staticmethod
    def test_single_roi_with_individual_params() -> None:
        """Test creating shapefile with individual parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_roi.shp")
            crs = "EPSG:4326"
            name = "test_polygon"
            coordinates = TestRoiToShp.create_test_polygon()

            result = roi_to_shp(file_path=file_path, crs=crs, name=name, coordinates=coordinates, return_path=True)

            assert isinstance(result, str)
            assert result.endswith(".shp")
            assert os.path.exists(result)

            # Verify the created shapefile
            gdf = gpd.read_file(result)
            assert len(gdf) == 1
            assert gdf.iloc[0]["name"] == name
            assert gdf.crs == crs

    @staticmethod
    def test_multiple_rois_with_roi_list() -> None:
        """Test creating shapefile with ROI list parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_rois.shp")
            crs = "EPSG:4326"
            roi_list = TestRoiToShp.create_test_roi_list()

            result = roi_to_shp(file_path=file_path, crs=crs, roi_list=roi_list, return_path=True)

            assert isinstance(result, str)
            assert os.path.exists(result)

            # Verify the created shapefile
            gdf = gpd.read_file(result)
            assert len(gdf) == 2
            assert gdf.iloc[0]["name"] == "test_roi_1"
            assert gdf.iloc[1]["name"] == "test_roi_2"
            assert gdf.crs == crs

    @staticmethod
    def test_return_none_when_return_path_false() -> None:
        """Test that function returns None when return_path=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_roi.shp")
            crs = "EPSG:4326"
            name = "test_polygon"
            coordinates = TestRoiToShp.create_test_polygon()

            result = roi_to_shp(file_path=file_path, crs=crs, name=name, coordinates=coordinates, return_path=False)

            assert result is None
            assert os.path.exists(file_path)

    @staticmethod
    def test_auto_closes_polygon() -> None:
        """Test that function automatically closes unclosed polygons."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_roi.shp")
            crs = "EPSG:4326"
            name = "test_polygon"
            # Unclosed polygon
            coordinates = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]

            result = roi_to_shp(file_path=file_path, crs=crs, name=name, coordinates=coordinates, return_path=True)

            assert os.path.exists(result)

            # Verify polygon was closed
            gdf = gpd.read_file(result)
            polygon = gdf.iloc[0].geometry
            assert polygon.is_valid

    @staticmethod
    def test_creates_nonexistent_directories() -> None:
        """Test that function creates non-existent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "nonexistent", "subdir", "test_roi.shp")
            crs = "EPSG:4326"
            name = "test_polygon"
            coordinates = TestRoiToShp.create_test_polygon()

            with pytest.warns(
                UserWarning,
                match="The specified path directory does not exist, the directory is created",
            ):
                result = roi_to_shp(
                    file_path=file_path,
                    crs=crs,
                    name=name,
                    coordinates=coordinates,
                    return_path=True,
                )

            assert os.path.exists(os.path.dirname(result))
            assert os.path.exists(result)

    @staticmethod
    def test_raises_error_missing_name_individual_params() -> None:
        """Test that function raises error when name is missing with individual params."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_roi.shp")
            crs = "EPSG:4326"
            coordinates = TestRoiToShp.create_test_polygon()

            with pytest.raises(ValueError, match="ROI name is not specified"):
                roi_to_shp(
                    file_path=file_path,
                    crs=crs,
                    name="",
                    coordinates=coordinates,  # Empty name
                )

    @staticmethod
    def test_raises_error_missing_coordinates_individual_params() -> None:
        """Test that function raises error when coordinates are missing with individual params."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_roi.shp")
            crs = "EPSG:4326"
            name = "test_polygon"

            with pytest.raises(ValueError, match="ROI polygon vertex coordinates is not provided"):
                roi_to_shp(
                    file_path=file_path,
                    crs=crs,
                    name=name,
                    coordinates=[],  # Empty coordinates
                )

    @staticmethod
    def test_raises_error_missing_name_in_roi_list() -> None:
        """Test that function raises error when name is missing in ROI list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_roi.shp")
            crs = "EPSG:4326"
            roi_list = [
                {
                    "name": "",  # Empty name
                    "type": "polygon",
                    "coordinates": TestRoiToShp.create_test_polygon(),
                }
            ]

            with pytest.raises(ValueError, match="ROI name is not specified"):
                roi_to_shp(file_path=file_path, crs=crs, roi_list=roi_list)

    @staticmethod
    def test_raises_error_duplicate_names_in_roi_list() -> None:
        """Test that function raises error for duplicate ROI names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_roi.shp")
            crs = "EPSG:4326"
            roi_list = [
                {
                    "name": "duplicate_name",
                    "type": "polygon",
                    "coordinates": TestRoiToShp.create_test_polygon(),
                },
                {
                    "name": "duplicate_name",  # Duplicate name
                    "type": "polygon",
                    "coordinates": TestRoiToShp.create_test_polygon(),
                },
            ]

            with pytest.raises(ValueError, match="ROI name must be unique"):
                roi_to_shp(file_path=file_path, crs=crs, roi_list=roi_list)

    @staticmethod
    def test_raises_error_insufficient_vertices() -> None:
        """Test that function raises error for polygons with insufficient vertices."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_roi.shp")
            crs = "EPSG:4326"
            name = "test_polygon"
            # Only 2 vertices (insufficient for polygon)
            coordinates = [[(0.0, 0.0), (1.0, 0.0)]]

            with pytest.raises(ValueError, match="At least 3 vertices must be defined"):
                roi_to_shp(file_path=file_path, crs=crs, name=name, coordinates=coordinates)

    @staticmethod
    def test_raises_error_unsupported_roi_type() -> None:
        """Test that function raises error for unsupported ROI types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_roi.shp")
            crs = "EPSG:4326"
            roi_list = [
                {
                    "name": "test_roi",
                    "type": "unsupported_type",  # Unsupported type
                    "coordinates": TestRoiToShp.create_test_polygon(),
                }
            ]

            with pytest.raises(ValueError, match="geom_type must be 'polygon'"):
                roi_to_shp(file_path=file_path, crs=crs, roi_list=roi_list)

    @staticmethod
    def test_multipolygon_geometry_creation() -> None:
        """Test that multipolygon geometries are created correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_roi.shp")
            crs = "EPSG:4326"
            name = "test_multipolygon"
            coordinates = TestRoiToShp.create_test_multipolygon()

            result = roi_to_shp(file_path=file_path, crs=crs, name=name, coordinates=coordinates, return_path=True)

            assert os.path.exists(result)

            # Verify multipolygon geometry
            gdf = gpd.read_file(result)
            geometry = gdf.iloc[0].geometry
            assert isinstance(geometry, MultiPolygon)
            assert len(geometry.geoms) == 2  # Should have 2 polygons


# %% Test - roi_to_shp

# TestRoiToShp.test_single_roi_with_individual_params()
# TestRoiToShp.test_multiple_rois_with_roi_list()
# TestRoiToShp.test_return_none_when_return_path_false()
# TestRoiToShp.test_auto_closes_polygon()
# TestRoiToShp.test_creates_nonexistent_directories()
# TestRoiToShp.test_raises_error_missing_name_individual_params()
# TestRoiToShp.test_raises_error_missing_coordinates_individual_params()
# TestRoiToShp.test_raises_error_missing_name_in_roi_list()
# TestRoiToShp.test_raises_error_duplicate_names_in_roi_list()
# TestRoiToShp.test_raises_error_insufficient_vertices()
# TestRoiToShp.test_raises_error_unsupported_roi_type()
# TestRoiToShp.test_multipolygon_geometry_creation()


# %% test functions : roi_to_shp


class TestLsdirRobust:
    """Simple test for lsdir_robust"""

    @staticmethod
    def create_test_files(path: str) -> str:
        """Create test files"""
        # Validate path
        if not os.path.exists(path):
            os.makedirs(path)

        # Dummy file names
        names = ["aaa.a", "bbb.b", "ccc.c", "a123.test", "a123.sss", "b123.test"]

        # Create dummy files
        for file in names:
            with open(os.path.join(path, file), "w") as f:
                f.write("dummy image content")

        return path

    @staticmethod
    def test_basic_functionality() -> None:
        """Test functionality consistency of the function and os.listdir"""
        # Create test files
        test_dir = tempfile.mkdtemp()
        _ = TestLsdirRobust.create_test_files(test_dir)

        files_listdir = os.listdir(test_dir)
        assert files_listdir is not None

        files_lsdir_r = lsdir_robust(test_dir)
        assert files_lsdir_r is not None

        assert len(list(files_listdir)) == len(list(files_lsdir_r))

        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    # @staticmethod
    # def test_minimal_required_num_of_fetch_results() -> None:
    #     """Test functionality consistency of the function and os.listdir"""
    #     # Create test files
    #     test_dir = tempfile.mkdtemp()
    #     _ = TestLsdirRobust.create_test_files(test_dir)

    #     files_lsdir_r = lsdir_robust(test_dir, 5, retry=1)
    #     assert files_lsdir_r is not None
    #     assert len(list(files_lsdir_r)) == 6

    #     with pytest.raises(ValueError, match="Failed to fetch required minimal number of results"):
    #         files_lsdir_r = lsdir_robust(test_dir, 6, retry=1)

    #     if os.path.exists(test_dir):
    #         shutil.rmtree(test_dir)


# %% Test - roi_to_shp

# TestLsdirRobust.test_basic_functionality()
# TestLsdirRobust.test_minimal_required_num_of_fetch_results()


# %% test functions : unc_path


class TestUncPathStatic(unittest.TestCase):
    """Static test methods for unc_path function with minimal mocking"""

    @staticmethod
    def create_long_path(base: str, length: int = 260) -> str:
        """Create a path longer than 255 characters"""
        # Calculate how many chars to add to exceed 255
        chars_needed = max(0, length - len(base))
        return base + 'x' * chars_needed

    @staticmethod
    def test_non_windows_unchanged() -> None:
        """Test that paths are unchanged on non-Windows platforms"""
        with patch('sys.platform', 'linux'):
            test_paths = [
                "/home/user/test",
                "/home/user/test/",
                "C:\\Users\\test",  # Even Windows-style paths on Linux
                "",
            ]

            for path in test_paths:
                result = unc_path(path)
                # Should return normalized path + slash if input had trailing slash
                expected = os.path.normpath(path)
                if path.endswith(('\\', '/')):
                    expected += '/'
                assert result == expected, f"Failed for: {path}"

    @staticmethod
    def test_windows_short_paths() -> None:
        """Test short paths on Windows (<= 255 chars)"""
        if sys.platform == 'win32':
            test_cases = [
                ("C:\\test", "C:\\test"),
                ("C:\\test\\", "C:\\test\\"),
                ("\\\\server\\share", "\\\\server\\share"),
                ("\\\\server\\share\\", "\\\\server\\share\\"),
                ("relative\\path", os.path.normpath("relative\\path")),
                ("C:/mixed/slashes\\test", os.path.normpath("C:/mixed/slashes\\test")),
            ]

            for input_path, expected in test_cases:
                result = unc_path(input_path)
                assert result == expected, f"Failed for: {input_path}"

    @staticmethod
    def test_windows_long_paths_with_support() -> None:
        """Test long paths on Windows with long path support enabled"""
        if sys.platform == 'win32':
            # Mock long path support
            with patch('swectral.specio._is_long_path_supported', return_value=True):
                # Test local long path
                long_local = TestUncPathStatic.create_long_path(r"C:\test")
                result = unc_path(long_local)
                assert result.startswith(r'\\?\C:\test')
                assert 'x' in result  # Contains long part

                # Test network long path
                long_network = TestUncPathStatic.create_long_path(r"\\server\share\test")
                result = unc_path(long_network)
                assert result.startswith(r'\\?\UNC\server\share\test')
                assert 'x' in result

                # Test with trailing slash
                result = unc_path(f'{long_local}\\')
                assert result.endswith('\\')
                assert result.startswith('\\\\?\\')

    @staticmethod
    def test_windows_long_paths_without_support() -> None:
        """Test that long paths raise ValueError without long path support"""
        if sys.platform == 'win32':
            # Mock NO long path support
            with patch('swectral.specio._is_long_path_supported', return_value=False):
                long_path = TestUncPathStatic.create_long_path(r"C:\test")
                with pytest.raises(ValueError, match="Windows does not enable long path"):
                    unc_path(long_path)

    @staticmethod
    def test_existing_unc_paths() -> None:
        """Test that already UNC-formatted paths are preserved"""
        if sys.platform == 'win32':
            with patch('swectral.specio._is_long_path_supported', return_value=True):
                # Test existing UNC local path
                existing_unc_local = r"\\?\C:\Users\test"
                result = unc_path(existing_unc_local)
                assert result == existing_unc_local

                # Test existing UNC network path
                existing_unc_network = r"\\?\UNC\server\share\test"
                result = unc_path(existing_unc_network)
                assert result == existing_unc_network

                # Test with trailing slash
                result = unc_path(existing_unc_local + '\\')
                assert result == existing_unc_local + '\\'

    @staticmethod
    def test_path_normalization() -> None:
        """Test that path normalization is always applied"""
        if sys.platform == 'win32':
            test_cases = [
                (r"C:/Users/../Users/test", r"C:\Users\test"),
                (r"C:\Users\.\test\..\file", r"C:\Users\file"),
                (r"\\server/share\path/..", r"\\server\share"),
            ]
            for input_path, expected in test_cases:
                result = unc_path(input_path)
                assert result == expected, f"Failed for: {input_path}"

    @staticmethod
    def test_edge_cases() -> None:
        """Test various edge cases"""
        if sys.platform == 'win32':
            with patch('swectral.specio._is_long_path_supported', return_value=True):
                # Test empty string
                result = unc_path("")
                assert result == "."

                # Test single dot
                result = unc_path(".")
                assert result == "."

                # Test root paths
                result = unc_path(r"C:\\")
                assert result == "C:\\"

                # Test exactly 255 characters (should not convert to UNC)
                exact_255 = "C:\\" + 'x' * 252
                result = unc_path(exact_255)
                assert not result.startswith('\\\\?\\')

    @staticmethod
    def test_relative_paths() -> None:
        """Test that relative paths are converted to absolute for UNC"""
        if sys.platform == 'win32':
            with patch('swectral.specio._is_long_path_supported', return_value=True):
                test_cases = [
                    (r"relative\path", os.path.normpath(r"relative\path")),
                    (r".\current", os.path.normpath(r".\current")),
                    (r"..\parent", os.path.normpath(r"..\parent")),
                ]
                for input_path, expected in test_cases:
                    result = unc_path(input_path)
                    assert os.path.normpath(result.replace('\\\\?\\', '')) == expected, f"Failed for: {input_path}"


# %% Test - unc_path

# TestUncPathStatic.test_non_windows_unchanged()
# TestUncPathStatic.test_windows_short_paths()
# TestUncPathStatic.test_windows_long_paths_with_support()
# TestUncPathStatic.test_windows_long_paths_without_support()
# TestUncPathStatic.test_existing_unc_paths()
# TestUncPathStatic.test_path_normalization()
# TestUncPathStatic.test_edge_cases()
# TestUncPathStatic.test_relative_paths()


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
