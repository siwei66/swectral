# -*- coding: utf-8 -*-
"""
Swectral - Spectra data I/O-related utilities

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os
import sys
import glob
import fnmatch

# import dill
import shutil

# Random
import random

# I/O
import io
from contextlib import redirect_stdout

# Warning
import warnings

# Typing
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
    Protocol,
    runtime_checkable,
)
from pydantic import validate_call

# Time
import time
from datetime import datetime

# Functions
import inspect
from functools import wraps

# Basic data
import numpy as np
import pandas as pd
import torch

# Geo
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from pyproj import CRS

# XML parsing
import re
from bs4 import BeautifulSoup


# %% simple_type_validator - Basic validator with serilization compatibility


def simple_type_validator(func: Callable) -> Callable:  # type: ignore[no-untyped-def]  # noqa: C901
    """
    Python function runtime native type validator for serilization of multiprocessing
    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]  # noqa: C901
        hints = get_type_hints(func, include_extras=True)
        sig = inspect.signature(func)

        def check_type(  # type: ignore[no-untyped-def]  # noqa: C901
            value: Any, expected_type: Any
        ) -> tuple[bool, str]:
            # Error msg in check_type
            err_msg = ""

            # Handle Any type - should always pass
            if expected_type is Any:
                return True, err_msg

            # Early return for None values
            if value is None:
                # Check if None is allowed (Optional[T] or Union[T, None])
                origin = get_origin(expected_type)
                if origin is Union and type(None) in get_args(expected_type):
                    return True, err_msg
                return False, err_msg

            # Handle special typing constructs
            origin = get_origin(expected_type)

            # Handle simple types
            if origin is None:  # Simple type (int, str, etc.)
                # Fix for serialization error for int
                if isinstance(value, expected_type):
                    return True, err_msg
                elif type(value) is expected_type:
                    return True, err_msg
                else:
                    return False, err_msg

            # Union[T1, T2, ...] or Optional[T] (which is Union[T, None])
            if origin is Union:
                return any(check_type(value, t)[0] for t in get_args(expected_type)), err_msg

            # Callable[[args], return]
            if origin is Callable:
                return callable(value), err_msg

            # Handle containers (list, tuple, set, etc.)
            if origin in (list, tuple, set, frozenset):
                if not isinstance(value, origin):
                    return False, err_msg

                type_args = get_args(expected_type)
                if not type_args:  # Unparameterized (just `tuple`, `list`, etc.)
                    return True, err_msg

                # Handle tuples (fixed-length)
                if origin is tuple:
                    if len(type_args) > 1:
                        if Ellipsis in type_args:
                            if (len(type_args) == 2) & (type_args[0] is not Ellipsis):
                                return (
                                    all(check_type(x, t)[0] for x, t in zip(value, [type_args[0]] * len(value))),
                                    err_msg,
                                )
                            else:
                                raise ValueError(
                                    f"Invalid tuple annotation with Ellipsis: \
                                        expected exactly one type before '...', got: {type_args}"
                                )
                        if len(value) != len(type_args):
                            return False, err_msg
                        return all(check_type(x, t)[0] for x, t in zip(value, type_args)), err_msg
                    elif len(type_args) == 1:
                        return (
                            all(check_type(x, t)[0] for x, t in zip(value, [type_args[0]] * len(value))),
                            err_msg,
                        )
                    else:
                        return isinstance(value, expected_type), err_msg

                # Handle list, set, etc. (all elements must match the first type arg)
                return all(check_type(x, type_args[0])[0] for x in value), err_msg

            # Handle Annotated[T, ...]
            if origin is Annotated:
                base_type, *validators = get_args(expected_type)
                if not check_type(value, base_type)[0]:
                    return False, err_msg
                for validator in validators:
                    try:
                        validator(value)
                    except Exception as e:
                        err_msg = f"\n\nValidator error: \n{e}"
                        return False, err_msg
                return True, err_msg

            # Handle Literal["val", ...]
            if origin is Literal:
                allowed_values = get_args(expected_type)
                if value in allowed_values:
                    return True, err_msg
                else:
                    err_msg = f"\n\nValue must be one of {allowed_values}"
                    return False, err_msg

            # Handle unrecognized types
            return isinstance(value, origin), err_msg

        # Validate all arguments
        bound_args = sig.bind(*args, **kwargs)
        for name, value in bound_args.arguments.items():
            if name in hints:
                is_valid, err_msg = check_type(value, hints[name])
                if not is_valid:
                    expected = hints[name]
                    raise TypeError(
                        f"Validation error for {name}\n\n "
                        f"Expected type: {expected}\n\n "
                        f"Got type: {type(value)}\n\n "
                        f"Got value: \n{repr(value)} "
                        f"{err_msg} "
                    )

        return func(*args, **kwargs)

    return wrapper


# %% Validator for numpy array-like


@simple_type_validator
def arraylike_validator(  # noqa: C901
    ndim: Optional[int] = None,
    shape: Optional[tuple[int, ...]] = None,
    as_type: Union[type, str, None] = None,
    d_type: Union[type, str, None] = None,
) -> Callable:
    """
    Pydantic validator for array-like.

    Parameters
    ----------
    ndim : int, optional
        ndim of the array. If not given, the criteria will not be applied.
        The default is None.

    shape : tuple[int,...], optional
        shape of the array. If not given, the criteria will not be applied. If both are given, both are applied.
        0 represents variable length, indicating the dimension can have any size.
        The default is None.

    dtype: type or str
        Validate simple datatypes of the values of the arraylike.

    as_type: type or str
        Convert simple datatypes of the values of the arraylike.
    """
    # Validate ndim
    if ndim is not None:
        if ndim < 0:
            raise ValueError(f"ndim cannot be negative, got: {ndim}")

    # Validate shape
    if shape is not None:
        for dimk in shape:
            if dimk < 0:
                raise ValueError(f"shape dimension cannot be negative. Got shape: {shape}\n")

    def arraylike_val(array_like_data: Any) -> np.ndarray:  # noqa: C901  # type: ignore[no-untyped-def]
        v = array_like_data

        # Validate conversion
        if isinstance(v, np.ndarray):
            arr = v
        elif isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
        elif isinstance(v, dict) or isinstance(v, set):
            raise TypeError(f"{type(v)} \n{v}\n cannot be directly converted to numpy.ndarray.")
        elif v is None:
            raise TypeError("None cannot be converted to numpy.ndarray.")
        elif callable(v):
            raise TypeError("Callable cannot be converted to numpy.ndarray.")
        else:
            try:
                arr = np.asarray(v)
            except Exception as e:
                raise ValueError(
                    f"Given data \n{v}\n with data type \n'{type(v)}'\n cannot be converted to numpy.ndarray."
                ) from e

        # Validate ndim
        if ndim is not None:
            if arr.ndim != ndim:
                raise ValueError(f"Given data has an incompatible ndim. Expected: {ndim}, got: {arr.ndim}\n")

        # Validate shape
        if shape is not None:
            if len(arr.shape) != len(shape):
                raise ValueError(f"Given data has an incompatible ndim. Expected: {len(shape)}, got: {arr.ndim}\n")
            for dimkt in enumerate(shape):
                if dimkt[1] is not None:
                    if dimkt[1] < 0:
                        raise ValueError(f"Shape dimensions cannot be negative. Got specified shape: {shape}\n")
                    if (dimkt[1] > 0) & (dimkt[1] != arr.shape[dimkt[0]]):
                        raise ValueError(f"Given data has an incompatible shape. Expected: {shape}, got: {arr.shape}\n")

        # Convert dtype
        if as_type is not None:
            try:
                arr = arr.astype(as_type)
            except Exception as e:
                raise ValueError(f"Failed to convert array data type to float: \n{str(e)}\n") from e

        # Validate dtype
        if d_type is not None:
            dtype_err = False
            # Check np numeric dtypes
            if arr.dtype != np.dtype(d_type):
                # Check np str
                if "U" in str(np.dtype(d_type)):
                    np_str = str(np.dtype(d_type))
                    if np_str[-1] in [str(i) for i in range(10)]:
                        if np_str != "<U0":
                            dtype_err = True
                        elif "U" not in str(arr.dtype):
                            dtype_err = True
                    else:
                        if "U" not in str(arr.dtype):
                            dtype_err = True
                else:
                    dtype_err = True
            if dtype_err:
                raise TypeError(f"Expect array data type: {np.dtype(d_type)}, but got: {arr.dtype}")

        return arr

    return arraylike_val


# %% Validator for pandas dataframe

pd_dtypes = [
    # Numeric types
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    # Boolean
    "bool",
    # Text/String
    "object",
    "O",
    "string",
    # Datetime/temporal
    "datetime64[ns]",
    "datetime64[ns, tz]",
    "timedelta64[ns]",
    # Categorical
    "category",
    # Nullable types (Pandas 1.0+)
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Float32",
    "Float64",
    "boolean",
    # Sparse
    "Sparse[int8]",
    "Sparse[int16]",
    "Sparse[int32]",
    "Sparse[int64]",
    "Sparse[float16]",
    "Sparse[float32]",
    "Sparse[float64]",
]


@simple_type_validator
def _pd_dtype_cond(pd_dtype: Any, target_type: Union[type, str]) -> bool:  # type: ignore[no-untyped-def]
    """
    Validate if a pandas dtype is compatible with target type.
    """
    # Native dtypes
    if (type(target_type) is type) or str(target_type).lower() in [
        "bool",
        "int",
        "float",
        "str",
        "string",
    ]:
        if (target_type is bool) | (str(target_type).lower() == "bool"):
            cond = pd.api.types.is_bool_dtype(pd_dtype)
        elif (target_type is int) | (str(target_type).lower() == "int"):
            cond = pd.api.types.is_integer_dtype(pd_dtype)
        elif (target_type is float) | (str(target_type).lower() == "float"):
            cond = pd.api.types.is_float_dtype(pd_dtype)
        elif (target_type is str) | (str(target_type).lower() == "str") | (str(target_type).lower() == "string"):
            cond = pd.api.types.is_string_dtype(pd_dtype)
        else:
            raise ValueError("If target_type is specified in type, target_type must be bool, int, float or str")

    elif type(target_type) is str:
        # Numeric dtypes
        if target_type.lower() == "numeric":
            cond = pd.api.types.is_numeric_dtype(pd_dtype)
        elif target_type not in pd_dtypes:
            raise ValueError("target_type is not a valid data type of pandas dataframe")
        # Other pandas dtypes in str
        else:
            cond = str(pd_dtype) == target_type

    else:
        raise TypeError(f"Invalid type of target_type, got: {type(target_type)}, expected: type or str")

    return bool(cond)


@simple_type_validator
def dataframe_validator(  # noqa: C901
    colname_dtypes_dict: Optional[dict[str, Union[type, str]]] = None,
    shape: Optional[tuple[int, int]] = None,
    dtype: Optional[Union[type, str]] = None,
    ncol: Optional[int] = None,
    nrow: Optional[int] = None,
    index: Optional[list[Union[int, str]]] = None,
) -> Callable:
    """
    Pydantic validator for Pandas dataframe.

    Parameters
    ----------
    colname_dtypes_dict : dict[str,Union[type,str]], optional
        Dtypes of the columns. The default is None.

    shape : tuple[int,int], optional
        Shape of the dataframe. If not given, the criteria will not be applied. If both are given, both are applied.
        0 represents variable length, indicating the dimension can have any size.
        The default is None.

    dtype : type
        Global dtype of the dataframe. The default is None.

    ncol :
        Number of columns of the dataframe. The default is None.

    nrow :
        Number of rows of the dataframe. The default is None.

    index : list[Union[int,str]], optional
        Dataframe index. The default is None.
    """
    if shape is not None:
        for dimk in shape:
            if dimk <= 0:
                raise ValueError(f"shape dimension must be positive. Got shape: {shape}\n")

    if nrow is not None:
        if nrow <= 0:
            raise ValueError(f"nrow must be positive. Got: {nrow}\n")

    if ncol is not None:
        if ncol <= 0:
            raise ValueError(f"ncol must be positive. Got: {ncol}\n")

    @simple_type_validator
    def dataframe_val(dataframe: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
        v = dataframe

        # Validate columns and dtypes
        if colname_dtypes_dict is not None:
            columns = list(colname_dtypes_dict.keys())
            dtype_values = list(colname_dtypes_dict.values())
            if dtype is not None:
                raise ValueError(
                    f"Redundant dtype specification.\
                        \nGlobal dtype : {dtype}\ncolname_dtypes_dict : {colname_dtypes_dict}"
                )
            if list(v.columns) != columns:
                raise ValueError(
                    f"Given dataframe has an incompatible column names.\
                        \nExpected: {columns}, \nGot: {v.columns}\n"
                )
            for coln, dtp, dte in zip(v.columns, v.dtypes, dtype_values):
                # Validate non-nested type
                nested = False
                tpcond = False
                if (type(dte) is type) | (type(dte) is str):
                    tpcond = _pd_dtype_cond(dtp, dte)

                # Validate nested type
                else:

                    @simple_type_validator
                    def nesttyp(v1: dte) -> None:  # type: ignore[valid-type]
                        pass

                    for vcoli in v[coln]:
                        try:
                            nesttyp(vcoli)
                            nested = True
                        except Exception as e:
                            raise ValueError(
                                f"Given dataframe has an incompatible nested data type:\
                                    \nColumn: {coln}\nValue: {vcoli}\n\nError:\n{e}\n"
                            ) from e

                if (not tpcond) and (not nested):
                    raise TypeError(
                        f"Given dataframe has an incompatible column data type:\nColumn: {coln},\
                            \nExpected dtype: {dte},\nGot dtype: {dtp}\n"
                    )

        elif dtype is not None:
            val_type = True

            for dtp in v.dtypes:
                val_type = val_type & _pd_dtype_cond(dtp, dtype)

            if not val_type:
                raise TypeError(
                    f"Given dataframe has incompatible data types:\nExpected: {dtype},\nGot dtype: {v.dtypes}\n"
                )

        # Validate shape
        if shape is not None:
            if v.shape != shape:
                raise ValueError(f"Given dataframe has an incompatible shape. Expected: {shape}, got: {v.shape}\n")

        if nrow is not None:
            if v.shape[0] != nrow:
                raise ValueError(
                    f"Given dataframe has an incompatible number of rows. Expected: {nrow}, got: {v.shape[0]}\n"
                )

        if ncol is not None:
            if v.shape[1] != ncol:
                raise ValueError(
                    f"Given dataframe has an incompatible number of columns. Expected: {ncol}, got: {v.shape[1]}\n"
                )

        # Validate index
        if index is not None:
            if list(v.index) != list(index):
                raise TypeError(
                    f"\nGiven dataframe has an incompatible index:\n\n\
                        Expected index: {index},\n\n Got index: {list(v.index)}\n"
                )

        return v

    return dataframe_val


# %% Validator for dictionary value types


# Dictionary value type validator
@simple_type_validator
def dict_value_validator(  # noqa: C901
    value_type_list: Optional[list[Any]] = None,
    key_type_list: Optional[list[Any]] = None,
    value_type_dict: Optional[dict[Any, type]] = None,
) -> Callable:
    """
    Validator for dictionary value and key types.

    Parameters
    ----------
    value_type_list : list[type]
        Dictionary value types, the types is ordered.

    key_type_list : Optional[list[type]], optional
        Key types, the types is ordered. The default is None.

    value_type_dict : Optional[dict[Any, type]], optional
        Dictionary value types according to key, order is not required.
        Simultaneously specifying value_type_list and value_type_dict is not allowed.
        The default is None.
    """
    # Validate value type specification
    if value_type_list is not None and value_type_dict is not None:
        raise ValueError(
            "Simultaneously specifying value_type_list and value_type_dict is not allowed."
            f"\ngot value_type_list: {value_type_list}"
            f"\ngot value_type_dict: {value_type_dict}"
        )

    # Validate key type and value types
    if key_type_list is not None:
        if value_type_list is not None:
            if len(key_type_list) != len(value_type_list):
                raise ValueError(
                    "Inconsistent length of key_type_list and value_type_list"
                    f"\ngot key_type_list: {key_type_list}, length: {len(key_type_list)}"
                    f"\ngot value_type_list: {value_type_list}, length: {len(value_type_list)}"
                )

        if value_type_dict is not None:
            key_list = list(value_type_dict.keys())
            if len(key_type_list) != len(key_list):
                raise ValueError(
                    "Inconsistent length of key_type_list and value_type_list"
                    f"\ngot key_type_list: {key_type_list}, length: {len(key_type_list)}"
                    f"\ngot value_type_list: {value_type_dict}, length: {len(key_list)}"
                )

    # Validate types
    @simple_type_validator
    def dict_val(dictionary: dict) -> dict:  # noqa: C901
        dict_key_list = list(dictionary.keys())
        dict_value_list = list(dictionary.values())

        # Validate value types
        if value_type_list is not None:
            # Length validation
            if len(dict_key_list) != len(value_type_list):
                raise ValueError(
                    "Inconsistent length of given dictionary and value_type_list"
                    f"\ngot dictionary: {dictionary}, length: {len(dictionary)}"
                    f"\ngot value_type_list: {value_type_list}, length: {len(value_type_list)}"
                )

            for i in range(len(dict_key_list)):
                value = dict_value_list[i]
                value_type = value_type_list[i]

                @simple_type_validator
                def value_type_val(v: value_type) -> None:  # type: ignore[valid-type]
                    pass

                value_type_val(value)

        # Validate key types
        if key_type_list is not None:
            # Length validation
            if len(dict_key_list) != len(key_type_list):
                raise ValueError(
                    "Inconsistent length of given dictionary and key_type_list"
                    f"\ngot dictionary: {dictionary}, length: {len(dictionary)}"
                    f"\ngot key_type_list: {key_type_list}, length: {len(key_type_list)}"
                )

            for i in range(len(dict_key_list)):
                value = dict_key_list[i]
                value_type = key_type_list[i]

                @simple_type_validator
                def key_type_val(v: value_type) -> None:  # type: ignore[valid-type]
                    pass

                key_type_val(value)

        # Validate dict value types
        if value_type_dict is not None:
            # Length validation
            if set(dict_key_list) != set(value_type_dict.keys()):
                raise ValueError(
                    "Inconsistent keys of given dictionary and value_type_dict"
                    f"\ngot dictionary keys: {dictionary.keys()}"
                    f"\ngot value_type_dict keys: {value_type_dict.keys()}"
                )

            for key in list(dict_key_list):
                value = dictionary[key]
                value_type = value_type_dict[key]

                @simple_type_validator
                def type_val(v: value_type) -> None:  # type: ignore[valid-type]
                    pass

                type_val(value)

        return dictionary

    return dict_val


# %% Search file in dir


@validate_call
def search_file(  # noqa: C901
    directory_path: str, search_pattern: str, end_with: str = "", exclude_list: Optional[list[str]] = None
) -> list[str]:
    """
    Search and list file paths in a directory.

    Parameters
    ----------
    path : str
        Directory path to search.

    search_pattern : str
        Pattern to search. Wildcards "*" / "[]" / "?" / "[!]" are supported.

    end_with : str
        Filter file names ending with this pattern, extension is included.
        Default is empty string, which uses name extension when exists.

    exclude_list : list of str
        Filter file names that contains any strings in the exclude_list.

    Returns
    -------
    list[str]
        A list of file paths that match the search criteria.

    Examples
    --------
    >>> search_file("/some_dir", "*abc.tif")
    >>> search_file("/some_dir", "abc?.tif")
    >>> search_file("/some_dir", "abc[123].tif")
    >>> search_file("/some_dir", "abc*", end_with=".tif")
    >>> search_file("/some_dir", "abc*", end_with=".tif", exclude_list=["mask", "test"])
    """  # noqa: E501
    # Init exclude_list
    if exclude_list is None:
        ex_list = []
    else:
        ex_list = exclude_list

    wildcardcheck = 0
    for wildcard in ["*", "?", "[", "]", "!"]:
        if wildcard in search_pattern:
            wildcardcheck = wildcardcheck + 1
    if wildcardcheck == 0:
        results = []
        for root, _, file_names in os.walk(directory_path):
            for file_name in file_names:
                if search_pattern is None or search_pattern in file_name:
                    results.append(os.path.join(root, file_name))
    else:
        search_path = os.path.join(directory_path, search_pattern)
        results = glob.glob(search_path)
        results = [str(res).replace("\\", "/") for res in results]

    # filter paths with unexpected ends
    if len(end_with) == 0:
        if "." in search_pattern:
            end_with = search_pattern.split(".")[-1]
    if "*" not in end_with:
        results_filtered = []
        for i in range(len(results)):
            result = results[i]
            if result[(len(result) - len(end_with)) : len(result)] == end_with:
                results_filtered.append(result)
        results = results_filtered
    elif end_with != "*":
        end_with_pattern = end_with.split("*")
        results_filtered = []
        for i in range(len(results)):
            result = results[i]
            result_ext = result.split(".")[-1]
            if len(end_with_pattern) == 2:
                pattern_ext1 = end_with_pattern[0]
                pattern_ext2 = end_with_pattern[1]
                if result_ext.startswith(pattern_ext1) & result_ext.endswith(pattern_ext2):
                    results_filtered.append(result)
            else:
                raise ValueError(
                    f"the number of '*' in file extension pattern cannot be larger than 1, "
                    f"but got pattern: {search_pattern}, "
                    f"with extension pattern: {end_with}"
                )
        results = results_filtered

    # filter paths excluding exclude_list items
    if len(ex_list) > 0:
        results_filtered2 = []
        for i in range(len(results)):
            resulti = results[i]
            result_name = resulti.split("/")[-1]
            iselect = True
            for eit in ex_list:
                if eit in result_name:
                    iselect = False
            if iselect:
                results_filtered2.append(resulti)
        results = results_filtered2

    # use same path format, replace \\ with /
    for ptid in range(len(results)):
        results[ptid] = str(results[ptid]).replace("\\", "/")

    return results


# %% filtering using wild cards


@overload
def names_filter(
    names: dict, pattern: str, dict_value_as_filename: bool = False, return_ids: bool = False
) -> tuple[dict[str, str], dict[str, str]]: ...


@overload
def names_filter(
    names: list[str],
    pattern: str,
    dict_value_as_filename: bool = False,
    return_ids: Literal[False] = False,
) -> tuple[list[str], list[str]]: ...


@overload
def names_filter(
    names: list[str],
    pattern: str,
    dict_value_as_filename: bool = False,
    return_ids: Literal[True] = True,
) -> tuple[list[int], list[int]]: ...


@validate_call
def names_filter(
    names: Union[list[str], dict],
    pattern: str,
    dict_value_as_filename: bool = False,
    return_ids: bool = False,
) -> Union[tuple[list[str], list[str]], tuple[list[int], list[int]], tuple[dict[str, str], dict[str, str]]]:
    """
    In a list or dictionary of names, filter names by a pattern.
    Returns matched names and unmatched names.

    Parameters
    ----------
    names : list[str] or dict[str,str]
        File names to filter.

    pattern : str
        Pattern to search. Unix-like patterns is supported (using fnmatch).

    dict_value_as_filename : bool
        For dictionary of names, set True if dictionary values instead of keys are the file names to filter.

    return_ids : bool
        For list of names, if True, return name indices of filtered results instead of result values. The default is False.

    Returns
    -------
    Union[tuple[list[str],list[str]], tuple[list[int],list[int]], tuple[dict[str,str],dict[str,str]]]
        Lists or dictionaries of matched name items and unmatched name items.

    Raises
    ------
    ValueError
        If names is not recognized as list or dictionary.

    """  # noqa: E501

    if type(names) is list:
        # For names
        selected_list = [fn for fnid, fn in enumerate(names) if fnmatch.fnmatch(fn, pattern)]
        removed_list = [fn for fnid, fn in enumerate(names) if not fnmatch.fnmatch(fn, pattern)]

        # For IDs
        selected_ids = [fnid for fnid, fn in enumerate(names) if fnmatch.fnmatch(fn, pattern)]
        removed_ids = [fnid for fnid, fn in enumerate(names) if not fnmatch.fnmatch(fn, pattern)]

        # Return
        if not return_ids:
            return selected_list, removed_list
        else:
            return selected_ids, removed_ids

    elif type(names) is dict:
        if dict_value_as_filename:
            fns_to_remove_ids: list = [fnid for fnid in names.keys() if fnmatch.fnmatch(names[fnid], pattern)]
        else:
            fns_to_remove_ids = [fn for fn in names.keys() if fnmatch.fnmatch(fn, pattern)]
        selected_dict: dict = {}
        removed_dict: dict = names
        for fnid in fns_to_remove_ids:
            selected_dict[fnid] = names[fnid]
            del removed_dict[fnid]
        return selected_dict, removed_dict

    else:
        raise ValueError("\n provided names is not recognized as a list or dictionary.")


# %% Extract ENVI ROI coordinates


# Read multipolygon vertex coordinate pairs from ENVI ROI files
@validate_call
def envi_roi_coords(roi_xml_path: str) -> list[dict[str, Any]]:
    """
    Get vertex coordinates of (multi-)polygon ROIs from ENVI ROI xml file.

    Parameters
    ----------
    rpath : str
        Path of ENVI xml ROI file.

    Returns
    -------
    list of dict
        A list of ROI dictionaries.
        Each ROI dictionary contains ROI name, ROI geometry type and lists of vertex coordinate pairs in tuples.

    Raises
    ------
    ValueError
        If no polygon for a ROI is found.

    Examples
    --------
    >>> coord_list = envi_roi_coords("/image_roi.xml")
    """
    # Read ENVI ROI xml file
    with open(unc_path(roi_xml_path), "r") as f:
        roi_data = f.read()
    soup = BeautifulSoup(roi_data, "xml")
    sroi = soup.find_all("Region")
    rn = len(sroi)
    if rn < 1:
        raise ValueError(f"No ROI is found in the provided xml file, got file content: \n\n{soup.prettify()}")
    # Parsing
    roi_list = []
    for n in range(rn):
        roii = sroi[n]
        roiname = roii.get("name")
        roipolys = roii.find_all("Polygon")
        if len(roipolys) == 1:
            polytype = "Polygon"
        elif len(roipolys) > 1:
            polytype = "MultiPolygon"
        else:
            raise ValueError(f"no polygon for ROI {roiname} is found")
        poly_coords = []
        for polygs in roipolys:
            roicoord_found = polygs.find("Coordinates")
            if roicoord_found is not None:
                roicoord = roicoord_found.text.strip()
            else:
                raise ValueError("No 'Coordinates' found in given xml file.")
            coord = re.findall(r"-?\d+\.?\d*[eE][+-]?\d+|-?\d+\.?\d*", roicoord)
            coordpairs = []
            for i in range(int(len(coord) / 2)):
                coordpairs.append(
                    (
                        float(coord[2 * i].replace("e", "E")),
                        float(coord[2 * i + 1].replace("e", "E")),
                    )
                )
            poly_coords.append(coordpairs)
        roi_list.append({"name": roiname, "type": polytype, "coordinates": poly_coords})
    return roi_list


# %% Extract shp ROI Coordinates


# Read QGIS shp polygon / multipolygon ROI Coordinate Pairs
@validate_call
def shp_roi_coords(roi_shp_path: str) -> list[dict[str, Any]]:
    """
    Get vertex coordinates of (multi-)polygon ROIs from shapefile.

    Parameters
    ----------
    shapefile_path : str
        Path to the input shapefile

    Returns
    -------
    list of dict
        A list of ROI dictionaries.
        Each ROI dictionary contains ROI name, ROI geometry type and lists of vertex coordinate pairs in tuples.

    Raises
    ------
    ValueError
        If geom_type is not supported.

    Examples
    --------
    >>> coord_list = shp_roi_coords("/image_roi.shp")
    """
    # Read ROI shapefile as geodataframe
    gdf = gpd.read_file(roi_shp_path)
    # Parsing
    roi_list = []
    for id1, rdata in gdf.iterrows():
        geometry = rdata.geometry
        name = rdata.get("name", f"ROI_{id1}")
        if geometry.geom_type == "Polygon":
            coordinates = list(geometry.exterior.coords)
            roi_list.append({"name": name, "type": "Polygon", "coordinates": [coordinates]})
        elif geometry.geom_type == "MultiPolygon":
            multi_coords = []
            for poly in geometry.geoms:
                poly_coords = list(poly.exterior.coords)
                multi_coords.append(poly_coords)
            roi_list.append({"name": name, "type": "MultiPolygon", "coordinates": multi_coords})
        else:
            raise ValueError(f"geom_type {geometry.geom_type} is not supported")
    return roi_list


# %% Write variables to dill file and load variables from dill file


# Helper: get object size
@simple_type_validator
def _get_base_size(obj: Any) -> int:
    """Helper: get object memory size"""
    result: int

    # If None
    if obj is None:
        result = 0
        return result

    # Labeled size check
    # For rasters
    elif isinstance(obj, dict) and "__specpipe_raster_meta_for_size_validation" in obj:
        import numpy as np

        d_size = np.dtype(obj['dtype']).itemsize
        result = int(obj['width'] * obj['height'] * obj['count'] * d_size)

    # Pandas
    elif hasattr(obj, "memory_usage"):
        try:
            result = int(obj.memory_usage(deep=True).sum())
        except Exception:
            result = sys.getsizeof(obj)
    # NumPy.NDArray
    elif hasattr(obj, "nbytes"):
        result = int(obj.nbytes)
    # PyTorch.Tensor
    elif hasattr(obj, "nelement") and hasattr(obj, "element_size"):
        try:
            result = int(obj.nelement() * obj.element_size())
        except Exception:
            result = sys.getsizeof(obj)

    # Recursive check for containers
    elif isinstance(obj, (dict, list, tuple)):
        result = sum(_get_base_size(i) for i in (obj.values() if isinstance(obj, dict) else obj))

    else:
        result = sys.getsizeof(obj)

    return result


# Disk space check and wait disk space
@simple_type_validator
def _wait_for_free_space(
    obj: Any,
    path: str,
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    *,
    min_sec_random_wait: float = 5.0,
    max_sec_random_wait: float = 5.0,
    obj_size_buffer_coeff: float = 1.1,
) -> None:
    """
    Disk free space validator for output files.
    Raises TimeoutError if space is not available within the timeout period.
    """
    start_time = time.time()
    # Check the parent directory since the file might not exist yet
    check_path = os.path.dirname(os.path.abspath(path))

    # Object size
    obj_size = int(_get_base_size(obj) * obj_size_buffer_coeff)

    while True:
        usage = shutil.disk_usage(check_path)
        predict_free = max(0, usage.free - obj_size)
        free_pct = (predict_free / usage.total) * 100

        if free_pct >= reserve_free_pct:
            return

        elapsed = time.time() - start_time
        if elapsed > space_wait_timeout:
            raise OSError(
                f"Disk space validation failed after {space_wait_timeout}s. "
                f"Hit: {reserve_free_pct}%, Current: {free_pct:.2f}% on {check_path}"
            )
        # Randomize for multiprocessing
        sleep_time = random.uniform(max(1.0, min_sec_random_wait), max(1.0, min_sec_random_wait, max_sec_random_wait))
        time.sleep(sleep_time)


# %% Dump and load python obj to dill file


# New dump vars with disk space validation and backup functionality
@simple_type_validator
def dump_dill(
    obj: Any,
    target_file_path: str,
    backup: bool = True,
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    *,
    min_sec_random_wait: float = 5.0,
    max_sec_random_wait: float = 5.0,
    obj_size_buffer_coeff: float = 1.1,
) -> None:
    """
    Dump variables to dill file with backup and disk space validation.
    The disk space validation estimates the size of the object to dump using (memory size * buffer coefficient) approach.

    Parameters
    ----------
    obj : Any
        Object to dump.
    target_file_path : str
        Full path for the target output file.
    backup : bool
        Whether to create a timestamped backup file.
    space_wait_timeout : int
        Seconds to wait for space to clear before raising an Error.
    reserve_free_pct : float
        Reserved disk free percentage required to proceed.
    min_sec_random_wait : float
        Minimum seconds of random wait before dump.
    max_sec_random_wait : float
        Maximum seconds of random wait before dump.
    """  # noqa: E501
    # Dependencies for multiprocessing
    import dill

    # Validate disk space before any I/O begins
    _wait_for_free_space(
        obj=obj,
        path=target_file_path,
        space_wait_timeout=space_wait_timeout,
        reserve_free_pct=reserve_free_pct,
        min_sec_random_wait=min_sec_random_wait,
        max_sec_random_wait=max_sec_random_wait,
        obj_size_buffer_coeff=obj_size_buffer_coeff,
    )

    # Validate extension and setup paths
    target_file_path_base = os.path.splitext(target_file_path)[0]
    target_file_path1 = target_file_path_base + ".dill"
    os.makedirs(os.path.dirname(target_file_path1), exist_ok=True)

    # Dill dump to a temporary file for atomicity
    pid = os.getpid()
    timestamp = str(time.time_ns())[7:-2]
    temp_path = f"{target_file_path_base}_{pid}{timestamp}.dill.tmp"

    # Dump file
    try:
        with open(unc_path(temp_path), "wb") as f:
            dill.dump(obj, f)
        # os.replace is atomic, preventing partial file reads
        os.replace(temp_path, target_file_path1)
    except Exception as e:
        # Cleanup temp file if the write failed
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

    # Dump backup file
    if backup:
        cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        target_file_path_backup = target_file_path_base + "_" + cts + ".dill"
        temp_path_backup = f"{target_file_path_backup}_{pid}{timestamp}.dill.tmp"
        try:
            with open(unc_path(temp_path_backup), "wb") as f:
                dill.dump(obj, f)
            os.replace(temp_path_backup, target_file_path_backup)
        except Exception as e:
            # Cleanup temp file if the write failed
            if os.path.exists(temp_path_backup):
                os.remove(temp_path_backup)
            raise e


# Load variables from dill file
@simple_type_validator
def load_dill(dill_file_path: str) -> Any:
    """
    Load variables from dill file.

    Parameters
    ----------
    file_path : str
        path of dill file to load.

    Raises
    ------
    ValueError
        If specified source file is not a 'dill' file.

    ValueError
        If source_file_path does not exist.

    Returns
    -------
    Any
        Stored object in the dill file.
    """
    # Dependencies for multiprocessing
    import dill

    dill_file_path = unc_path(dill_file_path)
    # Validate extension
    if os.path.splitext(dill_file_path)[1] != ".dill":
        raise ValueError(f"The specified source file must be a 'dill' file, got 'dill_file_path': \n{dill_file_path}")

    # Validate existence
    if not os.path.exists(dill_file_path):
        raise ValueError(f"Specified 'dill_file_path' does not exist: \n{dill_file_path}")

    with open(dill_file_path, "rb") as f:
        data = dill.load(f)

    return data


# %% _wait_for_free_space for multiprocessing


@simple_type_validator
def _safe_disk_wait(
    obj: Any,
    path: str,
    preprocess_status: Optional[dict[str, Any]] = None,
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    *,
    min_sec_random_wait: float = 5.0,
    max_sec_random_wait: float = 5.0,
    obj_size_buffer_coeff: float = 1.1,
) -> None:
    """Safe '_wait_for_free_space' compatible with multiprocessing '_wait_for_completion' for multiprocessing."""
    # Skip logic
    if preprocess_status is None:
        # Call plain _wait_for_free_space
        _wait_for_free_space(
            obj=obj,
            path=path,
            space_wait_timeout=space_wait_timeout,
            reserve_free_pct=reserve_free_pct,
            min_sec_random_wait=min_sec_random_wait,
            max_sec_random_wait=max_sec_random_wait,
            obj_size_buffer_coeff=obj_size_buffer_coeff,
        )
    # Apply with mp.Manager waiting list update
    else:
        waiting_list = preprocess_status['waiting_for_disk_space']
        lock = preprocess_status['lock']

        # Register waiting for the path
        with lock:
            waiting_list.append(path)

        # Wait
        try:
            _wait_for_free_space(
                obj=obj,
                path=path,
                space_wait_timeout=space_wait_timeout,
                reserve_free_pct=reserve_free_pct,
                min_sec_random_wait=min_sec_random_wait,
                max_sec_random_wait=max_sec_random_wait,
                obj_size_buffer_coeff=obj_size_buffer_coeff,
            )

        # Unregister waiting for the path
        finally:
            # Unregister
            with lock:
                if path in waiting_list:
                    try:
                        waiting_list.remove(path)
                    except ValueError:
                        # For concurrency error
                        pass


# %% Write to large csv


# Write pandas dataframe to csv with auto compression
@simple_type_validator
def df_to_csv(  # type: ignore[no-untyped-def]  # noqa: C901
    dataframe: Annotated[Any, dataframe_validator()],
    csv_path: str,
    index: bool = False,
    return_path: bool = False,
    overwrite: bool = True,
    compress_nvalue_threshold: int = 5000000,
    compress_shape_threshold: tuple[int, int] = (16384 - 1, 16384 - 1),
    compression_format: str = "zstd",
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    min_sec_random_wait: float = 5.0,
    max_sec_random_wait: float = 5.0,
    obj_size_buffer_coeff: float = 2.0,
    **kwargs,
) -> Optional[str]:
    """
    Write large Pandas dataframe to CSV file, automatically compress and optionally return write path.
    """

    # Validate disk space before any I/O begins
    _wait_for_free_space(
        obj=dataframe,
        path=csv_path,
        reserve_free_pct=reserve_free_pct,
        space_wait_timeout=space_wait_timeout,
        min_sec_random_wait=min_sec_random_wait,
        max_sec_random_wait=max_sec_random_wait,
        obj_size_buffer_coeff=obj_size_buffer_coeff,
    )

    compression_format = compression_format.lower()

    # Compresion formats {parameter : ext}
    ext_map = {"gzip": ".gz", "bz2": ".bz2", "zip": ".zip", "xz": ".xz", "zstd": ".zst", "infer": ""}

    # Validate compression configs
    if compression_format not in ext_map:
        raise ValueError(f"Compression_format must be one of {list(ext_map.keys())}, got: {compression_format}")

    # Validate compression
    nvalues = dataframe.size

    # For small table, compression is not applied by default
    if (
        (nvalues <= compress_nvalue_threshold)
        and (dataframe.shape[0] < compress_shape_threshold[0])
        and (dataframe.shape[1] < compress_shape_threshold[1])
    ):
        compression_format = "infer"

    # Validate path
    invalid_path: bool = False
    path_split = csv_path.split(".")
    if len(path_split) < 2:
        invalid_path = True
    # Path with specified compression extension
    elif (path_split[-1]).lower() != "csv":
        if len(path_split) < 3:
            invalid_path = True
        else:
            # Validate extension
            if compression_format == "infer":
                if ((path_split[-1]).lower() in set(ext_map.values())) and ((path_split[-2]).lower() == "csv"):
                    pass
                else:
                    invalid_path = True
            elif ((path_split[-1]).lower() == ext_map[compression_format][1:]) and ((path_split[-2]).lower() == "csv"):
                pass
            else:
                invalid_path = True
    # Plain csv extension - auto convert to compression extension according to compression_format
    else:
        csv_path = csv_path + ext_map[compression_format]
    # Raise for invalid path
    if invalid_path:
        raise ValueError(f"Invalid csv file path or compressed csv file path, got: \n{csv_path}\n")

    if not overwrite:
        if os.path.exists(csv_path):
            raise ValueError(f"File path '{csv_path}' already exists while overwrite is set {overwrite}")

    # Validate other parameters
    # Get accepted parameters
    sig = inspect.signature(dataframe.to_csv)
    accepted_params = sig.parameters.keys()
    # Filter kwargs and only allow accepted parameters
    filtered_params = {k: v for k, v in kwargs.items() if k in accepted_params}
    csv_path = unc_path(csv_path)
    filtered_params["path_or_buf"] = csv_path
    filtered_params["compression"] = compression_format
    filtered_params["index"] = index

    # Write CSV
    dataframe.to_csv(**filtered_params)
    if return_path:
        return csv_path
    else:
        return None


# Read compressed csv
@simple_type_validator
def df_from_csv(csv_path: str, **kwargs) -> pd.DataFrame:  # type: ignore[no-untyped-def]  # noqa: C901
    """
    Automatically read large CSV file with decompression to Pandas dataframe.
    Automatically find compressed file if plain CSV file path does not exist.
    """
    path = csv_path

    # Compresion formats {ext : parameter}
    extr = {"gz": "gzip", "bz2": "bz2", "zip": "zip", "xz": "xz", "zst": "zstd"}

    # Validate path
    find_path: bool = os.path.exists(path)
    if not find_path:
        # If user provided "data.csv", try "data.csv.gz", "data.csv.zst", etc.
        if path.lower().endswith(".csv"):
            for ext_suffix in extr.keys():
                potential_path = f"{path}.{ext_suffix}"
                if os.path.exists(potential_path):
                    path = potential_path
                    find_path = True
                    break
    if not find_path:
        raise ValueError(f"File path '{path}' is invalid.")

    # Validate path extension
    invalid_path: bool = False
    path_split = path.split(".")
    if len(path_split) < 2:
        invalid_path = True
    elif len(path_split) < 3 and (path_split[-1]).lower() != "csv":
        invalid_path = True
    elif ((path.split(".")[-1]).lower() != "csv") and (
        not (((path.split(".")[-2]).lower() == "csv") and ((path.split(".")[-1]).lower() in list(extr.keys())))
    ):
        invalid_path = True
    if invalid_path:
        raise ValueError(
            f"Invalid CSV file '{path}', \
                the file extension must be one of '.csv', '.csv.gz', '.csv.bz2', '.csv.zip', '.csv.xz' and '.csv.zst'"
        )

    # Parse compression
    if (path.split(".")[-1]).lower() != "csv":
        compression_format = extr[(path.split(".")[-1]).lower()]
    else:
        compression_format = "infer"

    # Validate other parameters
    # Get accepted parameters
    sig = inspect.signature(pd.read_csv)
    accepted_params = sig.parameters.keys()
    # Filter kwargs and only allow accepted parameters
    filtered_params = {k: v for k, v in kwargs.items() if k in accepted_params}
    path = unc_path(path)
    filtered_params["filepath_or_buffer"] = path
    filtered_params["compression"] = compression_format

    # Read CSV
    return pd.read_csv(**filtered_params)


# %% Write ROI coord lists to ROI files


# ROI coords to ENVI ROI xml
@simple_type_validator
def roi_to_envi(  # noqa: C901
    file_path: str,
    name: str = "",
    coordinates: Optional[list[list[tuple[Union[int, float], Union[int, float]]]]] = None,
    crs: Union[str, CRS] = "none",
    color: Optional[tuple[int, int, int]] = None,
    roi_type: str = "polygon",
    roi_list: Optional[list[dict[str, Any]]] = None,
    return_path: bool = True,
) -> Optional[str]:
    """
    Write one or more polygon Regions of Interest (ROIs) to an ENVI XML ROI file.

    This function supports writing a single ROI via individual arguments or multiple ROIs via ``roi_list``.
    When ``roi_list`` is provided, all single-ROI arguments are ignored.

    Parameters
    ----------
    file_path : str
        Path to the output ENVI XML ROI file.

    name : str, optional
        Name of the ROI when writing a single ROI. Ignored if ``roi_list`` is provided.

    coordinates : list of list of tuple of 2 (float or int), optional
        Vertex coordinate pairs defining the ROI geometry when writing a single ROI.

        Structure::

            [
                [ (x1, y1), (x2, y2), ..., (xn, yn), (x1, y1) ],  # Polygon 1
                [ (x1, y1), (x2, y2), ..., (xm, ym), (x1, y1) ],  # Polygon 2
                ...
            ]

        Each inner list represents a polygon (for multipart geometries), and each tuple is a vertex coordinate.

        Ignored if ``roi_list`` is provided.

    crs : str or CRS, optional
        Coordinate reference system of the ROI.

        Use ``"none"`` for image-space coordinates (ENVI default).

        Ignored if ``roi_list`` is provided.

    color : tuple of int, optional
        RGB color of the ROI specified as ``(R, G, B)``, where each value is in the range 0–255.

        If ``None``, a random color is assigned. Default is None.

        Ignored if ``roi_list`` is provided.

    roi_type : str, optional
        Type of ROI to write.

        Currently, only ``"polygon"`` is supported.
        This parameter has no effect and is reserved for future extensions.

        Ignored if ``roi_list`` is provided.

    roi_list : list of dict, optional
        List of ROI definitions for writing multiple ROIs.

        Each dictionary must contain the following keys::

            {
                "name": str,
                "crs": str or CRS,
                "color": tuple of int or None,
                "type": str,
                "coordinates": list of list of tuple of float
            }

        The ``coordinates`` entry represents one or more polygons as lists of vertex coordinate pairs.

    return_path : bool, optional
        If ``True``, return the path to the generated ENVI XML ROI file.

    Returns
    -------
    str or None
        Path of the generated ENVI XML ROI file if ``return_path`` is ``True``; otherwise, ``None``.

    Notes
    -----
    In ENVI, polygons, rectangles, and ellipses are all represented internally as polygon geometries.
    This function therefore writes all supported ROI types as polygons in the ENVI XML schema.

    Examples
    --------
    For single ROI with one polygon::

        >>> roi_to_envi(
        ...     file_path="image_roi.xml",
        ...     name="roi1",
        ...     coordinates=[[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]],
        ...     crs="EPSG:4326",
        ...     color=(255, 0, 0),
        ... )

    For single ROI with multiple polygons::

        >>> roi_to_envi(
        ...     file_path="image_roi.xml",
        ...     name="roi1",
        ...     coordinates=[
        ...             [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)],
        ...             [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
        ...         ],
        ...     crs="EPSG:4326",
        ...     color=(255, 0, 0)
        ... )

    For multiple ROis using ``roi_list``::

        >>> roi_to_envi(
        ...     file_path="image_roi.xml",
        ...     roi_list=[
        ...         {
        ...             "name": "test_roi_1",
        ...             "crs": "EPSG:4326",
        ...             "color": (255, 0, 0),
        ...             "type": "polygon",
        ...             "coordinates": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
        ...         },
        ...         {
        ...             "name": "test_roi_2",
        ...             "crs": "EPSG:3857",
        ...             "color": (0, 255, 0),
        ...             "type": "polygon",
        ...             "coordinates": [[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]],
        ...         },
        ...     ]
        ... )
    """  # noqa: E501

    # Initialize coordinate list
    if coordinates is None:
        coord_list = []
    else:
        coord_list = coordinates

    # Validate ROI item dict in roi_list for ENVI xml
    if roi_list is not None:
        roi_it_validator = dict_value_validator(
            [
                str,
                str,
                Optional[tuple[int, int, int]],
                str,
                list[list[tuple[Union[int, float], Union[int, float]]]],
            ]
        )
        for roi_item in roi_list:
            _ = roi_it_validator(roi_item)

    # Validate path
    file_path = str(file_path).replace("\\", "/")
    path_dir = os.path.dirname(file_path)
    if not os.path.exists(unc_path(path_dir)):
        warnings.warn(
            f"The specified path directory does not exist, the directory is created: {path_dir}",
            UserWarning,
            stacklevel=2,
        )
        os.makedirs(unc_path(path_dir))
    file_path = os.path.splitext(file_path)[0] + ".xml"

    # ROI type current version fixed to 'polygon'
    roi_type = "polygon"

    # Validate ROI parameters
    if roi_list is None:
        if name == "":
            raise ValueError("ROI name is not specified.")
        if len(coord_list) == 0:
            raise ValueError("ROI polygon vertex coordinates is not provided.")
        roi_list = [{"name": name, "crs": crs, "color": color, "type": roi_type, "coordinates": coord_list}]

    # Validate ROI name, color and coordinates
    roi_names = []
    for roi in roi_list:
        # Validate ROI name
        roin = roi["name"]
        if roin == "":
            raise ValueError("ROI name is not specified.")
        if roin in roi_names:
            raise ValueError(f"ROI name must be unique, got duplicated name: {roin}.")
        roi_names.append(roin)
        # Validate color
        if roi["color"] is not None:

            @simple_type_validator
            def val_color(v: tuple[int, int, int]) -> tuple[int, int, int]:
                return v

            roi["color"] = val_color(roi["color"])
            for cv in roi["color"]:
                if cv < 0 or cv > 255:
                    raise ValueError(f"RGB values must be in the range of 0 to 255, got: {roi['color']}")
        else:
            roi["color"] = tuple(np.random.randint(0, 256, 3))
        # Validate coordiantes
        for poly in roi["coordinates"]:
            if poly[0] != poly[-1]:
                poly.append(poly[0])
            if len(poly) < 4:
                raise ValueError(f"At least 3 vertices must be defined for a polygon geometry, but got: {poly}")

    # Write ROI xml file
    with open(unc_path(file_path), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>' + "\n")
        f.write('<RegionsOfInterest version="1.1">' + "\n")
        for roi in roi_list:
            roin = roi["name"]
            roic = roi["color"]
            roic_str = f"{roic[0]},{roic[1]},{roic[2]}"
            coord_sys = roi["crs"]
            f.write(f'  <Region name="{roin}" color="{roic_str}">' + "\n")
            f.write("    <GeometryDef>" + "\n")
            f.write(f"      <CoordSysStr>{coord_sys}</CoordSysStr>" + "\n")
            for poly in roi["coordinates"]:
                f.write("      <Polygon>" + "\n")
                f.write("        <Exterior>" + "\n")
                f.write("          <LinearRing>" + "\n")
                f.write("            <Coordinates>" + "\n")
                coord_str = ""
                for i, coord_pair in enumerate(poly):
                    coord_str = coord_str + str(coord_pair[0]) + " " + str(coord_pair[1])
                    if i < len(poly) - 1:
                        coord_str = coord_str + " "
                f.write(coord_str + "\n")
                f.write("            </Coordinates>" + "\n")
                f.write("          </LinearRing>" + "\n")
                f.write("        </Exterior>" + "\n")
                f.write("      </Polygon>" + "\n")
            f.write("    </GeometryDef>" + "\n")
            f.write("  </Region>" + "\n")
        f.write("</RegionsOfInterest>")

    if return_path:
        return file_path
    else:
        return None


# ROI coords to shp
@simple_type_validator
def roi_to_shp(  # noqa: C901
    file_path: str,
    crs: Union[str, CRS],
    name: str = "",
    coordinates: Optional[list[list[tuple[Union[int, float], Union[int, float]]]]] = None,
    roi_type: str = "polygon",
    roi_list: Optional[list[dict[str, Any]]] = None,
    return_path: bool = True,
) -> Optional[str]:
    """
    Write one or more polygon Regions of Interest (ROIs) to a Shapefile.

    This function supports writing a single ROI via individual arguments or multiple ROIs via ``roi_list``.
    When ``roi_list`` is provided, all single-ROI arguments are ignored.

    Parameters
    ----------
    file_path : str
        Path to the output ENVI XML ROI file.

    crs : str or CRS
        Coordinate reference system of the ROI(s).

        Note: CRS must be provided. Non-georeferenced image may require additional alignment in GIS softwares.

    name : str, optional
        Name of the ROI when writing a single ROI. Ignored if ``roi_list`` is provided.

    coordinates : list of list of tuple of 2 (float or int), optional
        Vertex coordinate pairs defining the ROI geometry when writing a single ROI.

        Structure::

            [
                [ (x1, y1), (x2, y2), ..., (xn, yn), (x1, y1) ],  # Polygon 1
                [ (x1, y1), (x2, y2), ..., (xm, ym), (x1, y1) ],  # Polygon 2
                ...
            ]

        Each inner list represents a polygon (for multipart geometries), and each tuple is a vertex coordinate.

        Ignored if ``roi_list`` is provided.

    roi_type : str, optional
        Type of ROI to write.

        Currently, only ``"polygon"`` is supported.
        This parameter has no effect and is reserved for future extensions.

        Ignored if ``roi_list`` is provided.

    roi_list : list of dict, optional
        List of ROI definitions for writing multiple ROIs.

        Each dictionary must contain the following keys::

            {
                "name": str,
                "type": str,
                "coordinates": list of list of tuple of float
            }

        The ``coordinates`` entry represents one or more polygons as lists of vertex coordinate pairs.

    return_path : bool, optional
        If ``True``, return the path to the generated ENVI XML ROI file.

    Returns
    -------
    str or None
        Path of the generated Shapefile if ``return_path`` is ``True``; otherwise, ``None``.

    Examples
    --------
    For single ROI with one polygon::

        >>> roi_to_shp(
        ...     file_path="image_roi.shp",
        ...     crs="EPSG:4326",
        ...     name="roi1",
        ...     coordinates=[[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]]
        ... )

    For single ROI with multiple polygons::

        >>> roi_to_shp(
        ...     file_path="image_roi.shp",
        ...     crs="EPSG:4326",
        ...     name="roi1",
        ...     coordinates=[
        ...             [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)],
        ...             [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
        ...         ]
        ... )

    For multiple ROis using ``roi_list``::

        >>> roi_to_shp(
        ...     file_path="image_roi.shp",
        ...     roi_list=[
        ...         {
        ...             "name": "test_roi_1",
        ...             "type": "polygon",
        ...             "coordinates": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
        ...         },
        ...         {
        ...             "name": "test_roi_2",
        ...             "type": "polygon",
        ...             "coordinates": [[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]],
        ...         },
        ...     ]
        ... )
    """  # noqa: E501

    # Initialize coordinate list
    if coordinates is None:
        coord_list = []
    else:
        coord_list = coordinates

    # Validate ROI item dict in roi_list for shapefile
    if roi_list is not None:
        roi_it_validator = dict_value_validator([str, str, list[list[tuple[Union[int, float], Union[int, float]]]]])
        for roi_item in roi_list:
            _ = roi_it_validator(roi_item)

    # ROI type (Only polygon for current version)
    roi_type = "polygon"

    # Validate path
    file_path = str(file_path).replace("\\", "/")
    path_dir = os.path.dirname(file_path)
    if not os.path.exists(unc_path(path_dir)):
        warnings.warn(
            f"The specified path directory does not exist, the directory is created: {path_dir}",
            UserWarning,
            stacklevel=2,
        )
        os.makedirs(unc_path(path_dir))
    file_path = os.path.splitext(file_path)[0] + ".shp"

    # Validate ROI parameters
    if roi_list is None:
        if name == "":
            raise ValueError("ROI name is not specified.")
        if len(coord_list) == 0:
            raise ValueError("ROI polygon vertex coordinates is not provided.")
        roi_list = [{"name": name, "type": roi_type, "coordinates": coord_list}]

    # Validate ROI name and coordinates
    roi_names = []
    for roi in roi_list:
        # Validate ROI name
        roin = roi["name"]
        if roin == "":
            raise ValueError("ROI name is not specified.")
        if roin in roi_names:
            raise ValueError(f"ROI name must be unique, got duplicated name: {roin}.")
        roi_names.append(roin)
        # Validate coordiantes
        for poly in roi["coordinates"]:
            if poly[0] != poly[-1]:
                poly.append(poly[0])
            if len(poly) < 4:
                raise ValueError(f"At least 3 vertices must be defined for a polygon geometry, but got: {poly}")

    # Write shp
    geometries = []
    attributes: dict = {"name": [], "type": []}

    for roi in roi_list:
        roin = roi["name"]
        geom_type = roi["type"]
        coords = roi["coordinates"]
        if geom_type == "polygon":
            polygons = [Polygon(poly) for poly in coords]
            geometries.append(MultiPolygon(polygons))
            attributes["name"].append(roin)
            attributes["type"].append(geom_type)
        else:
            raise ValueError(f"geom_type must be 'polygon', but got: '{geom_type}'")

    # Create gdf
    gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=crs)

    # Save to shapefile
    gdf.to_file(unc_path(file_path), driver="ESRI Shapefile")

    if return_path:
        return file_path
    else:
        return None


# %% Silent decorator to mute print and progress bar of functions


def silent(func: Callable) -> Callable:
    """
    Decorator to suppress print
    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        # Disable tqdm
        try:
            from tqdm import tqdm

            original_tqdm_init = tqdm.__init__

            def silent_tqdm_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                kwargs["disable"] = True
                return original_tqdm_init(self, *args, **kwargs)

            tqdm.__init__ = silent_tqdm_init
        except ImportError:
            pass
        # Redirect output
        with redirect_stdout(io.StringIO()):
            result = func(*args, **kwargs)
        # Reset tqdm
        tqdm.__init__ = original_tqdm_init
        return result

    return wrapper


# %% Robust listdir


@simple_type_validator
def lsdir_robust(  # noqa: C901
    path: str,
    fetch_number_gt: int = 0,
    *,
    retry: int = 5,
    time_wait_min: Union[int, float] = 0.5,
    time_wait_max: Union[int, float] = 20.0,
    include_hidden: bool = False,
) -> list:
    """
    Substitution of 'listdir' with retry for file-related testing using GitHub workflow actions.
    """
    path = unc_path(path)
    # Validate configs
    retry = max(int(retry), 1)
    time_wait_min = max(time_wait_min, 0.1)
    time_wait_max = max(time_wait_min, 0.2)
    fetch_number_gt = max(int(fetch_number_gt), 0)

    # Fetch loop
    for run_i in range(retry):
        # OS listdir method
        try:
            result = os.listdir(path)
            if result is not None:
                result1: list = list(result)
                # Filter hidden
                if not include_hidden:
                    result1 = _filepath_hiddenfilter(result1)
                if len(result1) > fetch_number_gt:
                    return result1

        except OSError:
            pass

        # Glob method
        try:
            pattern_path = os.path.join(path, "*")
            result = glob.glob(pattern_path)
            if result is not None:
                result1 = list(result)
                # Filter hidden
                if not include_hidden:
                    result1 = _filepath_hiddenfilter(result1)
                if len(result1) > fetch_number_gt:
                    return result1

        except OSError:
            pass

        if run_i < (retry - 1):
            # Wait time
            time_wait_coef: float = (time_wait_max / time_wait_min) ** (1 / max(1, retry - 1))
            wait_time: float = time_wait_min * (time_wait_coef**run_i)
            time.sleep(wait_time)

    # Not fetch required number of result
    if result is not None:
        # Filter hidden
        if not include_hidden:
            result = _filepath_hiddenfilter(result)
        return result
    else:
        return []


def _filepath_hiddenfilter(path_list: list[str]) -> list:
    "'lsdir_robust' helper to filter hidden files"
    # Hidden prefix and hidden suffix
    hpref = [".", "$", "_", "#", "%", "!", "="]
    hsuff = ["~"]
    # Filtering
    result = [
        item
        for item in path_list
        if os.path.basename(str(item))[0] not in hpref and os.path.basename(str(item))[-1] not in hsuff
    ]
    return result


# %% Numeric type validator


# Read number protocol
class RealNumberMeta(type(Protocol)):  # type: ignore[misc]
    def __instancecheck__(cls, instance: Any) -> bool:
        # Exclude numpy arrays
        if hasattr(instance, '__len__'):
            return False
        # Include RealNumber
        return hasattr(instance, '__mul__') and hasattr(instance, '__lt__')


@runtime_checkable
class RealNumber(Protocol, metaclass=RealNumberMeta):
    def __mul__(self, v: Any) -> Any: ...
    def __lt__(self, v: Any) -> bool: ...


# %% Convert to UNC path to support long path in Windows


def _is_long_path_supported() -> bool:
    """Check for long path support."""
    if sys.platform != 'win32':
        return True
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
        value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        winreg.CloseKey(key)
        return bool(value == 1)
    except Exception:
        return False


@simple_type_validator
def unc_path(path: str) -> str:
    """Convert long paths to UNC long path format on Windows."""
    if sys.platform != 'win32':
        # Check dir path trailing slash for join
        end_slash = ""
        if path.endswith("\\") or path.endswith("/"):
            end_slash = "/"
        return os.path.normpath(path).rstrip("/") + end_slash  # Unchanged if not Windows
    elif len(os.path.normpath(path)) <= 255:
        # Check dir path trailing slash for join
        end_slash = ""
        if path.endswith("\\") or path.endswith("/"):
            end_slash = "\\"
        return os.path.normpath(path).rstrip("\\") + end_slash  # Unchanged if not Windows
    else:
        # Check dir path trailing slash for join
        end_slash = ""
        if path.endswith("\\") or path.endswith("/"):
            end_slash = "\\"
        # To windows format
        path = os.path.normpath(path)
        # To absolute path
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        # Check length support
        if len(path) > 255 and not _is_long_path_supported():
            raise ValueError("Windows does not enable long path, long path must be enabled.")
        # Unchange for UNC path
        if path.startswith('\\\\?\\'):
            return path.rstrip("\\") + end_slash
        # Convert to UNC path format to support long path if not UNC path
        if path.startswith('\\\\'):  # Network path
            return '\\\\?\\UNC\\' + path[2:].rstrip("\\") + end_slash
        else:  # Local path
            return '\\\\?\\' + path.rstrip("\\") + end_slash
