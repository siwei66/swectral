# -*- coding: utf-8 -*-
"""
Swectral - Basic ROI descriptive statistics functions

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Warning
import warnings

# Typing
from typing import Annotated, Any, Callable, Optional, Union, overload
import inspect

# Basic data
import numpy as np
import pandas as pd
import torch

# Math
import math

# Raster
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon, box

# Local
from .specio import arraylike_validator, simple_type_validator, RealNumber


# %% Round to significant numbers


def num_sig_digit(v: RealNumber, sig_digit: int, mode: str = "round") -> float:
    """
    Round a value to given significant digits. Choose mode between 'round' / 'ceil' / 'floor'.
    """
    if sig_digit < 1:
        raise ValueError(f"sig_digit must be at least 1, got: {sig_digit}")
    if v == 0:
        return 0.0
    factor = 10 ** (sig_digit - 1 - math.floor(math.log10(abs(v))))  # type: ignore[arg-type]
    # unrecognized user-defined RealNumber type
    if mode == "round":
        rdfunc = round
    elif mode == "ceil":
        rdfunc = math.ceil  # type: ignore[assignment]
        # overloaded expression function
    elif mode == "floor":
        rdfunc = math.floor  # type: ignore[assignment]
        # overloaded expression function
    else:
        raise ValueError(f"mode must one of 'round', 'ceil' and 'floor', but got: '{mode}'")
    return float(rdfunc(v * factor) / factor)


def np_sig_digit(arr_like: Annotated[Any, arraylike_validator()], sig_digit: int, mode: str = "round") -> np.ndarray:
    """
    Round values to given significant digits for numeric NumPy arrays. Choose mode between 'round' / 'ceil' / 'floor'.
    """
    if sig_digit < 1:
        raise ValueError(f"sig_digit must be at least 1, got: {sig_digit}")
    arr = np.asarray(arr_like)
    with np.errstate(divide="ignore", invalid="ignore"):
        log10 = np.where(arr == 0, 0, np.floor(np.log10(np.abs(arr))))
    factor = 10 ** (sig_digit - 1 - log10)
    if mode == "round":
        rdfunc = np.round  # type: ignore[assignment]
        # overloaded expression function, following the same
    elif mode == "ceil":
        rdfunc = np.ceil  # type: ignore[assignment]
    elif mode == "floor":
        rdfunc = np.floor  # type: ignore[assignment]
    else:
        raise ValueError(f"mode must one of 'round', 'ceil' and 'floor', but got: '{mode}'")
    result: np.ndarray = np.asarray(rdfunc(arr_like * factor) / factor)
    return result


# For any scalar real number regardless of types - behavioral validation
@overload
def round_digit(value: RealNumber, sig_digit: int, mode: str) -> float: ...  # type: ignore[overload-cannot-match]


# Mypy failure on GitHub, the code works and passes local mypy validation, following the same


@overload
def round_digit(value: tuple, sig_digit: int, mode: str) -> tuple: ...  # type: ignore[overload-cannot-match]


@overload
def round_digit(value: list, sig_digit: int, mode: str) -> list: ...  # type: ignore[overload-cannot-match]


@overload
def round_digit(value: np.ndarray, sig_digit: int, mode: str) -> np.ndarray: ...  # type: ignore[overload-cannot-match]


@overload
def round_digit(  # type: ignore[overload-cannot-match]
    value: torch.Tensor, sig_digit: int, mode: str
) -> torch.Tensor: ...


@overload
def round_digit(  # type: ignore[overload-cannot-match]
    value: pd.DataFrame, sig_digit: int, mode: str
) -> pd.DataFrame: ...


@simple_type_validator
def round_digit(
    value: Union[RealNumber, Annotated[Any, arraylike_validator()]],
    sig_digit: int,
    mode: str = "round",
) -> Union[float, tuple, list, np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]:
    """
    Round values or values in arraylike to specified significant digits.

    Parameters
    ----------
    value : real number or array-like
        Real number or array-like of real numbers.

        Supported array-like types: tuple, list, ``numpy.ndarray``, ``pandas.Series``, ``torch.Tensor``.

    sig_digit : int
        Number of significant digits, must be at least 1.

    mode : str, optional
        Rounding mode, choose between:

        - ``"round"``
        - ``"ceil"``
        - ``"floor"``

        The default is ``"round"``.

    Returns
    -------
    result : float or tuple or list or numpy.ndarray or pandas.Series or pandas.DataFrame or torch.Tensor
        Rounded value or array-like of values.

        If the input is a number, returns a float.
        If the input is array-like, returns an object of the same type containing rounded values.

    Examples
    --------
    Round numbers::

        >>> round_digit(130.33, 2)
        >>> round_digit(0.323233, 2, 'ceil')
        >>> round_digit(138323, 2, 'floor')

    Round array-like of numbers::

        >>> round_digit([[1111, 2222], [9999, 7777]], 2)
    """
    if isinstance(value, (list, tuple, np.ndarray, pd.Series, pd.DataFrame, torch.Tensor)):
        value1 = np.asarray(value)
        result = np_sig_digit(value1, sig_digit, mode)
        if isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)):
            value[:] = result[:]
        elif isinstance(value, list):
            value = result.tolist()
        elif isinstance(value, torch.Tensor):
            value = torch.tensor(result)
        elif value1.ndim == 1:
            value = tuple(result)
        else:
            value = tuple(tuple(map(tuple, result.tolist())))  # noqa: C414
            # The double tuples are necessary
        return value
    else:
        return num_sig_digit(value, sig_digit, mode)


# %% Parameter binders


@simple_type_validator
def make_img_func(
    func: Callable,
    name_suffix: str = "img_only",
    *fixed_args: object,
    **fixed_kwargs: object,
) -> Callable:
    """
    Create a function from the given function that accepts image path only, fixing other parameters.

    Parameters
    ----------
    func : callable
        Original function with signature ``func(image_path, *args, **kwargs)``.
    name_suffix : str, optional
        The name suffix for the output function, default is ``"img_only"``.
    ``*fixed_args`` :
        Positional arguments to fix.
    ``**fixed_kwargs`` :
        Keyword arguments to fix.

    Returns
    -------
    callable
        Modified function that accepts only image path.

    Examples
    --------
    Suppose the original function is used as follows::

        >>> processed_img_path = image_processing_function("/image1.tif", param1=1, param2=2)

    A version that only requires image path is created as follows::

        >>> func_accept_img_path = make_img_func(image_processing_function, param1=1, param2=2)

    Usage of the modified function::

        >>> processed_img_path = func_accept_img_path("/image1.tif")
    """
    n_data_args = 1

    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) < n_data_args:
        raise TypeError("Function must accept at least 1 data parameter of image path.")

    data_params = params[:n_data_args]

    for i, p in enumerate(data_params, start=1):
        if p.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(f"Data parameter {i} must be positional.")

    @simple_type_validator
    def img_only(
        image_path: str,
    ) -> Any:
        result = func(image_path, *fixed_args, **fixed_kwargs)
        return result

    img_only.__name__ = f"{func.__name__}_{name_suffix}"
    img_only.__qualname__ = img_only.__name__

    return img_only


@simple_type_validator
def make_roi_func(
    func: Callable,
    name_suffix: str = "roi_only",
    *fixed_args: object,
    **fixed_kwargs: object,
) -> Callable:
    """
    Create a function from the given function that accepts only image path and region of interest (ROI) coordinates, fixing other parameters.

    Parameters
    ----------
    func : callable
        Original function with signature ``func(image_path, roi_coordinates, *args, **kwargs)``.
    name_suffix : str, optional
        The name suffix for the output function, default is ``"roi_only"``.
    ``*fixed_args`` :
        Positional arguments to fix.
    ``**fixed_kwargs`` :
        Keyword arguments to fix.

    Returns
    -------
    callable
        A function that accepts only image path and ROI coordinates.

    Examples
    --------
    Suppose the original function is used as follows::

        >>> roi_spectra_array = roi_processing_function(
        ...     "/image1.tif", [[(0, 0), (0, 10), (10, 0), (0, 0)]], param1=1, param2=2)

    A version that only requires image path and ROI coordinates is created as follows::

        >>> func_accept_img_roi = make_roi_func(roi_processing_function, param1=1, param2=2)

    Usage of the modified function::

        >>> roi_spectra_array = func_accept_img_roi("/image1.tif", [[(0, 0), (0, 10), (10, 0), (0, 0)]])
    """  # noqa: E501
    n_data_args = 2

    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) < n_data_args:
        raise TypeError(
            "Function must accept at least 2 data parameters: an image path and a list of ROI coordinate lists."
        )

    data_params = params[:n_data_args]

    for i, p in enumerate(data_params, start=1):
        if p.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(f"Data parameter {i} must be positional.")

    @simple_type_validator
    def roi_only(
        image_path: str,
        roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]],
    ) -> Any:
        result = func(image_path, roi_coordinates, *fixed_args, **fixed_kwargs)
        return result

    roi_only.__name__ = f"{func.__name__}_{name_suffix}"
    roi_only.__qualname__ = roi_only.__name__

    return roi_only


@simple_type_validator
def make_array_func(
    func: Callable,
    name_suffix: str = "x_only",
    *fixed_args: object,
    **fixed_kwargs: object,
) -> Callable:
    """
    Create a function from the given function that accepts an array-like only, fixing other parameters.

    Parameters
    ----------
    func : callable
        Original function with signature ``func(array-like, *args, **kwargs)``.
    name_suffix : str, optional
        The name suffix for the output function, default is 'x_only'.
    ``*fixed_args`` :
        Positional arguments to fix.
    ``**fixed_kwargs`` :
        Keyword arguments to fix.

    Returns
    -------
    callable
        A function that accepts array-like data only.

    Examples
    --------
    Suppose the original function is used as follows::

        >>> result = array_processing_function(array, param1=1, param2=2)

    A version that only requires the array argument is created as follows::

        >>> func_accept_arr = make_array_func(array_processing_function, param1=1, param2=2)

    Usage of the modified function::

        >>> result = func_accept_arr(array)
    """
    n_data_args = 1

    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) < n_data_args:
        raise TypeError("Function must accept at least 1 data parameter of data array.")

    data_params = params[:n_data_args]

    for i, p in enumerate(data_params, start=1):
        if p.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(f"Data parameter {i} must be positional.")

    @simple_type_validator
    def arr_only(
        data_array_2d: Annotated[Any, arraylike_validator()],
    ) -> Any:
        result = func(data_array_2d, *fixed_args, **fixed_kwargs)
        return result

    arr_only.__name__ = f"{func.__name__}_{name_suffix}"
    arr_only.__qualname__ = arr_only.__name__

    return arr_only


# %% Image and ROI functions


# Extract ROI spectra
@simple_type_validator
def roispec(
    image_path: str,
    roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]],
    as_type: Union[type, str] = "float32",
) -> Annotated[Any, arraylike_validator(ndim=2)]:
    """
    Extract spectra from a spectral image in a (multi-)polygon region of interest (ROI) defined with vertex coordinate pairs.

    Parameters
    ----------
    image_path : str
        Spectral image path.

    roi_coordinates : list of list of tuple of 2 (int or float)
        Coordinates of the ROI polygons.
        Structure::

            [
                [ (x1, y1), (x2, y2), ..., (xn, yn), (x1, y1) ],  # Polygon 1
                [ (x1, y1), (x2, y2), ..., (xm, ym), (x1, y1) ],  # Polygon 2
                ...
            ]

        Each inner list represents a polygon (for multipart geometries), and each tuple is a vertex coordinate.

    as_type : type or str, optional
        Desired numeric data type for the extracted spectral values.
        Supports ``numpy`` numeric dtypes. Default is 'float32'.

    Returns
    -------
    numpy.ndarray
        Array of pixel spectra in the specified ROI, each row represents the spectrum values of a pixel.

    Examples
    --------
    >>> roispec("/image1.tif", [[(0, 0), (10, 0), (0, 10), (0, 0)]])
    """  # noqa: E501

    # Silencing NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    # Validate roi_coordinates
    # Number of valid polygons
    npoly = 0
    if len(roi_coordinates) > 0:
        for roi_coords in roi_coordinates:
            if len(roi_coords) >= 3:
                npoly = npoly + 1
    if npoly == 0:
        raise ValueError(f"No valid polygon found in the given roi_coordinates: {roi_coordinates}")

    # For each polygon
    for i, polygon_coords in enumerate(roi_coordinates):
        # Create a Polygon object from the coordinates
        polygon = Polygon(polygon_coords)

        with rasterio.open(image_path) as src:
            # Get raster bounds
            raster_bbox = box(*src.bounds)
            # Validate Polygon
            if raster_bbox.intersects(polygon):
                # Apply the polygon mask to raster
                out_image, out_transform = mask(src, [polygon], crop=True, nodata=src.nodata)
                # Flatten the bands
                out_image = out_image.reshape(out_image.shape[0], -1)
                # Get valid pixel values
                out_image = out_image[:, np.sum(out_image, axis=0) > 0]
                out_image = out_image.astype(as_type)
            else:
                raise ValueError(
                    f"\nROI located out of raster image bounds, \ngot ROI bounds: {polygon.bounds} , \
                        \nimage bounds: {src.bounds}"
                )

        # Concatenate results if ROI is multipolygon
        if i == 0:
            out_image_total = out_image
        else:
            out_image_total = np.concatenate((out_image_total, out_image), axis=1)

    # Type conversion
    out_image_total = out_image_total.astype(as_type)

    # Return data default to row as sample
    out_image_total = out_image_total.T

    # Recover NotGeoreferencedWarning
    warnings.filterwarnings("default", category=rasterio.errors.NotGeoreferencedWarning)

    return out_image_total


# ROI pixel counter
@simple_type_validator
def pixcount(
    image_path: str,
    roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]],
    band: int = 1,
    threshold: Optional[tuple[Union[int, float], Union[int, float]]] = None,
) -> int:
    """
    Count valid pixel number within a region of interest (ROI) within a threshold at a specific band.

    Parameters
    ----------
    image_path : str
        Spectral raster image path.

    roi_coordinates : list of list of tuple of 2 (int or float)
        Coordinates of the ROI polygons.
        Structure::

            [
                [ (x1, y1), (x2, y2), ..., (xn, yn), (x1, y1) ],  # Polygon 1
                [ (x1, y1), (x2, y2), ..., (xm, ym), (x1, y1) ],  # Polygon 2
                ...
            ]

        Each inner list represents a polygon (for multipart geometries), and each tuple is a vertex coordinate.

    band : int, optional
        Reference band, if ``threshold`` value is provided, values at this band is used for ``threshold``.

    threshold : tuple of 2 (int or float) or None, optional
        The band value is compared with threshold, only pixel within the threshold are counted.

        If None, all pixels within the ROI are counted regardless of their band values.

    Returns
    -------
    int
        Valid pixel number within a ROI whose value at the specified band does not exceed the threshold.

    Examples
    --------
    >>> pixcount("/image1.tif", [[(0, 0), (10, 0), (0, 10), (0, 0)]], band=12)
    >>> pixcount("/image1.tif", [[(0, 0), (10, 0), (0, 10), (0, 0)]], band=12, threshold=(1000, 3000))
    """

    # Silencing NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    # For each polygon
    total_count = 0
    for polygon_coords in roi_coordinates:
        # Create a Polygon object from the coordinates
        polygon = Polygon(polygon_coords)

        with rasterio.open(image_path) as src:
            # Get raster bounds
            raster_bbox = box(*src.bounds)
            # Validate Polygon
            if raster_bbox.intersects(polygon):
                # Count
                masked, _ = mask(src, [polygon], crop=True, nodata=src.nodata, indexes=[band], all_touched=True)
                if threshold is None:
                    count: int = int(np.sum(masked[0] != src.nodata))
                else:
                    if threshold[0] >= threshold[1]:
                        raise ValueError(f"Invalid threshold range: {threshold}")
                    count = int(
                        np.sum((masked[0] != src.nodata) & (masked[0] >= threshold[0]) & (masked[0] <= threshold[1]))
                    )
            else:
                count = 0

        total_count = total_count + count

    # Recover NotGeoreferencedWarning
    warnings.filterwarnings("default", category=rasterio.errors.NotGeoreferencedWarning)

    return total_count


# ROI pixel counter
@simple_type_validator
def pixcounts(
    image_path: str,
    roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]],
    band: Union[int, list[int]],
    threshold: Union[
        tuple[Union[int, float], Union[int, float]], list[tuple[Union[int, float], Union[int, float]]], None
    ] = None,
) -> pd.DataFrame:
    """
    Count pixels within a region of interest (ROI), optionally filtered by value threshold(s) on specified spectral band(s).

    Pixels with nodata values in the specified band(s) are excluded from the count.

    Parameters
    ----------
    image_path : str
        Spectral raster image path.

    roi_coordinates : list of list of tuple of 2 (int or float)
        Coordinates of the ROI polygons.
        Structure::

            [
                [ (x1, y1), (x2, y2), ..., (xn, yn), (x1, y1) ],  # Polygon 1
                [ (x1, y1), (x2, y2), ..., (xm, ym), (x1, y1) ],  # Polygon 2
                ...
            ]

        Each inner list represents a polygon (for multipart geometries), and each tuple is a vertex coordinate.

    band : int or list of int
        Band index or indices used for threshold filtering.

        When ``threshold`` is provided, pixel values from these band(s) are compared against the corresponding threshold(s).

    threshold : tuple of 2 real numbers or list of tuple of 2 real numbers or None, optional
        Threshold range(s) used to filter pixels based on band values.

        - If a single tuple is provided, the same threshold is applied to all bands.
        - If a list of tuples is provided, each threshold corresponds to the respective band.
        - If None, all pixels within the ROI are counted regardless of band values.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing pixel count(s) at the specified band(s), restricted to pixels whose values fall within the specified threshold(s).

    Examples
    --------
    Count pixel for a single band::

        >>> pixcounts("/image1.tif", [[(0, 0), (10, 0), (0, 10), (0, 0)]], band=12)
        >>> pixcounts("/image1.tif", [[(0, 0), (10, 0), (0, 10), (0, 0)]], band=12, threshold=(1000, 3000))

    Count pixel for multiple bands::

        >>> pixcounts("/image1.tif", [[(0, 0), (10, 0), (0, 10), (0, 0)]], band=[2,3])
        >>> pixcounts("/image1.tif", [[(0, 0), (10, 0), (0, 10), (0, 0)]], band=[2,3], threshold=[(1, 200), (200, 400)])
    """  # noqa: E501

    # Validate bands and thresholds
    if isinstance(band, int):
        band = [band]
    if threshold is not None:
        if isinstance(threshold, tuple):
            threshold = [threshold]
        if len(threshold) != len(band):
            raise ValueError(
                f"Number of thresholds does not match the number of bands, \
                    got number of threshold: {len(threshold)}, got number of bands: {len(band)}."
            )
    else:
        threshold = [
            (-1e38, 1e38),
        ] * len(band)
    assert isinstance(threshold, list)

    band_names = []
    total_counts = []
    for band_i, threshold_i in zip(band, threshold):
        if band_i < 1:
            raise ValueError(f"band index must be at least 1, got: {band_i}")
        band_names.append(f"Band_{band_i}")
        if threshold is None:
            threshold_i = None
        total_counts.append(
            pixcount(
                image_path=image_path,
                roi_coordinates=roi_coordinates,
                band=band_i,
                threshold=threshold_i,
            )
        )

    df_counts = pd.DataFrame([total_counts], columns=band_names)

    return df_counts


# Find min sampling bbox from a ROI with min valid pixels
@simple_type_validator
def minbbox(
    image_path: str,
    roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]],
    minimum_valid_pixel: int,
    band: int = 1,
    valid_threshold: Optional[tuple[Union[int, float], Union[int, float]]] = None,
) -> list[list[tuple[float, float]]]:
    """
    Get minimum bounding box of a ROI that has at least some number of valid pixels with band value within a threshold.
    """
    # Get ROI bounding box
    carr = np.asarray(roi_coordinates[0])
    xmin, xmax = min(carr[:, 0]), max(carr[:, 0])
    ymin, ymax = min(carr[:, 1]), max(carr[:, 1])
    cdbounds = [[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]

    # Count total number
    total_count = pixcount(image_path, roi_coordinates, band, valid_threshold)
    incx, incy = (xmax - xmin), (ymax - ymin)
    if total_count < 4 * minimum_valid_pixel:
        return cdbounds
    while True:
        incx, incy = incx / 2, incy / 2
        for i in (0, 1):
            for j in (0, 1):
                xmin, xmax = (xmin + incx * i), (xmin + incx * (i + 1))
                ymin, ymax = (ymin + incx * j), (ymin + incx * (j + 1))
                mbounds = [[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]
                total_count = pixcount(image_path, mbounds, band, valid_threshold)
                if total_count > 4 * minimum_valid_pixel:
                    break
                else:
                    return mbounds
            if total_count > 4 * minimum_valid_pixel:
                break


# %% Stats metrics of extracted 2D array data


# Derivative calculator for 2d-array
@simple_type_validator
def nderiv(  # noqa: C901
    data_array_2d: Annotated[Any, arraylike_validator(ndim=2)],
    n: int,
    axis: int = 1,
    padding: Union[int, float, str, None] = 'nan',
) -> Annotated[Any, arraylike_validator(ndim=2)]:
    """
    Compute the n-th numerical derivative of 1D data series contained in a 2D array-like object.

    Each row or column is treated as an independent 1D series, depending on the selected axis.

    Parameters
    ----------
    data_array_2d : array-like of real numbers
        Two-dimensional array-like input data.

        The input must be convertible to a 2D array and support numeric operations.
        Each 1D slice along the computation axis is treated as an independent variable.

    n : int
        Order of the derivative. Must be a positive integer.

    axis : int, optional
        Axis along which to compute the derivative.

        - ``0`` : compute derivatives row-wise (each row is a variable, columns are samples)
        - ``1`` : compute derivatives column-wise (each column is a variable, rows are samples)

        Default is ``1``.

    padding : int or float or str or None, optional
        Boundary padding strategy applied to derivative arrays. Choose between:

        - ``"nan"`` : pad with NaN values
        - ``"edge"`` : pad using edge values
        - int or float : pad with the specified numeric value
        - ``None`` : no padding

        If ``None`` is specified, the output size along the computation axis is reduced by ``2 * n``.
        Default is ``"nan"``.

    Returns
    -------
    array-like
        The n-th derivative of the input data along the specified axis.

        If padding is applied, the output has the same shape as the input.
        Otherwise, the output size is reduced along the computation axis.

    Examples
    --------
    Basic usage::

        >>> x = [[1, 2, 3, 4],
        ...      [5, 6, 7, 8],
        ...      [9, 10, 11, 12],
        ...      [13, 14, 15, 16]]

        >>> nderiv(x, 1)
        >>> nderiv(x, 2)

    Compute along a different axis::

        >>> nderiv(x, 2, axis=0)

    Use different padding strategies::

        >>> nderiv(x, 2, padding="nan")
        >>> nderiv(x, 2, padding="edge")

    Pad with a custom numeric value::

        >>> nderiv(x, 2, padding=0)

    Disable padding::

        >>> nderiv(x, 2, padding=None)
    """

    data_array_2d = np.asarray(data_array_2d).astype("float32")

    if axis == 0:
        data_array_2d = data_array_2d.T

    # Validate length of data series.
    dlen = data_array_2d.shape[1]
    if dlen < 2 * n:
        raise ValueError(
            f"Insufficient data length. \
                The input data must have at least {2 * n} elements to compute the {n}th derivative."
        )

    # Compute nth derivative
    if n == 0:
        derivative = data_array_2d
    else:
        for od in range(1, n + 1):
            # Create an array to hold the n-th derivative
            derivative = np.zeros_like(data_array_2d)
            for i in range(od, dlen - od):
                derivative[:, i] = (data_array_2d[:, i + 1] - data_array_2d[:, i - 1]) / 2
            data_array_2d = derivative

        # Padding
        if padding is None:
            derivative = derivative[:, od : dlen - od]
        elif type(padding) is str:
            if padding is None or padding.lower() == 'nan':
                ed = np.nan
                derivative[:, 0:od] = ed
                derivative[:, dlen - od : dlen] = ed
            elif padding.lower() == 'edge':
                derivative[:, 0:od] = np.repeat(derivative[:, od : (od + 1)], od, axis=1)
                derivative[:, (dlen - od) : dlen] = np.repeat(derivative[:, (dlen - od - 1) : (dlen - od)], od, axis=1)
            else:
                raise ValueError(f"Invalid padding method name '{padding}'. Choose from 'nan' or 'edge'.")
        else:
            assert isinstance(padding, float) or isinstance(padding, int)
            ed = padding
            derivative[:, 0:od] = ed
            derivative[:, dlen - od : dlen] = ed

    if axis == 0:
        derivative = derivative.T

    return derivative


# Computing axis converter
@simple_type_validator
def axisconv(
    data_array_2d: Annotated[Any, arraylike_validator(ndim=2, as_type=float)],
    axis: int,
    astype: type,
) -> np.ndarray:
    """
    Transpose array to make computing axis 0.
    """
    arr: np.ndarray
    if axis == 0:
        arr = np.asarray(data_array_2d).astype(astype)
    elif axis == 1:
        arr = np.asarray(data_array_2d).astype(astype).T
    else:
        raise ValueError(f"Axis must be 0 or 1, got: {axis}")
    return arr


# Statistical moment
@simple_type_validator
def moment2d(  # noqa: C901
    data_array_2d: Annotated[Any, arraylike_validator(ndim=2, as_type=float)],
    n: int,
    standardized: bool = False,
    reference: Optional[tuple[Union[int, float]]] = None,
    axis: int = 0,
    zero_division: str = "add",
) -> tuple[float, ...]:
    """
    Compute the n-th statistical moment of 2D array-like data along a specified axis.

    Parameters
    ----------
    data_array_2d : array-like
        2D array-like of int or float. Nan value is omited.

    n : int
        Order of the statistical moment.

    standardized : bool, optional
        Whether to compute standardized moments.

        If True, the moment is scaled by the standard deviation of the data.
        Default is False.

    reference : tuple[Union[int,float]], optional
        Reference point of the moment.

        If not given, the reference point is 0 for first moment and mean for higher-order moments.

    axis : int, optional
        Axis along which the moment is computed.

        - ``0`` : compute moments column-wise (each column is a variable)
        - ``1`` : compute moments row-wise (each row is a variable)

        The default is ``0``.

    zero_division : str
        Choose between:

            - ``"add"`` : add a small number (1e-30) to denominator to avoid zero.
            - ``"replace"`` : replace zero with a small number (1e-30) in the denominator.
            - ``"nan"`` : return nan for zero divisions.
            - ``"numpy"`` : use default approach of numpy, i.e. return nan for 0 / 0, and +/-inf for non-zero values.
            - ``"raise"`` : raise Error for zero divisions.

        The default is ``"add"``.

    Returns
    -------
    tuple of float
        Target moment.

    Examples
    --------
    Basic usage::

        >>> x = np.array(
        ...     [[1, 2, 3, 4],
        ...      [5, 6, 7, 8],
        ...      [9, 10, 11, 12],
        ...      [13, 14, 15, 16]]
        ... )

        >>> moment2d(x, n=1)
        >>> moment2d(x, n=2)

    Compute along a different axis::

        >>> moment2d(x, n=2, axis=1)

    Change zero-division handling::

        >>> moment2d(x, n=2, zero_division="replace")

    Disable standardization::

        >>> moment2d(x, n=4, standardized=False)
    """

    # Standardize axis
    arr = axisconv(data_array_2d, axis, float)

    # Validate sample size
    nsample = arr.shape[0]
    if nsample < n:
        raise ValueError(f"Sample size must be at least {n} to compute {n}th moment, got: {nsample}")

    # Number of variables
    nvar = arr.shape[1]

    if (n == 1) & (reference is None):
        mom = np.nanmean(arr, axis=0)
    else:
        # Reference point
        if reference is None:
            ref = np.nanmean(arr, axis=0, keepdims=True)
        elif len(reference) == nvar:
            ref = np.asarray(reference).reshape(1, -1)
        else:
            raise ValueError(
                f"Reference and data_array_2d must have same number of variables, \
                    but given reference has {len(reference)}, while given data has {nvar}"
            )

        # Moment
        dev = arr - ref
        mom = np.nanmean(dev**n, axis=0)

        # Standardization
        if standardized:
            std = np.nanstd(arr, axis=0)
            denom = std**n
            # Zero division treatment
            if zero_division.lower() == "add":
                denom = denom + 1e-30
            elif zero_division.lower() == "replace":
                denom[denom == 0] = 1e-30
            elif zero_division.lower() == "nan":
                denom[denom == 0] = np.nan
            elif zero_division.lower() == "numpy":
                pass
            elif zero_division.lower() == "raise":
                if np.any(denom == 0):
                    raise ValueError(f"\nZero value found in moment denominator: \n\n{denom}\n")
            else:
                raise ValueError(
                    f"Invalid zero_division policy '{zero_division}', \
                        zero_division must be 'add'/'replace'/'nan'/'numpy' or 'raise'."
                )
            mom = mom / denom
    return tuple(mom)


# Calculate band value histogram values - band value distribution
@simple_type_validator
def bandquant(
    spec_array_2d: Annotated[Any, arraylike_validator(ndim=2, as_type=float)],
    band: int,
    bins: Union[int, Annotated[Any, arraylike_validator(ndim=1)]],
    axis: int = 0,
) -> tuple[float, ...]:
    """
    Compute quantile values of a specified spectral band from a 2D array-like spectral dataset.

    Quantiles may be specified either by the number of bins or by an explicit sequence of quantile levels.

    Parameters
    ----------
    spec_array_2d : array-like of real number
        Two-dimensional spectral data array.

    band : int
        Band index used for quantile computation.

    bins : int or array-like of real number
        Number of quantile bins to compute, or an explicit sequence of quantile levels.

        Quantile levels must be within the range of [0, 1].

    axis : int, optional
        Axis along which the computation is performed.

        - ``0`` : treat each array row as a sample
        - ``1`` : treat each array column as a sample

        Default is ``0``.

    Returns
    -------
    tuple of float
        Quantile values of the specified band at the given bins or quantile levels.

    Examples
    --------
    Basic usage with a fixed number of bins::

        >>> import numpy as np
        >>> x = np.random.randint(0, 10000, size=(100, 100))

        >>> bandquant(x, band=1, bins=10)

    Use explicit quantile levels::

        >>> bandquant(x, band=1, bins=[500, 1000, 1500, 2000, 2500, 3000])

    Compute along a different axis::

        >>> bandquant(x, band=1, bins=10, axis=1)
    """

    # Validate bins
    if not isinstance(bins, int):
        bins = list(np.array(bins))

    # Standardize axis
    arr = axisconv(spec_array_2d, axis, float)

    # Number of variables
    nvar = arr.shape[1]

    # Validate band
    if (band >= nvar) or (band < -nvar):
        raise ValueError(f"band index exceeds the number of bands, got: {band}, number of bands: {nvar}")

    # Bins to quantiles
    if type(bins) is int:
        if bins < 2:
            raise ValueError(f"number of bins must be greater than 1, got number of bins: {bins}")
        bs = [float(i / (bins - 1)) for i in range(bins)]
    elif type(bins) is list:
        if (max(bins) <= 1) & (min(bins) >= 0):
            bs = bins
        else:
            raise ValueError(f"Bin value must be in range of [0, 1], got range: [{min(bins)}, {max(bins)}]")
        if len(bins) < 2:
            raise ValueError(f"number of bins must be greater than 1, got number of bins: {len(bins)}")

    # Calculate quantile values
    values: tuple[float, ...] = ()
    for q in bs:
        v = float(np.nanquantile(arr[:, band], q, axis=0))
        values = values + (v,)

    return values


# %% Common and custom statistical measures for 2D array-like


# Helper: invalid measure
@simple_type_validator
def invalid_measure(
    v: Annotated[Any, arraylike_validator(ndim=2, as_type=float)],
    axis: Optional[int] = 0,
    *args: object,
    **kwargs: object,
) -> Union[np.ndarray, float]:
    if axis is None:
        return np.nan
    elif axis == 0:
        result = np.full(v.shape[1], np.nan, dtype=float)
    elif axis == 1:
        result = np.full(v.shape[0], np.nan, dtype=float)
    else:
        raise ValueError(f"axis must be 0, 1 or None, got: {axis}")
    assert isinstance(result, np.ndarray)
    return result


# Helper: stats measure opter
@simple_type_validator
def smopt(measure: str) -> Callable:  # noqa: C901
    """
    Return common statistical measure function by measure name.
    """

    # Skewness and kurtosis modified for zero std
    @simple_type_validator
    def stskew(v: Annotated[Any, arraylike_validator(ndim=2, as_type=float)], axis: int) -> tuple[float]:
        skewness: tuple[float] = moment2d(v, 3, standardized=True, axis=axis)
        return skewness

    @simple_type_validator
    def stkurt(v: Annotated[Any, arraylike_validator(ndim=2, as_type=float)], axis: int) -> tuple[float]:
        kurtosis: tuple[float] = moment2d(v, 4, standardized=True, axis=axis)
        return kurtosis

    # Single common measure function
    if type(measure) is str:
        if str(measure).lower() == "mean":
            mfunc = np.nanmean
        elif str(measure).lower() == "median":
            mfunc = np.nanmedian  # type: ignore[assignment]
            # overloaded expression function, follows assignment error the same reason
        elif str(measure).lower() == "min":
            mfunc = np.nanmin  # type: ignore[assignment]
        elif str(measure).lower() == "max":
            mfunc = np.nanmax  # type: ignore[assignment]
        elif str(measure).lower() == "var":
            mfunc = np.nanvar  # type: ignore[assignment]
        elif (str(measure).lower() == "std") | (str(measure).lower() == "stdev") | (str(measure).lower() == "sd"):
            mfunc = np.nanstd  # type: ignore[assignment]
        elif (str(measure).lower() == "skewness") | (str(measure).lower() == "skew"):
            mfunc = stskew
        elif (str(measure).lower() == "kurtosis") | (str(measure).lower() == "kurt"):
            mfunc = stkurt
        else:
            raise ValueError("Undefined measure name, please provide the measure function instead")

    def msfunc(arr: Annotated[Any, arraylike_validator(ndim=2, as_type=float)]) -> np.ndarray:
        result: np.ndarray = np.asarray(mfunc(arr, axis=0))
        return result

    return msfunc


# Helper: custom measure function validator
@simple_type_validator
def cmval(arr: Annotated[Any, arraylike_validator(ndim=2)], cfunc: Callable) -> Callable:
    """
    Custom measure function validator
    """
    cfunc_name = cfunc.__name__
    test_arr = np.asarray(arr).astype(float)[0:5, :]
    try:
        testv = cfunc(test_arr)
    except Exception as e:
        raise ValueError(
            f"Error in the testing given custom measure \n'{cfunc_name}' on the provided data: \n{e}"
        ) from e
    testv = np.asarray(testv)
    if np.issubdtype(testv.dtype, np.number):
        if testv.ndim > 1:
            raise ValueError(
                f"Given custom measure function \n'{cfunc_name}' \nreturned result shape {testv.shape}, expected (1,)"
            )
    else:
        raise TypeError(
            f"Given custom measure function \n'{cfunc_name}' \nmust return numbers, got datatype: {testv.dtype}"
        )
    return cfunc


# %% Common statistical measures for 2D array-like


# Common statistical measures for 2D array-like
class Stats2d:
    """
    Statistical measure calculator for 2D array-like data.

    Each row or column of the input is treated as a sample, depending on the selected axis.

    For parallel or standalone computation of individual statistics, prefer methods:

        ``mean``, ``std``, ``skew``, ``kurt``, ``minimum``, ``median``, ``maximum``

    instead of factory methods:

        ``summary`` and ``values``

    Attributes
    ----------
    axis : int, optional
        Axis along which statistics are computed.

        - ``0`` : treat each row as a sample
        - ``1`` : treat each column as a sample

        Default is ``0``.

    measure : str or callable or list of (str or callable) or None, optional
        Statistical measure(s) to be used by :meth:`Stats2d.values`.

        - Common measures may be specified by name: ``"mean"``, ``"std"``, ``"skewness"``, ``"kurtosis"``, ``"min"``, ``"median"``, ``"max"``.
        - Custom measures must be callables accepting a 2D array-like input and returning a statistical measure with same length as the number of variables.
        - Multiple measures must be provided as a list or tuple of measure names or measure callbles.

        If ``None``, all common measures listed above are computed. Default is ``None``.

    Methods
    -------
    summary
        Compute specified statistical measures for the provided 2D array and return the results as a dictionary.
    values
        Return a callable that computes a single specified statistical measure for a provided 2D array.
        The returned callable is named after the selected measure.
    mean
        Compute the arithmetic mean of a 2D array.
    std
        Compute the standard deviation of a 2D array.
    skew
        Compute the skewness of a 2D array.
    kurt
        Compute the kurtosis of a 2D array.
    minimum
        Compute the minimum value of a 2D array.
    median
        Compute the median value of a 2D array.
    maximum
        Compute the maximum value of a 2D array.

    Examples
    --------
    >>> stats2d = Stats2d(axis=0)
    """  # noqa: E501

    @simple_type_validator
    def __init__(self, axis: int = 0, measure: Union[str, Callable, list[Union[str, Callable]], None] = None) -> None:
        self.measure = measure
        self.axis = axis

    # Simplified measures
    @simple_type_validator
    def mean(self, data_array_2d: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Mean values of a 2D data array with each row or column as a sample.

        Parameters
        ----------
        data_array_2d : array-like
            2D array-like data. Nan value is omited.

        Returns
        -------
        numpy.ndarray
            Measure values of the samples.

        See Also
        --------
        Stats2d

        Examples
        --------
        >>> demo_array = np.random.randint(0, 1000, size=(10, 10))
        >>> Stats2d(axis=0).kurt(demo_array)
        """
        result: np.ndarray = np.asarray(np.nanmean(data_array_2d, axis=self.axis))
        return result

    @simple_type_validator
    def std(self, data_array_2d: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Standard deviation values of a 2D data array with each row or column as a sample.

        Parameters
        ----------
        data_array_2d : array-like
            2D array-like data. Missing value is omited.

        Returns
        -------
        numpy.ndarray
            Measure values of the samples.

        See Also
        --------
        Stats2d

        Examples
        --------
        >>> demo_array = np.random.randint(0, 1000, size=(10, 10))
        >>> Stats2d(axis=0).kurt(demo_array)
        """
        result: np.ndarray = np.asarray(np.nanstd(data_array_2d, axis=self.axis))
        return result

    @simple_type_validator
    def var(self, data_array_2d: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Variance values of a 2D data array with each row or column as a sample.

        Parameters
        ----------
        data_array_2d : array-like
            2D array-like data. Missing value is omited.

        Returns
        -------
        numpy.ndarray
            Measure values of the samples.

        See Also
        --------
        Stats2d

        Examples
        --------
        >>> demo_array = np.random.randint(0, 1000, size=(10, 10))
        >>> Stats2d(axis=0).kurt(demo_array)
        """
        result: np.ndarray = np.asarray(np.nanvar(data_array_2d, axis=self.axis))
        return result

    @simple_type_validator
    def skew(self, data_array_2d: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Skewness values of a 2D data array with each row or column as a sample.

        Parameters
        ----------
        data_array_2d : array-like
            2D array-like data. Missing value is omited.

        Returns
        -------
        numpy.ndarray
            Measure values of the samples.

        See Also
        --------
        Stats2d

        Examples
        --------
        >>> demo_array = np.random.randint(0, 1000, size=(10, 10))
        >>> Stats2d(axis=0).kurt(demo_array)
        """
        result: np.ndarray = np.asarray(moment2d(data_array_2d, 3, standardized=True, axis=self.axis))
        return result

    @simple_type_validator
    def kurt(self, data_array_2d: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Kurtosis values of a 2D data array with each row or column as a sample.

        Parameters
        ----------
        data_array_2d : array-like
            2D array-like data. Missing value is omited.

        Returns
        -------
        numpy.ndarray
            Measure values of the samples.

        See Also
        --------
        Stats2d

        Examples
        --------
        >>> demo_array = np.random.randint(0, 1000, size=(10, 10))
        >>> Stats2d(axis=0).kurt(demo_array)
        """
        result: np.ndarray = np.asarray(moment2d(data_array_2d, 4, standardized=True, axis=self.axis))
        return result

    @simple_type_validator
    def minimum(self, data_array_2d: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Minimum values of a 2D data array with each row or column as a sample.

        Parameters
        ----------
        data_array_2d : array-like
            2D array-like data. Missing value is omited.

        Returns
        -------
        numpy.ndarray
            Measure values of the samples.

        See Also
        --------
        Stats2d

        Examples
        --------
        >>> demo_array = np.random.randint(0, 1000, size=(10, 10))
        >>> Stats2d(axis=0).minimum(demo_array)
        """
        result: np.ndarray = np.asarray(np.nanmin(data_array_2d, axis=self.axis))
        return result

    @simple_type_validator
    def median(self, data_array_2d: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Median values of a 2D data array with each row or column as a sample.

        Parameters
        ----------
        data_array_2d : array-like
            2D array-like data. Missing value is omited.

        Returns
        -------
        numpy.ndarray
            Measure values of the samples.

        See Also
        --------
        Stats2d

        Examples
        --------
        >>> demo_array = np.random.randint(0, 1000, size=(10, 10))
        >>> Stats2d(axis=0).median(demo_array)
        """
        result: np.ndarray = np.asarray(np.nanmedian(data_array_2d, axis=self.axis))
        return result

    @simple_type_validator
    def maximum(self, data_array_2d: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Maximum values of a 2D data array with each row or column as a sample.

        Parameters
        ----------
        data_array_2d : array-like
            2D array-like data. Missing value is omited.

        Returns
        -------
        numpy.ndarray
            Measure values of the samples.

        See Also
        --------
        Stats2d

        Examples
        --------
        >>> demo_array = np.random.randint(0, 1000, size=(10, 10))
        >>> Stats2d(axis=0).maximum(demo_array)
        """
        result: np.ndarray = np.asarray(np.nanmax(data_array_2d, axis=self.axis))
        return result

    # Return stats values only, single measure only
    @simple_type_validator
    def _values(self, data_array_2d: Annotated[Any, arraylike_validator(ndim=2)]) -> np.ndarray:
        """
        Measure values of a 2D data array with each row or column as a sample. See Stats2d.
        This function accepts single measures only.
        """
        measure = self.measure
        if type(measure) is str:
            result: np.ndarray = np.asarray(Stats2d(measure=measure).summary(data_array_2d)[measure])
            return result
        elif callable(measure):
            result = np.asarray(Stats2d(measure=measure).summary(data_array_2d)[measure.__name__])
            return result
        elif measure is None:
            raise ValueError("'measure' must be specified for 'Stats2d' when calling this method, got None.")
        else:
            raise ValueError(f"'Stats2d.values' accepts single measure only, but got multiple measures: {measure}")

    # Return stats values with dynamic function name of the measure
    @property
    def values(self) -> Callable:
        """
        Returns a function computing the values of a single specified statistical measure for the provided 2D array.
        The function is named as the specified measure.

        Parameters
        ----------
        data_array_2d : array-like
            2D array-like data. Missing value is omited.

        Returns
        -------
        callable
            Function accepting 2D arraylike data that computes the specified statistical measure.

        Examples
        --------
        >>> demo_array = np.random.randint(0, 1000, size=(10, 10))
        >>> mean_function = Stats2d(axis=0, measure="mean").values
        >>> mean_function(demo_array)
        >>> mean_function.__name__
        """

        def wrapper(*args, **kwargs) -> np.ndarray:  # type: ignore[no-untyped-def]
            result: np.ndarray = np.asarray(self._values(*args, **kwargs))
            return result

        # Set dynamic name
        if callable(self.measure):
            dname = f"{self.measure.__name__}_values"
        else:
            dname = str(self.measure) + "_values"
        wrapper.__name__ = dname
        wrapper.__qualname__ = f"{self.__class__.__name__}.{dname}"
        return wrapper

    # Stats function, able to return multiple measures at one time
    @simple_type_validator
    def summary(  # noqa: C901
        self,
        data_array_2d: Annotated[Any, arraylike_validator(ndim=2)],
    ) -> dict[str, np.ndarray]:
        """
        Computes the statistical measures in the ``measure`` of this Stats2d instance for a 2D data array.

        Parameters
        ----------
        data_array_2d : array-like
            2D array-like data. Missing value is omited.

        Returns
        -------
        dict
            A dictionary mapping str measure names to resulting value arrays, where:

                - Keys : str
                    Names of the measures, e.g. "mean"
                - Values : ``numpy.ndarray``
                    Results for each measure.

        See Also
        --------
        Stats2d

        Examples
        --------
        >>> demo_array = np.random.randint(0, 1000, size=(10, 10))
        >>> Stats2d(axis=0).summary(demo_array)

        Customize computed measures:
        >>> Stats2d(axis=0, measure=["mean", "std"]).summary(demo_array)
        """
        measure = self.measure
        axis = self.axis

        # Standardize axis
        arr = axisconv(data_array_2d, axis, float)

        # Validate sample size
        nsample = arr.shape[0]
        if nsample < 5:
            # raise ValueError(f"Sample size must be at least 5, got: {nsample}")
            warnings.warn(
                f"Sample size {nsample} is too small for some statistical measures.",
                UserWarning,
                stacklevel=2,
            )

        # Stats configuration
        if measure is not None:
            if (type(measure) is str) or callable(measure):
                measure = [measure]
            if (type(measure) is list) or (type(measure) is tuple):
                mnames = []
                mfuncs = []
                for ms in measure:
                    if type(ms) is str:
                        mnames.append(ms)
                        mfuncs.append(smopt(ms))
                    elif callable(ms):
                        mnames.append(ms.__name__)
                        # Test run
                        ms = cmval(arr, ms)
                        mfuncs.append(ms)
                    else:
                        raise TypeError(
                            f"Type of measure must be string or Callable, got type: {type(ms)} for measure {ms}"
                        )
            else:
                raise TypeError(
                    f"Type of measure must be string or Callable or list of them, got type: {type(measure)}"
                )
        else:
            mnames = ["mean", "std", "skewness", "kurtosis", "min", "median", "max"]
            mfuncs = [smopt(mn) for mn in mnames]
            if nsample >= 3 and nsample < 5:
                mfuncs = mfuncs[0:2] + [invalid_measure] * 2 + mfuncs[4:8]
            if nsample < 3:
                mfuncs = mfuncs[0:1] + [invalid_measure] * 3 + mfuncs[4:8]

        # Computing measures
        mvalues: dict = {}
        for mn, mfunc in zip(mnames, mfuncs):
            mvalues[mn] = mfunc(arr)

        return mvalues


def roi_mean(image_path: str, roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]]) -> np.ndarray:
    """
    Computes spectral mean of the provided region of interests (ROI) in the provided image.

    Parameters
    ----------
    image_path : str
        Spectral raster image path.

    roi_coordinates : list of list of tuple of 2 (int or float)
        Coordinates of the ROI polygons.
        Structure::

            [
                [ (x1, y1), (x2, y2), ..., (xn, yn), (x1, y1) ],  # Polygon 1
                [ (x1, y1), (x2, y2), ..., (xm, ym), (x1, y1) ],  # Polygon 2
                ...
            ]

        Each inner list represents a polygon (for multipart geometries), and each tuple is a vertex coordinate.

    Returns
    -------
    numpy.ndarray
        An array of ROI spectral mean values.

    Examples
    --------
    >>> roi_mean("/image1.tif", [[(0, 0), (0, 10), (10, 0), (0, 0)]])
    """
    result: np.ndarray = np.asarray(Stats2d().mean(roispec(image_path, roi_coordinates, as_type="float64")))
    return result


def roi_std(image_path: str, roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]]) -> np.ndarray:
    """
    Computes spectral mean of the provided region of interests (ROI) in the provided image.

    Parameters
    ----------
    image_path : str
        Spectral raster image path.

    roi_coordinates : list of list of tuple of 2 (int or float)
        Coordinates of the ROI polygons.
        Structure::

            [
                [ (x1, y1), (x2, y2), ..., (xn, yn), (x1, y1) ],  # Polygon 1
                [ (x1, y1), (x2, y2), ..., (xm, ym), (x1, y1) ],  # Polygon 2
                ...
            ]

        Each inner list represents a polygon (for multipart geometries), and each tuple is a vertex coordinate.

    Returns
    -------
    numpy.ndarray
        An array of ROI spectral mean values.

    Examples
    --------
    >>> roi_std("/image1.tif", [[(0, 0), (0, 10), (10, 0), (0, 0)]])
    """
    result: np.ndarray = np.asarray(Stats2d().std(roispec(image_path, roi_coordinates, as_type="float64")))
    return result


def roi_median(image_path: str, roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]]) -> np.ndarray:
    """
    Computes spectral mean of the provided region of interests (ROI) in the provided image.

    Parameters
    ----------
    image_path : str
        Spectral raster image path.

    roi_coordinates : list of list of tuple of 2 (int or float)
        Coordinates of the ROI polygons.
        Structure::

            [
                [ (x1, y1), (x2, y2), ..., (xn, yn), (x1, y1) ],  # Polygon 1
                [ (x1, y1), (x2, y2), ..., (xm, ym), (x1, y1) ],  # Polygon 2
                ...
            ]

        Each inner list represents a polygon (for multipart geometries), and each tuple is a vertex coordinate.

    Returns
    -------
    numpy.ndarray
        An array of ROI spectral median values.

    Examples
    --------
    >>> roi_std("/image1.tif", [[(0, 0), (0, 10), (10, 0), (0, 0)]])
    """
    result: np.ndarray = np.asarray(Stats2d().median(roispec(image_path, roi_coordinates, as_type="float64")))
    return result


# %% ROI spectral angle tools


# Spectral angle calculator
def spectral_angle(
    spec_1: Annotated[Any, arraylike_validator(ndim=1)],
    spec_2: Annotated[Any, arraylike_validator(ndim=1)],
    invalid_raise: bool = False,
) -> float:
    """
    Compute the spectral angle between two one-dimensional spectra.

    The spectral angle is defined as the angle (in radians) between two vectors in spectral space.

    Parameters
    ----------
    spec_1 : 1D array-like of real numbers
        First input spectrum.

    spec_2 : 1D array-like of real numbers
        Second input spectrum.

    invalid_raise : bool, optional
        Whether to raise an error when the spectral angle is undefined (e.g. when the values of one spectrum are all zeros).

        If False, ``numpy.nan`` is returned for invalid inputs.
        If True, an error is raised for undefined spectral angle.
        Default is False.

    Returns
    -------
    float
        Spectral angle in radians.

    Examples
    --------
    Basic usage::

        >>> spectral_angle([1, 2, 3, 4], [2, 3, 4, 5])

    Raise an error for invalid spectra::

        >>> spectral_angle(
        ...     [1, 2, 3, 4],
        ...     [0, 0, 0, 0],
        ...     invalid_raise=True
        ... )
    """  # noqa: E501

    # Convert lists to numpy arrays
    spec_1 = np.asarray(spec_1)
    spec_2 = np.asarray(spec_2)

    spec_angle: float

    # Validate spectral vectors
    if np.all(spec_1 == np.zeros(spec_1.shape)) or np.all(spec_2 == np.zeros(spec_2.shape)):
        if not invalid_raise:
            spec_angle = float(np.nan)
            return spec_angle
        else:
            raise ValueError(f"Undefined spectral angle for given spectral vector: \n{spec_1} \nand \n{spec_2}")

    # Calculate the dot product
    dot_product = np.dot(spec_1, spec_2)

    # Calculate the magnitudes
    magnitude_1 = np.linalg.norm(spec_1)
    magnitude_2 = np.linalg.norm(spec_2)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_1 * magnitude_2)

    # Calculate the spectral angle in radians
    spec_angle = float(np.arccos(cos_theta))

    return spec_angle


# %% Spectral angle of spectra in 2d array to a reference spectrum


# Array spec angles
def spectral_angle_arr(
    spec_array_2d: Annotated[Any, arraylike_validator(ndim=2)],
    reference_spectrum: Annotated[Any, arraylike_validator(ndim=1)],
    axis: int = 0,
    invalid_raise: bool = False,
) -> np.ndarray:
    """
    Compute spectral angles between multiple spectra and a reference spectrum.

    Each spectrum in the input 2D array is compared against the reference spectrum.
    Rows or columns are treated as individual spectra depending on the selected axis.

    Parameters
    ----------
    spec_array_2d : 2D array-like
        2D array-like containing 1D spectra.

    reference_spectrum : 1D array-like
        1D reference spectrum used for comparison.

    axis : int, optional
        Axis along which spectra are defined.

        - ``0`` : each row data represents a spectrum sample.
        - ``1`` : each column data represents a spectrum sample.

        The default is ``0``.

    invalid_raise : bool
        Whether to raise an error when the spectral angle is undefined (e.g. when the values of one spectrum are all zeros).

        If False, ``numpy.nan`` is returned for invalid inputs.
        If True, an error is raised for undefined spectral angle.
        Default is False.

    Returns
    -------
    numpy.ndarray
        1D array containing the spectral angles (in radians) between each input spectrum and the reference spectrum.

    Examples
    --------
    Basic usage::

        >>> x = [[1, 2, 3, 4],
        ...      [5, 6, 7, 8],
        ...      [9, 10, 11, 12],
        ...      [13, 14, 15, 16]]
        >>> ref = [1, 1, 2, 2]

        >>> spectral_angle_arr(x, ref)

    Use different axis::

        >>> spectral_angle_arr(x, ref, axis=1)

    Raise an error for invalid spectra::

        >>> x_invalid = [[1, 2, 3, 4],
        ...              [5, 6, 7, 8],
        ...              [9, 10, 11, 12],
        ...              [0, 0, 0, 0]]

        >>> spectral_angle_arr(x_invalid, ref, invalid_raise=True)
    """  # noqa: E501

    # Data validation
    spec_array_2d = np.asarray(spec_array_2d)
    if spec_array_2d.ndim != 2:
        raise ValueError("input spec_array_2d must be 2d array like.")
    if np.isnan(spec_array_2d).any():
        raise ValueError("input spec_array_2d must not contain nan values.")
    if axis == 0:
        pass
    elif axis == 1:
        spec_array_2d = spec_array_2d.T
    else:
        raise ValueError("axis can only be 0 or 1.")
    reference_spectrum = np.asarray(reference_spectrum)
    if len(reference_spectrum) != spec_array_2d.shape[1]:
        raise ValueError("input spec_array_2d does not match with reference_spectrum.")

    # Calculating spec angles
    npix = spec_array_2d.shape[0]
    spec_angle: np.ndarray = np.zeros(npix)
    for i in range(npix):
        spec_angle[i] = spectral_angle(reference_spectrum, spec_array_2d[i, :], invalid_raise=invalid_raise)
    return spec_angle
