# -*- coding: utf-8 -*-
"""
Swectral - High-performance raster image operation toolset

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Interface
from tqdm import tqdm

# Typing
from typing import Callable, Optional, Union, overload, Any

# Basic data
import numpy as np

# Raster
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
from shapely.geometry import Polygon

# GPU acceleration
import torch

# Local
from .specio import simple_type_validator, unc_path, _wait_for_free_space, _safe_disk_wait


# %% Crop ROI


# Crop ROI from a raster image and save to a new image
@simple_type_validator
def croproi(
    raster_path: str,
    roi_coordinates: list[list[tuple[Union[int, float], Union[int, float]]]],
    output_path: str,
    *,
    _space_wait_timeout: int = 6,
    _reserve_free_pct: float = 5.0,
) -> None:
    """
    Crop a region of interest (ROI) from a raster image and save croped region to a new image.

    Parameters
    ----------
    raster_path : str
        Source raster path.

    roi_coordinates : list of list of tuple of 2 (float or int)
        Coordinates of the ROI polygons.
        Structure::

            [
                [ (x1, y1), (x2, y2), ..., (xn, yn), (x1, y1) ],  # Polygon 1
                [ (x1, y1), (x2, y2), ..., (xm, ym), (x1, y1) ],  # Polygon 2
                ...
            ]

        Each inner list represents a polygon (for multipart geometries), and each tuple is a vertex coordinate.

    output_path : str
        Output raster path.

    Examples
    --------
    >>> croproi("/image1.tif", [[(0, 0), (0, 10), (10, 0), (0, 0)]], output_path="/image1_cropped.tif")
    """
    # Create a list of Polygons from the coordinate lists
    polygons = [Polygon(poly_coords) for poly_coords in roi_coordinates]

    # Open the raster file
    with rasterio.open(unc_path(raster_path)) as src:
        # Crop the raster
        out_image, out_transform = mask(
            src,
            polygons,
            crop=True,
            all_touched=True,  # Include pixels that touch any polygon
        )

        # Copy the metadata
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        # Save the cropped raster
        # Validate available disk space
        raster_meta = out_meta.copy()
        raster_meta["__specpipe_raster_meta_for_size_validation"] = True
        _wait_for_free_space(
            obj=raster_meta,
            path=unc_path(output_path),
            space_wait_timeout=_space_wait_timeout,
            reserve_free_pct=_reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
            obj_size_buffer_coeff=1.05,
        )
        with rasterio.open(unc_path(output_path), "w", **out_meta) as dst:
            dst.write(out_image)


# %% Raster tile-wise apply


def dtype_mapper(  # noqa: C901
    dtype: Union[type, str, np.dtype, torch.dtype],
    map_type: str = "numpy",
    min_compatible: bool = True,
) -> Union[type, str, np.dtype, torch.dtype]:
    """
    Map data types between string representations, Python native types, NumPy, PyTorch, and GDAL raster formats.

    Parameters
    ----------
    dtype : Union[type, str]
        Input data type.
    map_type : str, optional
        Target format, can be 'numpy', 'torch' or 'raster'. The default is 'numpy'.
    min_compatible : bool, optional
        Whether minimal compatible data type is returned.
        For example, 'np.uint16' is not supported in PyTorch, then 'torch.int32' is returned.
        The default is True.

    Returns
    -------
    str or type
        The mapped data type in the target format.
    """
    torch_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "half": torch.float16,
        "double": torch.float64,
        float: torch.float32,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        "int": torch.int32,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        int: torch.int32,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        "uint8": torch.uint8,
        np.uint8: torch.uint8,
        "bool": torch.bool,
        bool: torch.bool,
        np.bool_: torch.bool,
    }
    torch_ext = {
        "uint16": torch.int32,
        "uint32": torch.int32,
        np.uint16: torch.int32,
        np.uint32: torch.int32,
    }
    np_map = {
        "float": np.float32,
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "half": np.float16,
        "double": np.float64,
        float: np.float32,
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        "int": np.int32,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        int: np.int32,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        "uint8": np.uint8,
        torch.uint8: np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
        bool: np.bool_,
        torch.bool: np.bool_,
    }
    raster_type = {
        "float": "float32",
        "double": "float32",
        float: "float32",
        torch.float32: "float32",
        torch.float64: "float64",
        np.float32: "float32",
        np.float64: "float64",
        int: "int32",
        torch.int16: "int16",
        torch.int32: "int32",
        np.int16: "int16",
        np.int32: "int32",
        torch.uint8: "uint8",
        np.uint8: "uint8",
        np.uint16: "uint16",
        bool: "bool",
    }
    raster_ext = {
        "float16": "float32",
        torch.float16: "float32",
        np.float16: "float32",
        "half": "float32",
        "int8": "int16",
        np.int8: "int16",
        torch.int8: "int16",
        "uint32": "int32",
        np.uint32: "int32",
    }
    if map_type.lower() == "numpy":
        if dtype in list(np_map.values()):
            result_np: Union[type, str, np.dtype, torch.dtype] = dtype
            return dtype
        elif dtype in list(np_map.keys()):
            result_np = np_map[dtype]
            return result_np
        else:
            raise ValueError(f"Unsupported NumPy data type: {dtype}")
    elif map_type.lower() == "torch":
        if dtype in list(torch_map.values()):
            result_torch: Union[type, str, np.dtype, torch.dtype] = dtype
            return result_torch
        elif dtype in list(torch_map.keys()):
            result_torch = torch_map[dtype]
            return result_torch
        elif min_compatible & (dtype in list(torch_ext.keys())):
            return torch_ext[dtype]
        else:
            raise ValueError(f"Unsupported PyTorch data type: {dtype}")
    elif map_type.lower() == "raster":
        if dtype in list(raster_type.values()):
            result_ras: Union[type, str, np.dtype, torch.dtype] = dtype
            return result_ras
        elif dtype in list(raster_type.keys()):
            result_ras = raster_type[dtype]
            return result_ras
        elif min_compatible & (dtype in list(raster_ext.keys())):
            result_ras = raster_ext[dtype]
            return result_ras
        else:
            raise ValueError(f"Unsupported raster data type: {dtype}")
    else:
        raise ValueError(f"Unsupported map_type, map_type must be 'numpy', 'torch' or 'raster', got: '{map_type}'")


@overload
def auto_fp(
    data: torch.Tensor,
    scaling: bool = False,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    safety_factor: float = 2.0,
) -> tuple[torch.Tensor, Optional[float]]: ...


@overload
def auto_fp(  # type: ignore[overload-cannot-match]
    data: np.ndarray,
    scaling: bool = False,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    safety_factor: float = 2.0,
) -> tuple[np.ndarray, Optional[float]]: ...


# Mypy failure on GitHub, the code works and passes local mypy validation


def auto_fp(  # noqa: C901
    data: Union[np.ndarray, torch.Tensor],
    scaling: bool = False,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    safety_factor: float = 2.0,
) -> tuple[Union[np.ndarray, torch.Tensor], Optional[float]]:
    """
    Cast numeric data in PyTorch tensor to type FP16 if compatible.
    If scaling set True, the function will try to scale the data to fit FP16, and scaling coefficient is returned if compatible.

    Parameters
    ----------
    data : Union[np.ndarray, torch.Tensor]
        Input data in NumPy ndarray or PyTorch tensor.

    scaling : bool, optional
        If data is scaled to fit FP16. The default is False.

    rtol : float, optional
        Relative tolerance for precision checks. The default is 1e-4.

    atol : float, optional
        Absolute tolerance for small-value checks. The default is 1e-5.

    safety_factor : float, optional
        Multiplier to avoid FP16 limits. The default is 2.0.

    Returns
    -------
    tuple[torch.Tensor, Optional[float]]
        - Scaled data in FP16, if FP16 is compatible after scaling, and scaling factor.
        - Data in FP16, if FP16 is natively compatible, and 1.0.
        - Original data, if FP16 is not compatible or not compatible despite scaling if scaling set True.
    """  # noqa: E501
    # FP16 limits
    fp16_max = 65504 / safety_factor
    fp16_min = 6.10e-5 * safety_factor

    ## For numpy array
    if type(data) is np.ndarray:
        short_types = [np.int8, np.uint8, np.int16, np.uint16, np.float16]
        long_types = [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64]

        if data.dtype in short_types:
            return (data, None)

        elif data.dtype in long_types:
            # Get min and max values of non_zeros
            max_val: float = float(np.max(np.abs(data)))
            non_zero_data = data[data != 0]
            if non_zero_data.size > 0:
                min_val: float = float(np.min(np.abs(non_zero_data)))
            else:
                min_val = fp16_min

            # If data fits in FP16 natively
            if max_val <= fp16_max and min_val >= fp16_min:
                arr_fp16: np.ndarray = data.astype(np.float16)
                if np.allclose(data, arr_fp16.astype(data.dtype), rtol=rtol, atol=atol):
                    return (arr_fp16, 1.0)

            # If data can be scaled into FP16 range
            if scaling:
                scale_coef = 1.0
                # Scale down if too large
                if max_val > fp16_max:
                    scale_coef = fp16_max / max_val
                # Scale up if too small
                elif min_val < fp16_min:
                    scale_coef = fp16_min / min_val
                # Scale up if value is in the FP16 range but too small to meet precision requirements in conversion
                elif (
                    np.allclose(data, arr_fp16.astype(data.dtype), rtol=rtol, atol=atol) is False
                    and (fp16_max * (1 / min_val)) < fp16_max
                ):
                    scale_coef = 1 / min_val

                # Check if scaled data preserves precision
                scaled_arr = data * scale_coef
                scaled_fp16_arr = scaled_arr.astype(np.float16)
                # Compare loss with tolerance threshold
                if np.allclose(
                    scaled_arr.astype(scaled_arr.dtype),
                    scaled_fp16_arr.astype(scaled_arr.dtype),
                    rtol=rtol,
                    atol=atol,
                ):
                    return (scaled_fp16_arr, scale_coef)

            # If FP16 is incompatible
            return (data, None)  # FP32 required, e.g. mixed large / small values

        else:
            raise ValueError(f"Unsupported data dtype '{data.dtype}'")

    ## For torch tensor
    elif type(data) is torch.Tensor:
        if data.dtype is torch.float16:
            return (data, None)

        elif (data.dtype is torch.float32) or (data.dtype is torch.float64):
            # Get min and max values of non_zeros
            max_val = float(data.abs().max())
            non_zero_data = data[data != 0]
            if non_zero_data.numel() > 0:
                min_val = float(non_zero_data.abs().min())
            else:
                min_val = fp16_min

            # If data fits in FP16 natively
            if max_val <= fp16_max and min_val >= fp16_min:
                torch_fp16 = data.to(torch.float16)
                if torch.allclose(data, torch_fp16.to(data.dtype), rtol=rtol, atol=atol):
                    return (torch_fp16, 1.0)

            # If data can be scaled into FP16 range or FP16 precision
            if scaling:
                # Scale down if too large
                if max_val > fp16_max:
                    scale_coef = fp16_max / max_val
                # Scale up if too small
                elif min_val < fp16_min:
                    scale_coef = fp16_min / min_val
                # Scale up if value is in the FP16 range but too small to meet precision requirements in conversion
                elif (
                    torch.allclose(data, torch_fp16.to(data.dtype), rtol=rtol, atol=atol) is False
                    and (fp16_max * (1 / min_val)) < fp16_max
                ):
                    scale_coef = 1 / min_val

                # Check if scaled data preserves precision
                scaled_tensor = data * scale_coef
                scaled_fp16_tensor = scaled_tensor.to(torch.float16)
                if torch.allclose(
                    scaled_tensor.to(scaled_tensor.dtype),
                    scaled_fp16_tensor.to(scaled_tensor.dtype),
                    rtol=rtol,
                    atol=atol,
                ):
                    return (scaled_fp16_tensor, scale_coef)

            # If FP16 is incompatible
            return (data, None)  # FP32 required, e.g. mixed large / small values

        else:
            raise ValueError(f"Unsupported data dtype '{data.dtype}'")

    else:
        raise ValueError(f"Unsupported data type '{type(data)}'")


# Tiled pixel-wise apply a 1D function
@simple_type_validator
def pixel_spec_apply(  # noqa: C901
    image_path: str,
    output_path: str,
    spectral_function: Callable,
    dtype: Union[type, str, np.dtype, torch.dtype] = "float32",
    tile_size: int = -1,
    *,
    progress: bool = True,
    override: bool = True,
    _space_wait_timeout: int = 6,
    _reserve_free_pct: float = 5.0,
    _preprocess_status: Optional[dict[str, Any]] = None,
) -> None:
    """
    Apply a function to the 1D spectra of every pixel of a raster image.
    The function must accept 1D arraylike as only required parameter, and return processed 1D arraylike data.
    If override False, function will not be executed for existed dst image.
    """

    # Dependencies for multiprocessing
    import os
    import numpy as np

    # Validate tile size and set default
    if tile_size == -1:
        tile_size = 32
    elif tile_size < 1:
        raise ValueError(f"If provided, tile_size must be positive, got: {tile_size}")

    # Validate dtype
    dtype = dtype_mapper(dtype)

    with rasterio.open(unc_path(image_path)) as src:
        # Validate dst raster
        if override:
            if os.path.exists(unc_path(output_path)):
                os.remove(unc_path(output_path))
        else:
            if os.path.exists(unc_path(output_path)):
                return

        # Get output number of bands
        test_data = src.read(window=Window(col_off=0, row_off=0, width=1, height=1))
        test_result = spectral_function(test_data[:, 0, 0])
        num_bands_out = len(test_result)

        # Get metadata from the source image
        meta = src.meta.copy()
        meta.update({"dtype": dtype_mapper(dtype, "raster"), "count": num_bands_out})

        # Apply compression if available
        if os.path.splitext(output_path)[1] == ".tiff" or os.path.splitext(output_path)[1] == ".tif":
            if "float32" in str(dtype_mapper(dtype, "raster")):
                meta.update({'compress': 'lzw', 'predictor': 2})
            else:
                meta.update({'compress': 'lzw'})

        # Validate available disk space
        raster_meta = meta.copy()
        raster_meta["__specpipe_raster_meta_for_size_validation"] = True
        _safe_disk_wait(
            obj=raster_meta,
            path=unc_path(output_path),
            preprocess_status=_preprocess_status,
            space_wait_timeout=_space_wait_timeout,
            reserve_free_pct=_reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
            obj_size_buffer_coeff=1.05,
        )

        # Create dst raster
        with rasterio.open(unc_path(output_path), "w", **meta) as dst:
            # Get image dimensions
            height, width = src.height, src.width

            # Process in tiles
            if progress:
                print(f"\nProcessing image with tile size {tile_size}x{tile_size}\n")
            for i in tqdm(range(0, height, tile_size), total=int(height / tile_size) + 1, disable=not progress):
                for j in range(0, width, tile_size):
                    # Define window for current tile
                    win = Window(
                        col_off=j,
                        row_off=i,
                        width=min(tile_size, width - j),
                        height=min(tile_size, height - i),
                    )
                    # Read all bands, shape: [bands, rows, cols]
                    tile_data = src.read(window=win)
                    tile_height, tile_width = tile_data.shape[1], tile_data.shape[2]
                    # Create array and save processed data
                    processed_tile = np.zeros((num_bands_out, tile_height, tile_width))
                    for m in range(tile_height):
                        for n in range(tile_width):
                            # Apply function
                            processed_spec = spectral_function(tile_data[:, m, n])
                            processed_tile[:, m, n] = np.asarray(processed_spec)
                    # Write array to output
                    dst.write(processed_tile, window=win)


# Tiled vectorized apply a 2D function
@simple_type_validator
def pixel_array_apply(  # noqa: C901
    image_path: str,
    output_path: str,
    spectral_function: Callable,
    dtype: Union[type, str, np.dtype, torch.dtype] = "float32",
    tile_size: int = -1,
    *,
    progress: bool = True,
    override: bool = True,
    _space_wait_timeout: int = 6,
    _reserve_free_pct: float = 5.0,
    _preprocess_status: Optional[dict[str, Any]] = None,
) -> None:
    """
    Apply a function to the 1D spectra of every pixel of a raster image.
    The function must accept 2D array as only required parameter, and return processed 2D array with same length.
    Each row of the 2D array represents an 1D spectra or processed data of a pixel.
    If override False, function will not be executed for existed dst image.
    """

    # Dependencies for multiprocessing
    import os
    import numpy as np

    # Validate tile size and set default
    if tile_size == -1:
        tile_size = 32
    elif tile_size < 1:
        raise ValueError(f"If provided, tile_size must be positive, got: {tile_size}")

    # Validate dtype
    dtype = dtype_mapper(dtype)

    with rasterio.open(unc_path(image_path)) as src:
        # Validate dst raster
        if override:
            if os.path.exists(unc_path(output_path)):
                os.remove(unc_path(output_path))
        else:
            if os.path.exists(unc_path(output_path)):
                return

        # Get output number of bands
        test_data = src.read(window=Window(col_off=0, row_off=0, width=2, height=2))
        # Reshape test data
        test_data_input = test_data.reshape(src.count, -1).T
        test_result = spectral_function(test_data_input)
        num_bands_out = np.asarray(test_result).shape[1]

        # Get metadata from the source image
        meta = src.meta.copy()
        meta.update({"dtype": dtype_mapper(dtype, "raster"), "count": num_bands_out})

        # Apply compression if available
        if os.path.splitext(output_path)[1] == ".tiff" or os.path.splitext(output_path)[1] == ".tif":
            if "float" in str(dtype_mapper(dtype, "raster")):
                meta.update({'compress': 'lzw', 'predictor': 2})
            else:
                meta.update({'compress': 'lzw'})

        # Validate available disk space
        raster_meta = meta.copy()
        raster_meta["__specpipe_raster_meta_for_size_validation"] = True
        _safe_disk_wait(
            obj=raster_meta,
            path=unc_path(output_path),
            preprocess_status=_preprocess_status,
            space_wait_timeout=_space_wait_timeout,
            reserve_free_pct=_reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
            obj_size_buffer_coeff=1.05,
        )

        # Create dst raster
        with rasterio.open(unc_path(output_path), "w", **meta) as dst:
            # Get image dimensions
            height, width = src.height, src.width

            # Process in tiles
            if progress:
                print(f"\nProcessing image with tile size {tile_size}x{tile_size}\n")
            for i in tqdm(range(0, height, tile_size), total=int(height / tile_size) + 1, disable=not progress):
                for j in range(0, width, tile_size):
                    # Define window for current tile
                    win = Window(
                        col_off=j,
                        row_off=i,
                        width=min(tile_size, width - j),
                        height=min(tile_size, height - i),
                    )
                    # Read all bands for current tile, shape: [bands, rows, cols]
                    tile_data = src.read(window=win)
                    original_shape = tile_data.shape
                    # Reshape src tile data
                    spectra_2d = tile_data.reshape(src.count, -1).T
                    spectra_2d = np.asarray(spectra_2d)
                    # Apply vectorized function
                    processed_spectra = spectral_function(spectra_2d)
                    # Reshape back
                    output_shape = (num_bands_out, original_shape[1], original_shape[2])
                    processed_tile = processed_spectra.T.reshape(output_shape)
                    # Write processed tile to output
                    dst.write(processed_tile, window=win)


# Tiled tensor apply a tensor function with calculation axis = 0
@simple_type_validator
def pixel_tensor_apply(  # noqa: C901
    image_path: str,
    output_path: str,
    spectral_function: Callable,
    dtype: Union[type, str, np.dtype, torch.dtype] = "float32",
    tile_size: int = -1,
    device: str = "cuda",
    *,
    progress: bool = True,
    override: bool = True,
    _space_wait_timeout: int = 6,
    _reserve_free_pct: float = 5.0,
    _preprocess_status: Optional[dict[str, Any]] = None,
) -> None:
    """
    Apply a function to the 1D spectra of every pixel of a raster image.
    The function must accept 3D PyTorch tenser as only required parameter, and return processed 3D tensor with same size.
    The function must compute along 0 axis. And the input and output tensor must have the same size except 0 axis.
    If override False, function will not be executed for existed dst image.
    """  # noqa: E501

    # Dependencies for multiprocessing
    import os
    import numpy as np
    import torch

    # Validate tile size and set default
    if tile_size == -1:
        tile_size = 64
    elif tile_size < 1:
        raise ValueError(f"If provided, tile_size must be positive, got: {tile_size}")

    # Validate dtype
    dtype = dtype_mapper(dtype, "torch")

    # Initialize torch device
    device = str(torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu"))
    if progress:
        print(f"\nUsing device: {device}")
        if "cuda" in str(device):
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Use CUDA Stream
    stream = torch.cuda.Stream() if "cuda" in str(device) else None

    with rasterio.open(unc_path(image_path)) as src:
        # Validate dst raster
        if override:
            if os.path.exists(unc_path(output_path)):
                os.remove(unc_path(output_path))
        else:
            if os.path.exists(unc_path(output_path)):
                return

        # Data for test running of given function
        test_data = src.read(window=Window(col_off=0, row_off=0, width=2, height=2))

        # Validate data type
        nctypes = [np.uint16, np.uint32, np.uint64]
        if test_data.dtype in nctypes:
            test_data = test_data.astype(dtype_mapper(dtype))
        test_tensor = torch.from_numpy(test_data).to(device)

        # Get output number of bands of given function
        test_tensor_out = spectral_function(test_tensor)
        num_bands_out = test_tensor_out.shape[0]
        del test_tensor_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get metadata from the source image
        meta = src.meta.copy()
        meta.update({"dtype": dtype_mapper(dtype, "raster"), "count": num_bands_out})

        # Apply compression if available
        if os.path.splitext(output_path)[1] == ".tiff" or os.path.splitext(output_path)[1] == ".tif":
            if "float" in str(dtype_mapper(dtype, "raster")):
                meta.update({'compress': 'lzw', 'predictor': 2})
            else:
                meta.update({'compress': 'lzw'})

        # Validate available disk space
        raster_meta = meta.copy()
        raster_meta["__specpipe_raster_meta_for_size_validation"] = True
        _safe_disk_wait(
            obj=raster_meta,
            path=unc_path(output_path),
            preprocess_status=_preprocess_status,
            space_wait_timeout=_space_wait_timeout,
            reserve_free_pct=_reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
            obj_size_buffer_coeff=1.05,
        )

        # Create output file with same number of bands
        with rasterio.open(unc_path(output_path), "w", **meta) as dst:
            # Get image dimensions
            height, width = src.height, src.width

            # Process in tiles
            if progress:
                print(f"\nProcessing image with tile size {tile_size}x{tile_size}\n")

            for i in tqdm(range(0, height, tile_size), total=int(height / tile_size) + 1, disable=not progress):
                for j in range(0, width, tile_size):
                    # Define window for current tile
                    win = Window(
                        col_off=j,
                        row_off=i,
                        width=min(tile_size, width - j),
                        height=min(tile_size, height - i),
                    )

                    # Read all bands for current tile, shape: [bands, rows, cols]
                    tile_data = src.read(window=win)

                    # Validate data type
                    if tile_data.dtype in nctypes:
                        tile_data = tile_data.astype(dtype_mapper(dtype))

                    # Create tensor from tile-data and move to GPU
                    with torch.cuda.stream(stream):
                        tile_tensor = torch.from_numpy(tile_data).pin_memory().to(device, non_blocking=True)

                    # Convert tensor dtype
                    if tile_tensor.dtype is not dtype_mapper(dtype, "torch"):
                        assert isinstance(dtype, (str, torch.dtype))
                        tile_tensor = tile_tensor.to(dtype)

                    # Apply tensor function
                    with (
                        torch.cuda.stream(stream),
                        torch.no_grad(),
                    ):  # Disable gradient tracking for memory efficiency
                        processed = spectral_function(tile_tensor)

                    # Move back to CPU and convert to numpy
                    with torch.cuda.stream(stream):
                        processed = processed.cpu().numpy()

                    # Synchronize before writing
                    if stream:
                        torch.cuda.synchronize()

                    # Write output
                    dst.write(processed, window=win)


# Row-wise tensor apply a tensor function with calculation axis = 1
@simple_type_validator
def pixel_tensor_hyper_apply(  # noqa: C901
    image_path: str,
    output_path: str,
    spectral_function: Callable,
    dtype: Union[type, str, np.dtype, torch.dtype] = "float32",
    tile_size: int = -1,
    device: str = "cuda",
    *,
    progress: bool = True,
    override: bool = True,
    _space_wait_timeout: int = 6,
    _reserve_free_pct: float = 5.0,
    _preprocess_status: Optional[dict[str, Any]] = None,
) -> None:
    """
    Apply a function to the 1D spectra of every pixel of a raster image, optimized for hyperspectral image data transfer.
    The function must accept 3D PyTorch tenser as only required parameter, and return processed 3D tensor with same size.
    The function must compute along 1 axis. And the input and output tensor must have the same size except 1 axis.
    If override False, function will not be executed for existed dst image.
    """  # noqa: E501

    # Dependencies for multiprocessing
    import os
    import numpy as np
    import torch

    # Validate tile size and set default
    if tile_size == -1:
        row_chunk = 4
    elif tile_size < 1:
        raise ValueError(f"Row_chunk must be positive, got: {tile_size}")
    else:
        row_chunk = tile_size

    # Validate dtype
    dtype = dtype_mapper(dtype, "torch")

    # Initialize torch device
    device = str(torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu"))
    if progress:
        print(f"\nUsing device: {device}")
        if "cuda" in str(device):
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Use CUDA Stream
    stream = torch.cuda.Stream() if "cuda" in str(device) else None

    # Validate dst raster
    if override:
        if os.path.exists(unc_path(output_path)):
            os.remove(unc_path(output_path))
    else:
        if os.path.exists(unc_path(output_path)):
            return

    with rasterio.open(unc_path(image_path)) as src:
        # Data for test running of given function
        test_data = src.read(window=Window(0, 0, src.width, row_chunk))

        # Validate data type
        nctypes = [np.uint16, np.uint32, np.uint64]
        if test_data.dtype in nctypes:
            test_data = test_data.astype(dtype_mapper(dtype))

        # Get output number of bands of given function
        test_tensor = torch.from_numpy(test_data).to(device)  # [C, 1, W]
        test_tensor = test_tensor.permute(1, 0, 2)  # [1, C, W]
        with torch.no_grad():
            test_out = spectral_function(test_tensor)
        num_bands_out = test_out.shape[1]
        del test_data, test_tensor, test_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get metadata from the source image
        meta = src.meta.copy()
        meta.update({"dtype": dtype_mapper(dtype, "raster"), "count": num_bands_out})

        # Apply compression if available
        if os.path.splitext(output_path)[1] == ".tiff" or os.path.splitext(output_path)[1] == ".tif":
            if "float" in str(dtype_mapper(dtype, "raster")):
                meta.update({'compress': 'lzw', 'predictor': 2})
            else:
                meta.update({'compress': 'lzw'})

        # Validate available disk space
        raster_meta = meta.copy()
        raster_meta["__specpipe_raster_meta_for_size_validation"] = True
        _safe_disk_wait(
            obj=raster_meta,
            path=unc_path(output_path),
            preprocess_status=_preprocess_status,
            space_wait_timeout=_space_wait_timeout,
            reserve_free_pct=_reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
            obj_size_buffer_coeff=1.05,
        )

        with rasterio.open(unc_path(output_path), "w", **meta) as dst:
            for row_start in tqdm(range(0, src.height, row_chunk), disable=not progress):
                # Read entire row chunk [C, row_chunk, W]
                window = Window(0, row_start, src.width, min(row_chunk, src.height - row_start))
                row_data = src.read(window=window)

                # Validate data type
                if row_data.dtype in nctypes:
                    row_data = row_data.astype(dtype_mapper(dtype))

                # Convert to [row_chunk, C, W] for processing
                with torch.cuda.stream(stream):
                    row_tensor = torch.from_numpy(row_data).pin_memory().to(device, non_blocking=True)
                row_tensor = row_tensor.permute(1, 0, 2)  # [row_chunk, C, W]

                # Convert tensor dtype
                if row_tensor.dtype is not dtype_mapper(dtype, "torch"):
                    assert isinstance(dtype, (str, torch.dtype))
                    row_tensor = row_tensor.to(dtype)

                # Apply tensor function
                with torch.cuda.stream(stream), torch.no_grad():
                    processed = spectral_function(row_tensor)

                # Convert back to [C, row_chunk, W] for writing
                with torch.cuda.stream(stream):
                    processed_cpu = processed.permute(1, 0, 2).cpu().numpy()

                # Synchronize before writing
                if stream:
                    torch.cuda.synchronize()

                # Write output
                dst.write(processed_cpu, window=window)


# Raster pixel apply - apply a function to every raster pixel
@simple_type_validator
def pixel_apply(
    image_path: str,
    spectral_function: Callable,
    function_type: str,
    output_path: Optional[str] = None,
    dtype: Union[type, str, np.dtype, torch.dtype] = "float32",
    tile_size: int = -1,
    *,
    progress: bool = True,
    return_output_path: bool = True,
    override: bool = True,
    _space_wait_timeout: int = 6,
    _reserve_free_pct: float = 5.0,
    _preprocess_status: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """
    Apply a function to process the 1D spectra of every pixel of a raster image.

    Parameters
    ----------
    image_path : str
        Input raster image path in string.

    spectral_function : callable, optional
        Function to apply to 1D spectra of every pixel.
        The type of the function is specified to parameter function_type

    function_type : callable, optional
        Specifies the type of spectral processing function to apply. Must be one of:

            - ``"spec"``
                Processes individual spectra.

                The function must:

                    Accept a 1D ``array-like`` as its only required input.
                    Return 1D ``numpy.ndarray`` of the processed spectra.

            - ``"array"``
                Processes batches of spectra as a 2D array.

                The function must:

                    Accept a 2D ``numpy.ndarray`` as its only required input, where each row represents a spectrum of a pixel.
                    Return a 2D ``numpy.ndarray`` with the same number of rows (processed spectra).

            - ``"tensor"``
                Processes data as a 3D ``torch.Tensor``, used for multispectral data with limited number of bands.

                "cuda" is applied if available.

                The function must:

                    Accept a 3D tensor as its only required input (shape: [Channels, Height, Width]).
                    Return a 3D tensor with identical ``Height`` and ``Width``.
                    Compute operations along the Channel dimension (axis 0).

            - ``"tensor_hyper"``
                Processes data as a 3D PyTorch tensor, optimized for hyperspectral data with large number of bands.

                "cuda" is applied if available.

                The function must:

                    Accept a 3D tensor as its only required input (shape: [Channels, Height, Width]).
                    Return a 3D tensor with identical ``Height`` and ``Width``.
                    Compute operations along the ``Height`` dimension (axis 1).

    output_path : str
        Output raster image path in string.
        Defaults to ``image_path`` combined with "_px_app_" and function name.

    dtype : type or str, optional
        Value data type of output array. The default is ``"float32"``.

    tile_size : int, optional
        Size of the processing tiles in pixels.

        Larger values may improve performance, but require more memory.

        The default is ``-1``, in which case the value is determined by ``function_type`` as follows:

            - ``"spec"``: 32
            - ``"array"``: 32
            - ``"tensor"``: 64
            - ``"tensor_hyper"``: 4

    progress : bool, optional
        Whether to show progress bar.

        If True, enables progress bar.

        If False, suppresses progress messages.

        Default is True.

    return_output_path : bool, optional
        Whether path of processed image is returned.
        Default is True.

    override : bool, optional
        Whether existed image of output_path is override.

        If False, function will not be applied if output_path existed. The default is True.

    Returns
    -------
    str
        Path of processed image.

    Examples
    --------
    Apply function accepting 2D array::

        >>> pixel_apply("/image1.tif", array_function, "array")

    Save to custom output path::

        >>> pixel_apply("/image1.tif", array_function, "array", output_path="/image1_processed.tif")

    Apply function accepting 3D hyperspectral tensor and computing along axis 0::

        >>> pixel_apply("/image1.tif", tensor_function, "tensor")

    Apply function accepting 3D hyperspectral tensor and computing along axis 1::

        >>> pixel_apply("/image1.tif", hypertensor_function, "tensor_hyper")

    Customize tile size::

        >>> pixel_apply("/image1.tif", array_function, "array", tile_size=128)
    """  # noqa: E501

    # Dependencies for multiprocessing
    import os

    # Default output path
    if output_path is None:
        ptx = os.path.splitext(image_path)
        output_path = ptx[0] + "_px_app_" + spectral_function.__name__ + ptx[1]

    # Validate path
    if not os.path.exists(unc_path(image_path)):
        raise ValueError(f"\nInvalid image_path: {image_path} \nImage file does not exist.")
    output_dir_path = os.path.dirname(output_path)
    if not os.path.exists(unc_path(output_dir_path)):
        raise ValueError(f"\nInvalid output directory: {output_dir_path} \nDirectory does not exist.")

    # Validate tile size and batch size
    if (tile_size != -1) & (tile_size < 1):
        raise ValueError(f"\nInvalid tile_size, got: {tile_size} \nIf provided, tile_size must be positive integer.")

    # Apply function
    # Spec apply
    if function_type.lower() == "spec":
        pixel_spec_apply(
            image_path,
            output_path,
            spectral_function,
            dtype,
            tile_size,
            progress=progress,
            override=override,
            _space_wait_timeout=_space_wait_timeout,
            _reserve_free_pct=_reserve_free_pct,
            _preprocess_status=_preprocess_status,
        )

    # Array apply
    elif function_type.lower() == "array":
        pixel_array_apply(
            image_path,
            output_path,
            spectral_function,
            dtype,
            tile_size,
            progress=progress,
            override=override,
            _space_wait_timeout=_space_wait_timeout,
            _reserve_free_pct=_reserve_free_pct,
            _preprocess_status=_preprocess_status,
        )

    # Tensor apply
    elif function_type.lower() == "tensor":
        pixel_tensor_apply(
            image_path,
            output_path,
            spectral_function,
            dtype,
            tile_size,
            "cuda",
            progress=progress,
            override=override,
            _space_wait_timeout=_space_wait_timeout,
            _reserve_free_pct=_reserve_free_pct,
            _preprocess_status=_preprocess_status,
        )

    # Hyper-tensor apply
    elif function_type.lower() == "tensor_hyper":
        pixel_tensor_hyper_apply(
            image_path,
            output_path,
            spectral_function,
            dtype,
            tile_size,
            "cuda",
            progress=progress,
            override=override,
            _space_wait_timeout=_space_wait_timeout,
            _reserve_free_pct=_reserve_free_pct,
            _preprocess_status=_preprocess_status,
        )

    # Else
    else:
        raise ValueError(
            f"Invalid function_type, function_type must be 'spec', 'array' or 'tensor', but got: {function_type}"
        )

    # Return path of processed image
    if return_output_path:
        return output_path
    else:
        return None
