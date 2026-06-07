# -*- coding: utf-8 -*-
"""
Example data generators and downloader for Swectral

The docstrings were created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os
import warnings

# Time
import time

# Typing
from typing import Optional, Union, Any

# Basic data processing
import numpy as np
import pandas as pd

# Rasters
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

# Local
from .specexp import SpecExp
from .specio import roi_to_envi, silent, simple_type_validator

# Download
import urllib


# %% test helper functions


# Test helper functions : create_test_raster
@simple_type_validator
def create_test_raster(
    raster_path: str,
    width: int = 100,
    height: int = 50,
    bands: int = 4,
    incl_nodata: Union[None, int, float] = None,
    nodata_value: Union[int, float] = 0,
    dtype: Union[str, type] = "uint16",
    data: Optional[np.ndarray] = None,
) -> str:
    """
    Create a synthetic raster file for demonstration or testing purposes.

    Generates a GeoTIFF raster with specified dimensions, number of bands, and data type.
    Custom data can also be provided.
    Optional nodata values can be applied to simulate missing data.

    By default, it creates a 100x50 raster with 4 bands containing deterministic pseudo-random data.

    Parameters
    ----------
    raster_path : str
        Path to save the raster. Must end with ``.tif`` or ``.tiff``.
        The directory must exist.

    width : int, optional
        Raster width in pixels. Default is 100.

    height : int, optional
        Raster height in pixels. Default is 50.

    bands : int, optional
        Number of bands. Default is 4.

    incl_nodata : int, float, or None, optional
        Value to fill selected bands to simulate nodata regions. Default is ``None``.

    nodata_value : int or float, optional
        Metadata nodata value. Default is 0.

    dtype : str or type, optional
        Data type of raster array. Default is ``'uint16'``.

    data : numpy.ndarray, optional
        Custom raster data of shape ``(bands, height, width)``.

        If ``None``, synthetic data is generated. Default is `None`.

    Returns
    -------
    str
        Path of the created raster file.

    Notes
    -----
    - The raster uses a simple linear gradient with pseudo-random noise.
    - No real CRS is assigned; transform is based on pixel coordinates.
    - If `incl_nodata` is set, first and last bands (if >3) are filled with this value.

    Examples
    --------
    Generate a default test raster::

        >>> create_test_raster("test_raster.tif")

    Generate a raster with custom dimensions and 3 bands::

        >>> create_test_raster("custom_raster.tif", width=200, height=100, bands=3)

    Provide custom data for the raster::

        >>> import numpy as np
        >>> data = np.random.randint(0, 256, size=(4, 50, 100), dtype='uint16')
        >>> create_test_raster("custom_data_raster.tif", data=data)
    """

    # Validate raster path
    raster_path = raster_path.replace("\\", "/").replace("//", "/")
    if (raster_path[-4:] != ".tif") and (raster_path[-5:] != ".tiff"):
        raise ValueError(f"raster_path must have .tif or .tiff extension, got: {raster_path}")
    if not os.path.exists(os.path.dirname(raster_path)):
        raise ValueError(f"raster path directory does not exist: {os.path.dirname(raster_path)}")

    # Validate raster data
    if data is None:
        # Mock image data
        np.random.seed(42)
        data = np.array(
            [
                [
                    (
                        np.array([1080 / bands * j + 360 / bands * int(i / 5)] * width)
                        + (np.random.rand(width) * 120 / bands)
                    )
                    .astype(dtype)
                    .tolist()
                    for i in range(height)
                ]
                for j in range(bands)
            ]
        )
    elif type(data) is np.ndarray:
        data = np.array(data).astype(dtype)
        if data.shape != (bands, height, width):
            raise ValueError(
                f"Given raster data shape {data.shape} does not match specified raster dimensions (bands={bands}, \
                    height={height}, width={width})."
            )
    else:
        raise ValueError(f"Given raster data must be numpy.ndarray, got type: {type(data)}.")

    # Set some nodata values
    if incl_nodata is not None:
        data[0, :, :] = incl_nodata
        if bands > 3:
            data[-1, :, :] = incl_nodata

    # Define transform (georeferencing)
    transform = rasterio.transform.from_bounds(0, 0, width, height, width=width, height=height)

    # CRS for direct coordinate mapping to avoid NotGeoreferencedWarning
    crs = CRS.from_wkt(
        """
    ENGCRS["Pixel Space",
        EDATUM["Unknown engineering datum"],
        CS[Cartesian,2],
            AXIS["x",east,
                ORDER[1],
                LENGTHUNIT["unity",1]],
            AXIS["y",north,
                ORDER[2],
                LENGTHUNIT["unity",1]]
    ]
    """
    )

    # Create metadata
    meta = {
        "driver": "GTiff",
        "dtype": dtype,
        "nodata": nodata_value,
        "width": width,
        "height": height,
        "count": bands,
        "crs": crs,
        "transform": transform,
    }

    # Write to file
    with rasterio.open(raster_path, "w", **meta) as dst:
        dst.write(data)

    return raster_path


# Alias
create_example_raster = create_test_raster


# Test helper functions : create_test_roi_xml
@simple_type_validator
def create_test_roi_xml(
    xml_path: str,
    raster_width: int = 10,
    raster_height: int = 50,
    roi_count: int = 10,
    roi_list: Optional[list[dict]] = None,
    return_path: bool = True,
    return_roi_list: bool = False,
) -> Union[str, list[dict], tuple[str, list[dict]], None]:
    """
    Create a synthetic ENVI ROI XML file for demonstration or testing purposes.

    Generates an ENVI-compatible ROI XML file containing a set of rectangular regions of interest (ROIs).
    Custom ROIs can be provided as a list of dictionaries.

    By default, it creates 10 ROIs evenly distributed along the raster height.

    Parameters
    ----------
    xml_path : str
        Path to save the ROI XML file.

    raster_width : int, optional
        Width of the raster used to define ROI coordinates. Default is 10.

    raster_height : int, optional
        Height of the raster used to define ROI coordinates. Default is 50.

    roi_count : int, optional
        Number of synthetic ROIs to generate if ``roi_list`` is not provided. Default is 10.

    roi_list : list of dict, optional
        Custom list of ROI dictionaries.

        Each dictionary must contain the following keys::

            {
                "name": str,
                "crs": str or CRS,
                "color": tuple of int or None,
                "type": str,
                "coordinates": list of list of tuple of float
            }

        If ``None``, synthetic ROIs are generated. Default is ``None``.

    return_path : bool, optional
        If True, return the path of the generated XML file. Default is True.

    return_roi_list : bool, optional
        If True, return the list of ROI dictionaries. Default is False.

    Returns
    -------
    str, list of dict, (str, list of dict), or None

        - If ``return_path`` is True and ``return_roi_list`` is False: returns the XML file path (str).
        - If ``return_path`` is False and ``return_roi_list`` is True: returns the ROI list (list of dict).
        - If both are True: returns a tuple ``(path, roi_list)``.
        - If both are False: returns None.

    Notes
    -----
    Coordinates are defined in raster pixel units, and CRS is set to ``"none"`` by default.
    This method is also available as ``create_example_roi_xml``.

    Examples
    --------
    Generate a default ROI XML file::

        >>> create_test_roi_xml("test_rois.xml")

    Generate a file with 5 ROIs for a raster of height 50::

        >>> create_test_roi_xml("small_rois.xml", raster_height=50, roi_count=5)

    Generate a file and also get the ROI list::

        >>> xml_path, rois = create_test_roi_xml("example_rois.xml", return_roi_list=True)

    Provide a custom ROI list::

        >>> custom_rois = [
            ...     {"name": "ROI_1", "type": "polygon", "coordinates": [[(0, 0), (5, 0), (0, 5), (0, 0)]]}
            ... ]
        >>> create_test_roi_xml("custom_rois.xml", roi_list=custom_rois)
    """

    # Create line of mock ROIs in the raster
    height_increment = int(raster_height / roi_count)
    if height_increment < 1:
        raise ValueError("raster_height / roi_count must be at least 1.")
    width_range = min(5, raster_width)

    if roi_list is None:
        roi_list = []
        # Test ROI data
        for i in range(roi_count):
            roi_dict = {
                "name": f"ROI_{i + 1}",
                "crs": "none",
                "color": None,
                "type": "polygon",
                "coordinates": [
                    [
                        (0.0, height_increment * i),
                        (width_range, height_increment * i),
                        (width_range, height_increment * (i + 1)),
                        (0.0, height_increment * (i + 1)),
                        (0.0, height_increment * i),
                    ]
                ],
            }
            roi_list.append(roi_dict)

    # Save ROI xml file
    path_out = roi_to_envi(file_path=xml_path, roi_list=roi_list)
    path_out = str(path_out)

    # Output
    result: Union[str, list[dict], tuple[str, list[dict]]]
    if return_path and not return_roi_list:
        result = path_out
    elif not return_path and return_roi_list:
        result = roi_list
    elif return_path and return_roi_list:
        result = path_out, roi_list
    else:
        return None
    return result


# Alias
create_example_roi_xml = create_test_roi_xml


# Test helper functions : create_test_spec_exp
@silent
@simple_type_validator
def create_test_spec_exp(
    dir_path: str,
    sample_n: int = 10,
    n_bands: int = 4,
    is_regression: bool = True,
    use_val_group: bool = False,
) -> SpecExp:
    """
    Create a standard test `SpecExp` instance for spectral experiments.

    This function generates a `SpecExp` data manager pre-populated with:

    - A test group.
    - A synthetic multispectral image.
    - Synthetic ROIs.
    - Synthetic sample labels and targets.

    It can be used for testing spectral processing pipelines, demonstration or development.

    Parameters
    ----------
    dir_path : str
        Directory where the test data (images and ROI XML) will be stored.
        Will be created if it does not exist.

    sample_n : int, optional
        Number of samples / ROIs to generate. Default is 10.

    n_bands : int, optional
        Number of bands in the synthetic image. Default is 4.

    is_regression : bool, optional
        If True, target values are numeric (regression).
        If False, targets are categorical (classification).
        Default is True.

    use_val_group : bool, optional
        Whether validation group is enabled. Default is False.

    Returns
    -------
    SpecExp
        A ``SpecExp`` instance populated with synthetic data, ready for spectral analysis.

    Notes
    -----
    - The synthetic image is a pseudo-random gradient with band-specific variations.
    - ROIs are rectangular and evenly distributed across the raster height.
    - Sample labels and target values are automatically generated based on ``sample_n``.

    Examples
    --------
    Create a default test ``SpecExp`` instance::

        >>> exp = create_test_spec_exp("test_spec_exp")

    Create a test instance with 20 samples and 6 spectral bands::

        >>> exp = create_test_spec_exp("test_spec_exp_dir", sample_n=20, n_bands=6)

    Create a classification-type ``SpecExp`` instance::

        >>> exp = create_test_spec_exp("class_spec_exp", is_regression=False)
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    exp1: SpecExp = SpecExp(dir_path, log_loading=False)

    # # Add test
    # group
    exp1.add_groups(["test_group"])

    # Mock image data
    np.random.seed(42)
    mock_img_data = np.array(
        [
            [
                (np.array([1080 / n_bands * j + 360 / n_bands * i] * 10) + (np.random.rand(10) * 120 / n_bands))
                .astype(np.uint16)
                .tolist()
                for i in range(5 * sample_n)
            ]
            for j in range(n_bands)
        ]
    )

    for i in range(5 * sample_n):
        mock_img_data[-int(n_bands / 3) :, i, :] = mock_img_data[-int(n_bands / 3) :, i, :] * (i // 5 % 2 + 0.5)

    # Mock image
    img_path = dir_path + "/test_img.tif"
    create_test_raster(
        raster_path=img_path,
        width=mock_img_data.shape[2],
        height=mock_img_data.shape[1],
        bands=n_bands,
        data=mock_img_data,
    )
    exp1.add_images_by_name(group="test_group", image_name=["test_img.tif"], image_directory=dir_path)

    # ROIs
    roi_path = dir_path + "/test_roi.xml"
    create_test_roi_xml(roi_path, roi_count=sample_n)
    exp1.add_rois_by_file(group="test_group", path=[roi_path], image_name="test_img.tif")

    # Samples
    dflb = exp1.ls_sample_labels()
    assert isinstance(dflb, pd.DataFrame)
    dflb.iloc[:, 1] = [f"sample_{str(i + 1)}" for i in range(len(dflb))]
    exp1.sample_labels = dflb  # type: ignore[assignment]
    # Auto-conversion in setter

    # Target values
    dft = exp1.ls_sample_targets()
    assert isinstance(dft, pd.DataFrame)
    if is_regression:
        dft["Target_value"] = list(range(len(dft)))
    else:
        dft["Target_value"] = [["a", "b"][int(i % 2)] for i in range(len(dft))]
    # Set validation groups
    if use_val_group:
        dft["Validation_group"] = [f"vg_{int(i / 4) + 1}" for i in range(len(dft))]
    exp1.sample_targets_from_df(dft)

    return exp1


# Alias
create_example_spec_exp = create_test_spec_exp


# %% Demo data downloader


@simple_type_validator
def download_demo_data(
    data_dir: str = os.getcwd(),
    demo_dir_url: str = 'https://raw.githubusercontent.com/siwei66/swectral/master/demo/demo_data/',
    files: tuple = ('demo_1.xml', 'demo_2.xml', 'demo_3.xml', 'demo_4.xml', 'demo_5.xml', 'demo.tiff'),
    retry_limit: int = 5,
) -> None:
    """
    Download real-world demo data files for demonstration or user testing purposes.

    Parameters
    ----------
    data_dir : str, optional
        Local directory where downloaded files will be saved.

        Default is the current working directory (``os.getcwd()``).

    demo_dir_url : str, optional
        Base URL of the remote demo data repository of ``Swectral`` package.

        Default is ``'https://raw.githubusercontent.com/siwei66/swectral/master/demo/demo_data/'``.

    files : tuple of str, optional
        Names of files to download from the repository.

        Default includes ``('demo_1.xml', 'demo_2.xml', 'demo_3.xml', 'demo_4.xml', 'demo_5.xml', 'demo.tiff')``.
        The files must exist in the remote repository.

    retry_limit : int, optional
        Maximum number of download retries per file in case of failure.
        Default is 5.

    Returns
    -------
    None
        The files are downloaded to ``data_dir``. No value is returned.

    Examples
    --------
    Download default demo files to the current directory::

        >>> download_demo_data()

    Download demo files to a custom directory::

        >>> download_demo_data(data_dir="my_demo_data")

    Download only a subset of files::

        >>> download_demo_data(files=("demo_1.xml", "demo.tiff"))
    """

    # Validate retry limit
    retry_limit = min(3, retry_limit)
    # Setup download directory
    data_dir = (data_dir.replace('\\', '/') + '/').replace('//', '/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Package demo dir path
    downloaded = []
    for file in files:
        url = demo_dir_url + file
        local_path = data_dir + file
        i = 0
        while i < retry_limit:
            try:
                urllib.request.urlretrieve(url, local_path)
                # Verify download
                if os.path.exists(local_path):
                    downloaded.append(file)
                    print(f"\nDownloaded: {local_path}")
                    break
            except Exception as e:
                i = i + 1
                print(f"\nDownload file '{file}' failed: {e}, \nRetry {i} in {2 ** i} seconds...")
                time.sleep(min(2**i, 16))
                if i >= retry_limit:
                    print(
                        f"\nDownload '{file}' failed, please download it manually from:\
                          https://github.com/siwei66/swectral/tree/master/demo/demo_data/"
                    )
    if set(downloaded) != set(files):
        raise Exception(
            "Download is not completed, plaese download the files manually from: \
                https://github.com/siwei66/swectral/tree/master/demo/demo_data/"
        )
    return None


# %% Create shaped rasters for SpecPipeTensor pipelines


@simple_type_validator
def create_test_raster_shaped(  # noqa: C901
    out_dir: str,
    n_samples: int = 20,
    width: int = 20,
    height: int = 20,
    n_bands: int = 4,
    task_type: str = "classification",
    n_classes: int = 2,
    nodata: Union[int, float, None] = None,
    nodata_cov: float = 0.0,
    dtype: str = "uint16",
) -> list[str]:
    """
    Generate mock hyperspectral raster images where a spatial shape acts as the image signal.

    This function generates synthetic raster datasets suitable for testing spatial-spectral machine learning pipelines.
    The average spectral signature of the images remains similar and indistinguishable.
    Instead, the task target is encoded in the spatial location of a localized 2D Gaussian point.

    For classification, the point stochasticly jitters around fixed coordinates corresponding to specific classes.
    For regression, the point moves continuously across the image extent with added location noise.
    Random Gaussian noise is applied globally to simulate sensor variations.

    Parameters
    ----------
    out_dir : str
        The base directory where the generated images will be saved.
    n_samples : int, optional
        The total number of sample images to generate. Default is 20.
    width : int, optional
        The width of the generated raster images in pixels. Default is 20.
    height : int, optional
        The height of the generated raster images in pixels. Default is 20.
    n_bands : int, optional
        The number of spectral bands (channels) for each image. Default is 4.
    task_type : str, optional
        The machine learning task type. Must be either "classification" or "regression".
        Default is "classification".
    n_classes : int, optional
        The number of distinct classes to generate (only applicable if task_type is "classification").
        Default is 2.
    nodata : int, float, or None, optional
        The specific value to be assigned to no-data pixels.
        If None, no-data masking is skipped regardless of `nodata_cov`.
        Default is None.
    nodata_cov : float, optional
        The coverage percentage of no-data pixels per image, expressed as a float between 0.0 and 1.0. Default is 0.0.
    dtype : str, optional
        The NumPy/Rasterio data type string for the output images (e.g., "uint16", "float32").
        Default is "uint16".

    Returns
    -------
    list of str
        A list of absolute file paths to the generated raster images.

    Raises
    ------
    ValueError
        If the `out_dir` does not exist, if `task_type` is invalid, or if the dimensions are too small to support the moving point requirement.
    """  # noqa: E501

    # 1. Validate output directory and dimensions
    if not os.path.isdir(out_dir):
        raise ValueError(f"The provided output directory path is invalid or does not exist: {out_dir}")

    if min(width, height) < 10:
        raise ValueError("Image dimensions must be at least 10x10 to fit the spatial Gaussian point.")

    target_dir = os.path.join(out_dir, "mock_rasters")
    os.makedirs(target_dir, exist_ok=True)

    task_type = task_type.lower()
    if task_type not in ("classification", "regression"):
        raise ValueError("task_type must be either 'classification' or 'regression'.")

    # 2. Define geometry for the 2D Gaussian point
    # Point size is set so its visible diameter (~4*sigma) is strictly < 0.5 * extent
    sigma = min(width, height) / 10.0
    padding = 3.0 * sigma  # Ensures the point is entirely present without edge clipping

    # 3. Generate target labels (y) and define spatial centers
    if task_type == "classification":
        if n_samples < n_classes * 3:
            warnings.warn(
                f"Number of samples ({n_samples}) is less than 3 times the number of classes ({n_classes}). "
                "This may lead to insufficient data for standard train/test/validation splits.",
                stacklevel=2,
            )
        samples_per_class = int(np.ceil(n_samples / n_classes))
        y_values: np.ndarray = np.repeat(np.arange(n_classes), samples_per_class)[:n_samples]

        # Pre-calculate fixed centers for each class using a fixed seed for reproducibility
        rng = np.random.RandomState(42)
        class_centers: list[tuple[float, float]] = []
        for _ in range(n_classes):
            class_cy = rng.uniform(padding, height - padding)
            class_cx = rng.uniform(padding, width - padding)
            class_centers.append((class_cy, class_cx))
    else:
        y_values = np.linspace(1.0, 10.0, n_samples)
        min_y, max_y = 1.0, 10.0

    np_dtype: np.dtype = np.dtype(dtype)
    file_paths: list[str] = []
    bands_x = np.linspace(0, n_bands - 1, n_bands)

    # A fixed spectral signature for the 2D Gaussian point to keep the average spectrum identical
    point_spectrum = np.exp(-((bands_x - n_bands / 2.0) ** 2) / ((n_bands / 4.0) ** 2)) * 5000.0

    transform = from_origin(0.0, 0.0, 1.0, 1.0)
    Y, X = np.ogrid[:height, :width]  # noqa: N806

    # 4. Generate images iteratively
    for i, y_val in enumerate(y_values):
        # Determine the base coordinate based on task_type
        if task_type == "classification":
            base_cy, base_cx = class_centers[int(y_val)]
            label_str = str(int(y_val))
        else:
            # Continuous trajectory from top-left to bottom-right bounds
            t = (y_val - min_y) / (max_y - min_y) if max_y > min_y else 0.5
            base_cy = padding + t * (height - 2.0 * padding)
            base_cx = padding + t * (width - 2.0 * padding)
            label_str = f"{y_val:.2f}".replace(".", "_")

        # Add stochastic location noise
        cy = base_cy + np.random.normal(0, sigma / 3.0)
        cx = base_cx + np.random.normal(0, sigma / 3.0)

        # Strictly clip to ensure the entire point is present and uncropped
        cy = np.clip(cy, padding, height - padding)
        cx = np.clip(cx, padding, width - padding)

        # Create the spatial Gaussian blob
        spatial_blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2))

        # Integrate spatial blob with fixed spectral signature over a noisy base background
        img = np.full((n_bands, height, width), 1000.0)
        for b in range(n_bands):
            img[b, :, :] += point_spectrum[b] * spatial_blob

        img += np.random.normal(loc=0.0, scale=200.0, size=(n_bands, height, width))

        # Safely clip and cast to target dtype
        if np.issubdtype(np_dtype, np.integer):
            dinfo = np.iinfo(np_dtype)
            img: np.ndarray = np.clip(img, dinfo.min, dinfo.max)
        img = img.astype(np_dtype)

        # Apply NoData masking
        if nodata_cov > 0.0 and nodata is not None:
            n_pixels = width * height
            n_nodata = int(n_pixels * nodata_cov)
            if n_nodata > 0:
                flat_indices = np.random.choice(n_pixels, n_nodata, replace=False)
                row_idx, col_idx = np.unravel_index(flat_indices, (height, width))
                img[:, row_idx, col_idx] = nodata

        # Write to disk
        filename = f"sample_{i}_y_{label_str}.tif"
        out_path = os.path.join(target_dir, filename)

        profile: dict[str, Any] = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": n_bands,
            "dtype": str(dtype),
            "transform": transform,
        }

        if nodata is not None:
            profile["nodata"] = nodata

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(img)

        file_paths.append(os.path.abspath(out_path))

    return file_paths


# Alias
create_example_raster_shaped = create_test_raster_shaped
