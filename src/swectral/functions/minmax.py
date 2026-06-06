# -*- coding: utf-8 -*-
"""
Swectral - process functions - MinMax (MinMax Normalization)

Copyright (c) 2025 Siwei Luo. MIT License.
"""

from typing import Annotated, Any
from ..specio import arraylike_validator, simple_type_validator


# %% MinMax


@simple_type_validator
def minmax(data: Annotated[Any, arraylike_validator()]) -> Any:
    """
    MinMax (MinMax Normalization) function.

    For image pixel spectrum correction in SpecPipe pipelines:

        Set process input data level: 2 / 'pixel_specs_array'

        Set process output data level: 2 / 'pixel_specs_array'

    For ROI spectrum normalization in SpecPipe pipelines:

        Set process input data level: 6 / 'roispecs'

        Set process output data level: 6 / 'roispecs'

    For sample spectrum normalization in SpecPipe pipelines:

        Set process input data level: 7 / 'spec1d'

        Set process output data level: 7 / 'spec1d'

    For image tensor in SpecPipeTensor pipelines:

        Set process input data level: 1 / 'function'

    Parameters
    ----------
    data : 1D, 2D, 3D, or 4D array-like or torch.Tensor
        1D (n_bands,), 2D (n_samples, n_bands), 3D (C, H, W) or 4D (N, C, H, W) spectral data to be processed.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        MinMax transformed spectral data computed natively as either numpy or torch based on the input type.

    Examples
    --------
    >>> minmax([[1, 2, 3, 4, 5, 6], [2, 2, 4, 4, 6, 6]])

    Incorporation into pipeline for image processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(2, 2, 0, minmax)

    Incorporation into pipeline for ROI spectra processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(6, 6, 0, minmax)

    Incorporation into pipeline for 1D spectra processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(7, 7, 0, minmax)

    Incorporation into pipeline for deterministic image tensor transformation, for SpecPipeTensor instance ``pipe_tensor``:

        >>> pipe_tensor.add_process(0, 1, 0, minmax)
    """  # noqa: E501

    is_tensor = False
    has_torch_module = False
    try:
        import torch

        has_torch_module = True

        if isinstance(data, torch.Tensor):
            is_tensor = True
    except ImportError:
        pass

    if is_tensor and has_torch_module:
        ndim = data.dim()
        if ndim in (1, 3):
            axis = 0
        elif ndim in (2, 4):
            axis = 1
        else:
            raise ValueError(f"Expected 1D, 2D, 3D, or 4D tensor, got dimension: {ndim}")

        if not data.is_floating_point():
            data = data.float()

        inf_tensor = torch.tensor(float("inf"), dtype=data.dtype, device=data.device)

        data_min = torch.where(torch.isnan(data), inf_tensor, data)
        vmin = torch.amin(data_min, dim=axis, keepdim=True)

        data_max = torch.where(torch.isnan(data), -inf_tensor, data)
        vmax = torch.amax(data_max, dim=axis, keepdim=True)

        result1: torch.Tensor = (data - vmin) / (vmax - vmin + 1e-15)

        return result1

    else:

        import numpy as np

        data = np.asarray(data)
        ndim = data.ndim

        if ndim in (1, 3):
            axis = 0
        elif ndim in (2, 4):
            axis = 1
        else:
            raise ValueError(f"Expected 1D, 2D, 3D, or 4D array-like, got dimension: {ndim}")

        vmin = np.nanmin(data, axis=axis, keepdims=True)
        vmax = np.nanmax(data, axis=axis, keepdims=True)

        minmax_values = (data - vmin) / (vmax - vmin + 1e-15)

        result: np.ndarray = np.asarray(minmax_values)

        return result
