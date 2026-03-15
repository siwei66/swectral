# -*- coding: utf-8 -*-
"""
Swectral - process functions - SNV (Standard Normal Variate)

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import numpy as np

from typing import Annotated, Any
from ..specio import arraylike_validator, simple_type_validator


# %% SNV


@simple_type_validator
def snv(data: Annotated[Any, arraylike_validator()]) -> np.ndarray:
    """
    SNV (Standard Normal Variate) function.

    For image pixel spectrum correction in SpecPipe pipelines:

        Set process input data level: 2 / 'pixel_specs_array'

        Set process output data level: 2 / 'pixel_specs_array'

    For ROI spectrum normalization:

        Set process input data level: 6 / 'roispecs'

        Set process output data level: 6 / 'roispecs'

    For sample spectrum normalization:

        Set process input data level: 7 / 'spec1d'

        Set process output data level: 7 / 'spec1d'

    Parameters
    ----------
    data : 2D array-like (n_samples, n_bands) or 1D array-like (n_bands,)
        1D or 2D array-like spectral data to be processed.

    Returns
    -------
    numpy.ndarray
        SNV transformed spectral data.

    Examples
    --------
    >>> snv([[1, 2, 3, 4, 5, 6], [2, 2, 4, 4, 6, 6]])

    Incorporation into pipeline for image processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(2, 2, 0, snv)

    Incorporation into pipeline for ROI spectra processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(6, 6, 0, snv)

    Incorporation into pipeline for 1D spectra processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(7, 7, 0, snv)
    """
    import numpy as np  # noqa: W291

    data = np.asarray(data)

    if data.ndim == 2:
        vmean = np.nanmean(data, axis=1, keepdims=True)
        vstd = np.nanstd(data, axis=1, keepdims=True)
        snv_values = (data - vmean) / (vstd + 1e-15)
    elif data.ndim == 1:
        vmean = np.nanmean(data)
        vstd = np.nanstd(data)
        snv_values = (data - vmean) / (vstd + 1e-15)
    else:
        raise ValueError(f"Expected 1D or 2D array-like, got dimension: {data.ndim}")

    result: np.ndarray = np.asarray(snv_values)

    return result
