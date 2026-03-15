# -*- coding: utf-8 -*-
"""
Swectral - process functions - AUC (Area Under Curve) normalization

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import numpy as np

from typing import Annotated, Any
from ..specio import arraylike_validator, simple_type_validator


# %% AUC normalization


@simple_type_validator
def aucnorm(data: Annotated[Any, arraylike_validator()]) -> np.ndarray:
    """
    AUC (Area Under Curve) normalization function.

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
        AUC normalization transformed spectral data.

    Examples
    --------
    >>> aucnorm([[1, 2, 3, 4, 5, 6], [2, 2, 4, 4, 6, 6]])

    Incorporation into pipeline for image processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(2, 2, 0, aucnorm)

    Incorporation into pipeline for ROI spectra processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(6, 6, 0, aucnorm)

    Incorporation into pipeline for 1D spectra processing, for SpecPipe instance ``pipe``:

        >>> pipe.add_process(7, 7, 0, aucnorm)
    """
    import numpy as np  # noqa: W291

    data = np.asarray(data)

    if data.ndim == 2:
        areas = np.sum(np.abs(data), axis=1, keepdims=True)
        auc_normalized = data / (areas + 1e-15)
    elif data.ndim == 1:
        areas = np.sum(np.abs(data))
        auc_normalized = data / (areas + 1e-15)
    else:
        raise ValueError(f"Expected 1D or 2D array-like, got dimension: {data.ndim}")

    result: np.ndarray = np.asarray(auc_normalized)

    return result
