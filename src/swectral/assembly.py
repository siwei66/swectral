# -*- coding: utf-8 -*-
"""
Swectral - Pipeline basic sample data assembly tools

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import numpy as np

from typing import Any, Annotated

from .specio import simple_type_validator, arraylike_validator

# %% identity_assembly with type validation


@simple_type_validator
def identity_assembly(
    sample_list: list[
        tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, Annotated[Any, arraylike_validator(ndim=1)]]
    ],
) -> list[tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, Annotated[Any, arraylike_validator(ndim=1)]]]:
    """Identity assembly process for pass-through with data type validation."""
    return sample_list
