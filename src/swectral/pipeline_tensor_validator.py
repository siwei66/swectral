# -*- coding: utf-8 -*-
"""
Swectral - Pipeline process meta and process method validators

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Typing
from typing import Union, Callable, Optional

# Data
import torch
import torch.nn as nn

# Warning
import warnings

# Local
from .specio import simple_type_validator


# %% Data level validator


@simple_type_validator
def _dl_val(data_level: Union[str, int]) -> tuple[int, str]:
    """
    Data level validator for SpecPipeTensor.
    Input data level name or index number, return (data level index, data level name).

    Valid levels: 0:'function', 1:'fittable', 2:'model'
    """

    # Define mapping using a dictionary for O(1) lookups
    level_map = {0: "function", 1: "fittable", 2: "model"}

    # Reverse mapping for string-to-int lookups
    name_map = {v: k for k, v in level_map.items()}

    if isinstance(data_level, str):
        key = data_level.lower()
        if key not in name_map:
            raise ValueError(f"Invalid data_level string: '{data_level}'. " f"Expected one of {list(name_map.keys())}.")
        dlind = name_map[key]

    elif isinstance(data_level, int):
        if data_level not in level_map:
            raise ValueError(f"Invalid data_level index: {data_level}. " f"Expected one of {list(level_map.keys())}.")
        dlind = data_level

    else:
        raise TypeError(f"data_level must be int or str, but got {type(data_level).__name__}")

    return (dlind, level_map[dlind])


# %% Process parameter validators


@simple_type_validator
def _data_level_seq_validator(
    input_data_level: Union[str, int], output_data_level: Union[str, int], application_sequence: int
) -> tuple[int, str, int, str]:
    """
    Validates the data levels and application sequence for a pipeline process.
    Returns the parsed indices and names: (dl_in_ind, dl_in_name, dl_out_ind, dl_out_name).
    """
    # Any invalid type or value will automatically raise the appropriate
    # ValueError or TypeError from within _dl_val
    dl_in_ind, dl_in_name = _dl_val(input_data_level)
    dl_out_ind, dl_out_name = _dl_val(output_data_level)

    # --- Validate Logical Flow ---
    if dl_out_ind < dl_in_ind:
        raise ValueError(
            f"output_data_level ({dl_out_name}, {dl_out_ind}) cannot be lower than input_data_level "
            f"({dl_in_name}, {dl_in_ind})."
        )

    # --- Validate application_sequence ---
    if not isinstance(application_sequence, int):
        raise TypeError("application_sequence must be an integer.")

    if not (0 <= application_sequence <= 1000000):
        raise ValueError(f"application_sequence must be between 0 and 1000000. Got {application_sequence}.")

    return dl_in_ind, dl_in_name, dl_out_ind, dl_out_name


def _training_parameter_validator(  # noqa: C901
    methods: list[Union[Callable, nn.Module]],
    batch_sizes: list[int],
    shuffle: Optional[bool],
    criteria: list[torch.nn.Module],
    optimizers: list[Union[torch.optim.Optimizer, None]],
    schedulers: list[Union[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler._LRScheduler, None]],
) -> None:
    """
    Validates that necessary training parameters are provided and correctly typed if the process includes an nn.Module.
    """
    # Check if any of the provided methods are an nn.Module
    has_module = any(isinstance(m, nn.Module) for m in methods)
    has_batch_size = False
    has_shuffle = False
    has_criterion = False
    has_optimizer = False
    defined_scheduler = False

    if has_module:
        # 1. Validate batch_size
        for batch_size in batch_sizes:
            if batch_size is None:
                raise ValueError("Missing 'batch_size': Must be specified when adding an nn.Module.")
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError(f"Invalid 'batch_size': Must be a positive integer, got {batch_size}.")
        has_batch_size = True

        # 2. Validate shuffle
        if shuffle is None:
            raise ValueError("Missing 'shuffle': Must be specified (True/False) when adding an nn.Module.")
        if not isinstance(shuffle, bool):
            raise TypeError(f"Invalid 'shuffle': Must be a boolean, got {type(shuffle).__name__}.")
        has_shuffle = True

        # 3. Validate criterion
        for criterion in criteria:
            if criterion is None:
                raise ValueError("Missing 'criterion': A loss function must be specified when adding an nn.Module.")
            if not isinstance(criterion, nn.Module):
                raise TypeError(
                    f"Invalid 'criterion': Must be an instance of torch.nn.Module, got {type(criterion).__name__}."
                )
        has_criterion = True

        # 4. Validate optimizers
        if len(optimizers) == 0:
            raise ValueError("Missing 'optimizer': Must be specified when adding an nn.Module.")
        elif any((opt is None) for opt in optimizers):
            raise ValueError("Missing 'optimizer': Must be specified when adding an nn.Module.")
        else:
            has_optimizer = True
            if len(optimizers) != len(methods):
                raise ValueError("Length of optimizer list must match the length of provided methods.")
            for opt in optimizers:
                if not isinstance(opt, torch.optim.Optimizer):
                    raise TypeError(
                        "Invalid 'optimizer': Must be an instance of torch.optim.Optimizer, "
                        f"got {type(opt).__name__}."
                    )

        # 5. Validate schedulers
        # Handle PyTorch version differences for LRScheduler base classes
        valid_schedulers: tuple
        try:
            from torch.optim.lr_scheduler import LRScheduler

            valid_schedulers = (LRScheduler, torch.optim.lr_scheduler._LRScheduler, type(None))
        except ImportError:
            valid_schedulers = (torch.optim.lr_scheduler._LRScheduler, type(None))

        if len(schedulers) == 0:
            raise ValueError("Missing 'scheduler': Must be specified when adding an nn.Module.")
        else:
            defined_scheduler = True
            if len(schedulers) != len(methods):
                raise ValueError("Length of scheduler list must match the length of provided methods.")
            for sch in schedulers:
                if not isinstance(sch, valid_schedulers):
                    raise TypeError(
                        "Invalid 'scheduler': Must be a valid PyTorch learning rate scheduler, "
                        f"got {type(sch).__name__}."
                    )

    else:
        # If no nn.Module is provided, check if the user accidentally passed training args
        provided_params = []
        if has_batch_size:
            provided_params.append("batch_size")
        if has_shuffle:
            provided_params.append("shuffle")
        if has_criterion:
            provided_params.append("criterion")
        if has_optimizer:
            provided_params.append("optimizer")
        if defined_scheduler:
            provided_params.append("scheduler")

        if len(provided_params) > 0:
            warnings.warn(
                f"Training parameters {provided_params} were provided, but the method is a deterministic Callable. "
                "These parameters are ignored for non-trainable functions.",
                UserWarning,
                stacklevel=2,
            )
