# -*- coding: utf-8 -*-
"""
Swectral - Pipeline management and implemention module for spectral image processing and modeling - GPU version

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Temporarily ignore of dependency F401 for development
# ruff: noqa: F401

# OS
import os
import sys
import shutil

# Interface
from tqdm import tqdm

# Warning
import warnings

# Typing
from typing import Annotated, Any, Callable, Literal, Optional, Union, overload
from types import ModuleType

# Time
import time
from datetime import datetime

# Functions
from functools import partial

# Basic data
from operator import itemgetter
import copy
import json
import numpy as np
import pandas as pd

# GPU computation
import torch

# Raster
# import rasterio
# from rasterio.windows import Window
from rasterio.errors import NotGeoreferencedWarning

# Visualization
import matplotlib.pyplot as plt

# Multiprocessing
import dill
from pathos.helpers import mp
from pathos.multiprocessing import ProcessingPool, cpu_count
import multiprocessing as nmp

# Local
from .groupstats import (
    sample_group_stats,
    performance_metrics_summary,
    performance_marginal_stats,
)
from .modeleva import ModelEva
from .rasterop import croproi

# from .rasterop import pixel_apply
from .resultcli import group_stats_report, core_chain_report
from .roistats import roispec, minbbox
from .specexp import SpecExp, _spec_exp_validator
from .specio import (
    # arraylike_validator,
    dataframe_validator,
    dump_dill,
    load_dill,
    RealNumber,
    simple_type_validator,
    unc_path,
    df_to_csv,
)
from .pipeline_validator import (
    _target_type_validation_for_serialization,
    _dl_val,
    _data_level_seq_validator,
    _process_validator,
    _classifier_validator,
    _regressor_validator,
    _num_image_chains,
    _estimate_img_output_size,
    _estimate_report_plot_size,
    _pipeline_process_seq_validator,
    _assembly_method_validator,
    _pre_assembly_data_validator,
)
from .pipeline_processor import (
    _preprocessing_sample,
    _ModelMethod,
    _model_evaluator,
    _model_evaluator_mp,
    _DummyLock,
    _DummyManager,
    _load_disk_backed_data,
    _sample_list_constructor,
    _single_preprocess_assembly,
)
from .pipeline_tqdm import PipelineFileTqdm, DirProgressObserver
from .modelconnector import (
    combined_model_marginal_stats,
)
from .pipeline import SpecPipe

# For multiprocessing
global ModelEva


# %% Spectral Modeling Pipeline Class GPU version - SpecPipeTensor


class SpecPipeTensor:
    """
    Design and implement processing and modeling pipelines on spectral experiment datasets.

    Attributes
    ----------
    spec_exp : SpecExp
        Instance of SpecExp configuring spectral experiment datasets. See ``SpecExp`` for details.

    report_directory : str
        Root directory where reports are stored.
        This value is automatically derived from the ``report_directory`` attribute of the provided ``spec_exp`` instance.

    space_wait_timeout : int
        Number of seconds to wait for disk space to become available before raising an error when the disk is full.
        Default is 36000 (10 hours).

    reserve_free_pct : float
        Minimum percentage of free disk space required to proceed with processing.
        Default is 5.0 (5% of total storage capacity).

    process : list of tuple
        Added process items.
        Each tuple represents a process definition and contains:

            process_id : str
            process_label : str
            input_data_level : str
            output_data_level : str
            application_sequence : int
            method : callable
            full_application_sequence : int
            alternative_number : int

    process_steps : list of tuple of str
        Processes of each pipeline step, each tuple represents a step.
        The processes are represented in process ID.

    process_chains : list of tuple of str
        Generated full-factorial processing chains, each tuple represents a processing chain.
        The processes are represented in process ID.

    custom_chains : list of tuple of str
        Customized subset of the full-factorial ``process_chains``.

    create_time : str
        Creation date and time of this SpecPipe isntance.

    See Also
    --------
    SpecExp

    Methods
    -------
    add_process
        Add a processing method with defined input/output data levels and application sequence.
        The method can be a preprocessing function or a model for evaluation.

    ls_process
        List process items filtered by process properties.

    rm_process
        Remove added processes by ID, process_label, data_level, application_sequence, or method.

    add_model
        Add a model to the pipeline.

    ls_model
        List model process items filtered by given properties.

    rm_model
        Remove added models by model_id, model_label, or model method (object).

    process_chains_to_df
        List process chains in a dataframe. Returns default full-factorial process chains.
        Each row represents a processing chain with process IDs.

        Alias: ls_process_chains

    custom_chains_from_df
        Customize processing chains and update chains using a chain dataframe.
        Custom chains are prioritized over default full-factorial chains.

    custom_chains_to_df
        List custom chains in a dataframe.

        Alias: ls_custom_chains

    ls_chains
        List process chains in the pipeline execution.
        Returns custom chains if configured, otherwise default full-factorial chains.

    save_pipe_config
        Save current pipeline configurations to files in the root of the report directory.

        Alias: save_config

    load_pipe_config
        Load SpecPipe configurations from a dill file.

        Alias: load_config

    test_run
        Run all processing chains using simplified test data.
        Executed automatically prior to each formal run.

    preprocessing
        Apply preprocessing steps of all chains on the entire dataset and output modeling-ready sample_list data to files.

    assembly
        Apply assembly process to introduce cross-sample interactions prior to modeling.

    model_evaluation
        Evaluate added models on sample data from all preprocessing chains.

    run
        Run the pipelines of given processes on a ``SpecExp`` instance.

    report_summary
        Retrieve summary of reports in the console, including performance summary and marginal performances among processes.

    report_chains
        Retrieve major model evaluation reports of every processing chain in the console.

    Examples
    --------
    Create a ``SpecPipe`` instance using a prepared ``SpecExp`` instance ``exp``::

        >>> pipe = SpecPipe(exp)
    """  # noqa: E501

    @simple_type_validator
    def __init__(
        self,
        spec_exp: SpecExp,
        space_wait_timeout: int = 36000,
        reserve_free_pct: float = 5.0,
    ) -> None:  # noqa: C901

        # Validate SpecExp integrity
        _spec_exp_validator(spec_exp)

        ## Private internal attributes
        self.__sample_targets: list[tuple[str, str, Union[str, bool, int, float], str, str, np.int8, np.int8]] = (
            spec_exp.sample_targets
        )  # noqa: E501
        self.__is_target_numeric: bool = self._check_target_numeric(spec_exp)
        self.__band_wavelength: Optional[tuple[Union[int, float], ...]] = None
        self.__pretest_data: Optional[dict[str, Any]] = None
        self.__sample_data: list[dict[str, Any]] = []
        self.__tested: bool = False
        self.__preprocess_result_path: list[str] = []

        ## Experiment data manager - SpecExp

        # SpecExp
        # SpecExp._groups: [0 group]
        # SpecExp._images: [0 id, 1 group, 2 image_name, 3 image_use_type, 4 image_path]
        # SpecExp._rois: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
        self._spec_exp: SpecExp = spec_exp

        # File output parameters
        self._spec_exp._space_wait_timeout = max(0, space_wait_timeout)
        self._spec_exp._reserve_free_pct = max(0.01, reserve_free_pct)
        self._space_wait_timeout: int = max(0, space_wait_timeout)
        self._reserve_free_pct: float = max(0.01, reserve_free_pct)

        # Report directory
        self._report_directory: str = self._spec_exp._report_directory

        # Processes
        # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
        self._process: list[tuple[str, str, str, str, int, Union[Callable, object], int, int]] = []

        # Generated process chain for full factors
        # [(process 1 ID of step 1, process 2 ID of step 1,...), (process 1 ID of step 2, process 2 ID of step 2,...), ...]  # noqa: E501
        self._process_steps: list[list[str]] = []
        # [(process 1 ID of step 1, process 1 ID of step 2,...), (process 2 ID of step 1, process 1 ID of step 2,...), ...]  # noqa: E501
        self._process_chains: list[tuple[str, ...]] = []
        # Custom chains for custom partly test
        self._custom_chains: list = []

        # Pipeline creating time
        self._create_time: str = datetime.now().strftime("created_at_%Y-%m-%d_%H-%M-%S")

    ## Mutable properties
    @property
    def space_wait_timeout(self) -> int:
        return self._space_wait_timeout

    @space_wait_timeout.setter
    @simple_type_validator
    def space_wait_timeout(self, value: int) -> None:
        if len(self.process) > 0:
            if len([proc for proc in self.process if proc[3] == "model"]) > 0:
                warnings.warn(
                    "Found model evaluation process. "
                    + "Remove and re-add the models to make the change effective for model evaluation processes.",
                    UserWarning,
                    stacklevel=2,
                )
        self._space_wait_timeout = max(0, value)

    @property
    def reserve_free_pct(self) -> float:
        return self._reserve_free_pct

    @reserve_free_pct.setter
    @simple_type_validator
    def reserve_free_pct(self, value: float) -> None:
        if len(self.process) > 0:
            if len([proc for proc in self.process if proc[3] == "model"]) > 0:
                warnings.warn(
                    "Found model evaluation process. "
                    + "Remove and re-add the models to make the change effective for model evaluation processes.",
                    UserWarning,
                    stacklevel=2,
                )
        self._reserve_free_pct = max(0.01, value)

    @property
    def report_directory(self) -> str:
        return self._report_directory

    @report_directory.setter
    @simple_type_validator
    def report_directory(self, value: str) -> None:
        if os.path.exists(unc_path(value)):
            warning_msg = (
                "The current report_directory is shared with the SpecExp's report_directory. "
                "It is recommended to set the report_directory directly in SpecExp instead. "
                "Note that modifying it here will not update SpecExp's report_directory, "
                "which may result in test reports being saved in two different locations."
            )
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
            value = (str(value).replace("\\", "/") + "/").replace("//", "/")
            self._report_directory = value
        else:
            raise ValueError(f"Given report_directory is invalid: {value}")

    @property
    def spec_exp(self) -> SpecExp:
        return self._spec_exp

    @spec_exp.setter
    def spec_exp(self, spec_exp: SpecExp) -> None:
        if isinstance(spec_exp, SpecExp):
            self.__tested = False
            self._spec_exp_updater(spec_exp)
        else:
            raise ValueError(f"{self.__class__.__name__}.spec_exp must be a SpecExp instance")

    @property
    def _band_wavelength(self) -> Optional[tuple[Union[int, float], ...]]:
        return self._band_wavelength

    @_band_wavelength.setter
    def _band_wavelength(self, value: Optional[tuple[Union[int, float], ...]]) -> None:
        if value is not None:
            value = tuple(value)
            if self.__pretest_data is None:
                raise ValueError(
                    "Internal Error: 'SpecPipe._pretest_data' is None. "
                    "Pre-execution test data initialization fails. Please report."
                )
            if len(value) != len(self.__pretest_data["spec1d"]):
                raise ValueError(
                    f"The number of band wavelengths ({len(value)}) does not match "
                    f"the number of bands ({len(self.__pretest_data['spec1d'])})."
                )
            v0: Union[int, float] = 0
            for v in value:
                if (type(v) is not float) | (type(v) is not int):
                    raise TypeError(f"Band wavelengths must be numeric, got type: {type(v)}.")
                if v <= 0:
                    raise ValueError(f"Band wavelengths must be positive, got: {v}")
                if v > v0:
                    v0 = v
                else:
                    raise ValueError(f"Band wavelengths must be in an ascending order without ties, got: {value}")
        self.__band_wavelength = value

    ## Read only or immuatable properties
    @property
    def _sample_targets(self) -> list[tuple[str, str, Union[str, bool, int, float], str, str, np.int8, np.int8]]:
        return self.__sample_targets

    @_sample_targets.setter
    def _sample_targets(
        self, value: list[tuple[str, str, Union[str, bool, int, float], str, str, np.int8, np.int8]]
    ) -> None:
        raise ValueError("_sample_targets cannot be modified in SpecPipe, please update using 'SpecExp' instead")

    @property
    def _is_target_numeric(self) -> bool:
        return self.__is_target_numeric

    @_is_target_numeric.setter
    def _is_target_numeric(self, value: bool) -> None:
        raise ValueError("_is_target_numeric cannot be modified in SpecPipe, please update using 'SpecExp' instead")

    @property
    def process(self) -> list[tuple[str, str, str, str, int, Union[Callable, object], int, int]]:
        return self._process

    @process.setter
    def process(self, value: list[tuple[str, str, str, str, int, Union[Callable, object], int, int]]) -> None:
        raise ValueError("process cannot be modified directly, use 'add_process' and 'rm_process' instead")

    @property
    def process_steps(self) -> list:
        return self._process_steps

    @process_steps.setter
    def process_steps(self, value: list) -> None:
        raise ValueError("process_steps cannot be modified")

    @property
    def process_chains(self) -> list:
        return self._process_chains

    @process_chains.setter
    def process_chains(self, value: list) -> None:
        raise ValueError("process_chains cannot be modified")

    @property
    def custom_chains(self) -> list:
        return self._custom_chains

    @custom_chains.setter
    def custom_chains(self, value: list) -> None:
        raise ValueError("custom_chains cannot be modified directly, use 'custom_chains_from_df' to set custom_chains")

    @property
    def _sample_data(self) -> list[dict[str, Any]]:
        return self.__sample_data

    @_sample_data.setter
    def _sample_data(self, value: list[dict[str, Any]]) -> None:
        raise ValueError("_sample_data cannot be modified")

    @property
    def _pretest_data(self) -> Optional[dict]:
        return self.__pretest_data

    @_pretest_data.setter
    def _pretest_data(self, value: Optional[dict]) -> None:
        raise ValueError("_pretest_data cannot be modified")

    @property
    def _preprocess_result_path(self) -> list[str]:
        return self.__preprocess_result_path

    @_preprocess_result_path.setter
    def _preprocess_result_path(self, value: list[str]) -> None:
        raise ValueError("_preprocess_result_path cannot be modified")

    @property
    def _tested(self) -> bool:
        return self.__tested

    @_tested.setter
    def _tested(self, value: bool) -> None:
        raise ValueError("_tested is immutable and cannot be modified")

    @property
    def create_time(self) -> str:
        return self._create_time

    @create_time.setter
    def create_time(self, value: str) -> None:
        raise ValueError("create_time is immutable and cannot be modified")

    # SpecExp target value numeric validator
    @staticmethod
    @simple_type_validator
    def _check_target_numeric(spec_exp: SpecExp) -> bool:
        # Read target values
        sample_target_values = [spt[2] for spt in spec_exp.sample_targets]
        # Validate whether numeric
        is_numeric: bool = True
        for yi in sample_target_values:
            if not isinstance(yi, RealNumber):
                is_numeric = False
        return is_numeric

    # Alias method
    @simple_type_validator
    def update_spec_exp(self, spec_exp: SpecExp) -> None:
        self.spec_exp = spec_exp

    # SpecExp-related initializer / updater
    @simple_type_validator
    def _spec_exp_updater(self, spec_exp: SpecExp) -> None:
        # Backup current spec_exp
        spec_exp_old = copy.deepcopy(self.spec_exp)
        # Update data and test
        try:
            _spec_exp_validator(spec_exp)
        except Exception as e:
            raise ValueError("Given SpecExp instance is invalid") from e
        try:
            self._spec_exp = spec_exp
            # Set SpecExp saving parameters
            self._spec_exp._space_wait_timeout = self._space_wait_timeout
            self._spec_exp._reserve_free_pct = self._reserve_free_pct
            # Update SpecExp-related SpecPipe data
            self.__sample_targets = spec_exp.sample_targets
            self.__is_target_numeric = self._check_target_numeric(spec_exp)
            self._report_directory = spec_exp._report_directory
            # self._pretest_data_init()
            # n_chains = len(self.process_chains)
            # if n_chains > 0:
            #     self.test_run(dump_result=False, model_test_coverage=(min(100, n_chains) / n_chains))
        except Exception as e:
            # Roll back when fail in test
            self._spec_exp = spec_exp_old
            self.__sample_targets = spec_exp_old.sample_targets
            self.__is_target_numeric = self._check_target_numeric(spec_exp_old)
            self._report_directory = spec_exp_old._report_directory
            # self._pretest_data_init()
            raise ValueError("Given SpecExp failed in test_run, spec_exp configuration rolls back.") from e
