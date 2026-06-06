# -*- coding: utf-8 -*-
"""
Swectral - Pipeline management and implemention module for spectral image processing and modeling - GPU version

The following source code was created with AI assistance and has been human reviewed and edited.

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
import itertools

# Basic data
from operator import itemgetter
import copy
import json
import numpy as np
import pandas as pd

# GPU computation
import torch
import torch.nn as nn

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
from .pipeline_tensor_validator import (
    _dl_val,
    _data_level_seq_validator,
    _training_parameter_validator,
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
        # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_data, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
        self._process: list[tuple[str, str, str, str, int, Union[Callable, dict[str, Any]], int, int]] = []

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
    def process(self) -> list[tuple[str, str, str, str, int, Union[Callable, dict[str, Any]], int, int]]:
        return self._process

    @process.setter
    def process(self, value: list[tuple[str, str, str, str, int, Union[Callable, dict[str, Any]], int, int]]) -> None:
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

    # Process Management ===============================================================================================
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_data, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    @simple_type_validator
    def add_process(
        self,
        # Process general parameters
        input_data_level: Union[str, int],
        output_data_level: Union[str, int],
        application_sequence: int,
        method: Union[Callable, nn.Module, list[Callable], tuple[Callable], list[nn.Module], tuple[nn.Module]],
        process_label: Union[str, list[str], tuple[str]] = "",
        *,
        test_error_raise: bool = True,
        # Fittable training parameters
        batch_size: Union[int, tuple[int], list[int], None] = None,
        shuffle: Optional[bool] = None,
        criterion: Union[torch.nn.Module, tuple[torch.nn.Module], list[torch.nn.Module], None] = None,
        optimizer: Union[torch.optim.Optimizer, list[torch.optim.Optimizer], None] = None,
        scheduler: Union[
            torch.optim.lr_scheduler.LRScheduler,
            torch.optim.lr_scheduler._LRScheduler,
            list[Union[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler._LRScheduler]],
            None,
        ] = None,
        # Modeling parameters
        is_regression: Optional[bool] = None,
        validation_method: str = "2-fold",
        unseen_threshold: Optional[float] = 0.0,
        x_shape: Optional[tuple[int]] = None,
        result_backup: bool = False,
        # Model validation and evaluation configurations
        data_split_config: Union[str, dict[str, Any]] = "default",
        validation_config: Union[str, dict[str, Any]] = "default",
        metrics_config: Union[str, dict[str, Any], None] = "default",
        roc_plot_config: Union[str, dict, None] = "default",
        scatter_plot_config: Union[str, dict[str, Any], None] = "default",
        residual_config: Union[str, dict[str, Any], None] = "default",
        residual_plot_config: Union[str, dict[str, Any], None] = "default",
        influence_analysis_config: Union[str, dict[str, Any], None] = "default",
        save_application_model: bool = True,
    ) -> None:
        """
        Add a processing method with defined input/output data levels and application sequence to the pipeline.
        A processing method can be a preprocessing function or a model for evaluation.

        Parameters
        ----------
        input_data_level : int or str
            Input data level for the process. Available options:

                ``0`` or ``"function"``
                    If the provided method is applied to input image tensors as a deterministic preprocessing function before any trainable ``nn.Module`` processing.

                ``1`` or ``"fittable"``
                    If the provided method is applied to input image tensors as an adaptive transformer or estimator trained from data.

        output_data_level : int or str
            Output data level. Available options:

                ``0`` or ``"function"``
                    If the output tensors are intended for further deterministic preprocessing before any trainable ``nn.Module`` processing.

                ``1`` or ``"fittable"``
                    If the output tensors are intended for further adaptive transformation or representation-learning ``nn.Module`` processing.

                ``2`` or ``"model"``
                    If the output tensors are intended for final predictive modeling and performance evaluation.

                    The final process in the pipeline must be an estimator or predictive ``nn.Module`` with ``"model"`` as its ``output_data_level``.

        application_sequence : int
            Sequence number of the method within the same input data level. Lower numbers execute first.

        method : callable or list of (callable or nn.Module)
            Method applied for the step.
            If Callable, the method will be stored directly in the attribute of the instance.
            If nn.Module, the method will be stored in a file and the file path is stored in the attribute of the instance.

        process_label : str or list of str, optional
            Custom label(s) for the process.

            If provided:

                - If a single method is provided, must be a single str of label.
                - If multiple methods are provided, must be a list of str with a length equal to the number of methods.

            Default is an empty string, which automatically generates label(s) using the Callable name(s) or the estimator class name(s).

        test_error_raise : bool, optional
            Whether to raise error when the process fails in validation using simplified mock data before added to the pipeline.

            If True, an exception is raised, otherwise only a warning is issued. Default is True.

        batch_size : int or list of int, optional
            Number of samples processed in a single training batch when training an ``nn.Module``.
            See ``add_model`` for details. Default is None.

            Ignored for deterministic functions.

        shuffle : bool, optional
            Whether to shuffle the dataset at the beginning of each training epoch.
            See ``add_model`` for details. Default is None.

            Ignored for deterministic functions.

        criterion : torch.nn.Module or list of torch.nn.Module, optional
            The loss function used to evaluate the error during training (e.g., ``torch.nn.CrossEntropyLoss``).
            See ``add_model`` for details. Default is None.

            Ignored for deterministic functions.

        optimizer : torch.optim.Optimizer or list of torch.optim.Optimizer, optional
            The optimizer used to update model weights (e.g., ``torch.optim.Adam``).
            See ``add_model`` for details. Default is None.

            Ignored for deterministic functions.

        scheduler : torch.optim.lr_scheduler.LRScheduler or list of torch.optim.lr_scheduler.LRScheduler, optional
            The learning rate scheduler used to dynamically adjust the learning rate during training.
            See ``add_model`` for details. Default is None.

            Ignored for deterministic functions.

        is_regression : bool, optional
            Whether the model is a regression model.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        validation_method : str, optional
            Validation strategy for model evaluation.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        unseen_threshold : float, optional
            Classification-only parameter.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        x_shape : tuple of int, optional
            Expected shape of independent variables for models requiring structured input. Currently ignored.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        result_backup : bool, optional
            Whether to save timestamped backup copies of result files.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        data_split_config : str or dict, optional
            Additional data splitting configuration.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        validation_config : str or dict, optional
            Validation behavior configuration.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        metrics_config : str or dict or None, optional
            Metrics computation configuration.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        roc_plot_config : str or dict or None, optional
            Receiver Operating Characteristic (ROC) plotting configuration for classification models.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        scatter_plot_config : str or dict or None, optional
            Scatter plot configuration for regression models.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        residual_config : str or dict or None, optional
            Residual analysis configuration.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        residual_plot_config : str or dict or None, optional
            Residual plot configuration for regression models.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        influence_analysis_config : str or dict or None, optional
            Influence analysis configuration.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        save_application_model : bool, optional
            Influence analysis configuration.
            See ``add_model`` for details.

            Ignored for deterministic functions.

        See Also
        --------
        add_model

        Returns
        -------
        None

        Examples
        --------
        For prepared ``SpecExp`` instance ``exp``::

            >>> pipe = SpecPipeTensor(exp)

        Add an image processor accepting image path and returning processed path::

            >>> pipe.add_process('function', 'function', 0, snv_transform)

        Or using numeric level indices::

            >>> pipe.add_process(0, 0, 0, snv_transform)

        Customize method name::

            >>> pipe.add_process(0, 0, 0, snv_transform, process_label='snv')

        Output for model training::

            >>> from swectral.functions import snv as snv_transform
            >>> pipe.add_process('function', 'model', 0, snv_transform)

        Or using numeric level indices::

            >>> pipe.add_process(0, 2, 0, snv_transform)

        Add multiple alternative methods for a step::

            >>> pipe.add_process(1, 2, 0, [snv_transform, aucnorm_transform])
        """  # noqa: E501

        # 1. Validate and Parse Levels and Sequence
        dl_in_ind, dl_in_name, dl_out_ind, dl_out_name = _data_level_seq_validator(
            input_data_level=input_data_level,
            output_data_level=output_data_level,
            application_sequence=application_sequence,
        )

        # 2. Standardize Methods, Labels, Batch Sizes, Criteria, Optimizers and Schedulers to iterables
        methods: list = list(method) if isinstance(method, (list, tuple)) else [method]

        if isinstance(process_label, (list, tuple)):
            labels: list = list(process_label)
            if len(labels) != len(methods):
                raise ValueError("Length of process_label list must match the length of provided methods.")
        else:
            labels = [process_label] * len(methods)

        criteria: list = list(criterion) if isinstance(criterion, (list, tuple)) else [criterion]
        if len(criteria) == 1 and len(methods) > 1:
            criteria = criteria * len(methods)
        batch_sizes: list = list(batch_size) if isinstance(batch_size, (list, tuple)) else [batch_size]
        if len(batch_sizes) == 1 and len(methods) > 1:
            batch_sizes = batch_sizes * len(methods)

        optimizers: list = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers: list = list(scheduler) if isinstance(scheduler, (list, tuple)) else [scheduler]
        if all((sch is None) for sch in schedulers):
            schedulers = [None] * len(methods)

        # 3. Validate Training Parameters
        _training_parameter_validator(
            methods=methods,
            batch_sizes=batch_sizes,
            shuffle=shuffle,
            criteria=criteria,
            optimizers=optimizers,
            schedulers=schedulers,
        )

        # 4. Calculate _Full_app_sequence
        fapp_seq = int(2000000 * dl_out_ind - 1000000 * int(dl_in_ind != dl_out_ind) + application_sequence)

        # 5. Find current maximum alternative number for this step
        existing_alts = [
            p[7] for p in self._process if p[6] == fapp_seq or (p[2] == dl_in_name and p[4] == application_sequence)
        ]
        alt_num = max(existing_alts) + 1 if existing_alts else 0

        # 6. Process each defined method
        for i, m in enumerate(methods):

            # Determine actual label
            lbl = labels[i]
            actual_label = lbl if lbl else getattr(m, '__name__', m.__class__.__name__)

            # Formulate Process ID
            process_id = f"{dl_in_ind}_{application_sequence}_%#{alt_num}"

            # Formulate Method_data
            method_data: Union[Callable, dict[str, Any]]
            if isinstance(m, nn.Module) and (dl_in_ind == 1 or dl_out_ind == 2):
                # Training parameters
                bs = batch_sizes[i]
                crit = criteria[i]
                opt = optimizers[i]
                sch = schedulers[i]
                # Save the nn.Module to a file for lazy-loading
                # Create save dir
                model_dir = self.report_directory + "SpecPipe_configuration/models/"
                os.makedirs(unc_path(model_dir), exist_ok=True)
                # Create save path
                file_name = model_dir + f"{process_id}_{actual_label}.pt"
                # Save model
                torch.save(m, file_name)

                # Determine 'use' type based on output expectation
                if dl_out_name == 'model':
                    use_type = 'regression' if is_regression else 'classification'
                else:
                    use_type = 'preprocessing'

                evaluation_dict = {
                    'is_regression': is_regression,
                    'validation_method': validation_method,
                    'unseen_threshold': unseen_threshold,
                    'x_shape': x_shape,
                    'result_backup': result_backup,
                    'data_split_config': data_split_config,
                    'validation_config': validation_config,
                    'metrics_config': metrics_config,
                    'roc_plot_config': roc_plot_config,
                    'scatter_plot_config': scatter_plot_config,
                    'residual_config': residual_config,
                    'residual_plot_config': residual_plot_config,
                    'influence_analysis_config': influence_analysis_config,
                    'save_application_model': save_application_model,
                }

                method_data = {
                    'use': use_type,
                    'batch_size': bs,
                    'shuffle': shuffle,
                    'criterion': crit,
                    'optimizer': opt,
                    'scheduler': sch,
                    'module': file_name,
                    'module_name': getattr(m, '__name__', m.__class__.__name__),
                    'evaluation': evaluation_dict,
                }
            elif dl_in_ind == 0 and (dl_out_ind < 2):
                # Deterministic callable functions get stored directly
                assert callable(m)
                method_data = m
            else:
                raise ValueError(
                    "Incompatible input and output data level and method object.\n"
                    f"Got input data level: {dl_in_name}\n"
                    f"Got output data level: {dl_in_name}\n"
                    f"Got method object: {m}\n"
                )

            # 7. Append constructed tuple to process attribute
            # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_data, 6 _Full_app_sequence, 7 _Alternative_number]  # noqa: E501
            new_process = (
                process_id,
                actual_label,
                dl_in_name,
                dl_out_name,
                application_sequence,
                method_data,
                fapp_seq,
                alt_num,
            )

            self._process.append(new_process)
            alt_num += 1

        self._generate_process_steps()
        self._generate_process_chains()

    @simple_type_validator
    def _get_process(  # noqa: C901
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Union[int, tuple[int, int], None] = None,
        method: Union[str, Callable, dict[str, Any], None] = None,
        full_application_sequence: Union[int, tuple[int, int], None] = None,
        exact_match: bool = True,
    ) -> tuple[
        list[tuple[str, str, str, str, int, Union[Callable, dict[str, Any]], int, int]],
        list[tuple[str, str, str, str, int, Union[Callable, dict[str, Any]], int, int]],
    ]:
        """
        Get matched and unmatched process items by process properties.
        """
        # Validate processes
        if len(self._process) == 0:
            print("No process added!")
            return ([], [])

        # Standardize data level properties to string names matching self._process storage
        ind_to_name = {0: 'function', 1: 'fittable', 2: 'model'}
        if input_data_level is not None:
            input_data_level = (
                ind_to_name.get(input_data_level, input_data_level)
                if isinstance(input_data_level, int)
                else input_data_level
            )
        if output_data_level is not None:
            output_data_level = (
                ind_to_name.get(output_data_level, output_data_level)
                if isinstance(output_data_level, int)
                else output_data_level
            )

        prps: list = [
            process_id,
            process_label,
            input_data_level,
            output_data_level,
            application_sequence,
            method,
            full_application_sequence,
        ]

        # Retrieve removed and remained image items
        get_items = []
        rest_items = []
        for pit in self._process:
            # Validate properties - retrieve and validate
            rcond = True
            # Matches indices 0 to 6
            # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_data, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
            for i, prop in enumerate(pit[0:-1]):
                # Condition - not applied
                if prps[i] is None:
                    cond = True
                # Condition - Application_sequence (4) and Full_application_sequence (6)
                elif ((i == 4) or (i == 6)) and isinstance(prps[i], tuple):
                    if prps[i][0] >= prps[i][1]:
                        raise ValueError(f"Invalid value range [{prps[i][0]}, {prps[i][1]})")
                    else:
                        cond = (prop >= prps[i][0]) and (prop < prps[i][1])
                # Condition - Method Callable or Dict (5)
                elif i == 5:
                    if callable(prps[i]):
                        cond = prop is prps[i]
                    elif isinstance(prps[i], str):
                        # Extract a string name to compare against depending on method type
                        if callable(prop):
                            prop_name = getattr(prop, '__name__', str(prop))
                        elif isinstance(prop, dict):
                            prop_name = prop.get('module_name', prop.get('module', ''))  # Name of nn.module
                        else:
                            prop_name = str(prop)

                        cond = ((prps[i] == prop_name) and exact_match) or ((prps[i] in prop_name) and not exact_match)
                    else:
                        cond = ((prps[i] == prop) and exact_match) or ((str(prps[i]) in str(prop)) and not exact_match)
                # Condition - Standard String/Int matching
                else:
                    cond = ((prps[i] == prop) and exact_match) or ((str(prps[i]) in str(prop)) and not exact_match)

                rcond = rcond and cond

            # Retrieval
            if rcond:
                get_items.append(pit)
            else:
                rest_items.append(pit)

        return get_items, rest_items

    # List / view added processes
    @overload
    def ls_process(
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Optional[int] = None,
        method: Union[str, Callable, dict[str, Any], None] = None,
        full_application_sequence: Optional[int] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: Literal[True] = True,
    ) -> pd.DataFrame: ...

    @overload
    def ls_process(
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Optional[int] = None,
        method: Union[str, Callable, dict[str, Any], None] = None,
        full_application_sequence: Optional[int] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: Literal[False] = False,
    ) -> None: ...

    # List / view added processes
    @simple_type_validator
    def ls_process(
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Optional[int] = None,
        method: Union[str, Callable, dict[str, Any], None] = None,
        full_application_sequence: Optional[int] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        List process items based on filtering conditions.
        If a filter criterion is ``None``, the corresponding filter is not applied.

        Parameters
        ----------
        process_id : str, optional
            Process ID.
            The default is ``None``.

        process_label : str, optional
            Custom process label.
            The default is ``None``.

        input_data_level : str or int, optional
            Input data level of the process.

            See ``add_process`` for available options.
            The default is ``None``.

        output_data_level : str or int, optional
            Output data level of the process.

            See ``add_process`` for available options.
            The default is ``None``.

        application_sequence : int or tuple of int, optional
            Exact sequence number or a sequence number range within a data level.

            Ranges must be specified as a tuple.
            The default is ``None``.

        method : str or callable or object, optional
            Method function, method name, or method object.
            The default is ``None``.

        full_application_sequence : int or tuple of int, optional
            Exact sequence number or a sequence number range within the entire pipeline.
            Ranges must be specified as a tuple.
            The default is ``None``.

        exact_match : bool, optional
            If False, processes whose property values partially match the specified value are included.
            The default is ``True``.

        print_result : bool, optional
            Whether to print simplified matched process items.
            The default is ``True``.

        return_result : bool, optional
            Whether to return a dataframe of matched process items.
            The default is ``False``.

        Returns
        -------
        pandas.DataFrame or None
            If ``return_result=True``, returns a pandas DataFrames of matched process items.

            If ``return_result=False``, returns None.

        See Also
        --------
        add_process

        Examples
        --------
        For prepared ``SpecExp`` instance ``exp``::

            >>> pipe = SpecPipeTensor(exp)
            >>> from swectral.functions import snv
            >>> pipe.add_process(0, 0, 0, snv)

        List all added processes::

            >>> pipe.ls_process()

        List processes by input data level::

            >>> pipe.ls_process(input_data_level=0)

        List processes by output data level::

            >>> pipe.ls_process(output_data_level=1)

        List processes by method name::

            >>> pipe.ls_process(method='snv')

        List processes by partial method name::

            >>> pipe.ls_process(method='nv', exact_match=False)

        Return results instead of printing::

            >>> df_process = pipe.ls_process(print_result=False, return_result=True)
        """

        # Get matched processes
        matched = self._get_process(
            process_id,
            process_label,
            input_data_level,
            output_data_level,
            application_sequence,
            method,
            full_application_sequence,
            exact_match,
        )[0]

        # Full matched in dataframe
        df_proc = pd.DataFrame(
            matched,
            columns=[
                "ID",
                "Process_label",
                "Input_data_level",
                "Output_data_level",
                "Application_sequence",
                "Method",
                "Sequence_in_complete_process",
                "Alternative_number",
            ],
        )

        # Format the Method column to prevent dataframe rendering crashes
        for i in range(df_proc.shape[0]):
            method_val = df_proc.iloc[i, -3]  # Method is 3rd column from the end
            if callable(method_val):
                df_proc.iloc[i, -3] = getattr(method_val, '__name__', str(method_val))
            elif isinstance(method_val, dict):
                # Represent nn.Module configurations securely for logging
                file_name = method_val.get('module', 'Unknown_Module.pt')
                df_proc.iloc[i, -3] = f"<nn.Module: {file_name}>"
            else:
                df_proc.iloc[i, -3] = str(method_val)

        # Print simple df
        if print_result:
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                df_proc_simple = df_proc.iloc[:, [0, 1, 2, 3, 4, 5]]
                print(df_proc_simple)

        # Return df
        if return_result:
            return df_proc
        elif print_result:
            return None
        else:
            raise ValueError("At least one of return_result or print_result must be True.")

    # Remove matched added processes
    @simple_type_validator
    def rm_process(  # noqa: C901
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Optional[int] = None,
        method: Union[str, Callable, dict[str, Any], None] = None,
        # full_application_sequence: Optional[int] = None,
        exact_match: bool = True,
    ) -> None:
        """
        Remove process items based on filtering conditions.
        If a filter criterion is not provided, the corresponding filter is not applied.

        Parameters
        ----------
        process_id : str, optional
            Process ID.
            The default is None.

        process_label : str, optional
            Custom process label.
            The default is None.

        input_data_level : str or int, optional
            Input data level of the process.
            See ``add_process`` for available options.
            The default is None.

        output_data_level : str or int, optional
            Output data level of the process.
            See ``add_process`` for available options.
            The default is None.

        application_sequence : int or tuple of int, optional
            Exact sequence number or a sequence number range within a data level.
            Ranges must be specified as a tuple.
            The default is None.

        method : str or callable or object, optional
            Method function, method name, or method object.
            The default is None.

        full_application_sequence : int or tuple of int, optional
            Exact sequence number or a sequence number range within the entire pipeline.
            Ranges must be specified as a tuple.
            The default is None.

        exact_match : bool, optional
            If False, processes whose property values partially match the specified
            value are included.
            The default is True.

        print_df : bool, optional
            Whether to print simplified matched process items.
            The default is True.

        return_df : bool, optional
            Whether to return a dataframe of matched process items.
            The default is False.

        See Also
        --------
        add_process

        Examples
        --------
        For prepared ``SpecExp`` instance ``exp``::

            >>> pipe = SpecPipe(exp)
            >>> from swectral.functions import snv
            >>> pipe.add_process(0, 0, 0, snv)

        Remove all added processes::

            >>> pipe.rm_process()

        Remove processes by input data level::

            >>> pipe.rm_process(input_data_level=0)

        Remove processes by output data level::

            >>> pipe.rm_process(output_data_level=0)

        Remove processes by method::

            >>> pipe.rm_process(method='snv')
        """

        # Not applied parameters
        full_application_sequence = None
        # Change process test status
        self.__tested = False

        # Filter processes
        matched, unmatched = self._get_process(
            process_id,
            process_label,
            input_data_level,
            output_data_level,
            application_sequence,
            method,
            full_application_sequence,
            exact_match,
        )

        if len(unmatched) == 0:
            print("\nAll processes are removed")

            # Clean up all disk files before clearing lists
            for p in self._process:
                method_callable = p[5]
                if isinstance(method_callable, dict) and 'module' in method_callable:
                    try:
                        os.remove(method_callable['module'])
                    except OSError:
                        pass

            self._process = []
            self._process_steps = []
            self._process_chains = []
            self._custom_chains = []

        else:
            if len(matched) > 0:
                # Clean up specific disk files for matched/removed processes
                for p in matched:
                    method_callable = p[5]
                    if isinstance(method_callable, dict) and 'module' in method_callable:
                        try:
                            os.remove(method_callable['module'])
                        except OSError:
                            pass

                # Update process
                self._process = unmatched
                # Sort processes and update process_steps and process_chains
                self._generate_process_steps()
                self._generate_process_chains()
                self._update_custom_chains()

                # Print report
                df_proc = pd.DataFrame(
                    matched,
                    columns=[
                        "ID",
                        "Process_label",
                        "Input_data_level",
                        "Output_data_level",
                        "Application_sequence",
                        "Method",
                        "Sequence_in_complete_process",
                        "Alternative_number",
                    ],
                )

                # Format Method Column Safely
                for i in range(df_proc.shape[0]):
                    method_val = df_proc.iloc[i, -3]  # Method is 3rd column from the end
                    if callable(method_val):
                        df_proc.iloc[i, -3] = getattr(method_val, '__name__', str(method_val))
                    elif isinstance(method_val, dict):
                        # Represent nn.Module configurations securely for logging
                        file_name = method_val.get('module', 'Unknown_Module.pt')
                        df_proc.iloc[i, -3] = f"<nn.Module: {file_name}>"
                    else:
                        df_proc.iloc[i, -3] = str(method_val)

                with pd.option_context("display.max_rows", None, "display.max_columns", None):
                    df_proc_simple = df_proc.iloc[:, [1, 2, 3, 4, 5]]
                    print(f"\nFollowing processes are removed from the pipeline: \n{df_proc_simple}")
            else:
                print("\nNo matched process found")

        self._generate_process_steps()
        self._generate_process_chains()

    # Processing chain composition =====================================================================================
    # Custom chain updater
    def _update_custom_chains(self) -> None:
        """
        Updates custom process chains by keeping only those that still exist in the generated full process chains.
        """
        if len(self._custom_chains) > 0:
            # Convert to set for O(1) fast lookup
            valid_chains = set(self._process_chains)

            # List comprehension preserves order while filtering non-existent chains
            self._custom_chains = [chain for chain in self._custom_chains if chain in valid_chains]

    # Sort added processes
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_data, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    def _sort_proc(self) -> None:
        """
        Sort added processes internally.
        """  # noqa: E501
        if len(self._process) > 1:
            # Native Python sort is significantly faster than pandas DataFrame sorting
            # Sorts primarily by _Full_app_seq (idx 6), then by _Alternative_number (idx 7)
            self._process.sort(key=lambda pit: (pit[6], pit[7]))

    # Generate processing steps and processes for each step
    def _generate_process_steps(self) -> None:
        """
        Generates process steps of Process_IDs, representing a sequential workflow of added process items.
        Length of the chain list represents the total number of processing steps.
        """
        if len(self._process) > 0:
            # Ensure the process list is sorted before grouping
            self._sort_proc()

            # Use a dictionary to group Process IDs by their Full Application Sequence
            steps_dict: dict = {}
            for p_item in self._process:
                process_id = p_item[0]
                fseq = p_item[6]

                if fseq not in steps_dict:
                    steps_dict[fseq] = []
                steps_dict[fseq].append(process_id)

            # Convert the grouped dictionary values back to a list of lists
            self._process_steps = list(steps_dict.values())
        else:
            # Clear steps safely if no process exists
            self._process_steps = []

    # Compose processing chains from _process_steps
    def _generate_process_chains(self) -> None:
        """
        Generates a list of all actual chains of added process items in execution.
        Each chain corresponds to a unique final result.
        """
        if len(self._process_steps) > 0:
            self._process_chains = list(itertools.product(*self._process_steps))
        else:
            self._process_chains = []

        # Ensure custom chains are synchronized after base chains are rebuilt
        self._update_custom_chains()

    # Model Management - from Process Management =======================================================================
    @simple_type_validator
    def add_model(
        self,
        # Process general parameters - part 1
        method: Union[Callable, nn.Module, list[Callable], tuple[Callable], list[nn.Module], tuple[nn.Module]],
        # Fittable training parameters
        batch_size: int,
        shuffle: bool,
        criterion: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, list[torch.optim.Optimizer]],
        scheduler: Union[
            torch.optim.lr_scheduler.LRScheduler,
            torch.optim.lr_scheduler._LRScheduler,
            list[
                Union[
                    torch.optim.lr_scheduler.LRScheduler,
                    torch.optim.lr_scheduler._LRScheduler,
                ]
            ],
        ],
        # Process general parameters - part 2
        input_data_level: Union[str, int, None] = None,
        process_label: Union[str, list[str], tuple[str]] = "",
        *,
        test_error_raise: bool = True,
        # Modeling parameters
        is_regression: Optional[bool] = None,
        validation_method: str = "2-fold",
        unseen_threshold: Optional[float] = 0.0,
        x_shape: Optional[tuple[int]] = None,
        result_backup: bool = False,
        # Model validation and evaluation configurations
        data_split_config: Union[str, dict[str, Any]] = "default",
        validation_config: Union[str, dict[str, Any]] = "default",
        metrics_config: Union[str, dict[str, Any], None] = "default",
        roc_plot_config: Union[str, dict, None] = "default",
        scatter_plot_config: Union[str, dict[str, Any], None] = "default",
        residual_config: Union[str, dict[str, Any], None] = "default",
        residual_plot_config: Union[str, dict[str, Any], None] = "default",
        influence_analysis_config: Union[str, dict[str, Any], None] = "default",
        save_application_model: bool = True,
    ) -> None:
        """
        Add a model evaluation process to the processing pipeline.

        Parameters
        ----------
        method : callable or list of (callable or nn.Module)
            Method applied for the step.
            If Callable, the method will be stored directly in the attribute of the instance.
            If nn.Module, the method will be stored in a file and the file path is stored in the attribute of the instance.

        batch_size : int or list of int, optional
            Number of samples processed in a single training batch when training an ``nn.Module``.
            Must be specified if an ``nn.Module`` is added. Default is None.

            If provided:

                - If a single batch size is provided, it will be applied to all input methods.
                - If a list of batch sizes are provided, the length must be equal to the number of methods.

        shuffle : bool, optional
            Whether to shuffle the dataset at the beginning of each training epoch.
            Must be specified if an ``nn.Module`` is added. Default is None.

        criterion : torch.nn.Module or list of torch.nn.Module, optional
            The loss function used to evaluate the error during training (e.g., ``torch.nn.CrossEntropyLoss``).
            Must be specified if an ``nn.Module`` is added. Default is None.

            If provided:

                - If a single criterion is provided, it will be applied to all input methods.
                - If a list of criteria are provided, the length must be equal to the number of methods.

        optimizer : torch.optim.Optimizer or list of torch.optim.Optimizer, optional
            The optimizer used to update model weights (e.g., ``torch.optim.Adam``).
            Must be specified if an ``nn.Module`` is added. Default is None.

            If provided:

                - If a single method is provided, must be a single optimizer.
                - If multiple methods are provided, must be a list of optimizers with a length equal to the number of methods.

        scheduler : torch.optim.lr_scheduler.LRScheduler or list of torch.optim.lr_scheduler.LRScheduler, optional
            The learning rate scheduler used to dynamically adjust the learning rate during training.
            If None, no scheduler will be used in training.
            Ignored for deterministic functions.
            Default is None.

            If provided:

                - If a single method is provided, must be a single scheduler.
                - If multiple methods are provided, must be a list of schedulers with a length equal to the number of methods.

        input_data_level : int or str or None, optional
            Input data level for the process.

            If ``None``, the ``input_data_level`` is deduced automatically from the previously added processes.
            If any trainable transformation (where ``output_data_level=1`` or ``"fittable"``) exists in the pipeline, it defaults to ``1`` (``"fittable"``).
            Otherwise, it defaults to ``0`` (``"function"``).

            Default is None.

            See ``add_process`` for details.

        process_label : str or list of str, optional
            Custom label(s) for the process.

            If provided:

                - If a single method is provided, must be a single str of label.
                - If multiple methods are provided, must be a list of str with a length equal to the number of methods.

            Default is an empty string, which automatically generates label(s) using the Callable name(s) or the estimator class name(s).

        is_regression : bool, optional
            Whether the model is a regression model.

            If None, the model type is inferred from sample target values.
            Default is None.

        validation_method : str, optional
            Validation strategy for model evaluation.
            Supported formats include:

            - ``"loo"`` for leave-one-out cross-validation
            - ``"k-fold"`` (e.g. ``"5-fold"``) for k-fold cross-validation
            - ``"m-n-split"`` (e.g. ``"70-30-split"``) for train-test split

            Default is ``"2-fold"``.

        unseen_threshold : float, optional
            Classification-only parameter.

            If the highest predicted class probability of a sample is below this threshold, the sample is assigned to an unknown class.
            Default is 0.0.

        x_shape : tuple of int, optional
            Expected shape of independent variables for models requiring structured input. Default is None.

            Currently ignored.

        result_backup : bool, optional
            Whether to save timestamped backup copies of result files.
            Default is False.

        data_split_config : str or dict, optional
            Additional data splitting configuration.

            If a dictionary of parameters is provided, it may include:

                ``random_state`` : int
                    Random state for splitting and shuffling.

            Default is ``"default"``, which uses the default data splitting behavior.

        validation_config : str or dict, optional
            Validation behavior configuration.

            If a dictionary of parameters is provided, it may include:

                ``unseen_threshold`` : float
                    If an unseen class for the training data exists, a test sample is predicted to the unseen class if the predicted probabilities of seen classes is lower than this threshold.
                    Default is 0 (Only predict seen classes).
                ``use_original_shape`` : bool
                    Whether data shape is applied for the model. Currently no use. Default is False.
                ``save_fold_model`` : bool
                    Whether models of the validation folds are saved to files. Default is True.
                ``save_fold_data`` : bool
                    Whether models of the validation folds are saved to files. Default is True.

            Default is ``"default"``, which uses the default validation behavior.

        metrics_config : str or dict or None, optional
            Metrics computation configuration.

            If None, metric computation is skipped.
            Default is ``"default"``. Currently only ``"default"`` is supported.

        roc_plot_config : str or dict or None, optional
            Receiver Operating Characteristic (ROC) plotting configuration for classification models.

            No use for regression models.

            If None, ROC plot generation is skipped.

            If a dictionary of parameters is provided, it may include:

                ``plot_title`` : str
                    title of the ROC plot. Default is 'ROC Curve'.
                ``title_size`` : int or float
                    font size of the plot title. Default is 26.
                ``title_pad`` : int or float or None
                    padding between the title and the plot. Default is None.
                ``figure_size`` : tuple of 2 (float or int)
                    figure size as (width, height). Default is (8, 8).
                ``plot_margin`` : tuple of 4 float
                    plot margins as (left, right, top, bottom). Default is (0.15, 0.95, 0.9, 0.13).
                ``plot_line_width`` : int or float
                    line width of the ROC curve. Default is 3.
                ``plot_line_alpha`` : float
                    alpha value of the ROC curve line. Default is 0.8.
                ``diagnoline_width`` : int or float
                    line width of the diagonal reference line. Default is 3.
                ``x_axis_limit`` : tuple of 2 (float or int) or None
                    x-axis limits as (min, max). Default is None.
                ``x_axis_label`` : str
                    label of the x-axis. Default is 'False Positive Rate'.
                ``x_axis_label_size`` : int or float
                    font size of the x-axis label. Default is 26.
                ``x_tick_size`` : int or float
                    font size of x-axis tick labels. Default is 24.
                ``x_tick_number`` : int
                    number of x-axis ticks. Default is 6.
                ``y_axis_limit`` : tuple of 2 (float or int) or None
                    y-axis limits as (min, max). Default is None.
                ``y_axis_label`` : str
                    label of the y-axis. Default is 'True Positive Rate'.
                ``y_axis_label_size`` : int or float
                    font size of the y-axis label. Default is 26.
                ``y_tick_size`` : int or float
                    font size of y-axis tick labels. Default is 24.
                ``y_tick_number`` : int
                    number of y-axis ticks. Default is 6.
                ``axis_line_size_left`` : int or float or None
                    line width of the left axis spine. Default is 1.5.
                ``axis_line_size_right`` : int or float or None
                    line width of the right axis spine. Default is 1.5.
                ``axis_line_size_top`` : int or float or None
                    line width of the top axis spine. Default is 1.5.
                ``axis_line_size_bottom`` : int or float or None
                    line width of the bottom axis spine. Default is 1.5.
                ``legend`` : bool
                    whether to display the legend. Default is True.
                ``legend_location`` : str
                    legend location string accepted by matplotlib. Default is 'lower right'.
                ``legend_fontsize`` : int or float
                    font size of legend entries. Default is 20.
                ``legend_title`` : str
                    legend title text. Default is empty.
                ``legend_title_fontsize`` : int or float
                    font size of the legend title. Default is 24.
                ``background_grid`` : bool
                    whether to show a background grid. Default is False.
                ``show_plot`` : bool
                    whether to display the plot interactively. Default is False.

            Default is ``"default"``, which uses the default plotting behavior.

        scatter_plot_config : str or dict or None, optional
            Scatter plot configuration for regression models.

            If None, scatter plot generation is skipped.

            If a dictionary of parameters is provided, it may include:

                ``plot_title`` : str
                    plot title text. Default is ''.
                ``title_size`` : int or float
                    font size of the plot title. Default is 26.
                ``title_pad`` : int or float or None
                    padding between the title and the plot. Default is None.
                ``figure_size`` : tuple of 2 (float or int)
                    figure size in inches as (width, height). Default is (8, 8).
                ``plot_margin`` : tuple of 4 float
                    plot margins as (left, right, top, bottom). Default is (0.2, 0.95, 0.95, 0.15).
                ``plot_line_width`` : int or float
                    line width of plotted curves. Default is 3.
                ``point_size`` : int or float
                    size of plotted points. Default is 120.
                ``point_color`` : str
                    color of plotted points. Default is 'firebrick'.
                ``point_alpha`` : float
                    transparency of plotted points. Default is 0.7.
                ``x_axis_limit`` : tuple of 2 (float or int) or None
                    limits of the x-axis. Default is None.
                ``x_axis_label`` : str
                    label of the x-axis. Default is 'Predicted target values'.
                ``x_axis_label_size`` : int or float
                    font size of the x-axis label. Default is 26.
                ``x_tick_values`` : list of int or float or None
                    explicit tick values for the x-axis. Default is None.
                ``x_tick_size`` : int or float
                    font size of x-axis ticks. Default is 24.
                ``x_tick_number`` : int
                    number of x-axis ticks. Default is 5.
                ``y_axis_limit`` : tuple of 2 (float or int) or None
                    limits of the y-axis. Default is None.
                ``y_axis_label`` : str
                    label of the y-axis. Default is 'Residuals'.
                ``y_axis_label_size`` : int or float
                    font size of the y-axis label. Default is 26.
                ``y_tick_values`` : list of int or float or None
                    explicit tick values for the y-axis. Default is None.
                ``y_tick_size`` : int or float
                    font size of y-axis ticks. Default is 24.
                ``y_tick_number`` : int
                    number of y-axis ticks. Default is 5.
                ``axis_line_size_left`` : int or float or None
                    line width of the left axis spine. Default is 1.0.
                ``axis_line_size_right`` : int or float or None
                    line width of the right axis spine. Default is 1.5.
                ``axis_line_size_top`` : int or float or None
                    line width of the top axis spine. Default is 1.5.
                ``axis_line_size_bottom`` : int or float or None
                    line width of the bottom axis spine. Default is 1.5.
                ``background_grid`` : bool
                    whether to display background grid lines. Default is False.
                ``show_plot`` : bool
                    whether to display the plot immediately. Default is False.

            Default is ``"default"``, which uses the default plotting behavior.

        residual_config : str or dict or None, optional
            Residual analysis configuration.

            If None, residual analysis is skipped.
            Default is ``"default"``, which uses the default residual analysis behavior.

        residual_plot_config : str or dict or None, optional
            Residual plot configuration for regression models.

            If None, residual plot generation is skipped.

            If a dictionary of parameters is provided, the available parameters are same as ``scatter_plot_config``.

            Default is ``"default"``, which uses the default plotting behavior.

        influence_analysis_config : str or dict or None, optional
            Influence analysis configuration. When enabled, computes a Cook's distance–like influence measure for each sample using a Leave-One-Out (LOO) approach.

            If None, influence analysis is skipped.

            Note: This computation can be very time-consuming for large datasets. For such cases, consider using a simple validation method or setting this option to None.

            If a dictionary of parameters is provided, it may include:

                ``validation_method`` : bool, optional
                    whether to use independent validation for leave-one-out influence analysis.
                ``random_state`` : int or None, optional
                    random state for data splitting.

            Default is ``"default"``, which uses the default influence analysis behavior.

        save_application_model : bool, optional
            Whether application model is trained on all data and stored in the chain report.
            Default is True.

        See Also
        --------
        add_process

        Examples
        --------
        Create a ``SpecPipe`` instance from an existing SpecExp object::

            >>> pipe = SpecPipe(exp)

        Add a model with a specified validation method::

            >>> pipe.add_model(cnn_classifier, validation_method="5-fold")

        Use different validation strategies::

            >>> pipe.add_model(cnn_classifier, validation_method="60-40-split")
            >>> pipe.add_model(cnn_classifier, validation_method="loo")
        """  # noqa: E501
        # 1. Automatic Input Data Level Detection
        if input_data_level is None:
            # Check if any fittable (trainable) transformations exist in the pipeline
            fittable_processes = self.ls_process(output_data_level=1, return_result=True, print_result=False)
            if fittable_processes is None or len(fittable_processes) == 0:
                # No fittable transforms, input comes directly from function/raw data (Level 0)
                input_data_level = 0
            else:
                # Fittable transforms exist, model takes input from them (Level 1)
                input_data_level = 1
        elif input_data_level not in (0, 1, "function", "fittable"):
            raise ValueError(
                f"Input_data_level of model process must be 0 ('function') or 1 ('fittable'), "
                f"but got: {input_data_level}"
            )

        # 2. Hardcoded logic for model (Level 2)
        output_data_level = 2
        application_sequence = 0

        # 3. Pass through to add_process
        self.add_process(
            input_data_level=input_data_level,
            output_data_level=output_data_level,
            application_sequence=application_sequence,
            method=method,
            process_label=process_label,
            # Fittable training parameters
            batch_size=batch_size,
            shuffle=shuffle,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            # Modeling parameters
            is_regression=is_regression,
            validation_method=validation_method,
            unseen_threshold=unseen_threshold,
            x_shape=x_shape,
            result_backup=result_backup,
            # Model evaluation configurations
            data_split_config=data_split_config,
            validation_config=validation_config,
            metrics_config=metrics_config,
            roc_plot_config=roc_plot_config,
            scatter_plot_config=scatter_plot_config,
            residual_config=residual_config,
            residual_plot_config=residual_plot_config,
            influence_analysis_config=influence_analysis_config,
            save_application_model=save_application_model,
        )

    # List model
    @overload
    def ls_model(
        self,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
        model_method: Union[str, dict[str, Any], None] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: Literal[False] = False,
    ) -> None: ...

    @overload
    def ls_model(
        self,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
        model_method: Union[str, dict[str, Any], None] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: Literal[True] = True,
    ) -> pd.DataFrame: ...

    @simple_type_validator
    def ls_model(
        self,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
        model_method: Union[str, dict[str, Any], None] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        List added model evaluation processes based on filtering conditions.
        Acts as a shortcut to `ls_process` filtering for output_data_level=2 ("model").
        """
        # Call ls_process with hardcoded model output level
        if return_result:
            return self.ls_process(
                process_id=model_id,
                process_label=model_label,
                output_data_level=2,
                application_sequence=None,
                method=model_method,
                full_application_sequence=None,
                exact_match=exact_match,
                print_result=print_result,
                return_result=True,
            )
        else:
            self.ls_process(
                process_id=model_id,
                process_label=model_label,
                output_data_level=2,
                application_sequence=None,
                method=model_method,
                full_application_sequence=None,
                exact_match=exact_match,
                print_result=print_result,
                return_result=False,
            )
            return None

    # Remove model
    @simple_type_validator
    def rm_model(
        self,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
        model_method: Union[str, dict[str, Any], None] = None,
        exact_match: bool = True,
    ) -> None:
        """
        Remove added model evaluation processes from this pipeline based on filtering conditions.
        Acts as a shortcut to `rm_process` filtering for output_data_level=2 ("model").
        """
        self.rm_process(
            process_id=model_id,
            process_label=model_label,
            output_data_level=2,
            application_sequence=None,
            method=model_method,
            exact_match=exact_match,
        )

    # Model Management - from Process Management =======================================================================
    @simple_type_validator
    def ls_process_chains(
        self, print_label: bool = True, return_label: bool = False
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame], None]:
        """
        List process chains. Returns the default full-factorial process chains.

        Returns a dataframe where each row represents a processing chain with process IDs.
        For custom chains, use ``ls_custom_chains``.

        Parameters
        ----------
        print_label : bool, optional
            If True, prints chains using chain label. Default is True.

        return_label : bool, optional
            If True, returns an additional dataframe of process labels. Default is False.

        Returns
        -------
        pandas.DataFrame or tuple of pandas.DataFrame or None
            If ``return_label=False``, returns a ``pandas.DataFrame`` of process chains in process IDs.

            If ``return_label=True``, returns a tuple of 2 ``pandas.DataFrame`` of process chains in IDs and labels.

            If no process is added to this SpecPipeTensor instance, returns None.

        See Also
        --------
        ls_custom_chains

        Notes
        -----
        This method is also available as ``process_chains_to_df``.

        Examples
        --------
        For prepared ``SpecPipeTensor`` instance ``pipe``::

            >>> pipe.ls_process_chains()

        Or equivalent::

            >>> pipe.process_chains_to_df()

        Return label display in addition to process ID display::

            >>> pipe.ls_process_chains(return_label=True)
        """  # noqa: E501
        if len(self._process_chains) > 0:
            df_chains = pd.DataFrame(
                self._process_chains,
                columns=["Step_" + str(i) for i in range(len(self._process_chains[0]))],
            ).drop_duplicates()

            if return_label or print_label:
                # Optimized Vectorized Label Mapping
                id_to_label = {p[0]: p[1] for p in self._process}
                df_chains_label = df_chains.replace(id_to_label)

                if print_label:
                    with pd.option_context("display.max_rows", None, "display.max_columns", None):
                        print(df_chains_label)
                if return_label:
                    return df_chains, df_chains_label
                else:
                    return df_chains
            else:
                return df_chains
        else:
            print("No process chain found")
            return None

    # Alias
    process_chains_to_df = ls_process_chains

    # Read process chains from dataframe
    @simple_type_validator
    def custom_chains_from_df(self, process_chain_dataframe: Annotated[Any, dataframe_validator()]) -> None:
        """
        Customize processing chains and update chains using a chain dataframe.

        Once custom chains are created, SpecPipeTensor will prioritize their execution, bypassing the original full-factorial chains.

        Parameters
        ----------
        process_chain_dataframe : pandas.DataFrame-like
            A process chain dataframe.

            Must be a subset of the original full-factorial chains, and each chain must be complete.
            Columns must be ['Step_1', 'Step_2', ...] and the length must match the column length of ``process_chains``.
            And all values must be valid process IDs of SpecPipeTensor.

            It is recommended to modify the dataframe obtained from ``ls_process_chains`` or ``process_chains_to_df`` to construct a customized process chain dataframe.

        See Also
        --------
        ls_process_chains
        process_chains_to_df

        Examples
        --------
        For prepared ``SpecPipeTensor`` instance ``pipe``::

            >>> df_chain = pipe.process_chains_to_df()

        After modification, load the modified dataframe::

            >>> pipe.custom_chains_from_df(df_chain_modified)
        """  # noqa: E501
        process_chain_dataframe = dataframe_validator(dtype="str", ncol=len(self._process_chains[0]))(
            process_chain_dataframe
        )

        # Convert chain df to list
        cchain = [tuple(row) for row in process_chain_dataframe.to_numpy()]

        # We use a set for O(1) lookup time during validation, massively speeding up large pipelines
        full_chain_set = set(self._process_chains)

        # Validate given custom chains
        for ind, ccr in enumerate(cchain):
            if ccr not in full_chain_set:
                raise ValueError(
                    f"\nInvalid process chain in given chains: \n{ccr}, \nRow index of invalid chain: {ind}"
                )

        # Change process test status
        self.__tested = False

        # Update
        self._custom_chains = cchain

    @simple_type_validator
    def ls_custom_chains(
        self, print_label: bool = True, return_label: bool = False
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame], None]:
        """
        List customized process chains.

        Returns a dataframe where each row represents a processing chain with process IDs.

        Parameters
        ----------
        print_label : bool, optional
            If True, prints chains using chain label. Default is True.

        return_label : bool, optional
            If True, returns an additional dataframe of process labels. Default is False.

        Returns
        -------
        pandas.DataFrame or None
            If ``return_label=False``, returns a ``pandas.DataFrame`` of process chains in process IDs.

            If ``return_label=True``, returns a tuple of 2 ``pandas.DataFrame`` of process chains in IDs and labels.

            If no custom chain is specified in this SpecPipeTensor instance, returns None.

        See Also
        --------
        ls_process_chains

        Notes
        -----
        This method is also available as ``custom_chains_to_df``.

        Examples
        --------
        For prepared ``SpecPipeTensor`` instance ``pipe``::

            >>> df_chain = pipe.ls_custom_chains()
        """  # noqa: E501
        if len(self._custom_chains) > 0:
            df_chains = pd.DataFrame(
                self._custom_chains,
                columns=["Step_" + str(i) for i in range(len(self._custom_chains[0]))],
            ).drop_duplicates()

            if return_label or print_label:
                # Optimized Vectorized Label Mapping
                id_to_label = {p[0]: p[1] for p in self._process}
                df_chains_label = df_chains.replace(id_to_label)

                if print_label:
                    with pd.option_context("display.max_rows", None, "display.max_columns", None):
                        print(df_chains_label)
                if return_label:
                    return df_chains, df_chains_label
                else:
                    return df_chains
            else:
                return df_chains
        else:
            print("No custom chain configured")
            return None

    # Alias
    custom_chains_to_df = ls_custom_chains

    # List default chains
    @simple_type_validator
    def ls_chains(
        self, print_label: bool = True, return_label: bool = False
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame], None]:
        """
        List process chains for the pipeline execution.

        Returns custom process chains if they are specified; otherwise, returns the default full-factorial process chains.
        Returns a dataframe where each row represents a processing chain with process IDs.

        Parameters
        ----------
        print_label : bool, optional
            If True, prints chains using chain label. Default is True.

        return_label : bool, optional
            If True, returns an additional dataframe of process labels. Default is False.

        Returns
        -------
        pandas.DataFrame or None
            If ``return_label=False``, returns a ``pandas.DataFrame`` of process chains in process IDs.

            If ``return_label=True``, returns a tuple of 2 ``pandas.DataFrame`` of process chains in IDs and labels.

            If no process chain exists in this SpecPipeTensor instance, returns None.

        See Also
        --------
        ls_process_chains
        ls_custom_chains

        Examples
        --------
        For created ``SpecPipeTensor`` instance ``pipe``::

            >>> df_chain = pipe.ls_chains()
        """  # noqa: E501
        if len(self._custom_chains) > 0:
            return self.ls_custom_chains(print_label=print_label, return_label=return_label)
        else:
            return self.ls_process_chains(print_label=print_label, return_label=return_label)

    # Compose pipelines ================================================================================================
    @simple_type_validator
    def compose_pipeline(
        self,
        step_methods: list[
            Union[
                tuple[
                    tuple[Union[str, int], Union[str, int]],
                    Union[Callable, nn.Module, list[Union[Callable, nn.Module]], dict[str, Union[Callable, nn.Module]]],
                ],
                tuple[
                    tuple[Union[str, int], Union[str, int]],
                    Union[Callable, nn.Module, list[Union[Callable, nn.Module]], dict[str, Union[Callable, nn.Module]]],
                    dict[str, Any],
                ],
            ]
        ],
    ) -> None:
        """
        Compose pipelines by given structure and methods of each step.

        This method constructs one or more processing pipelines directly from an explicit structural description.
        Each pipeline step is defined by its input/output data levels with one or more alternative callable(s) or nn.Modules responsible for processing at that step.

        Parameters
        ----------
        step_methods : list of tuples
            A list describing the pipeline structure and the processing logic for each step.
            Each element of the list has the form::

                ((input_data_level, output_data_level), methods, params)

            where:

                ``input_data_level`` : int or str
                    Input data level in number or name (e.g., 0 or 'function', 1 or 'fittable').
                ``output_data_level`` : int or str
                    Output data level in number or name.
                ``methods`` : callable or nn.Module or list or dict
                    A single callable or module defining one processing method.
                    A list of callables or modules representing alternative methods for the step.
                    A dictionary mapping method names to callables or modules, allowing named alternatives.
                ``params`` : dict, optional
                    Optional dictionary of additional parameters applied to the methods at the step (e.g., batch_size, criterion, optimizer).

        See Also
        --------
        add_process
        add_model

        Examples
        --------
        For an initialized SpecPipeTensor instance ``pipe``::

            >>> pipe.compose_pipeline(
            ...     [
            ...         # Step 1: Deterministic transforms (0 -> 0)
            ...         ((0, 0), [snv_transformation, minmax_transformation]),
            ...         # Step 2: Extract to fittable (0 -> 1)
            ...         ((0, 1), snv_transformation),
            ...         # Step 3: Fittable Models (1 -> 2)
            ...         (
            ...             (1, 2),
            ...             {'Model_A': cnn1, 'Model_B': cnn2},
            ...             {'batch_size': 32, 'shuffle': True, 'criterion': nn.CrossEntropyLoss(),
            ...              'optimizer': torch.optim.Adam(cnn1.parameters())}
            ...         )
            ...     ]
            ... )
        """  # noqa: E501
        # Dictionary to track the next application_sequence for each unique input_data_level safely
        appseq_tracker = {}

        for step in step_methods:
            dl_in_ind = _dl_val(step[0][0])[0]
            dl_out_ind = _dl_val(step[0][1])[0]

            # Initialize tracker for this data level if not present
            if dl_in_ind not in appseq_tracker:
                appseq_tracker[dl_in_ind] = 0

            application_sequence = appseq_tracker[dl_in_ind]

            # Extract additional params if provided
            if len(step) == 3:
                params = step[2].copy()  # Use copy to avoid mutating the user's dictionary
                # Pop out reserved keys so they don't conflict with explicit kwargs in add_process
                for key in ["input_data_level", "output_data_level", "application_sequence", "method", "process_label"]:
                    params.pop(key, None)
            else:
                params = {}

            # Handle the method block based on its type
            if isinstance(step[1], dict):
                # Unpack dictionary into labels and methods
                process_label = list(step[1].keys())
                method: Union[object, list, dict] = list(step[1].values())

                self.add_process(
                    input_data_level=dl_in_ind,
                    output_data_level=dl_out_ind,
                    application_sequence=application_sequence,
                    method=method,
                    process_label=process_label,
                    **params,
                )
            else:
                # Direct method or list of methods
                method = step[1]
                self.add_process(
                    input_data_level=dl_in_ind,
                    output_data_level=dl_out_ind,
                    application_sequence=application_sequence,
                    method=method,
                    **params,
                )

            # Increment the sequence counter for this input level
            appseq_tracker[dl_in_ind] += 1

    # Save pipeline configurations =====================================================================================
    @simple_type_validator
    def save_pipe_config(self, copy: bool = False, save_spec_exp_config: bool = True) -> None:
        """
        Save the current pipeline configuration files to the root of the report directory.

        Parameters
        ----------
        copy : bool, optional
            Whether to create a backup copy of the configuration files.
            The default is True.

        save_spec_exp_config : bool, optional
            Whether to save the data configuration of the associated ``SpecExp`` instance of this SpecPipe instance.
            The default is True.

        Notes
        -----
        This method is also available as ``save_config``.

        Examples
        --------
        For a created SpecPipe instance ``pipe``::

            >>> pipe.save_pipe_config()

        Or equivalently::

            >>> pipe.save_config()

        Save a backup copy as well::

            >>> pipe.save_pipe_config(copy=True)
        """

        # Create save dir
        report_dir = self.report_directory + "SpecPipe_configuration/"
        os.makedirs(unc_path(report_dir), exist_ok=True)

        # Get configs
        df_process = self.ls_process(print_result=False, return_result=True)
        df_full_chains, df_full_chains_label = self.ls_process_chains(print_label=False, return_label=True)
        df_exec_chains, df_exec_chains_label = self.ls_chains(print_label=False, return_label=True)

        # Save configs
        df_to_csv(
            dataframe=df_process,
            csv_path=unc_path(report_dir + "SpecPipe_added_process.csv"),
            index=False,
            space_wait_timeout=self.space_wait_timeout,
            reserve_free_pct=self.reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )
        df_to_csv(
            dataframe=df_full_chains,
            csv_path=unc_path(report_dir + "SpecPipe_full_factorial_chains_in_ID.csv"),
            index=False,
            space_wait_timeout=self.space_wait_timeout,
            reserve_free_pct=self.reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )
        df_to_csv(
            dataframe=df_full_chains_label,
            csv_path=unc_path(report_dir + "SpecPipe_full_factorial_chains_in_label.csv"),
            index=False,
            space_wait_timeout=self.space_wait_timeout,
            reserve_free_pct=self.reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )
        df_to_csv(
            dataframe=df_exec_chains,
            csv_path=unc_path(report_dir + "SpecPipe_exec_chains_in_ID.csv"),
            index=False,
            space_wait_timeout=self.space_wait_timeout,
            reserve_free_pct=self.reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )
        df_to_csv(
            dataframe=df_exec_chains_label,
            csv_path=unc_path(report_dir + "SpecPipe_exec_chains_in_label.csv"),
            index=False,
            space_wait_timeout=self.space_wait_timeout,
            reserve_free_pct=self.reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )

        # Save SpecPipe
        config_dill_path = unc_path(f"{report_dir}SpecPipe_pipeline_configuration_{self.create_time}.dill")
        dump_dill(
            self,
            target_file_path=config_dill_path,
            backup=False,
            space_wait_timeout=self.space_wait_timeout,
            reserve_free_pct=self.reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )

        # Save copies
        if copy:
            # Prevent duplication
            time.sleep(1.0)
            # Dump copy
            cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            df_to_csv(
                dataframe=df_process,
                csv_path=unc_path(report_dir + f"SpecPipe_added_process_{cts}.csv"),
                index=False,
                space_wait_timeout=self.space_wait_timeout,
                reserve_free_pct=self.reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )
            df_to_csv(
                dataframe=df_full_chains,
                csv_path=unc_path(report_dir + f"SpecPipe_full_factorial_chains_in_ID_{cts}.csv"),
                index=False,
                space_wait_timeout=self.space_wait_timeout,
                reserve_free_pct=self.reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )
            df_to_csv(
                dataframe=df_full_chains_label,
                csv_path=unc_path(report_dir + f"SpecPipe_full_factorial_chains_in_label_{cts}.csv"),
                index=False,
                space_wait_timeout=self.space_wait_timeout,
                reserve_free_pct=self.reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )
            df_to_csv(
                dataframe=df_exec_chains,
                csv_path=unc_path(report_dir + f"SpecPipe_exec_chains_in_ID_{cts}.csv"),
                index=False,
                space_wait_timeout=self.space_wait_timeout,
                reserve_free_pct=self.reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )
            df_to_csv(
                dataframe=df_exec_chains_label,
                csv_path=unc_path(report_dir + f"SpecPipe_exec_chains_in_label_{cts}.csv"),
                index=False,
                space_wait_timeout=self.space_wait_timeout,
                reserve_free_pct=self.reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )
            # Save SpecPipe copy
            config_dill_path_copy = unc_path(
                report_dir + f"SpecPipe_pipeline_configuration_{self.create_time}_copy_at_{cts}.dill"
            )
            dump_dill(
                self,
                target_file_path=config_dill_path_copy,
                backup=False,
                space_wait_timeout=self.space_wait_timeout,
                reserve_free_pct=self.reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )

        # Save SpecExp
        if save_spec_exp_config:
            self.spec_exp.save_data_config(
                copy=copy,
                _space_wait_timeout=self.space_wait_timeout,
                _reserve_free_pct=self.reserve_free_pct,
            )

        # Print output path
        print(
            "\nSpecPipe configurations saved to:\n"
            f"{report_dir}SpecPipe_pipeline_configuration_{self.create_time}.dill\n"
        )

    # Alias
    save_config = save_pipe_config

    # Load pipeline configurations
    @simple_type_validator
    def load_pipe_config(self, config_file_path: str = "") -> None:
        """
        Load SpecPipe configuration from a dill file.

        Parameters
        ----------
        config_file_path : str, optional
            Path to the SpecPipe configuration dill file.

            Can be a file path or the file name in the report directory of this SpecPipe instance.

            If not provided or empty, the path will be:

                ``(SpecPipe.spec_exp.report_directory)/SpecPipe_configuration/SpecPipe_pipeline_configuration_created_at_(SpecExp.create_time).dill``.

            Default is empty string.

        See Also
        --------
        save_pipe_config

        Notes
        -----
        This method is also available as ``load_config``.

        Examples
        --------
        For a created SpecPipe instance ``pipe``::

            >>> pipe.save_pipe_config()

        Load from the default configuration path::

            >>> pipe.load_pipe_config()

        Or equivalently::

            >>> pipe.load_config()

        Load from a custom configuration file path::

            >>> pipe.load_pipe_config("/pipe_config.dill")
        """  # noqa: E501

        # Load path
        if config_file_path == "":
            dump_path0 = (
                self.report_directory
                + "SpecPipe_configuration/"
                + f"SpecPipe_pipeline_configuration_{self.create_time}.dill"
            )
        elif ("/" not in config_file_path) & ("\\" not in config_file_path):
            dump_path0 = self.report_directory + "SpecPipe_configuration/" + config_file_path
        else:
            dump_path0 = config_file_path

        # Load to instance
        loaded_instance = load_dill(unc_path(dump_path0))
        for key, value in loaded_instance.__dict__.items():
            object.__setattr__(self, key, value)

    # Alias
    load_config = load_pipe_config
