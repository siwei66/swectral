# -*- coding: utf-8 -*-
"""
Swectral - Pipeline management and implemention module for spectral image processing and modeling

Copyright (c) 2025 Siwei Luo. MIT License.
"""

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
from collections.abc import Iterable

# Time
import time
from datetime import datetime

# Functions
from functools import partial

# Basic data
import copy
import json
import numpy as np
import pandas as pd
import torch

# Raster
import rasterio
from rasterio.windows import Window
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
from .rasterop import croproi, pixel_apply
from .resultcli import group_stats_report, core_chain_report
from .roistats import roispec, minbbox
from .specexp import SpecExp
from .specio import (
    arraylike_validator,
    dataframe_validator,
    df_to_csv,
    dump_vars,
    load_vars,
    RealNumber,
    simple_type_validator,
    unc_path,
)
from .pipeline_validator import (
    _target_type_validation_for_serialization,
    _dl_val,
    _data_level_seq_validator,
    _spec_exp_validator,
    _process_validator,
    _classifier_validator,
    _regressor_validator,
    _num_image_chains,
    _estimate_img_output_size,
    _estimate_report_plot_size,
)
from .pipeline_processor import (
    _preprocessing_sample,
    _ModelMethod,
    _model_evaluator,
    _model_evaluator_mp,
    _DummyLock,
    _DummyManager,
    _load_disk_backed_data,
)
from .pipeline_tqdm import PipelineFileTqdm, DirProgressObserver
from .modelconnector import (
    combined_model_marginal_stats,
)

# For multiprocessing
global ModelEva


# %% Initialize multiprocessing


_INITIALIZED: bool = False


def _mp_init() -> None:
    """Initialize multiprocessing"""
    # Run once only
    global _INITIALIZED
    if _INITIALIZED:
        return

    # Use dill serialization multiprocessing
    try:
        nmp.reduction.ForkingPickler = dill.Pickler  # type: ignore[misc]
    except AttributeError:
        pass

    # Enforce Linux spawn for multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
        nmp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    _INITIALIZED = True


# %% Spectral Modeling Pipeline Class - SpecPipe


class SpecPipe:
    """
    Design and implement processing and modeling pipelines on spectral experiment datasets.

    Attributes
    ----------
    spec_exp : SpecExp
        Instance of SpecExp configuring spectral experiment datasets. See ``SpecExp`` for details.

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
    def __init__(self, spec_exp: SpecExp) -> None:  # noqa: C901

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
        # Validate SpecExp integrity
        # _spec_exp_validator(spec_exp)

        # SpecExp
        # SpecExp._groups: [0 group]
        # SpecExp._images: [0 id, 1 group, 2 image_name, 3 image_use_type, 4 image_path]
        # SpecExp._rois: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
        self._spec_exp: SpecExp = spec_exp

        # Sample target values
        # [0 fixed sample id, 1 user assinged labels, 2 Target values]
        # self.__sample_targets: list[tuple[str, str, Union[str, bool, int, float], str]] = spec_exp.sample_targets

        # Is_target_numeric according to sample target values
        # self.__is_target_numeric: bool = self._check_target_numeric(spec_exp)

        # Report directory
        self._report_directory: str = self.spec_exp._report_directory

        # Processes
        # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
        self._process: list[tuple[str, str, str, str, int, Callable, int, int]] = []

        # Generated process chain for full factors
        # [(process 1 ID of step 1, process 2 ID of step 1,...), (process 1 ID of step 2, process 2 ID of step 2,...), ...]  # noqa: E501
        self._process_steps: list = []
        # [(process 1 ID of step 1, process 1 ID of step 2,...), (process 2 ID of step 1, process 1 ID of step 2,...), ...]  # noqa: E501
        self._process_chains: list = []
        # Custom chains for custom partly test
        self._custom_chains: list = []

        # Band wavelengths of the pixel spectra of the images (_pretest_data['spec1d']) - No use
        # self.__band_wavelength: Optional[tuple[Union[int, float], ...]] = None

        # Pre-execution test data
        # Sample data format - ROI: {ID, label, target, img_path, roi_coords}
        # Sample data format - standalone spec: {ID, label, target, spec1d: tuple}
        # Sample data format - test: {img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d}
        # self.__pretest_data: Optional[dict[str, Any]] = None
        self._pretest_data_init()

        # Pipeline creating time
        self._create_time: str = datetime.now().strftime("created_at_%Y-%m-%d_%H-%M-%S")

        # Input sample data
        # self.__sample_data: list[dict[str, Any]] = []

        # test_run status
        # Set False for any process modification, set True after successful calling of 'test_run'
        # self.__tested: bool = False

        # Sample preprocessing result paths
        # self.__preprocess_result_path: list[str] = []

    ## Mutable properties
    @property
    def report_directory(self) -> str:
        return self._report_directory

    @report_directory.setter
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
                    "Internal Error: 'SpecPipe._pretest_data' is None. \
                        Pre-execution test data initialization fails. Please report."
                )
            if len(value) != len(self.__pretest_data["spec1d"]):
                raise ValueError(
                    f"The number of band wavelengths ({len(value)}) does not match \
                        the number of bands ({len(self.__pretest_data['spec1d'])})."
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
    def process(self) -> list[tuple[str, str, str, str, int, Callable, int, int]]:
        return self._process

    @process.setter
    def process(self, value: list[tuple[str, str, str, str, int, Callable, int, int]]) -> None:
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
            self.__sample_targets = spec_exp.sample_targets
            self.__is_target_numeric = self._check_target_numeric(spec_exp)
            self._report_directory = spec_exp._report_directory
            self._pretest_data_init()
            n_chains = len(self.process_chains)
            if n_chains > 0:
                self.test_run(dump_result=False, model_test_coverage=(min(100, n_chains) / n_chains))
        except Exception as e:
            # Roll back when fail in test
            self._spec_exp = spec_exp_old
            self.__sample_targets = spec_exp_old.sample_targets
            self.__is_target_numeric = self._check_target_numeric(spec_exp_old)
            self._report_directory = spec_exp_old._report_directory
            self._pretest_data_init()
            raise ValueError("Given SpecExp failed in test_run, spec_exp configuration rolls back.") from e

    # Process configuration ===================================

    # Generate test data of each level for process validation
    # From data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    # Pretest_data: [ID, label, target, validation_group, test, train, img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d]  # noqa: E501
    @simple_type_validator
    def _pretest_data_init(self) -> None:
        """
        Initialize test data of each level for process validation before method execution.
        """
        ### For image samples
        if len(self.spec_exp.standalone_specs_sample) == 0:
            # Get first sample ROI item
            sroi_test = self._spec_exp._rois_sample[0]
            roicoords = sroi_test[5]

            # Get associated image item - raster image path

            img_test0 = self._spec_exp._get_images([sroi_test[2]], sroi_test[1], None)[0]
            if len(img_test0) == 1:
                img_test = img_test0[0]
                img_path = img_test[4]
            else:
                raise ValueError(f"\nSpecExp data error: duplicated image items: \n{img_test0}")

            # Get ROI spec table
            try:
                bdmin = minbbox(img_path, roicoords, 100)
                roitable = roispec(img_path, bdmin, as_type=float)
            except Exception as e:
                raise ValueError(
                    f"\nUnable to retrieve ROI spectra from image:\n\nROI:\n{sroi_test}\n\nImage:\n{img_test}\n\n"
                ) from e

            # Get test image ROI
            test_roi_coords = [
                [
                    (0, 0),
                    (max(int(bdmin[0][1][0] - bdmin[0][0][0]), 1), 0),
                    (
                        max(int(bdmin[0][1][0] - bdmin[0][0][0]), 1),
                        max(int(bdmin[0][2][1] - bdmin[0][1][1]), 1),
                    ),
                    (0, max(int(bdmin[0][2][1] - bdmin[0][1][1]), 1)),
                    (0, 0),
                ]
            ]

            # Single spec
            spec1d = tuple(roitable[0, :])

            ## Save to file
            # Output path
            sdir = self._spec_exp.report_directory + "Pre_execution_test_data/"
            os.makedirs(unc_path(sdir), exist_ok=True)

            # Save test image
            test_img_path = sdir + "test_images." + img_path.split(".")[-1]
            with open(unc_path(sdir + "pre_execution_data.json"), "w") as f:
                # Save test image
                croproi(raster_path=img_path, roi_coordinates=bdmin, output_path=test_img_path)
                # Convert np.float32 to native float for json dump
                # Save test spectra
                td1 = {
                    "img_path": img_path,
                    "test_img_path": test_img_path,
                    "roi_coords": bdmin,
                    "test_roi_coords": test_roi_coords,
                    "spec1d": spec1d,
                }
                # Dump
                json.dump(td1, f)

            # Testing spectrum table
            td2 = pd.DataFrame(roitable, columns=[("Band_" + str(i + 1)) for i in range(roitable.shape[1])])
            td2.to_csv(unc_path(sdir + "Pre_execution_data_roi_specs.csv"), index=False)

            # Output for pre-execution testing data
            test_data_pre = {
                "ID": "test_run",
                "label": "test_run",
                "target": None,
                "validation_group": "test_run",
                "test": np.int8(1),
                "train": np.int8(1),
                "img_path": img_path,
                "test_img_path": test_img_path,
                "roi_coords": bdmin,
                "test_roi_coords": test_roi_coords,
                "roi_specs": roitable,
                "spec1d": spec1d,
            }

            self.__pretest_data = test_data_pre

        ### For standalone spectral samples
        # Pretest_data: [ID, label, target, validation_group, img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d]  # noqa: E501
        else:
            # Config test item data
            img_path = None
            test_img_path = None
            bdmin = None
            test_roi_coords = None
            roitable = None
            spec1d = tuple(self.spec_exp.standalone_specs_sample[0][-1])

            ## Save to file
            # Output path
            sdir = self._spec_exp.report_directory + "Pre_execution_test_data/"
            os.makedirs(unc_path(sdir), exist_ok=True)

            # Testing spectrum table
            td2 = pd.DataFrame([list(spec1d)], columns=[("Band_" + str(i + 1)) for i in range(len(spec1d))])
            td2.to_csv(unc_path(sdir + "Pre_execution_data_standalone_specs.csv"), index=False)

            # Output for pre-execution testing data
            test_data_pre = {
                "ID": "test_run",
                "label": "test_run",
                "target": None,
                "validation_group": "test_run",
                "test": np.int8(1),
                "train": np.int8(1),
                "img_path": img_path,
                "test_img_path": test_img_path,
                "roi_coords": bdmin,
                "test_roi_coords": test_roi_coords,
                "roi_specs": roitable,
                "spec1d": spec1d,
            }
            self.__pretest_data = test_data_pre

    # Process function validator - pretest validator
    # Data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    # Pretest_data: [ID, label, target, validation_group, img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d]  # noqa: E501
    @simple_type_validator
    def _process_validator(  # noqa: C901
        self,
        method: Callable,
        input_data_level: Union[str, int],
        output_data_level: Union[str, int],
    ) -> Callable:
        """
        Validate process method of specified input data level before execution of entire processing chain.
        """
        # Pretest_data validation for static typing
        if self.__pretest_data is None:
            raise ValueError(
                "Internal Error: 'SpecPipe._pretest_data' is None. \
                    Pre-execution test data initialization fails. Please report."
            )

        # Applied only for image samples
        if len(self.spec_exp.standalone_specs_sample) == 0:
            # Validate data_level
            dl_in = _dl_val(input_data_level)[0]
            dl_out = _dl_val(output_data_level)[0]

            # Test image path
            test_img_path = self.__pretest_data["test_img_path"]

            # Test data
            if dl_out < 8:
                if dl_in == 0:
                    # Output dst image path
                    img_path = os.path.splitext(str(test_img_path).replace("\\", "/").replace("//", "/"))
                    img_name = img_path[0].split("/")[-1]
                    output_path = (
                        self.report_directory
                        + "/Pre_execution_test_data/"
                        + img_name
                        + "_processed_by_"
                        + method.__name__
                        + img_path[1]
                    )
                    # Process image and return path of processed image
                    result = method(test_img_path, output_path)
                elif dl_in == 1:
                    # Output dst image path
                    img_path = os.path.splitext(str(test_img_path).replace("\\", "/").replace("//", "/"))
                    img_name = img_path[0].split("/")[-1]
                    output_path = (
                        self.report_directory
                        + "/Pre_execution_test_data/"
                        + img_name
                        + "_px_app_"
                        + method.__name__
                        + img_path[1]
                    )
                    # Process image and return path of processed image
                    result = pixel_apply(test_img_path, method, "spec", output_path, progress=False)
                elif dl_in == 2:
                    # Output dst image path
                    img_path = os.path.splitext(str(test_img_path).replace("\\", "/").replace("//", "/"))
                    img_name = img_path[0].split("/")[-1]
                    output_path = (
                        self.report_directory
                        + "/Pre_execution_test_data/"
                        + img_name
                        + "_px_app_"
                        + method.__name__
                        + img_path[1]
                    )
                    # Process image and return path of processed image
                    result = pixel_apply(test_img_path, method, "array", output_path, progress=False)
                elif dl_in == 3:
                    # Output dst image path
                    img_path = os.path.splitext(str(test_img_path).replace("\\", "/").replace("//", "/"))
                    img_name = img_path[0].split("/")[-1]
                    output_path = (
                        self.report_directory
                        + "/Pre_execution_test_data/"
                        + img_name
                        + "_px_app_"
                        + method.__name__
                        + img_path[1]
                    )
                    # Process image and return path of processed image
                    result = pixel_apply(test_img_path, method, "tensor", output_path, progress=False)
                elif dl_in == 4:
                    # Output dst image path
                    img_path = os.path.splitext(str(test_img_path).replace("\\", "/").replace("//", "/"))
                    img_name = img_path[0].split("/")[-1]
                    output_path = (
                        self.report_directory
                        + "/Pre_execution_test_data/"
                        + img_name
                        + "_px_app_"
                        + method.__name__
                        + img_path[1]
                    )
                    # Process image and return path of processed image
                    result = pixel_apply(test_img_path, method, "tensor_hyper", output_path, progress=False)
                elif dl_in == 5:
                    result = method(test_img_path, self.__pretest_data["roi_coords"])
                elif dl_in == 6:
                    testing_data = self.__pretest_data["roi_specs"]
                    result = method(testing_data)
                elif dl_in == 7:
                    testing_data = self.__pretest_data["spec1d"]
                    result = method(testing_data)
                else:
                    raise ValueError("Input data level cannot be 'model' or 8 (corresponding index).")
            else:
                # Model method is not validated here
                return method

            # Output validation
            if result is None:
                raise ValueError(
                    f"Method '{method.__name__}' returns no data. \
                        The added method must have a return. \
                            For image processing methods, absolute path of resulting image must be returned."
                )

            # For raster image path and image file output
            if dl_out <= 4:
                # Raster file validation
                if os.path.exists(unc_path(result)):
                    # Open raster validation
                    try:
                        with rasterio.open(result) as src:
                            # Raster validation
                            if src is None:
                                raise ValueError("Invalid raster: raster is None.")
                            elif (src.width == 0) or (src.height == 0) or (src.count == 0):
                                raise ValueError(
                                    f"Invalid raster, \
                                        got dimensions: {src.width} x {src.height}, got number of bands: {src.count}."
                                )
                            else:
                                # Raster value validation
                                all_no_data = True
                                sample = src.read(min(int(src.count / 2), 1))
                                if np.all(sample == src.nodata) or np.all(np.isnan(sample)):
                                    sample = src.read(
                                        window=(
                                            (
                                                max(int(src.height / 2) - 1, 0),
                                                min(int(src.height / 2) + 1, src.height),
                                            ),
                                            (
                                                max(int(src.width / 2) - 1, 0),
                                                min(int(src.width / 2) + 1, src.width),
                                            ),
                                        )
                                    )
                                    if np.all(sample == src.nodata) or np.all(np.isnan(sample)):
                                        for i in range(0, src.height, 32):
                                            for j in range(0, src.width, 32):
                                                # Define window for current tile
                                                win = Window(
                                                    col_off=j,
                                                    row_off=i,
                                                    width=min(32, src.width - j),
                                                    height=min(32, src.height - i),
                                                )
                                                # Read all bands for current tile (shape: [bands, rows, cols])
                                                sample = src.read(window=win)
                                                if not (np.all(sample == src.nodata) or np.all(np.isnan(sample))):
                                                    all_no_data = False
                                                    break
                                    else:
                                        all_no_data = False
                                else:
                                    all_no_data = False
                                if all_no_data:
                                    raise ValueError("All raster values are NoData")
                    except Exception as e:
                        raise ValueError(
                            f"Failed to open resulting raster image of {method.__name__}.\
                                \nGot path:\n{result}"
                        ) from e
                else:
                    raise ValueError(f"Resulting file path is invalid: {result}")

            # For array-like output
            if (dl_out >= 6) & (dl_out <= 7):
                result = arraylike_validator()(result)
                if type(result) is np.ndarray:
                    if np.issubdtype(result.dtype, np.number):
                        if (dl_out == 6) and (result.ndim != 2):
                            raise ValueError(
                                f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                                    must return an 2D array, got array dimension: {result.ndim}"
                            )
                        else:
                            result = np.array(result)
                            if (dl_out == 7) and (result.ndim != 1):
                                raise ValueError(
                                    f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                                        must return an 1D array-like, got array dimension: {result.ndim}"
                                )
                    else:
                        raise ValueError(
                            f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                                must return an array of numbers, got array dtype: {result.dtype}"
                        )
                else:
                    raise TypeError(
                        f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                            must return an NumPy array-like, got: {type(result)}"
                    )

            return method

        else:
            # Validate data_level
            dl_in = _dl_val(input_data_level)[0]
            dl_out = _dl_val(output_data_level)[0]
            if dl_in != 7:
                raise ValueError(
                    f"Method for one-dimensional standalone spectra must have input data level of 7 ('spec1d'), \
                        but got: {input_data_level}"
                )
            if dl_out < 7:
                raise ValueError(
                    f"Method for one-dimensional standalone spectra cannot have output data level below 7 ('spec1d'), \
                        but got level number: {dl_out}"
                )
            if dl_out == 8:
                # Model method is not validated here
                return method

            testing_data = self.__pretest_data["spec1d"]
            result = method(testing_data)

            # Output validation
            if result is None:
                raise ValueError(
                    f"Method '{method.__name__}' returns no data. The added method must have a return. \
                        For image processing methods, absolute path of resulting image must be returned."
                )

            # For array-like output
            result = arraylike_validator()(result)
            if type(result) is np.ndarray:
                if np.issubdtype(result.dtype, np.number):
                    result = np.array(result)
                    if (dl_out == 7) and (result.ndim != 1):
                        raise ValueError(
                            f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                                must return an 1D array-like, got array dimension: {result.ndim}"
                        )
                else:
                    raise ValueError(
                        f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                            must return an array of numbers, got array dtype: {result.dtype}"
                    )
            else:
                raise TypeError(
                    f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' \
                        must return an NumPy array-like, got: {type(result)}"
                )

            return method

    # Add model - from add_process
    @simple_type_validator
    def add_model(
        self,
        model_method: Union[object, list[object], tuple[object]],
        model_label: Union[str, list[str], tuple[str]] = "",
        # Process parameters
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
    ) -> None:
        """
        Add a model evaluation process to the processing pipeline.

        The added model operates on 1D data (data level ``7`` / ``"spec1d"``) and producing model-level output (data level ``8`` / ``"model"``).
        All models share a unified application sequence within the pipeline.

        Parameters
        ----------
        model_method : object
            Sklearn-style model object.

            Regression models must implement ``fit`` and ``predict``.
            Classification models must additionally implement ``predict_proba``.

        model_label : str, optional
            Custom label for the added model.

            If empty string, a label is automatically generated.

        test_error_raise : bool, optional
            Whether to raise error when the model fails in validation using simplified mock data before added to the pipeline.

            If True, an exception is raised, otherwise only a warning is issued. Default is True.

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

        See Also
        --------
        add_process

        Examples
        --------
        Create a ``SpecPipe`` instance from an existing SpecExp object::

            >>> pipe = SpecPipe(exp)

        Add a model with a specified validation method::

            >>> from sklearn.neighbors import KNeighborsClassifier
            >>> knn = KNeighborsClassifier(n_neighbors=3)
            >>> pipe.add_model(knn, validation_method="5-fold")

        Use different validation strategies::

            >>> pipe.add_model(knn, validation_method="60-40-split")
            >>> pipe.add_model(knn, validation_method="loo")
        """  # noqa: E501

        # Process general parameters
        input_data_level = 7
        output_data_level = 8
        # Application sequence of non-model processes with input data level spec1d / 7
        preprocess_dl7 = self.ls_process(input_data_level=7, return_result=True, print_result=False)
        app_seqs = [proci[4] for proci in np.array(preprocess_dl7[preprocess_dl7["Output_data_level"] != "model"])]
        if len(app_seqs) > 0:
            application_sequence = max(app_seqs) + 1
        else:
            application_sequence = 0
        process_label = model_label
        method = model_method

        # Add model processes
        self.add_process(
            input_data_level=input_data_level,
            output_data_level=output_data_level,
            application_sequence=application_sequence,
            method=method,
            process_label=process_label,
            test_error_raise=test_error_raise,
            # Modeling parameters
            is_regression=is_regression,
            validation_method=validation_method,
            unseen_threshold=unseen_threshold,
            x_shape=x_shape,
            result_backup=result_backup,
            # Model evaluation parameters
            data_split_config=data_split_config,
            validation_config=validation_config,
            metrics_config=metrics_config,
            roc_plot_config=roc_plot_config,
            scatter_plot_config=scatter_plot_config,
            residual_config=residual_config,
            residual_plot_config=residual_plot_config,
            influence_analysis_config=influence_analysis_config,
        )

    # List model - from ls_process
    @overload
    def ls_model(
        self,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
        model_method: Union[str, object, None] = None,
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
        model_method: Union[str, object, None] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: Literal[True] = True,
    ) -> pd.DataFrame: ...

    # List model - from ls_process
    @simple_type_validator
    def ls_model(
        self,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
        model_method: Union[str, object, None] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        List added model evaluation processes based on filtering conditions.

        If a filter criterion is not provided or None, the corresponding filter is not applied.

        Parameters
        ----------
        model_id : str, optional
            Model evaluation process ID. Default is None.

        model_label : str, optional
            Custom model label. Default is None.

        model_method : str or object, optional
            Model object or method. Default is None.

        exact_match : bool, optional
            If False, any process with a property value containing the specified value is included. Default is True.

        print_result : bool, optional
            If True, simplified results are printed. Default is True.

        return_result : bool, optional
            If True, a complete resulting DataFrame is returned. Default is False.

        Returns
        -------
        pandas.DataFrame or None
            If ``return_result=True``, returns a pandas DataFrame of matched model evaluation processes.

            If ``return_result=False``, returns None.

        Examples
        --------
        For a prepared ``SpecExp`` instance ``exp``::

            >>> pipe = SpecPipe(exp)
            >>> from sklearn.neighbors import KNeighborsClassifier
            >>> knn = KNeighborsClassifier(n_neighbors=3)
            >>> pipe.add_model(knn, validation_method='2-fold')

        List all models::

            >>> pipe.ls_model()

        Return model items as a DataFrame::

            >>> df = pipe.ls_model(return_result=True)

        Filter results by model label::

            >>> pipe.ls_model(model_label='KNeighbor', exact_match=False)
        """

        # List models
        if return_result:
            return self.ls_process(
                process_id=model_id,
                process_label=model_label,
                input_data_level="spec1d",
                output_data_level="model",
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
                input_data_level="spec1d",
                output_data_level="model",
                application_sequence=None,
                method=model_method,
                full_application_sequence=None,
                exact_match=exact_match,
                print_result=print_result,
                return_result=False,
            )
            return None

    # Remove model - from rm_process
    @simple_type_validator
    def rm_model(
        self,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
        model_method: Union[str, object, None] = None,
        exact_match: bool = True,
    ) -> None:
        """
        Remove added model evaluation processes from this SpecPipe instance based on filtering conditions.

        If a filter criterion is not provided or None, the corresponding filter is not applied.

        Parameters
        ----------
        model_id : str, optional
            Model evaluation process ID. Default is None.

        model_label : str, optional
            Custom model label. Default is None.

        model_method : str or object, optional
            Method object or model. Default is None.

        exact_match : bool, optional
            If False, any process with a property value containing the specified value is removed. Default is True.

        Examples
        --------
        For a prepared ``SpecExp`` instance ``exp``::

            >>> pipe = SpecPipe(exp)
            >>> from sklearn.neighbors import KNeighborsClassifier
            >>> knn = KNeighborsClassifier(n_neighbors=3)
            >>> pipe.add_model(knn, validation_method='2-fold')

        Remove all models::

            >>> pipe.rm_model()

        Remove a specific model::

            >>> pipe.rm_model(model_label='KNeighbor')
        """

        self.rm_process(
            process_id=model_id,
            process_label=model_label,
            input_data_level=7,
            output_data_level=8,
            application_sequence=None,
            method=model_method,
            exact_match=exact_match,
        )

    # Add processes
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    @simple_type_validator
    def add_process(  # noqa: C901
        self,
        # Process general parameters
        input_data_level: Union[str, int],
        output_data_level: Union[str, int],
        application_sequence: int,
        method: Union[Callable, object, list[Callable], tuple[Callable], list[object], tuple[object]],
        process_label: Union[str, list[str], tuple[str]] = "",
        *,
        test_error_raise: bool = True,
        # Modeling parameters
        is_regression: Optional[bool] = None,
        validation_method: str = "2-fold",
        unseen_threshold: Optional[float] = 0.0,
        x_shape: Optional[tuple[int]] = None,
        result_backup: bool = False,
        # Model evaluation parameters
        data_split_config: Union[str, dict[str, Any]] = "default",
        validation_config: Union[str, dict[str, Any]] = "default",
        metrics_config: Union[str, dict[str, Any], None] = "default",
        roc_plot_config: Union[str, dict, None] = "default",
        scatter_plot_config: Union[str, dict[str, Any], None] = "default",
        residual_config: Union[str, dict[str, Any], None] = "default",
        residual_plot_config: Union[str, dict[str, Any], None] = "default",
        influence_analysis_config: Union[str, dict[str, Any], None] = "default",
    ) -> None:
        """
        Add a processing method with defined input/output data levels and application sequence to the pipeline.
        A processing method can be a preprocessing function or a model for evaluation.

        Parameters
        ----------
        input_data_level : int or str
            Input data level for the process. Available options:

                ``0`` or ``"image"``
                    If the process callable is applied to raster images.
                    The corresponding callable must accept the input raster path as the first argument and the output path as the second argument.
                ``1`` or ``"pixel_spec"``
                    If the process callable is applied to 1D spectra of each pixel.
                ``2`` or ``"pixel_specs_array"``
                    If the process callable is applied to 2D ``numpy.ndarray`` of pixel spectra. Each row is a pixel spectrum.
                ``3`` or ``"pixel_specs_tensor"``
                    If the process callable is applied to 3D ``torch.tensor`` (Shape: ``(C, H, W)``), with computation performed along axis 0.
                ``4`` or ``"pixel_hyperspecs_tensor"``
                    If the process callable is applied to 3D hyperspectral ``torch.tensor`` (Shape: ``(C, H, W)``), with computation performed along axis 1.
                ``5`` or ``"image_roi"``
                    If the process callable is applied to a region of interest (ROI) within a raster image.
                    The callable must receive the raster path and ROI coordinates provided by provided ``SpecExp`` instance.
                ``6`` or ``"roi_specs"``
                    If the process callable is applied to 2D ``numpy.ndarray`` of ROI spectra, each row is a pixel.
                ``7`` or ``"spec1d"``
                    If the process callable is applied to 1D array-like sample spectra or flattened data, such as ROI spectral statistics.

            Note:

                Input data levels ``0`` through ``4`` share a single, common ``application_sequence`` scheme.
                These data levels do not maintain independent application sequence series, in contrast to input data levels ``5``, ``6``, and ``7``.

                For example, a process defined with input data level ``0`` (``"image"``) and ``application_sequence=0`` and a process defined with input data
                level ``2`` (``"pixel_specs_array"``) and ``application_sequence=0`` are treated as parallel operations within the same image-processing step.

        output_data_level : int or str
            Output data level. Available options:

                ``0`` or ``"image"``
                    Returns raster image path.
                ``1`` or ``"pixel_spec"``
                    Same as input "pixel_spec".
                ``2`` or ``"pixel_specs_array"``
                    Same as input "pixel_specs_array".
                ``3`` or ``"pixel_specs_tensor"``
                    Same as input "pixel_specs_tensor".
                ``4`` or ``"pixel_hyperspecs_tensor"``
                    Same as input "pixel_hyperspecs_tensor".
                ``5`` or ``"image_roi"``
                    Currently unavailable.
                ``6`` or ``"roi_specs"``
                    Returns 2D ``numpy.ndarray`` of ROI spectra.
                ``7`` or ``"spec1d"``
                    Returns 1D array-like spectral data.
                ``8`` or ``"model"``
                    Used for modeling, accepts "spec1d" input.

        application_sequence : int
            Sequence number of the method within the same input data level. Lower numbers execute first.

        method : callable or object or list of (callable or object)
            Processing method, a processing function or sklearn-style estimator.

            The callable or estimator must accept inputs and produce outputs that conform to the configured data levels.

            If an estimator is provided, it must follow the sklearn estimator interface, implementing `fit` and `predict`, and `predict_proba` for classifiers.

            A list of methods may be provided to specify multiple methods that share the same configuration.

        process_label : str or list of str, optional
            Custom label(s) for the process.

            If provided:

                - If a single method is provided, must be a single str of label.
                - If multiple methods are provided, must be a list of str with a length equal to the number of methods.

            Default is an empty string, which automatically generates label(s) using the Callable name(s) or the estimator class name(s).

        test_error_raise : bool, optional
            Whether to raise error when the process fails in validation using simplified mock data before added to the pipeline.

            If True, an exception is raised, otherwise only a warning is issued. Default is True.

        is_regression : bool, optional
            Whether the model is a regression model.
            See ``add_model`` for details.

        validation_method : str, optional
            Validation strategy for model evaluation.
            See ``add_model`` for details.

        unseen_threshold : float, optional
            Classification-only parameter.
            See ``add_model`` for details.

        x_shape : tuple of int, optional
            Expected shape of independent variables for models requiring structured input. Currently ignored.
            See ``add_model`` for details.

        result_backup : bool, optional
            Whether to save timestamped backup copies of result files.
            See ``add_model`` for details.

        data_split_config : str or dict, optional
            Additional data splitting configuration.
            See ``add_model`` for details.

        validation_config : str or dict, optional
            Validation behavior configuration.
            See ``add_model`` for details.

        metrics_config : str or dict or None, optional
            Metrics computation configuration.
            See ``add_model`` for details.

        roc_plot_config : str or dict or None, optional
            Receiver Operating Characteristic (ROC) plotting configuration for classification models.
            See ``add_model`` for details.

        scatter_plot_config : str or dict or None, optional
            Scatter plot configuration for regression models.
            See ``add_model`` for details.

        residual_config : str or dict or None, optional
            Residual analysis configuration.
            See ``add_model`` for details.

        residual_plot_config : str or dict or None, optional
            Residual plot configuration for regression models.
            See ``add_model`` for details.

        influence_analysis_config : str or dict or None, optional
            Influence analysis configuration.
            See ``add_model`` for details.

        See Also
        --------
        add_model
        make_img_func
        make_roi_func
        make_array_func
        pixel_apply

        Examples
        --------
        For prepared ``SpecExp`` instance ``exp``::

            >>> pipe = SpecPipe(exp)

        Add an image processor accepting image path and returning processed path::

            >>> pipe.add_process('image', 'image', 0, img_processor)

        Or using numeric level indices::

            >>> pipe.add_process(0, 0, 0, img_processor)

        Customize method name::

            >>> pipe.add_process(0, 0, 0, img_processor, process_label='img_proc')

        Apply function to pixel spectra array::

            >>> from swectral.functions import snv
            >>> pipe.add_process('pixel_specs_array', 'pixel_specs_array', 0, snv)

        GPU processing example::

            >>> from swectral.functions import snv_hyper
            >>> pipe.add_process(4, 4, 0, snv_hyper)

        Denoiser on ROI spectra::

            >>> from swectral.denoiser import LocalPolynomial
            >>> pipe.add_process(6, 6, 0, LocalPolynomial(5, polynomial_order=2).savitzky_golay_filter)

        Process 1D sample spectra::

            >>> pipe.add_process(7, 7, 0, LocalPolynomial(5, polynomial_order=2).savitzky_golay_filter)
        """  # noqa: E501

        # Validate processes and process_label
        if isinstance(method, (list, tuple)):
            methods = method
            if isinstance(process_label, (list, tuple)):
                if len(process_label) == len(methods):
                    process_labels = process_label
                else:
                    raise ValueError(
                        f"The number of provided process labels ({len(process_label)}) do not match "
                        f"the number of process ({len(methods)})."
                    )
            elif process_label == "":
                process_labels = [""] * len(methods)
            elif len(methods) == 1:
                process_labels = [process_label]
            else:
                raise TypeError(
                    "If multiple methods are provided,"
                    "process_label must be a list of str with a length equal to the number of methods, "
                    f"but got: {process_label}, type: {type(process_label)}"
                )
        elif (hasattr(method, "fit") and hasattr(method, "predict")) or callable(method):
            methods = [method]
            if not isinstance(process_label, str):
                if len(process_label) != 1:
                    raise ValueError(
                        f"'process_label' must be str or list with one str for single method, got: {process_label}"
                    )
                else:
                    process_labels = process_label
            else:
                process_labels = [process_label]
        else:
            raise TypeError(f"'method' must be a model or Callable or list of them, got: {type(method)}")

        # Add processes
        for method_i, process_label_i in zip(methods, process_labels):
            if not ((hasattr(method_i, "fit") and hasattr(method_i, "predict")) or callable(method_i)):
                raise TypeError(f"'method' must be a model or Callable or list of them, got: {type(method_i)}")
            self._add_process(
                input_data_level=input_data_level,
                output_data_level=output_data_level,
                application_sequence=application_sequence,
                method=method_i,  # method
                process_label=process_label_i,  # process label
                test_error_raise=test_error_raise,
                is_regression=is_regression,
                validation_method=validation_method,
                model_label=process_label_i,  # model_label
                unseen_threshold=unseen_threshold,
                x_shape=x_shape,
                result_backup=result_backup,
                data_split_config=data_split_config,
                validation_config=validation_config,
                metrics_config=metrics_config,
                roc_plot_config=roc_plot_config,
                scatter_plot_config=scatter_plot_config,
                residual_config=residual_config,
                residual_plot_config=residual_plot_config,
                influence_analysis_config=influence_analysis_config,
            )

    # Add process
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    @simple_type_validator
    def _add_process(  # noqa: C901
        self,
        # Process general parameters
        input_data_level: Union[str, int],
        output_data_level: Union[str, int],
        application_sequence: int,
        method: Union[Callable, object],
        process_label: str = "",
        *,
        test_error_raise: bool = True,
        # Modeling parameters
        is_regression: Optional[bool] = None,
        validation_method: str = "2-fold",
        model_label: str = "",
        unseen_threshold: Optional[float] = 0.0,
        x_shape: Optional[tuple[int]] = None,
        result_backup: bool = False,
        # Model evaluation parameters
        data_split_config: Union[str, dict[str, Any]] = "default",
        validation_config: Union[str, dict[str, Any]] = "default",
        metrics_config: Union[str, dict[str, Any], None] = "default",
        roc_plot_config: Union[str, dict, None] = "default",
        scatter_plot_config: Union[str, dict[str, Any], None] = "default",
        residual_config: Union[str, dict[str, Any], None] = "default",
        residual_plot_config: Union[str, dict[str, Any], None] = "default",
        influence_analysis_config: Union[str, dict[str, Any], None] = "default",
    ) -> None:
        """
        Add a single processing method with defined input/output data levels and application sequence to the pipeline.
        """
        # Validate Data_level
        dl_in = _dl_val(input_data_level)
        dl_in_name = dl_in[1]
        dl_in_ind = dl_in[0]

        dl_out = _dl_val(output_data_level)
        dl_out_name = dl_out[1]
        dl_out_ind = dl_out[0]

        # Full application sequence
        fapp_seq = 2000000 * dl_in_ind * (dl_in_ind > 4) + 1000000 * int(dl_in_ind != dl_out_ind) + application_sequence

        # Validate data levels and data level sequence
        _data_level_seq_validator(
            input_data_level=input_data_level,
            output_data_level=output_data_level,
            application_sequence=application_sequence,
            full_application_sequence=fapp_seq,
            existed_process=self.process,
        )

        # Generate process number
        existed_proc_num = [0]
        if len(self.process) > 0:
            for pr in self.process:
                # Get existed process number (repeat number) for dl>4 sharing same dl_in, dl_out and app_seq
                if dl_in_ind > 4:
                    if (pr[2] == dl_in_name) & (pr[3] == dl_out_name) & (pr[4] == application_sequence):
                        existed_proc_num.append(pr[7])
                # Get existed process number (repeat number) for dl 0~4 sharing same app_seq
                else:
                    if (_dl_val(pr[2])[0] <= 4) & (pr[4] == application_sequence):
                        existed_proc_num.append(pr[7])
        proc_num_new = max(existed_proc_num) + 1

        # Auto bump application sequence if fapp_seq higher than others of same dl_in
        existed_appseq = [0]
        existed_fappseq = [0]
        appseq_for_id = -1
        if len(self.process) > 0:
            for pr in self.process:
                # If identical positioned process existed
                if (pr[2] == dl_in_name) & (pr[3] == dl_out_name) & (pr[4] == application_sequence):
                    appseq_for_id = int(str(pr[0]).split("_")[1])
                if pr[2] == dl_in_name:
                    existed_appseq.append(pr[4])
                    existed_fappseq.append(pr[6])
        if (fapp_seq > max(existed_fappseq)) & (application_sequence < max(existed_appseq)):
            appseq_for_id = max(existed_appseq) + 1
        elif appseq_for_id == -1:
            appseq_for_id = application_sequence

        # Build process ID
        proc_id = str(dl_in_ind) + "_" + str(appseq_for_id) + "_%#" + str(proc_num_new)

        # Add preprocess - validate preprocess method
        if dl_out_ind < 8:
            try:
                proc_method = _process_validator(
                    method=method,
                    input_data_level=input_data_level,
                    output_data_level=output_data_level,
                    pretest_data=self._pretest_data,
                    standalone_specs_sample=self.spec_exp.standalone_specs_sample,
                    report_directory=self.report_directory,
                )
            except Exception as e:
                if test_error_raise:
                    raise ValueError("Method testing fails") from e
                else:
                    warnings.warn(f"Method fails on '{input_data_level}' testing data", UserWarning, stacklevel=2)

        # Add model - model method constructor
        else:
            # Get report dir
            report_dir = self.report_directory

            # Validate model labels
            existed_model_labels = [proc[1] for proc in self.process if proc[3] == 'model' and proc[1] != ''] + [
                proc[5].model_label for proc in self.process if proc[3] == 'model' and hasattr(proc[5], 'model_label')
            ]
            if model_label == "":
                # Validate for duplication
                model_name = method.__class__.__name__
                name_k = 0
                while model_name in existed_model_labels:
                    name_k = name_k + 1
                    model_name = method.__class__.__name__ + f"_{name_k}"
                model_label = model_name
            else:
                # Validate for duplication
                name_k = 0
                while model_label in existed_model_labels:
                    name_k = name_k + 1
                    model_label = method.__class__.__name__ + f"_{name_k}"
                process_label = model_label

            # Validate is_regression
            if is_regression is None:
                try:
                    self._sample_targets[0][2] + 1  # type: ignore[operator]
                    # Behavior-based type check after serialization, safe approach for serialization
                    is_regression = True
                except Exception:
                    is_regression = False

            # Validate model instance
            if is_regression:
                _regressor_validator(method)
            else:
                _classifier_validator(method)

            # Generate process method for model evaluation
            proc_method = _ModelMethod(
                model_label=model_label,
                is_regression=is_regression,
                x_shape=x_shape,
                report_dir=report_dir,
                method=method,
                result_backup=result_backup,
                validation_method=validation_method,
                unseen_threshold=unseen_threshold,
                data_split_config=data_split_config,
                validation_config=validation_config,
                metrics_config=metrics_config,
                roc_plot_config=roc_plot_config,
                scatter_plot_config=roc_plot_config,
                residual_config=residual_config,
                residual_plot_config=residual_plot_config,
                influence_analysis_config=influence_analysis_config,
            )

        # Change process test status
        self.__tested = False

        # Add process item
        proc_item = (
            proc_id,
            process_label,
            dl_in_name,
            dl_out_name,
            application_sequence,
            proc_method,
            fapp_seq,
            proc_num_new,
        )
        self._process.append(proc_item)
        # Sort processes and update process_steps and process_chains
        self._generate_process_steps()
        self._generate_process_chains()

    # Get matched and unmatched process items
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    # Data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    @simple_type_validator
    def _get_process(  # noqa: C901
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Union[int, tuple[int, int], None] = None,
        method: Union[str, Callable, object, None] = None,
        full_application_sequence: Union[int, tuple[int, int], None] = None,
        exact_match: bool = True,
    ) -> tuple[
        list[tuple[str, str, str, str, int, Callable, int, int]],
        list[tuple[str, str, str, str, int, Callable, int, int]],
    ]:
        """
        Get matched and unmatched process items by process properties.
        """
        # Validate processes
        if len(self._process) == 0:
            print("No process added!")
            return ([], [])

        # Properties
        if input_data_level is not None:
            input_data_level = _dl_val(input_data_level)[1]
        if output_data_level is not None:
            output_data_level = _dl_val(output_data_level)[1]
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
            for i, prop in enumerate(pit[0:-1]):
                # Condition - not applied
                if prps[i] is None:
                    cond = True
                # Condition - Application_sequence and Full_application_sequence
                # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
                elif ((i == 4) or (i == 6)) and (type(prps[i]) is tuple):
                    if prps[i][0] >= prps[i][1]:
                        raise ValueError(f"Invalid value range [{prps[i][0]}, {prps[i][1]})")
                    else:
                        cond = (prop >= prps[i][0]) & (prop < prps[i][1])
                elif i == 5:
                    if type(prps[i]) is Callable:
                        cond = prop is prps[i]
                    else:
                        assert hasattr(prop, '__name__')
                        cond = ((prps[i] == prop.__name__) & exact_match) | (
                            (prps[i] in prop.__name__) & (not exact_match)
                        )
                else:
                    cond = ((prps[i] == prop) & exact_match) | ((str(prps[i]) in str(prop)) & (not exact_match))
                rcond = rcond & cond

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
        method: Union[str, Callable, object, None] = None,
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
        method: Union[str, Callable, object, None] = None,
        full_application_sequence: Optional[int] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: Literal[False] = False,
    ) -> None: ...

    # List / view added processes
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    # Data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    @simple_type_validator
    def ls_process(
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Optional[int] = None,
        method: Union[str, Callable, object, None] = None,
        full_application_sequence: Optional[int] = None,
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_result: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        List process items based on filtering conditions.
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
            If False, processes whose property values partially match the specified value are included.
            The default is True.

        print_df : bool, optional
            Whether to print simplified matched process items.
            The default is True.

        return_df : bool, optional
            Whether to return a dataframe of matched process items.
            The default is False.

        Returns
        -------
        pandas.DataFrame or None
            If ``return_df=True``, returns a pandas DataFrames of matched process items.

            If ``return_df=False``, returns None.

        See Also
        --------
        add_process

        Examples
        --------
        For prepared ``SpecExp`` instance ``exp``::

            >>> pipe = SpecPipe(exp)
            >>> from swectral.functions import snv
            >>> pipe.add_process(2, 2, 0, snv)

        List all added processes::

            >>> pipe.ls_process()

        List processes by input data level::

            >>> pipe.ls_process(input_data_level=2)

        List processes by output data level::

            >>> pipe.ls_process(output_data_level=2)

        List processes by method::

            >>> pipe.ls_process(method='snv')

        List processes by partial method name::

            >>> pipe.ls_process(method='nv', exact_match=False)

        Return results instead of printing::

            >>> df_process = pipe.ls_process(print_df=False, return_df=True)
        """  # noqa: E501

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
        for i in range(df_proc.shape[0]):
            if callable(df_proc.iloc[i, -3]):
                df_proc.iloc[i, -3] = df_proc.iloc[i, -3].__name__
            else:
                try:
                    df_proc.iloc[i, -3] = df_proc.iloc[i, -3].model_label
                except Exception as e:
                    raise ValueError(
                        f"\nInvalid processing method with type '{type(df_proc.iloc[i, -3])}' detected in process:\
                            \n{df_proc.drop('Method', axis=1).iloc[i, :]}\n\n"
                    ) from e
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

    # Remove process
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    # Data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    @simple_type_validator
    def rm_process(
        self,
        process_id: Optional[str] = None,
        process_label: Optional[str] = None,
        input_data_level: Union[str, int, None] = None,
        output_data_level: Union[str, int, None] = None,
        application_sequence: Optional[int] = None,
        method: Union[str, Callable, object, None] = None,
        # full_application_sequence : Optional[int] = None,
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
            >>> pipe.add_process(2, 2, 0, snv)

        Remove all added processes::

            >>> pipe.rm_process()

        Remove processes by input data level::

            >>> pipe.rm_process(input_data_level=2)

        Remove processes by output data level::

            >>> pipe.rm_process(output_data_level=2)

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
            self._process = []
            self._process_steps = []
            self._process_chains = []
            self._custom_chains = []

        else:
            if len(matched) > 0:
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
                for i in range(df_proc.shape[0]):
                    if callable(df_proc.iloc[i, -3]):
                        df_proc.iloc[i, -3] = df_proc.iloc[i, -3].__name__
                    else:
                        try:
                            df_proc.iloc[i, -3] = df_proc.iloc[i, -3].model_label
                        except Exception as e:
                            raise ValueError(
                                f"\nInvalid processing method with type '{type(df_proc.iloc[i, -3])}' \
                                    detected in process: \n{df_proc.drop('Method', axis=1).iloc[i, :]}\n\n"
                            ) from e
                with pd.option_context("display.max_rows", None, "display.max_columns", None):
                    df_proc_simple = df_proc.iloc[:, [1, 2, 3, 4, 5]]
                    print(f"\nFollowing processes are removed from the pipeline: \n{df_proc_simple}")
            else:
                print("\nNo matched process found")

    # Update custom process chains
    def _update_custom_chains(self) -> None:
        if len(self._custom_chains) > 0:
            # Remove non-existed chains
            new_chains = []
            for chain in self._custom_chains:
                if chain in self._process_chains:
                    new_chains.append(chain)
            # Update
            self._custom_chains = new_chains

    # Sort processes
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    def _sort_proc(self) -> None:
        """
        Sort added processes
        """
        if len(self._process) > 1:
            # Sort processes
            proc_df = pd.DataFrame(
                self._process,
                columns=[
                    "Process_ID",
                    "Process_label",
                    "Input_data_level",
                    "Output_data_level",
                    "Application_sequence",
                    "Method_callable",
                    "Full_app_seq",
                    "Alternative_number",
                ],
            )
            proc_sorted = proc_df.sort_values(by=["Full_app_seq", "Alternative_number"])
            proc_sorted = np.array(proc_sorted)
            proc = [tuple(pit) for pit in proc_sorted]
            # Update processes
            self._process = proc

    # Generate process steps
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    def _generate_process_steps(self) -> None:
        """
        Generates process steps of Process_IDs, representing a sequential workflow of added process items.
        Length of the chain list represents the total number of proccessing steps.
        """
        if len(self._process) > 0:
            # Sort process
            self._sort_proc()
            # Grow a process tree
            pchain = []
            fseq = -1
            pti = -1
            p_item: tuple[str, str, str, str, int, Callable, int, int]
            for p_item in self._process:
                if p_item[6] != fseq:
                    pchain.append([p_item[0]])
                    fseq = p_item[6]
                    pti = pti + 1
                else:
                    pchain[pti].append(p_item[0])
            # Update tree
            self._process_steps = pchain
        else:
            raise ValueError("No process added")

    # Number to mixed radix vector
    @simple_type_validator
    def _num_to_mixed_radix(self, number: int, shape: Union[list[int], tuple[int]]) -> list[int]:
        """
        Converts a given integer number into the index of a mixed-radix numeral system with given shape
        """
        digits = []
        for radix in reversed(shape):
            digits.append(number % radix)
            number = number // radix
        return list(reversed(digits))

    # Generate process chains
    def _generate_process_chains(self) -> None:
        """
        Generates a list of all actual chains of added process items in execution.
        Each chain corresponds to an unique final results.
        """
        process_steps = self._process_steps
        # Get number of chains and chain shape
        nchain = 1
        chain_shape = []
        for proc_ids in process_steps:
            nids = len(proc_ids)
            chain_shape.append(nids)
            nchain = nchain * nids
        # Fill the chain table
        pcs = np.full((nchain, len(chain_shape)), np.nan, dtype="O")
        for i in range(pcs.shape[0]):
            sidi = self._num_to_mixed_radix(i, chain_shape)
            for j in range(pcs.shape[1]):
                pcs[i, j] = process_steps[j][sidi[j]]
        # Update process chains
        self._process_chains = [tuple(pc) for pc in pcs]

    # Generate process id - label reference table
    def _process_id_label_ref(self) -> np.ndarray:
        """Generate process id - label reference table, return the table [proc_id, label]"""
        ref_table = []
        for proc in self.process:
            if proc[1] != "":
                label_name = proc[1]
            else:
                label_name = proc[5].__name__
            ref_table.append((proc[0], label_name))
        result: np.ndarray = np.array(ref_table)
        return result

    # Convert process ID to process label
    def _process_id_to_label(self, process_id: str, ref_table: np.ndarray) -> str:
        """Convert process ID to process label"""
        proc_label_item = ref_table[ref_table[:, 0] == process_id]
        if len(proc_label_item) == 1:
            return str(proc_label_item[0, 1])
        elif len(proc_label_item) == 0:
            raise ValueError(f"Got invalid process ID '{process_id}', no corresponding process label found.")
        else:
            raise ValueError(
                f"Process ID '{process_id}' has multiple label references: {proc_label_item[:, 1]}. \
                    This indicates corrupted process data."
            )

    # Build pipelines directly from structure
    @simple_type_validator
    def build_pipeline(
        self,
        step_methods: list[
            Union[
                tuple[
                    tuple[Union[str, int], Union[str, int]],
                    Union[Callable, object, list[Union[Callable, object]], dict[str, Union[Callable, object]]],
                ],
                tuple[
                    tuple[Union[str, int], Union[str, int]],
                    Union[Callable, object, list[Union[Callable, object]], dict[str, Union[Callable, object]]],
                    dict[str, Any],
                ],
            ]
        ],
    ) -> None:
        """
        Build pipelines by given structure and methods of each step.

        This method constructs one or more processing pipelines directly from an explicit structural description.
        Each pipeline step is defined by its input/output data levels with one or more alternative callable(s) or objects responsible for processing at that step.

        Parameters
        ----------
        step_methods : list of ((str or int, str or int), callable or object or list of (callable or object) or dict of str to (callable or object), None or dict of str to Any)
            A list describing the pipeline structure and the processing logic for each step.
            Each element of the list has the form::

                ((input_data_level, output_data_level), methods, params)

            where:

                ``input_data_level`` : int or str
                    Input data level in number or name. See ``add_process`` for details.
                ``output_data_level`` : int or str
                    Input data level in number or name. See ``add_process`` for details.
                ``methods`` : callable or object or list or dict
                    A single callable or object defining one processing method.
                    A list of callables or objects representing alternative methods for the step.
                    A dictionary mapping method names to callables or objects, allowing multiple named methods.
                ``params`` : dict, optional
                    Optional dictionary of additional parameters applied to the methods at the step.

        See Also
        --------
        add_process
        add_model

        Examples
        --------
        For an initialized SpecPipe instance ``pipe``::

            >>> from swectral import roi_mean
            >>> from swectral.functions import snv, minmax, aucnorm

            >>> pipe.build_pipeline(
            ...     [
            ...         ((2, 2), [snv, minmax, aucnorm]),
            ...         ((5, 7), roi_mean),
            ...         ((7, 8), {'RF': RandomForestRegressor(n_estimators=6), 'KNN': KNeighborsRegressor(n_neighbors=3)}, {'validation_method': '5-fold'})
            ...     ]
            ... )
        """  # noqa: E501

        dl_in_p, dl_out_p, appseq = 0, 0, 0
        for step in step_methods:
            input_data_level = _dl_val(step[0][0])[0]
            output_data_level = _dl_val(step[0][1])[0]
            if not (
                (input_data_level == dl_in_p and output_data_level == dl_out_p) or (_dl_val(input_data_level)[0] <= 4)
            ):
                dl_in_p = input_data_level
                dl_out_p = output_data_level
                appseq = 0
            application_sequence = appseq
            # Validate additional params
            if len(step) == 3:
                params = step[2]
                for key in ["input_data_level", "output_data_level", "application_sequence", "method", "process_label"]:
                    _ = params.pop(key, None)
            else:
                params = {}
            # Add processes
            if isinstance(step[1], dict):
                process_label = list(step[1].keys())
                method: Union[Callable, object, list[Callable], tuple[Callable], list[object], tuple[object]] = list(
                    step[1].values()
                )
                self.add_process(
                    input_data_level=input_data_level,
                    output_data_level=output_data_level,
                    application_sequence=application_sequence,
                    method=method,
                    process_label=process_label,
                    **params,
                )
            else:
                method = step[1]
                self.add_process(
                    input_data_level=input_data_level,
                    output_data_level=output_data_level,
                    application_sequence=application_sequence,
                    method=method,
                    **params,
                )
            # Update dl
            if (input_data_level == dl_in_p and output_data_level == dl_out_p) or (_dl_val(input_data_level)[0] <= 4):
                appseq = appseq + 1

    # List process chains
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

            If no process is added to this SpecPipe instance, returns None.

        See Also
        --------
        ls_custom_chains

        Notes
        -----
        This method is also available as ``process_chains_to_df``.

        Examples
        --------
        For prepared ``SpecPipe`` instance ``pipe``::

            >>> pipe.ls_process_chains()

        Or equivalent::

            >>> pipe.process_chains_to_df()

        Return label display in addition to process ID display::

            >>> pipe.ls_process_chains(return_label=True)
        """

        if len(self._process_chains) > 0:
            df_chains = pd.DataFrame(
                self._process_chains,
                columns=["Step_" + str(i) for i in range(len(self._process_chains[0]))],
            )
            if return_label or print_label:
                df_chains_label = copy.deepcopy(df_chains)
                ref_table = self._process_id_label_ref()
                for i in range(df_chains.shape[0]):
                    for j in range(df_chains.shape[1]):
                        df_chains_label.iloc[i, j] = self._process_id_to_label(df_chains.iloc[i, j], ref_table)
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
    def custom_chains_from_df(
        self,
        # process_chain_dataframe: Annotated[Any, AfterValidator(dataframe_validator())]
        process_chain_dataframe: Annotated[Any, dataframe_validator()],
    ) -> None:
        """
        Customize processing chains and update chains using a chain dataframe.

        Once custom chains are created, SpecPipe will prioritize their execution, bypassing the original full-factorial chains.

        Parameters
        ----------
        process_chain_dataframe : pandas.DataFrame-like
            A process chain dataframe.

            Must be a subset of the original full-factorial chains.
            Columns must be ['Step_1', 'Step_2', ...] and the length must match the column length of ``process_chains``.
            And all values must be valid process IDs of SpecPipe.

            It is recommended to modify the dataframe obtained from ``process_chains_to_df`` to get the custom process chain dataframe.

        See Also
        --------
        ls_process_chains
        process_chains_to_df

        Examples
        --------
        For prepared ``SpecPipe`` instance ``pipe``::

            >>> df_chain = pipe.process_chains_to_df()

        After modification, load the modified dataframe::

            >>> pipe.custom_chains_from_df(df_chain_modified)
        """  # noqa: E501

        # Validate chain df
        process_chain_dataframe = dataframe_validator(dtype="str", ncol=len(self._process_chains[0]))(
            process_chain_dataframe
        )

        # Convert chain df to list
        cchain = [tuple(row) for row in process_chain_dataframe.to_numpy()]
        full_chain = self._process_chains

        # Validate given custom chains
        for ind, ccr in enumerate(cchain):
            if ccr not in full_chain:
                raise ValueError(
                    f"\nInvalid process chain in given chains: \n{ccr}, \nRow index of invalid chain: {ind}"
                )

        # Change process test status
        self.__tested = False

        # Update
        self._custom_chains = cchain

    # List custom process chains
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

            If no custom chain is specified in this SpecPipe instance, returns None.

        See Also
        --------
        ls_process_chains

        Notes
        -----
        This method is also available as ``custom_chains_to_df``.

        Examples
        --------
        For prepared ``SpecPipe`` instance ``pipe``::

            >>> df_chain = pipe.ls_custom_chains()
        """

        if len(self._custom_chains) > 0:
            df_chains = pd.DataFrame(
                self._custom_chains,
                columns=["Step_" + str(i) for i in range(len(self._custom_chains[0]))],
            )
            if return_label or print_label:
                df_chains_label = copy.deepcopy(df_chains)
                ref_table = self._process_id_label_ref()
                for i in range(df_chains.shape[0]):
                    for j in range(df_chains.shape[1]):
                        df_chains_label.iloc[i, j] = self._process_id_to_label(df_chains.iloc[i, j], ref_table)
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

            If no custom chain is specified in this SpecPipe instance, returns None.

        See Also
        --------
        ls_process_chains
        ls_custom_chains

        Examples
        --------
        For created ``SpecPipe`` instance ``pipe``::

            >>> df_chain = pipe.ls_chains()
        """  # noqa: E501
        if len(self._custom_chains) > 0:
            result = self.ls_custom_chains(print_label, return_label)
        else:
            result = self.ls_process_chains(print_label, return_label)
        return result

    # Save pipeline configurations
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
        df_process.to_csv(unc_path(report_dir + "SpecPipe_added_process.csv"), index=False)
        df_full_chains.to_csv(unc_path(report_dir + "SpecPipe_full_factorial_chains_in_ID.csv"), index=False)
        df_full_chains_label.to_csv(unc_path(report_dir + "SpecPipe_full_factorial_chains_in_label.csv"), index=False)
        df_exec_chains.to_csv(unc_path(report_dir + "SpecPipe_exec_chains_in_ID.csv"), index=False)
        df_exec_chains_label.to_csv(unc_path(report_dir + "SpecPipe_exec_chains_in_label.csv"), index=False)

        # Save SpecPipe
        with open(unc_path(f"{report_dir}SpecPipe_pipeline_configuration_{self.create_time}.dill"), 'wb') as f:
            dill.dump(self, f)

        # Save copies
        if copy:
            # Prevent duplication
            time.sleep(1.0)
            # Dump copy
            cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            df_process.to_csv(unc_path(report_dir + f"SpecPipe_added_process_{cts}.csv"), index=False)
            df_full_chains.to_csv(unc_path(report_dir + f"SpecPipe_full_factorial_chains_in_ID_{cts}.csv"), index=False)
            df_full_chains_label.to_csv(
                unc_path(report_dir + f"SpecPipe_full_factorial_chains_in_label_{cts}.csv"), index=False
            )
            df_exec_chains.to_csv(unc_path(report_dir + f"SpecPipe_exec_chains_in_ID_{cts}.csv"), index=False)
            df_exec_chains_label.to_csv(unc_path(report_dir + f"SpecPipe_exec_chains_in_label_{cts}.csv"), index=False)
            with open(
                unc_path(report_dir + f"SpecPipe_pipeline_configuration_{self.create_time}_copy_at_{cts}.dill"), 'wb'
            ) as f:
                dill.dump(self, f)

        # Save SpecExp
        if save_spec_exp_config:
            self.spec_exp.save_data_config(copy=copy)

        # Print output path
        print(
            f"\nSpecPipe configurations saved to: \
              {report_dir}SpecPipe_pipeline_configuration_{self.create_time}.dill\n"
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
        with open(unc_path(dump_path0), 'rb') as f:
            loaded_instance = dill.load(f)
        self.__dict__.update(loaded_instance.__dict__)

    # Alias
    load_config = load_pipe_config

    # Construct initial sample data from SpecExp
    # Sample data format - ROI: {ID, label, target, img_path, roi_coords}
    # Sample data format - standalone spec: {ID, label, target, spec1d: tuple}
    def _sample_data_constructor(self) -> None:
        """
        Constructing the initial sample data from SpecExp.
        If the sample is defined as ROIs of spectral images, the data constructing information is initialized, but the data itself is not loaded (for lazy-loading).
        """  # noqa: E501
        # ROI level (level ind 5)
        # ROI item format: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
        # Img item format: [0 id, 1 group, 2 image_name, 3 mask_of, 4 image_path]
        # Target value item format: [0 fixed sample id, 1 user assinged labels, 2 Target values]
        if (len(self.spec_exp.rois_sample) > 0) and (len(self.spec_exp.standalone_specs_sample) == 0):
            sample_data: list[dict] = []
            for roit in self.spec_exp.rois_sample:
                sdata: dict[str, Any] = {}
                sdata["ID"] = roit[0]
                sdata["label"] = [lbt[1] for lbt in self.spec_exp.sample_labels if lbt[0] == roit[0]][0]
                sdata["target"] = [tg[2] for tg in self.spec_exp.sample_targets if tg[0] == roit[0]][0]
                sdata["validation_group"] = [tg[4] for tg in self.spec_exp.sample_targets if tg[0] == roit[0]][0]
                sdata["train"] = [tg[5] for tg in self.spec_exp.sample_targets if tg[0] == roit[0]][0]
                sdata["test"] = [tg[6] for tg in self.spec_exp.sample_targets if tg[0] == roit[0]][0]
                sdata["img_path"] = [
                    imgt[4] for imgt in self.spec_exp.images if ((imgt[1] == roit[1]) & (imgt[2] == roit[2]))
                ][0]
                sdata["roi_coords"] = roit[5]
                sample_data.append(sdata)
            self.__sample_data = sample_data

        # Standalone spectra (level ind 7)
        # Standalone spectrum item format: [0 id, 1 group, 2 use_type, 3 sample_name, 4 spectral_data_list]
        elif (len(self.spec_exp.standalone_specs_sample) > 0) and (len(self.spec_exp.rois_sample) == 0):
            sample_data = []
            for st in self.spec_exp.standalone_specs_sample:
                sdata = {}
                sdata["ID"] = st[0]
                sdata["label"] = [lbt[1] for lbt in self.spec_exp.sample_labels if lbt[0] == st[0]][0]
                sdata["target"] = [tg[2] for tg in self.spec_exp.sample_targets if tg[0] == st[0]][0]
                sdata["validation_group"] = [tg[4] for tg in self.spec_exp.sample_targets if tg[0] == roit[0]][0]
                sdata["train"] = [tg[5] for tg in self.spec_exp.sample_targets if tg[0] == roit[0]][0]
                sdata["test"] = [tg[6] for tg in self.spec_exp.sample_targets if tg[0] == roit[0]][0]
                sdata["spec1d"] = tuple(st[4])
                sample_data.append(sdata)
            self.__sample_data = sample_data

        else:
            raise ValueError(
                "Hybrid samples from both standalone spectra and spectral images \
                    is not allowed by SpecPipe pipeline.\
                    \nPlease provide pure image ROI samples or standalone spectrum samples"
            )

    # Test run of all chains
    # Format of associated attribute:
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    # Data levels:
    # 0 - image (path), \
    # 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
    # 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
    # Pretest_data: [ID, label, target, validation_group, img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d]  # noqa: E501
    @simple_type_validator
    def test_run(
        self,
        test_modeling: bool = True,
        return_result: bool = False,
        model_test_coverage: float = 1.0,
        dump_result: bool = True,
        dump_backup: bool = False,
        save_preprocessed_images: bool = False,
        num_type: Union[str, type] = np.float32,
    ) -> Optional[dict]:
        """
        Run the pipeline of all processing chains using simplified test data.
        This method is executed automatically prior to each formal run.

        Parameters
        ----------
        test_modeling : bool, optional
            Whether added models are tested.
            If False, only the first chain is tested. The default is True.

        return_result : bool, optional
            Whether results of the processes are returned.
            If True, results of all tested steps are returned in a list. The default is False.

        model_test_coverage : float, optional
            Fraction of modeling pipelines to test.
            Set to a value < 1.0 to reduce test runtime by randomly sampling preprocessing results without replacement.

                - If 1.0, all pipelines will be tested.
                - If < 1.0, only the specified fraction of pipelines will be tested.

            Default is 1.0.

        dump_result : bool, optional
            Whether test results are stored in the chains. The default is True.

        dump_backup : bool, optional
            Whether backup of the step results is stored.
            The backup file is named with the datetime of dumping.

        num_type: str or type, optional
            Numeric data type for array-like data storage, supporting numeric ``numpy`` data types.
            Default is ``numpy.float32``.

        Returns
        -------
        list of dict of list or None
            List of all tested steps in a dictionary.

            Dictionary keys are the list of applied processes of a processing chain.

            Dictionary values are lists of processing results of the applied processes for all steps of the processing chain.

        Examples
        --------
        For a created SpecPipe instance ``pipe``::

            >>> pipe.test_run()
        """  # noqa: E501

        # Test data
        if self.__pretest_data is not None:
            test_data: dict[str, Any] = copy.deepcopy(self.__pretest_data)
        else:
            raise ValueError(
                "Internal Error: 'SpecPipe._pretest_data' is None. \
                    Pre-execution test data initialization fails. Please report."
            )

        # Preprocessed test image dir
        preprocessed_img_dir = self._spec_exp.report_directory + "test_run/Preprocessed_images/"
        os.makedirs(unc_path(preprocessed_img_dir), exist_ok=True)

        # Preprocessing test
        status_results: dict = _preprocessing_sample(  # type: ignore[call-overload]
            sample_data=test_data,
            process=self.process,
            custom_chains=self.custom_chains,
            process_chains=self.process_chains,
            specpipe_report_directory=self.spec_exp.report_directory,
            preprocess_status=self._init_preprocessing_status(n_processor=1),
            num_type=num_type,
            dump_result=dump_result,
            return_result_path=False,
            dump_backup=dump_backup,
            return_step_result=True,
            final_result_only=False,
            is_test_run=True,
        )
        # Multiple dynamic bool switches passed to '_preprocessing_sample' that fails for type 'Literal' in overloads

        # If not save preprocessed images, remove after test
        if not save_preprocessed_images:
            shutil.rmtree(preprocessed_img_dir)

        # Modeling test
        if test_modeling:
            if len(self.ls_model(print_result=False, return_result=True)) > 0:
                self._test_model(
                    status_results=status_results,
                    specpipe_report_directory=self.spec_exp.report_directory,
                    test_coverage=model_test_coverage,
                )
            # Change process test status
            self.__tested = True

        if return_result:
            return status_results
        else:
            return None

    # Test modeling
    def _test_model(  # noqa: C901
        self,
        status_results: dict,
        specpipe_report_directory: str,
        test_coverage: float = 1.0,
    ) -> None:
        """
        Test models added on a simulated testing targets based on the given preprocessing results.
        """
        # Get model processes
        model_procs = [procit for procit in self._process if procit[3] == "model"]

        # Get model processes with input data level index dl_in_ind
        # Process: [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
        for modelit in model_procs:
            # Get test model
            model_methodi = modelit[5]
            assert hasattr(model_methodi, 'validation_method')

            # Get minimum mock sample size according to validation fold
            val_method = model_methodi.validation_method
            val_method = str(val_method).lower()
            if "fold" in val_method:
                sample_min_size = max(10, int(val_method.split("-")[0]))
            elif "split" in val_method:
                a = float(val_method.split("-")[0])
                b = float(val_method.split("-")[1])
                sample_min_size = max(10, int(max(a, b) / min(a, b)) + 1)
            elif "loo" in val_method:
                sample_min_size = 10

            # Get model input data level
            dl_in_name = _dl_val(modelit[2])[1]
            dl_in_ind = _dl_val(modelit[2])[0]

            # Get model input data shape
            assert hasattr(model_methodi, "input_shape")
            ts_shape = model_methodi.input_shape

            ## Construct test samples list and implement model testing
            # Sampled tesing - fast testing
            preprocessing_results = status_results["status_results"][-1]
            if not (0 < test_coverage <= 1.0):
                raise ValueError(f"test_coverage must be in (0, 1], got: {test_coverage}")
            if test_coverage < 1.0:
                total_npchain = len(preprocessing_results)
                sample_npchain = max(1, int(round(total_npchain * test_coverage)))
                sample_indices = np.random.choice(total_npchain, size=sample_npchain, replace=False)
                preprocessing_results = [preprocessing_results[i] for i in sample_indices]
            # Model testing
            # Status result: (0 - step_id, 1 step_procs, 2 dl_in, 3 dl_out, 4 sample_result)
            for pci, status_result in enumerate(preprocessing_results):
                # Get sample data of the model input data level
                if status_result[3] == dl_in_ind:
                    # Get test sample data
                    tsample = np.array(_load_disk_backed_data(status_result[4]))

                    # Validate data shape
                    if ts_shape is None:
                        ts_shape = tsample.shape
                    elif np.prod(tsample.shape) != np.prod(ts_shape):
                        raise ValueError(
                            f"Cannot reshape sample data with shape {tsample.shape} into specified input data shape \
                                {ts_shape} of the model.\
                                \nInput step data ID: {status_result[0]}\nModel label: {modelit[1]}"
                        )

                    # Save preprocess chain of sample_list
                    with open(unc_path(specpipe_report_directory + f"test_run/Preprocess_chain_#{pci}.txt"), "w") as f:
                        for ppci, pproc in enumerate(status_result[1]):
                            if ppci < (len(status_result[1]) - 1):
                                f.write(f"{pproc}\n")
                            else:
                                f.write(f"{pproc}")

                    # For data level of numeric values
                    # Sample_list item: (0 - Sample id, 1 - Sample label, 2 - Validation group, 3 - Test mask, 4 - Train mask, 5 - Original shape, 6 - Target value, 7 - Sample predictor values)  # noqa: E501
                    # if (dl_in_ind != 0) & (dl_in_ind != 5):
                    if dl_in_ind == 7:
                        test_data_range: float = float(np.nanmax(tsample) - np.nanmin(tsample))
                        # Model report dir
                        model_report_dir = specpipe_report_directory + "test_run/"
                        os.makedirs(unc_path(model_report_dir), exist_ok=True)
                        # Build testing sample list
                        assert hasattr(model_methodi, 'is_regression')
                        if model_methodi.is_regression:
                            # Regression mock data
                            test_samples: list[
                                tuple[str, str, str, np.int8, np.int8, Any, Union[float, int, bool, str], np.ndarray]
                            ]
                            test_samples = [
                                (
                                    str(i),
                                    str(i),
                                    str(i),
                                    np.int8(1),
                                    np.int8(1),
                                    ts_shape,
                                    float(i),
                                    tsample
                                    * (
                                        1
                                        + test_data_range
                                        * (0.5 / sample_min_size)
                                        * (i + 0.1 * i * np.random.rand(1)[0])
                                    ),
                                )
                                for i in range(sample_min_size)
                            ]
                        else:
                            # Classification mock data
                            test_samples = [
                                (
                                    str(i),
                                    str(i),
                                    str(i),
                                    np.int8(1),
                                    np.int8(1),
                                    ts_shape,
                                    str(["a", "b"][int(i % 2)]),
                                    tsample * (1 + test_data_range * 0.25 * (i % 2 + 0.1 * np.random.rand(1)[0])),
                                )
                                for i in range(sample_min_size)
                            ]
                        # Test modeling
                        assert hasattr(model_methodi, 'evaluation')
                        model_methodi.evaluation(
                            sample_list=test_samples,
                            data_label="test_chain_" + str(pci),
                            report_directory=model_report_dir,
                        )
                    else:
                        raise ValueError(f"Model only accepts data level 'spec1d' as input, but got: {dl_in_name}")

    # Space validator
    @simple_type_validator
    def _val_disk_space_preprocessing(self, image: bool = True, plot: bool = True, error_raise: bool = True) -> None:
        """Output image and plot size estimator"""

        # Estimate output image size
        # Get number of image processing chains
        applied_chains = self.ls_chains(print_label=False)
        n_img_chains = _num_image_chains(applied_chains)
        # Get image paths
        img_paths = self.spec_exp.ls_images(print_result=False, return_dataframe=True)["Path"].tolist()
        # Check image to roispecs process
        n_roispecs_process = len(
            self.ls_process(input_data_level=5, output_data_level=6, print_result=False, return_result=True)
        )
        has_roispecs = n_roispecs_process > 0
        # Compute total size
        output_img_size = _estimate_img_output_size(
            img_paths=img_paths,
            n_img_chains=n_img_chains,
            roispecs=has_roispecs,
        )

        # Estimate output plot size
        output_plot_size = _estimate_report_plot_size(len(applied_chains), is_regression=self._is_target_numeric)

        # Output size
        if image & plot:
            output_size = output_img_size + output_plot_size
        elif image & (not plot):
            output_size = output_img_size
        elif plot & (not image):
            output_size = output_plot_size
        else:
            output_size = 0
        # Available space
        available_space: int = shutil.disk_usage(self.report_directory).free
        # Validate available space
        if available_space < output_size:
            if error_raise:
                raise OSError(
                    f"Insufficient disk space: estimated output size is {output_size} bytes, "
                    f"which may exceed the available disk space ({available_space} bytes)."
                )
            else:
                warnings.warn(
                    f"Insufficient disk space: estimated output size is {output_size} bytes, "
                    f"which may exceed the available disk space ({available_space} bytes).",
                    ResourceWarning,
                    stacklevel=2,
                )
        else:
            return None

    # Run entire pipeline
    # Sample data format - ROI: {ID, label, target, img_path, roi_coords}
    # Sample data format - standalone spec: {ID, label, target, spec1d: tuple}
    # Process: [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    # Status_result: (0 - step_id, 1 step_procs, 2 dl_in, 3 dl_out, 4 sample_result)
    # Sample_list item: (0 - Sample id, 1 - Sample label, 2 - Validation group, 3 - Test mask, 4 - Train mask, 5 - Original shape, 6 - Target value, 7 - Sample predictor values)  # noqa: E501
    @simple_type_validator
    def preprocessing(  # noqa: C901
        self,
        result_directory: str = "",
        n_processor: int = 1,
        num_type: Union[str, type] = np.float32,
        dump_backup: bool = False,
        step_result: bool = True,
        resume: bool = False,
        to_csv: bool = True,
        show_progress: bool = True,
        save_config: bool = True,
        summary: bool = True,
        geo_reference_warning: bool = False,
        skip_test: bool = False,
        check_space: bool = True,
    ) -> None:
        """
        Run preprocessing steps of all processing chains on the entire dataset and output modeling-ready sample_list data to files.

        Parameters
        ----------
        result_directory : str, optional
            Directory to save the final preprocess results.
            Default is the ``report_directory`` of the ``SpecExp`` instance of this ``SpecPipe`` instance.

        n_processor : int
            Number of processors to use in preprocessing.

            Default is ``1`` (parallel processing is not applied).

            Windows note: when using ``n_processor > 1`` on Windows, all excecutable code in the working script must be placed within::

                if __name__ == '__main__':

        num_type: str or type, optional
            Numeric data type for array-like data storage, supporting numeric ``numpy`` data types.
            Default is ``numpy.float32``.

        dump_backup : bool, optional
            Create backup files of results with timestamp. Default is False.

        step_result : bool, optional
            Whether to retain intermediate step results for each processing chain.

            If False, intermediate results are removed after preprocessing.
            Default is True.

        resume : bool
            If True, computation resumes from preprocessing progress logs.
            Apply ``resume`` to avoid repeated preprocessing after interruption.
            Default is False.

        to_csv : bool
            If True, final preprocessing results are also saved to CSV files in addition to ``dill`` files.
            Default is True.

        show_progress : bool, optional
            Show processing progress. Default is True.

        save_config : bool, optional
            Save ``SpecPipe`` configurations. Default is True.

        summary : bool, optional
            Whether to summarize preprocessed data and target values. Default is True.

        geo_reference_warning : bool, optional
            Whether to suppress GeoReferenceWarning. If False, the warning is suppressed.
            Default is False.

        skip_test : bool, optional
            Whether to skip test execution completely.
            Test execution validates every processing chain and serves as a safeguard against runtime errors in long formal execution.
            Default is False.

        check_space : bool, optional
            Whether to validate available disk space against the estimated output size.
            If True, an error is raised when the estimate exceeds the available space.
            If False, a warning is issued instead.
            Default is True.

        Examples
        --------
        For created ``SpecPipe`` instance ``pipe``::

            >>> pipe.preprocessing()

        Pipeline-level multiprocessing::

            >>> pipe.preprocessing(n_processor=10)
        """  # noqa: E501

        # Setup multiprocessing
        if n_processor > 1:
            _mp_init()

        # Validate computation status
        finish_status_path = self._report_directory + "SpecPipe_configuration/.__preprocessing_complete.s"
        if os.path.exists(finish_status_path):
            if resume:
                print("Preprocessing is completed. Set resume=False to recompute.")
                return None
            else:
                os.remove(finish_status_path)

        # Validate disk space for output
        self._val_disk_space_preprocessing(image=True, plot=False, error_raise=check_space)

        # Prompt "if __name__ == '__main__':" protection for windows multiprocessing
        if n_processor > 1:
            if os.name == "nt":
                print(
                    "\n\n\n\
                    >>> WARNING: Windows users must run multiprocessing within block \n\nif __name__ == '__main__':\
                    \n\n Please make sure all of your main codes in the script are placed within this block. \n\n",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(2)
                warn_msg = (
                    "\nWindows users must run multiprocessing within block \n\nif __name__ == '__main__':"
                    + "Please make sure all of your main codes in the script are placed within this block."
                )
                warnings.warn(warn_msg, UserWarning, stacklevel=2)

        # Suppress NotGeoreferencedWarning
        if not geo_reference_warning:
            warnings.simplefilter("ignore", NotGeoreferencedWarning)

        # Save configurations
        if save_config:
            self.save_pipe_config(copy=True)

        # Added chain testing
        if not self._tested and not skip_test:
            self.test_run(test_modeling=False, return_result=False, dump_result=False, dump_backup=False)

        # Default result_directory
        if result_directory == "":
            result_directory = self._spec_exp.report_directory
        # Preprocessed image dir for data level 0~4
        preprocessed_img_dir = result_directory + "Preprocessing/Preprocessed_images/"
        os.makedirs(unc_path(preprocessed_img_dir), exist_ok=True)

        # Start preprocessing
        if show_progress:
            print("\nPreprocess samples ...")

        # Pipeline progress observation for multiprocessing
        step_file_root = unc_path(self.report_directory + "Preprocessing/Step_results/Intermediate_step_results/")
        if n_processor > 1:
            # Pipeline and resulting file configuration
            n_pre_step = len(self.ls_process_chains(print_label=False).iloc[0, :].tolist()) - 1
            step_file_patterns = [f"_step_{i}" for i in range(n_pre_step)]
            df_pchains = self.ls_chains(print_label=False)
            step_branches = [len(df_pchains.iloc[:, i].unique()) for i in range(n_pre_step)]
            # Progress observing process by preprocessing intermediate data files
            observer = PipelineFileTqdm(
                root=step_file_root,
                patterns=step_file_patterns,
                nbranches=step_branches,
                nsample=len(self._sample_targets),
            )
            observer.start()

        # Preprocessing
        final_result_only = not step_result
        self._preprocessor(
            result_directory=result_directory,
            n_processor=n_processor,
            dump_backup=dump_backup,
            final_result_only=final_result_only,
            resume=resume,
            show_progress=show_progress,
            num_type=num_type,
        )

        # Complete preprocessing progress monitoring
        open(step_file_root + ".__finished.s", "w").close()
        if n_processor > 1:
            observer.join()

        # Construct sample_list
        print("\n\nConstructing sample list...")
        self._sample_list_constructor(result_directory=result_directory, to_csv=to_csv, show_progress=show_progress)

        # Clear step result data after sample list construction after finishing sample_list construction
        if not step_result:
            self._cl_step_result(result_directory=result_directory)

        # Print result dir
        if len(result_directory) > 0:
            result_dir = str(result_directory)
        else:
            result_dir = self.report_directory
        self._save_sample_targets(result_dir)
        print(f"\n\nPipeline preprocessing complete, result stored in:\n{result_dir}")

        # Group stats for preprocessing
        if summary:
            _ = sample_group_stats(self.report_directory, is_regression=self._is_target_numeric)

        # Recover NotGeoreferencedWarning
        warnings.simplefilter("default", NotGeoreferencedWarning)

        # Store computation status
        open(finish_status_path, "w").close()

    @simple_type_validator
    def _preprocessor(  # noqa: C901
        self,
        result_directory: str = "",
        n_processor: int = 1,
        dump_backup: bool = False,
        final_result_only: bool = True,
        resume: bool = False,
        show_progress: bool = True,
        num_type: Union[str, type] = np.float32,
        *,
        # Dependencies for multiprocessing
        copy: ModuleType = copy,
        os: ModuleType = os,
        time: ModuleType = time,
        datetime: type = datetime,
        np: ModuleType = np,
        torch: ModuleType = torch,
    ) -> None:
        """
        Preprocessing of all processing chains on the entire dataset.
        Output sample results from all processing chains to files and update the resulting file paths.
        """
        # Store SpecExp configs
        self.spec_exp.save_data_config(copy=dump_backup)

        # Validate report directory
        if result_directory == "":
            result_directory = self._spec_exp.report_directory

        # Construct sample data
        self._sample_data_constructor()

        ## Preprocessing
        # Validate result_directory / report_directory
        # Step dir for chain results
        step_dir = result_directory + "Preprocessing/Step_results/"
        os.makedirs(unc_path(step_dir), exist_ok=True)
        # log dir for resume
        log_dir_path = step_dir + "Preprocess_progress_logs/"
        os.makedirs(unc_path(log_dir_path), exist_ok=True)

        # Check running log and subset sample data
        if not resume:
            rest_sample_data = self._sample_data
            # Clear log before start new preprocessing
            if os.path.exists(unc_path(log_dir_path)):
                shutil.rmtree(unc_path(log_dir_path))
        else:
            if not os.path.exists(unc_path(log_dir_path)):
                os.makedirs(unc_path(log_dir_path), exist_ok=True)
                finished_samples = []
            else:
                finished_samples = [
                    f.split(".")[0]
                    for f in os.listdir(unc_path(log_dir_path))
                    if os.path.isfile(unc_path(log_dir_path + f))
                ]
                existed_samples = [sd["ID"] for sd in self._sample_data]
                finished_samples = [sdid for sdid in finished_samples if sdid in existed_samples]
            if len(finished_samples) > 0:
                rest_sample_data = [sd for sd in self._sample_data if sd["ID"] not in finished_samples]
                # Quit if all finished
                if len(rest_sample_data) == 0:
                    return
            else:
                rest_sample_data = self._sample_data

        # Initialize preprocessing status for image operation
        preprocess_status = self._init_preprocessing_status(n_processor=n_processor)

        # Preprocessing of all data and generate sample_list data of all chains
        self.__preprocess_result_path = []

        # Sequential compute
        if n_processor <= 1:
            for sd_ind in tqdm(
                range(len(rest_sample_data)),
                total=len(rest_sample_data),
                disable=(not show_progress),
            ):
                sdata = rest_sample_data[sd_ind]
                pti = _preprocessing_sample(
                    sample_data=sdata,
                    process=self.process,
                    custom_chains=self.custom_chains,
                    process_chains=self.process_chains,
                    specpipe_report_directory=result_directory,
                    dump_result=True,
                    return_result_path=True,
                    dump_backup=dump_backup,
                    return_step_result=False,
                    final_result_only=final_result_only,
                    is_test_run=False,
                    dump_directory=step_dir,
                    update_progress_log=True,
                    preprocess_status=preprocess_status,
                    num_type=num_type,
                )
                self.__preprocess_result_path.append(pti)

        # Parallel compute
        else:
            # Initialize errorlogs dir for err handling in parallel computing
            errorlog_path = result_directory + "Preprocessing/Step_results/Error_logs/"
            if os.path.exists(unc_path(errorlog_path)):
                shutil.rmtree(unc_path(errorlog_path))
            # Validate number of processors to use
            ncpu_max = max(cpu_count() - 1, 1)
            ncpu = min(n_processor, ncpu_max)
            # Bind constant arguments for _preprocessing_sample - Result dumped in _preprocessing_sample
            preprocessing_sample_it = partial(
                _preprocessing_sample,
                process=self.process,
                custom_chains=self.custom_chains,
                process_chains=self.process_chains,
                specpipe_report_directory=result_directory,
                dump_result=True,
                return_result_path=True,
                dump_backup=dump_backup,
                return_step_result=False,
                final_result_only=final_result_only,
                is_test_run=False,
                dump_directory=step_dir,
                update_progress_log=True,
                preprocess_status=preprocess_status,
                num_type=num_type,
                # Explicitly assign dependencies for multiprocessing
                copy=copy,
                os=os,
                datetime=datetime,
                np=np,
                torch=torch,
            )
            # Processing - multiprocessing for loop
            with ProcessingPool(nodes=ncpu) as pool:
                preprocess_result_paths = list(pool.imap(preprocessing_sample_it, rest_sample_data))
            # Collect and print errors if exist
            if os.path.exists(unc_path(errorlog_path)):
                raise ValueError(
                    f"\nPreprocessing errors, please check error logs in the following path:\n{errorlog_path}"
                )
            # Validate returning path list
            if len(preprocess_result_paths) != len(rest_sample_data):
                raise ValueError(
                    f"\nIncomplete preprocessing results, expected number of results: {len(self._sample_data)}, \
                        got: {len(preprocess_result_paths)}"
                )
            for pti in preprocess_result_paths:
                if type(pti) is not str:
                    raise TypeError(f"\nResult file path must be str, got path: {pti}, with type: {type(pti)}")
                elif not os.path.exists(unc_path(pti)):
                    raise ValueError(f"\nGot invalid path: {pti}")
            # Update preprocess result file paths
            self.__preprocess_result_path = preprocess_result_paths

    # Step results to modeling-ready sample_list data
    @simple_type_validator
    def _sample_list_constructor(  # noqa: C901
        self, result_directory: str = "", to_csv: bool = True, show_progress: bool = False
    ) -> None:
        """
        Convert Step_result data from file to modeling-ready sample_list data.
        """
        # Validate report directory
        if result_directory == "":
            result_directory = self._spec_exp.report_directory
        preprocess_result_dir = result_directory + "Preprocessing/"
        os.makedirs(unc_path(preprocess_result_dir), exist_ok=True)

        # Validate preprocessing result file paths
        sd_paths = [
            f"{preprocess_result_dir}Step_results/PreprocessingResult_sample_{sd['label']}.dill"
            for sd in self._sample_data
        ]
        for sdp in sd_paths:
            if not os.path.exists(unc_path(sdp)):
                raise ValueError(f"\nPreprocessing step result file path not found : \n{sdp}\n")

        ## Chain results to sample_list data
        # Get preprocess chains of all preprocessing steps
        pchains = []
        for pchain in self.process_chains:
            if pchain[:-1] not in pchains:
                pchains.append(pchain[:-1])

        ## Loop preprocess chains across all samples and transform to modeling data
        if show_progress:
            print("\nConstruct chain sample list ...\nChain :")
        # pci - process chain id
        for pci in tqdm(range(len(pchains)), total=len(pchains), disable=(not show_progress)):
            pchain = pchains[pci]
            # Preprocessing results
            pre_results = []
            for spath in sd_paths:
                sdata = load_vars(unc_path(spath))
                status_results = sdata["status_results"][-1]
                # Sample ID and sample target value
                sample_id = sdata["ID"]
                sample_label = sdata["label"]
                sample_y = sdata["target"]
                sample_vg = sdata["validation_group"]  # sample validation group
                sample_te = sdata["test"]
                sample_tr = sdata["train"]
                # Sample data
                for status_result in status_results:
                    if tuple(status_result[1]) == tuple(pchain):
                        # Construct sample_list item
                        step_data = _load_disk_backed_data(status_result[4])
                        step_data_shape = np.array(step_data).shape
                        # Validate step output data level
                        step_dl_out = status_result[3]
                        if step_dl_out != 7:
                            raise ValueError("Input data level of modeling step")
                        pre_results.append(
                            (
                                sample_id,
                                sample_label,
                                sample_vg,
                                sample_te,
                                sample_tr,
                                step_data_shape,
                                sample_y,
                                step_data,
                            )
                        )  # noqa: E501

            ## Save resutls to files
            # Create file name
            chain_name = ""
            for proc_name in pchain:
                chain_name = chain_name + proc_name + "-"
            chain_name1 = f"PreprocessingChainResult_chain_ind_{str(pci)}"

            # Dump results to dill
            # chain_res == sample_list
            # Sample_list item: (0 - Sample id, 1 - Sample label, 2 - Validation group, 3 - Test mask, 4 - Train mask, 5 - Original shape, 6 - Target value, 7 - Sample predictor values)  # noqa: E501
            # Typing: list[tuple[str, str, str, np.int8, np.int8, tuple[int], Union[str,int,bool,float], Annotated[Any,arraylike_validator(ndim=1)]]]  # noqa: E501
            res_path_dill = preprocess_result_dir + chain_name1 + ".dill"
            dump_vars(unc_path(res_path_dill), {"chain_ind": pci, "chain_procs": pchain, "chain_res": pre_results})

            # Save results to CSV
            if to_csv:
                # Results to table (df)
                chain_res_table = []
                for pres in pre_results:
                    pres_data = pres[-1]
                    if isinstance(pres_data, Iterable):
                        pres_data_tuple: tuple = tuple(pres_data)
                    else:
                        pres_data_tuple = (pres_data,)
                    chain_res_table.append(
                        (pres[0], str(pres[1]), str(pres[2]), pres[3], pres[4], str(pres[5]), pres[6]) + pres_data_tuple
                    )
                arr_chain_res = np.array(chain_res_table)

                coln_chain_res = ["Sample_ID", "Label", "Validation_group", "Test", "Train", "X_shape", "y"] + [
                    f"x{i}" for i in range(arr_chain_res.shape[1] - 7)
                ]

                df_chain_res = pd.DataFrame(arr_chain_res, columns=coln_chain_res)
                # Add chain name to table content (as first col)
                df_chain_res = pd.concat(
                    [
                        pd.DataFrame(
                            [[chain_name]] + [[""]] * (df_chain_res.shape[0] - 1),
                            columns=["Preprocessing_chain"],
                        ),
                        df_chain_res,
                    ],
                    ignore_index=True,
                    axis=1,
                )
                # Recover colnames
                df_chain_res.columns = ["Preprocessing_chain"] + coln_chain_res
                # Save table to CSV
                res_path_csv = preprocess_result_dir + chain_name1 + ".csv"
                df_to_csv(df_chain_res, res_path_csv, index=False, return_path=False)

        # Add line after progress bar
        print("")

    ## Clear generated step results
    @simple_type_validator
    def _cl_step_result(self, result_directory: str) -> None:
        # Step dir for chain results
        step_dir = result_directory + "Preprocessing/Step_results/"
        if os.path.exists(unc_path(step_dir)):
            for item in os.listdir(step_dir):
                if "PreprocessingChainResult_" in str(item):
                    item_path = os.path.join(step_dir, item)
                    try:
                        if os.path.isfile(unc_path(item_path)):
                            os.remove(item_path)
                    except Exception as e:
                        print(f"Error in removing '{item_path}': \n{e}")
        else:
            warnings.warn(
                f"Step_results path is invalid:\n{result_directory}\nNo 'PreprocessingChainResult' is cleared.",
                UserWarning,
                stacklevel=3,
            )

    # Preprocessing image processing status
    @staticmethod
    def _init_preprocessing_status(n_processor: int) -> dict:
        """Initialize shared image processing status for preprocessing."""
        if n_processor > 1:
            manager = mp.Manager()
        else:
            manager = _DummyManager()
        preprocess_status: dict = {
            'start_status': manager.list(),
            'completion_status': manager.list(),
            'processed_image_init': manager.list(),
            'lock': manager.Lock(),
            'preprocess_resume_test_num': manager.Value('i', -1),
        }
        return preprocess_status

    # Run modeling on single dataset
    # chain_res == sample_list
    # Sample_list item: (0 - Sample id, 1 - Sample label, 2 - Validation group, 3 - Test mask, 4 - Train mask, 5 - Original shape, 6 - Target value, 7 - Sample predictor values)  # noqa: E501
    @simple_type_validator
    def model_evaluation(  # noqa: C901
        self,
        n_processor: int = 1,
        resume: bool = False,
        report_directory: str = "",
        show_progress: bool = True,
        save_config: bool = True,
        summary: bool = True,
        check_space: bool = True,
    ) -> None:
        """
        Evaluate added models on processed sample data from all preprocessing chains.
        Modeling and evaluation behavior is configured when models are added to the pipeline.

        Parameters
        ----------
        n_processor : int
            Number of processors to use in preprocessing.

            Default is ``1`` (parallel processing is not applied).

            Windows note: when using ``n_processor > 1`` on Windows, all excecutable code in the working script must be placed within::

                if __name__ == '__main__':

        resume : bool
            If True, computation resumes from preprocessing progress logs.

            Use ``resume`` to avoid repeated preprocessing after interruption.
            Default is False.

        report_directory : str, optional
            Directory to save modeling and evaluation reports.
            Default is the ``report_directory`` of the input ``SpecExp`` instance of this ``SpecPipe`` instance.

        show_progress : bool, optional
            Show processing progress. Default is True.

        save_config : bool, optional
            Save ``SpecPipe`` configurations. Default is True.

        summary : bool, optional
            Whether to summarize overall and marginal performance.
            Marginal performance metrics at each processing step is compared using the Mann–Whitney U test.
            Default is True.

        check_space : bool, optional
            Whether to validate available disk space against the estimated output size.
            If True, an error is raised when the estimate exceeds the available space.
            If False, a warning is issued instead.
            Default is True.

        See Also
        --------
        preprocessing

        Examples
        --------
        For a prepared ``SpecPipe`` instance ``pipe``::

            >>> pipe.preprocessing()
            >>> pipe.model_evaluation()

        Pipeline-level multiprocessing::

            >>> pipe.model_evaluation(n_processor=10)
        """  # noqa: E501

        # Setup multiprocessing
        if n_processor > 1:
            _mp_init()

        # Validate computation status
        finish_status_path = self._report_directory + "SpecPipe_configuration/.__modeling_complete.s"
        if os.path.exists(finish_status_path):
            if resume:
                print("Model evaluation is completed. Set resume=False to recompute.")
                return None
            else:
                os.remove(finish_status_path)

        # Validate disk space for output
        self._val_disk_space_preprocessing(image=False, plot=True, error_raise=check_space)

        # Prompt "if __name__ == '__main__':" protection for windows multiprocessing
        if n_processor > 1:
            if os.name == "nt":
                print(  # noqa: E501
                    "\n\n\n\
                    >>> WARNING: Windows users must run multiprocessing within block \n\nif __name__ == '__main__':\
                    \n\n Please make sure all of your main codes in the script are placed within this block. \n\n",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(2)
                warnings.warn(
                    "Windows users must run multiprocessing within block \n\nif __name__ == '__main__': \n\n\
                    Please make sure all of your main codes in the script are placed within this block.",
                    UserWarning,
                    stacklevel=2,
                )

        # Save configurations
        if save_config:
            self.save_pipe_config(copy=True)

        # Close existing pyplot to save memory
        plt.close("all")

        # Validate report directory
        result_directory = report_directory
        if result_directory == "":
            result_directory = self._spec_exp.report_directory
        # Preprocessing dir
        preprocess_result_dir = result_directory + "Preprocessing/"
        if not os.path.exists(unc_path(preprocess_result_dir)):
            raise ValueError(f"\nPreprocessing result directory not found, got path:\n{preprocess_result_dir}")
        # Modeling dir
        model_result_dir = result_directory + "Modeling/"
        os.makedirs(unc_path(model_result_dir), exist_ok=True)

        # Validate models
        model_procs = [proc for proc in self.process if proc[3] == "model"]
        if len(model_procs) < 1:
            raise ValueError("No model added.")

        # Validate preprocessing result file paths
        # Get preprocess chains of all preprocessing steps
        pchains = []
        for chain in self.process_chains:
            pchain = chain[:-1]
            if pchain not in pchains:
                pchains.append(pchain)
        # Validate result file existence
        cd_paths = [
            f"{preprocess_result_dir}PreprocessingChainResult_chain_ind_{pci}.dill" for pci in range(len(pchains))
        ]
        pchains_f = []
        for pci, cdp in enumerate(cd_paths):
            if not os.path.exists(unc_path(cdp)):
                raise ValueError(f"\nPreprocessing result file of chain {pchains[pci]} not found, path : \n{cdp}\n")
            cprocs = load_vars(unc_path(cdp))["chain_procs"]
            pchains_f.append(cprocs)
        spcs = set(pchains)
        spcf = set(pchains_f)
        if spcs != spcf:
            raise ValueError(
                f"\nPreprocessing chains and preprocessing results do not match, \
                    unmatched chains:\n\n{list((spcs | spcf) - (spcs & spcf))}\n"
            )

        # Check running log and subset sample data
        model_subdir_root = unc_path(self.report_directory + "Modeling/Model_evaluation_reports/")
        log_path = unc_path(model_subdir_root + "modeling_progress_log.dill")
        if not resume:
            rest_cd_paths = cd_paths
            # Clear log
            self._clear_model_log(log_path)
            modeling_finished = False
        else:
            if not os.path.exists(unc_path(model_subdir_root + ".__finished.s")):
                if not os.path.exists(unc_path(log_path)):
                    rest_cd_paths = cd_paths
                else:
                    modeling_progress_log = load_vars(unc_path(log_path))["modeling_progress_log"]
                    rest_cd_paths = []
                    for cdp in cd_paths:
                        cprocs = load_vars(unc_path(cdp))["chain_procs"]
                        if cprocs not in modeling_progress_log:
                            rest_cd_paths.append(cdp)
                if len(rest_cd_paths) == 0:
                    modeling_finished = True
                else:
                    modeling_finished = False
            else:
                modeling_finished = True

        # Pipeline progress observation for multiprocessing
        if n_processor > 1:
            # Pipeline and resulting file configuration
            n_chains = len(self.ls_chains(print_label=False))
            model_subdir_pattern = "Data_chain_Preprocessing_#"
            # Progress observing process by model subdirs
            observer = DirProgressObserver(
                root=model_subdir_root,
                total=n_chains,
                pattern=model_subdir_pattern,
                progress_prefix="Model evaluation",
            )
            observer.start()

        # Modeling by chain sample_list
        if not modeling_finished:
            # Sequential processing modeling
            if n_processor <= 1:
                lock = _DummyLock()
                for pci, cdp in enumerate(rest_cd_paths):
                    try:
                        # Progress
                        if show_progress:
                            print(f"\nModeling preprocessing result {pci + 1}/{len(rest_cd_paths)} :")
                        pc_it = load_vars(unc_path(cdp))
                        pc_sample_list = pc_it["chain_res"]
                        pc_sample_list = _target_type_validation_for_serialization(pc_sample_list)
                        pchain = pc_it["chain_procs"]
                        # Use preprocess chain ID as chain label
                        pproc_chain_label = [f"Preprocessing_#{pci}" for pci, pc in enumerate(pchains) if pc == pchain][
                            0
                        ]  # noqa: E501
                        _model_evaluator(
                            preprocess_result=pc_sample_list,
                            preprocess_chain=pchain,
                            preprocess_chain_label=pproc_chain_label,
                            model_processes=model_procs,
                            specpipe_report_directory=self.report_directory,
                            result_directory=result_directory,
                            lock=lock,
                            update_progress_log=True,
                        )

                    # Error handling
                    except Exception as e:
                        # Validate report directory
                        model_result_dir = result_directory + "Modeling/"
                        os.makedirs(unc_path(model_result_dir), exist_ok=True)
                        errdir = model_result_dir + "Error_logs/"
                        os.makedirs(unc_path(errdir), exist_ok=True)
                        cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        pid = os.getpid()
                        error_log_path = errdir + f"error_{cts}_pid_{pid}.log"
                        err_msg = f"\nFailed in the modeling of preprocessing chain from path '{cdp}', \
                                    error message: \n\n{str(e)}\n"
                        with open(unc_path(error_log_path), "w") as f:
                            f.write(err_msg)
                        raise ValueError(f"\nFailed in the modeling of preprocessing chain from path '{cdp}'") from e
            # Multiprocessing modeling
            else:
                lock = mp.Manager().Lock()
                # Initialize errorlogs dir for err handling in parallel computing
                errorlog_path = model_result_dir + "Error_logs/"
                if os.path.exists(unc_path(errorlog_path)):
                    shutil.rmtree(unc_path(errorlog_path))
                # Validate number of processors to use
                ncpu_max = max(cpu_count() - 1, 1)
                ncpu = min(n_processor, ncpu_max)
                # Bind constant arguments for _model_evaluator_it
                _model_evaluator_it = partial(
                    _model_evaluator_mp,
                    pchains=pchains,
                    model_processes=model_procs,
                    specpipe_report_directory=self.report_directory,
                    result_directory=result_directory,
                    lock=lock,
                    update_progress_log=True,
                    # Assign applied functions
                    _model_evaluator=_model_evaluator,
                    _dl_val=_dl_val,
                    unc_path=unc_path,
                    load_vars=load_vars,
                    dump_vars=dump_vars,
                    _target_type_validation_for_serialization=_target_type_validation_for_serialization,
                    modeleva=ModelEva,
                    silent_all=True,
                )
                # Processing - multiprocessing for loop
                with ProcessingPool(nodes=ncpu) as pool:
                    for _ in pool.imap(_model_evaluator_it, rest_cd_paths):
                        pass
                # Collect and print errors if exist
                if os.path.exists(unc_path(errorlog_path)):
                    raise ValueError(
                        f"\nPreprocessing errors, please check error logs in the following path:\n{errorlog_path}"
                    )

            # Complete modeling progress monitoring
            open(unc_path(model_subdir_root + ".__finished.s"), "w").close()
            if n_processor > 1:
                observer.join()

        # Print result dir and save corresponding targets
        if len(result_directory) > 0:
            result_dir = str(result_directory)
        else:
            result_dir = self.report_directory
        self._save_sample_targets(result_dir)
        print(f"\n\nPipeline model evaluation complete, result stored in:\n{result_dir}")

        # Performance summary and marginal performance stats
        if summary:
            pipeline_config_dir = f"{self.report_directory}/SpecPipe_configuration/"
            model_evaluation_report_dir = f"{self.report_directory}/Modeling/Model_evaluation_reports/"
            metrics_dict = performance_metrics_summary(pipeline_config_dir, model_evaluation_report_dir)
            _ = performance_marginal_stats(
                report_directory=self.report_directory,
                metrics_dict=metrics_dict,
            )
            _ = combined_model_marginal_stats(
                report_directory=self.report_directory,
                metrics_dict=metrics_dict,
            )

        # Store computation status
        open(finish_status_path, "w").close()

    @staticmethod
    def _clear_model_log(log_path: str) -> None:
        # Clear log when finished
        if os.path.exists(unc_path(log_path)):
            try:
                os.remove(unc_path(log_path))
            except PermissionError as e:
                raise PermissionError(f"\nNo permission to clear existed running log : \n'{log_path}'.\n") from e
            except Exception as e:
                raise ValueError("\nError in clearing existed running log\n") from e

    def _save_sample_targets(self, result_dir: str) -> None:
        # Save dir
        save_dir = f"{result_dir}/Modeling/".replace("//", "/")
        # Validate save dir
        if os.path.exists(unc_path(save_dir)):
            self.spec_exp.sample_targets_to_csv(unc_path(save_dir + "sample_targets.csv"))
        else:
            os.makedirs(unc_path(save_dir))
            self.spec_exp.sample_targets_to_csv(unc_path(save_dir + "sample_targets.csv"))
            # raise ValueError(f"Invalid modeling result directory: {save_dir}")

    @simple_type_validator
    def run(  # noqa: C901
        self,
        result_directory: str = "",
        n_processor: int = -1,
        num_type: Union[str, type] = np.float32,
        test_model: bool = True,
        model_parallel: bool = True,
        dump_backup: bool = False,
        step_result: bool = True,
        resume: bool = False,
        sample_data_to_csv: bool = True,
        show_progress: bool = True,
        save_config: bool = True,
        summary: bool = True,
        geo_reference_warning: bool = False,
        model_test_coverage: float = 1.0,
        skip_test: bool = False,
        check_space: bool = True,
    ) -> None:
        """
        Run entire pipelines of specified processes of this ``SpecPipe`` instance on provided ``SpecExp`` instance.
        Processes are configured using method ``add_process`` and ``add_model``.

        Parameters
        ----------
        result_directory : str, optional
            Directory to save preprocessing and model evaluation reports.
            Default is the ``report_directory`` of the input ``SpecExp`` instance of this ``SpecPipe`` instance.

        n_processor : int, optional
            Number of processors to use during pipeline execution.

            Default is -1, which does not apply parallel execution on Windows and applies parallel execution using (maximum available CPUs - 1) processors on other operating systems.

            Set to -2 to force (maximum available CPUs - 1) processors on Windows.

            Windows note: when using ``n_processor > 1`` or ``n_processor = -2`` on Windows, all excecutable code in the working script must be placed within::

                if __name__ == '__main__':

        num_type: str or type, optional
            Numeric data type for array-like data storage, supporting numeric ``numpy`` data types.
            Default is ``numpy.float32``.

        test_model : bool, optional
            Whether to test added models before formal execution.

            If False, model testing is skipped.
            Tests use minimal sample sizes, which may cause errors for some models.
            Default is True.

        model_parallel : bool, optional
            Whether to enable pipeline-level parallelism during modeling.

            Set to False when the modeling method already uses multiprocessing or GPU acceleration to avoid nested parallel execution.
            Default is True.

        dump_backup : bool, optional
            Whether to create timestamped backup files of results.
            Default is False.

        step_result : bool, optional
            Whether to retain intermediate step results for each processing chain.

            If False, intermediate results are removed after preprocessing.
            Default is True.

        resume : bool, optional
            Whether to resume execution from the last saved preprocessing checkpoint.

            This avoids redundant processing after interruptions.
            Default is False.

        sample_data_to_csv : bool, optional
            Whether to additionally save preprocessed sample data as CSV files.
            Default is True.

        show_progress : bool, optional
            Whether to display execution progress.
            Default is True.

        save_config : bool, optional
            Whether to save SpecPipe configuration files.
            Default is True.

        summary : bool, optional
            Whether to summarize preprocessed data, performance metrics, and marginal performance metrics.
            Marginal performance at each step is compared using the Mann–Whitney U test.
            Default is True.

        geo_reference_warning : bool, optional
            Whether to suppress GeoReferenceWarning messages.
            If False, warnings are suppressed.
            Default is False.

        skip_test : bool, optional
            Whether to skip test execution entirely.
            Test execution validates all processing chains and serves as a safeguard against runtime errors during long executions.
            Default is False.

        check_space : bool, optional
            Whether to validate available disk space against the estimated output size.
            If True, an error is raised when the estimate exceeds the available space.
            If False, a warning is issued instead.
            Default is True.

        See Also
        --------
        preprocessing
        model_evaluation

        Examples
        --------
        For a prepared ``SpecPipe`` instance ``pipe``::

            >>> pipe.run()

        Pipeline-level multiprocessing::

            >>> pipe.run(n_processor=10)

        Automatically determine CPU usage::

            >>> pipe.run(n_processor=-1)

        Windows multiprocessing::

            >>> if __name__ == '__main__':
            ...     pipe.run(n_processor=10)
            >>> if __name__ == '__main__':
            ...     pipe.run(n_processor=-2)
        """  # noqa: E501

        # Validate disk space for output
        self._val_disk_space_preprocessing(image=True, plot=True, error_raise=check_space)

        # Validate processor
        if n_processor < 0:
            if os.name == "nt" and n_processor != -2:
                n_processor = 1
            else:
                n_processor = max(1, cpu_count() - 1)

        # Prompt "if __name__ == '__main__':" protection for windows multiprocessing
        if n_processor > 1:
            if os.name == "nt":
                print(
                    "\n\n\n\
                    >>> WARNING: Windows users must run multiprocessing within block \n\nif __name__ == '__main__':\
                    \n\n Please make sure all of your main codes in the script are placed within this block. \n\n",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(2)
                warnings.warn(
                    "Windows users must run multiprocessing within block \n\nif __name__ == '__main__': \n\n\
                    Please make sure all of your main codes in the script are placed within this block.",
                    UserWarning,
                    stacklevel=2,
                )

        # Suppress NotGeoreferencedWarning
        if not geo_reference_warning:
            warnings.simplefilter("ignore", NotGeoreferencedWarning)

        # Save configurations
        if save_config:
            self.save_pipe_config(copy=True)

        # Test process
        if not skip_test:
            print("\n========= Test added chains =========\n")
            self.test_run(test_modeling=test_model, model_test_coverage=model_test_coverage)

        # Preprocessing
        print("\n========= Preprocessing samples =========\n")
        self.preprocessing(
            result_directory=result_directory,
            n_processor=n_processor,
            num_type=num_type,
            dump_backup=dump_backup,
            step_result=step_result,
            resume=resume,
            to_csv=sample_data_to_csv,
            show_progress=show_progress,
            save_config=False,
            summary=summary,
            geo_reference_warning=geo_reference_warning,
            skip_test=skip_test,
            check_space=check_space,
        )

        # Model evaluation
        model_procs = [proc[0] for proc in self.process if proc[3] == "model"]
        if len(model_procs) > 0:
            if model_parallel:
                ncpu_model = n_processor
            else:
                ncpu_model = 1
            print("\n\n========= Model evaluation =========\n")
            self.model_evaluation(
                n_processor=ncpu_model,
                resume=resume,
                report_directory=result_directory,
                show_progress=show_progress,
                save_config=False,
                summary=summary,
                check_space=check_space,
            )
        else:
            print("\nNo model added, pipeline complete with preprocessing results.\n")

        # Print result dir and save corresponding targets
        if len(result_directory) > 0:
            result_dir = str(result_directory)
        else:
            result_dir = self.report_directory
        self._save_sample_targets(result_dir)
        print(f"\n\nPipeline running complete, result stored in:\n{result_dir}")

        warnings.simplefilter("default", NotGeoreferencedWarning)

    # For get results in console
    def report_summary(self) -> dict:
        """
        Retrieve summary of generated reports in the console.
        The summary includes performance summary and marginal performances among added processes of each pipeline step.

        Returns
        -------
        dict
            A dictionary of pandas DataFrames with the following contents:

            For regression:

            - Performance summary.
            - Marginal R2 of the steps with multiple processes.

            For classification:

            - Macro- and micro-average performance summary.
            - Marginal macro- and micro-average AUC of the steps with multiple processes.

        Examples
        --------
        For ``SpecPipe`` instance ``pipe`` after running::

            >>> result_summary = pipe.report_summary()
        """  # noqa: E501

        # Validate pipeline running completion
        if os.path.exists(unc_path(self.report_directory + "Modeling/sample_targets_stats.csv")):
            result = group_stats_report(self.report_directory)
            assert isinstance(result, dict)
            return result
        else:
            print("Unfinished or incomplete pipeline running results. Cannot retrieve summary reports.")
            return {}

    # For get results in console
    def report_chains(self) -> list[dict]:
        """
        Retrieve major model evaluation reports of every processing chain in the console.

        Returns
        -------
        list of dict
            Each dictionary contains the reports of one processing chain.

            For regression pipelines, the reports include:

            - Processes of the chain
            - Validation results
            - Performance metrics
            - Residual analysis
            - Influence analysis (if available)
            - Scatter plot
            - Residual plot

            For classification pipelines, the reports include:

            - Processes of the chain
            - Validation results
            - Performance metrics
            - Residual analysis
            - Influence analysis (if available)
            - ROC curves

        Examples
        --------
        For ``SpecPipe`` instance ``pipe`` after running::

            >>> chain_results = pipe.report_chains()
        """
        # Validate pipeline running completion
        if os.path.exists(unc_path(self.report_directory + "Modeling/sample_targets_stats.csv")):
            result = core_chain_report(self.report_directory)
            assert isinstance(result, list)
            return result
        else:
            print("Unfinished or incomplete pipeline running results. Cannot retrieve chain reports.")
            return []
