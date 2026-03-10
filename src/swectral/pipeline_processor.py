# -*- coding: utf-8 -*-
"""
Swectral - Pipeline iterators, processors and other helpers for spectral image preprocessing and modeling

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os

# Warning
import warnings

# Interface
from tqdm import tqdm

# Typing
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    ContextManager,
    overload,
    Iterable,
)
from types import ModuleType

# Time
import time
from datetime import datetime

# Basic data
import copy
import numpy as np
import pandas as pd
import torch

# Local
from .modeleva import ModelEva
from .rasterop import pixel_apply
from .specio import (
    arraylike_validator,
    dump_dill,
    load_dill,
    simple_type_validator,
    unc_path,
    _wait_for_free_space,
    df_to_csv,
)
from .pipeline_validator import (
    _target_type_validation_for_serialization,
    _dl_val,
)

# For multiprocessing
global ModelEva


# %% DummyManager to replace multiproccessing manager of pathos for single processing


class _DummyLock(ContextManager[None]):
    """Lock for _DummyManager."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        return False  # do nothing


class _DummyManager:
    """Pass-through manager that mimics pathos.helpers.Manager() for single processing."""

    @staticmethod
    def Lock() -> ContextManager[None]:  # noqa: N802
        """Return lock."""
        return _DummyLock()

    class Value:
        """Simple value holder to mimic Manager().Value."""

        def __init__(self, typecode: str, value: Any) -> None:
            self.value: Any = value

    @staticmethod
    def list() -> list[Any]:
        """Simple list to mimic Manager().list()."""
        return []


# %% Static functions for SpecPipe


# Preprocessing of single sample using all chains
@overload
def _preprocessing_sample(
    sample_data: dict,
    process: list[tuple[str, str, str, str, int, Any, int, int]],
    custom_chains: list,
    process_chains: list,
    specpipe_report_directory: str,
    preprocess_status: dict,
    num_type: Union[str, type],
    *,
    dump_result: Literal[True] = True,
    return_result_path: Literal[True] = True,
    dump_backup: bool = False,
    return_step_result: Literal[False] = False,
    final_result_only: bool = True,
    is_test_run: bool = False,
    dump_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    # Explicitly load function for multiprocessing
    _dl_val: Callable = _dl_val,
    pixel_apply: Callable = pixel_apply,
    dump_dill: Callable = dump_dill,
    unc_path: Callable = unc_path,
    # Dependencies for multiprocessing
    copy: ModuleType = copy,
    os: ModuleType = os,
    time: ModuleType = time,
    datetime: type = datetime,
    np: ModuleType = np,
    torch: ModuleType = torch,
) -> str: ...


@overload
def _preprocessing_sample(
    sample_data: dict,
    process: list[tuple[str, str, str, str, int, Any, int, int]],
    custom_chains: list,
    process_chains: list,
    specpipe_report_directory: str,
    preprocess_status: dict,
    num_type: Union[str, type],
    *,
    dump_result: Literal[True] = True,
    return_result_path: Literal[True] = True,
    dump_backup: bool = False,
    return_step_result: Literal[True] = True,
    final_result_only: bool = True,
    is_test_run: bool = False,
    dump_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    # Explicitly load function for multiprocessing
    _dl_val: Callable = _dl_val,
    pixel_apply: Callable = pixel_apply,
    dump_dill: Callable = dump_dill,
    unc_path: Callable = unc_path,
    # Dependencies for multiprocessing
    copy: ModuleType = copy,
    os: ModuleType = os,
    time: ModuleType = time,
    datetime: type = datetime,
    np: ModuleType = np,
    torch: ModuleType = torch,
) -> tuple[str, dict]: ...


@overload
def _preprocessing_sample(
    sample_data: dict,
    process: list[tuple[str, str, str, str, int, Any, int, int]],
    custom_chains: list,
    process_chains: list,
    specpipe_report_directory: str,
    preprocess_status: dict,
    num_type: Union[str, type],
    *,
    dump_result: Literal[True] = True,
    return_result_path: Literal[False] = False,
    dump_backup: bool = False,
    return_step_result: Literal[True] = True,
    final_result_only: bool = True,
    is_test_run: bool = False,
    dump_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    # Explicitly load function for multiprocessing
    _dl_val: Callable = _dl_val,
    pixel_apply: Callable = pixel_apply,
    dump_dill: Callable = dump_dill,
    unc_path: Callable = unc_path,
    # Dependencies for multiprocessing
    copy: ModuleType = copy,
    os: ModuleType = os,
    time: ModuleType = time,
    datetime: type = datetime,
    np: ModuleType = np,
    torch: ModuleType = torch,
) -> dict: ...


@overload
def _preprocessing_sample(
    sample_data: dict,
    process: list[tuple[str, str, str, str, int, Any, int, int]],
    custom_chains: list,
    process_chains: list,
    specpipe_report_directory: str,
    preprocess_status: dict,
    num_type: Union[str, type],
    *,
    dump_result: Literal[True] = True,
    return_result_path: Literal[False] = False,
    dump_backup: bool = False,
    return_step_result: Literal[False] = False,
    final_result_only: bool = True,
    is_test_run: bool = False,
    dump_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    # Explicitly load function for multiprocessing
    _dl_val: Callable = _dl_val,
    pixel_apply: Callable = pixel_apply,
    dump_dill: Callable = dump_dill,
    unc_path: Callable = unc_path,
    # Dependencies for multiprocessing
    copy: ModuleType = copy,
    os: ModuleType = os,
    time: ModuleType = time,
    datetime: type = datetime,
    np: ModuleType = np,
    torch: ModuleType = torch,
) -> None: ...


@overload
def _preprocessing_sample(
    sample_data: dict,
    process: list[tuple[str, str, str, str, int, Any, int, int]],
    custom_chains: list,
    process_chains: list,
    specpipe_report_directory: str,
    preprocess_status: dict,
    num_type: Union[str, type],
    *,
    dump_result: Literal[False] = False,
    return_result_path: bool = True,
    dump_backup: bool = False,
    return_step_result: bool = False,
    final_result_only: bool = True,
    is_test_run: bool = False,
    dump_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    # Explicitly load function for multiprocessing
    _dl_val: Callable = _dl_val,
    pixel_apply: Callable = pixel_apply,
    dump_dill: Callable = dump_dill,
    unc_path: Callable = unc_path,
    # Dependencies for multiprocessing
    copy: ModuleType = copy,
    os: ModuleType = os,
    time: ModuleType = time,
    datetime: type = datetime,
    np: ModuleType = np,
    torch: ModuleType = torch,
) -> dict: ...


# Preprocessing of single sample using all chains
# SpecPipe attributes - argumente relation:
#     sample_data = SpecPipe.sample_data | SpecPipe.pretest_data
#     process = SpecPipe.process
#     custom_chains = SpecPipe.custom_chains
#     process_chains = SpecPipe.process_chains
#     specpipe_report_directory = SpecPipe.spec_exp.report_directory
# Sample data format - ROI: {ID, label, target, img_path, roi_coords}
# Sample data format - standalone spec: {ID, label, target, spec1d: tuple}
# Sample data format - test: {img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d}
@simple_type_validator
def _preprocessing_sample(  # noqa: C901
    sample_data: dict,
    process: list[tuple[str, str, str, str, int, Any, int, int]],
    custom_chains: list,
    process_chains: list,
    specpipe_report_directory: str,
    preprocess_status: dict,
    num_type: Union[str, type],
    *,
    dump_result: bool = True,
    return_result_path: bool = True,
    dump_backup: bool = False,
    return_step_result: bool = False,
    final_result_only: bool = True,
    is_test_run: bool = False,
    dump_directory: str = "",
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    # Explicitly load function for multiprocessing
    _dl_val: Callable = _dl_val,
    pixel_apply: Callable = pixel_apply,
    dump_dill: Callable = dump_dill,
    unc_path: Callable = unc_path,
    # Dependencies for multiprocessing
    copy: ModuleType = copy,
    os: ModuleType = os,
    time: ModuleType = time,
    datetime: type = datetime,
    np: ModuleType = np,
    torch: ModuleType = torch,
) -> Union[str, dict, tuple[str, dict], None]:
    """
    Preprocessing of single sample using all chains of SpecPipe.

    Parameters
    ----------
    sample_data : dict
        'sample_data' or 'pretest_data' of SpecPipe.

    process : list[tuple[str, str, str, str, int, Callable, int, int]]
        'process' of SpecPipe.

    custom_chains : list
        'custom_chains' of SpecPipe.

    process_chains : list
        'process_chains' of SpecPipe.

    specpipe_report_directory : str
        'report_directory' of SpecExp specified for SpecPipe.

    dump_result : bool, optional
        Whether result is dumped to dill file. The default is True.

    return_result_path : bool, optional
        Whether the result file path is returned. The default is True.

    dump_backup : bool, optional
        Whether backup for the step result is dumped. The backup file is named with the datetime of dumping.

    return_step_result : bool, optional
        Whether the step result is returned. The default is False. If both path and step_result are to be returned, (path, step_result) is returned.

    final_result_only : bool, optional
        Whether only the final result of chains is dumped, if True, only the result from the last preprocessing step is dumped.
        The default is True.

    is_test_run : bool, optional
        True if test data is applied for chain testing. The default is False.

    dump_directory : str, optional
        If '', default directory in specpipe_report_directory will be used as result file directory, or this directory is used for resulting files. The default is ''.

    update_progress_log : bool = False
        Whether to update progress log files, use to enable resume. The default is False.

    num_type : str or type
        Data type for array-like data storage, supports data types supported by numpy.
    """  # noqa: E501
    try:
        # Resume testing
        # Resume testing - initial break status
        env_preprocess_resume_test_num = int(os.getenv("SPECPIPE_PREPROCESS_RESUME_TEST_NUM", "-1"))
        if env_preprocess_resume_test_num > 0 and not is_test_run:
            with preprocess_status['lock']:
                preprocess_resume_test_num = preprocess_status['preprocess_resume_test_num']
                if env_preprocess_resume_test_num > int(preprocess_resume_test_num.value):
                    preprocess_resume_test_num.value = env_preprocess_resume_test_num
                # Resume testing - conditional break
                if preprocess_resume_test_num.value > 1:
                    raise ValueError("Preprocessing resume test raise")
                # Resume testing - status update
                preprocess_resume_test_num.value = preprocess_resume_test_num.value + 1
                os.environ["SPECPIPE_PREPROCESS_RESUME_TEST_NUM"] = str(preprocess_resume_test_num.value)

        # Validate sample data label
        if (sample_data["label"] == "") or (sample_data["label"] == "-"):
            sample_data_label = sample_data["ID"]
        else:
            sample_data_label = sample_data["label"]

        # Get methods
        methods = np.array(process)

        # Get testing chain
        if len(custom_chains) > 0:
            chains = custom_chains
        elif len(process_chains) > 0:
            chains = process_chains
        else:
            raise ValueError("\nNo process added")

        # Validate chains
        chain_length = len(chains[0])
        for chain in chains:
            if len(chain) != chain_length:
                raise ValueError(
                    f"Inconsistent steps of processing in chain: {chain} \
                        \nExpected number of steps: {chain_length}, got: {len(chain)}"
                )

        # Chains
        # [(process 1 ID of step 1, process 1 ID of step 2,...), (process 2 ID of step 1, process 1 ID of step 2,...), ...]  # noqa: E501
        # Status vector: Check previous steps,
        # once identical or previous process completed, avoid computing repeatly but use the previous result.
        # [[Step1:[preceding processes 1],[preceding processes 2],...],[Step2:...],...]

        # TODO: removed
        # model_ids = [pit[0] for pit in process if pit[3] == "model"]
        # if len(model_ids) > 0:
        #     n_non_preprocess_step = 1
        # else:
        #     n_non_preprocess_step = 0
        # TODO: new
        model_ids = [pit[0] for pit in process if pit[3] == "model"]
        assembly_ids = [pit[0] for pit in process if pit[3] == "assembly"]

        # Number of preprocessing steps
        # TODO: removed
        # preprocess_chain_length = len(chains[0]) - n_non_preprocess_step
        # TODO: new
        n_non_preprocess_step = len(
            [proc_id for proc_id in chains[0] if (proc_id in assembly_ids) or (proc_id in model_ids)]
        )
        preprocess_chain_length = len(chains[0]) - n_non_preprocess_step

        if preprocess_chain_length < 1:
            raise ValueError("No preprocessing process found.")
        calc_status: list[list] = [[] for _ in range(preprocess_chain_length)]
        status_results: list[list] = [[] for _ in range(preprocess_chain_length)]

        # Validate preprocessed image dir
        # Preprocessed image dir for data level 0~4
        # Formal run
        if not is_test_run:
            preprocessed_img_dir = specpipe_report_directory + "Preprocessing/Preprocessed_images/"
            if not os.path.exists(unc_path(preprocessed_img_dir)):
                raise ValueError(f"Invalid preprocessed image directory path: {preprocessed_img_dir}")
        # Test run
        else:
            preprocessed_img_dir = specpipe_report_directory + "test_run/Preprocessed_images/"
            os.makedirs(unc_path(preprocessed_img_dir), exist_ok=True)

        # Validate step result data dir
        if is_test_run:
            dir_name = "test_run"
        else:
            dir_name = "Preprocessing"
        # Step_results dir path
        if len(dump_directory) > 0:
            sdir = dump_directory
        else:
            sdir = specpipe_report_directory + f"{dir_name}/Step_results/"
        os.makedirs(unc_path(sdir), exist_ok=True)
        # Intermediate step_results dir path
        inter_sdir = sdir + "Intermediate_step_results/"
        os.makedirs(unc_path(inter_sdir), exist_ok=True)

        # Implement processing pipeline for every chain of chains
        status_results = _chain_step_processor(
            sample_data_label=sample_data_label,
            chains=chains,
            n_non_preprocess_step=n_non_preprocess_step,
            calc_status=calc_status,
            methods=methods,
            sample_data=sample_data,
            preprocessed_img_dir=preprocessed_img_dir,
            inter_sdir=inter_sdir,
            status_results=status_results,
            preprocess_status=preprocess_status,
            final_result_only=final_result_only,
            num_type=num_type,
            space_wait_timeout=space_wait_timeout,
            reserve_free_pct=reserve_free_pct,
        )

        # Collect test preprocessing results of current chain (chain i)
        if final_result_only:
            status_results = status_results[-1:]
        status_results_out = {
            "ID": sample_data["ID"],
            "label": sample_data_label,  # Dataset label
            "target": sample_data["target"],
            "sample_label": sample_data["label"],
            "validation_group": sample_data["validation_group"],
            "test": sample_data["test"],
            "train": sample_data["train"],
            "status_results": status_results,
        }

        # Dump step final results
        if is_test_run:
            file_name = "PreprocessingTestingResult"
        else:
            file_name = f"PreprocessingResult_sample_{sample_data_label}"
        if dump_result:
            chain_result_path = sdir + f"{file_name}.dill"
            # TODO: changed
            dump_dill(
                status_results_out,
                target_file_path=unc_path(chain_result_path),
                backup=dump_backup,
                space_wait_timeout=space_wait_timeout,
                reserve_free_pct=reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )

        # Update progress
        step_dir = specpipe_report_directory + "Preprocessing/Step_results/"
        log_dir_path = step_dir + "Preprocess_progress_logs/"
        log_fp = log_dir_path + sample_data["ID"]
        if update_progress_log:
            os.makedirs(unc_path(log_dir_path), exist_ok=True)
            try:
                with open(unc_path(log_fp), "x") as f:  # Write if not exist
                    f.write("")
            except FileExistsError:
                pass

        # Return result file path and step results if required
        if dump_result:
            # Return dumped file path and step results
            if return_result_path & return_step_result:
                return chain_result_path, status_results_out
            elif return_result_path & (not return_step_result):
                return chain_result_path
            elif (not return_result_path) & return_step_result:
                return status_results_out
            else:
                return None
        # Return step results only
        else:
            return status_results_out
            if not return_step_result:
                raise warnings.warn(
                    "When dump_result is False, \
                        the result is always returned and the return_step_result argument is ignored.",
                    UserWarning,
                    stacklevel=3,
                )

    # Error handling
    except Exception as e:
        # Log directory
        if is_test_run:
            dir_name = "test_run"
        else:
            dir_name = "Preprocessing"
        errdir = specpipe_report_directory + f"{dir_name}/Step_results/Error_logs/"
        os.makedirs(unc_path(errdir), exist_ok=True)
        assert hasattr(datetime, 'now')
        cts: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # PID for multiprocessing
        pid = os.getpid()
        error_log_path = errdir + f"error_{cts}_pid_{pid}.log"
        # Validate sample data label
        if (sample_data["label"] == "") or (sample_data["label"] == "-"):
            sample_data_label = sample_data["ID"]
        else:
            sample_data_label = sample_data["label"]
        # Write error log
        err_msg = f"Failed in the preprocessing of '{sample_data_label}', error message: \n\n{str(e)}"
        with open(unc_path(error_log_path), "w") as f:
            f.write(err_msg)
        raise ValueError(f"Failed in the preprocessing of '{sample_data_label}'") from e


# Chain step iterating processor
def _chain_step_processor(  # noqa: C901
    sample_data_label: str,
    chains: list,
    n_non_preprocess_step: int,
    calc_status: list[list],  # empty data container
    methods: np.ndarray,
    sample_data: dict,
    preprocessed_img_dir: str,
    inter_sdir: str,
    status_results: list[list],  # empty data container
    preprocess_status: dict,
    final_result_only: bool,
    num_type: Union[str, type],
    *,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    # Dependencies for multiprocessing
    copy: ModuleType = copy,
    os: ModuleType = os,
    time: ModuleType = time,
    datetime: type = datetime,
    np: ModuleType = np,
    torch: ModuleType = torch,
) -> list[list]:
    """Iterates the chains and steps to perform corresponding processing."""
    n_chains = len(chains)
    n_step = len(chains[0]) - n_non_preprocess_step

    # Chain-step results and its processes as result IDs
    chain_results_per_chain: list[list] = [[] for _ in range(n_chains)]
    step_procs_per_chain: list[list] = [[] for _ in range(n_chains)]

    # For every step exclude modeling step
    for stepi in range(n_step):

        # Chain and step loop
        for chain_ind, chain in enumerate(chains):

            # Step process ID
            step = chain[stepi]

            # Load existed result and process
            chain_result = chain_results_per_chain[chain_ind]
            step_procs = step_procs_per_chain[chain_ind]

            # Create new step_procs and sample result for the step
            # Shallow copy to continue local append safely
            step_procs_local = step_procs.copy()
            # Note down processes of every step in a chain
            step_procs_local.append(step)
            # Convert to tuple for immutability & hashability and avoid repeated deepcopy for memory efficiency
            step_procs_key = tuple(step_procs_local)

            # Reuse computed result to avoid repeating calculation
            if step_procs_key in calc_status[stepi]:
                chain_result = [srt[4] for srt in status_results[stepi] if srt[1] == step_procs_key]
            # New step calculation
            else:
                # Get method and method info
                # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
                method_item = methods[methods[:, 0] == step, :][0]
                dl_in = _dl_val(method_item[2])[0]
                dl_out = _dl_val(method_item[3])[0]
                method_func = method_item[5]
                assert callable(method_func)

                # Get input data of the step
                # Pretest_data: [img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d]
                if len(chain_result) == 0:
                    if dl_in <= 4:
                        step_input_data = sample_data["img_path"]
                    elif dl_in == 5:
                        step_input_data = sample_data["img_path"]
                    elif dl_in == 6:
                        step_input_data = sample_data["roitable"]
                    elif dl_in == 7:
                        step_input_data = sample_data["spec1d"]
                else:
                    step_input_data = chain_result[-1]
                roi_coords = sample_data["roi_coords"]

                # Preprocessing computing
                try:
                    status_results, calc_status = _single_process_handler(
                        sample_data_label=sample_data_label,
                        dl_in=dl_in,
                        chain_result=chain_result,
                        method_func=method_func,
                        step_input_data=step_input_data,
                        roi_coords=roi_coords,
                        step_procs_key=step_procs_key,
                        stepi=stepi,
                        chain_ind=chain_ind,
                        dl_out=dl_out,
                        preprocessed_img_dir=preprocessed_img_dir,
                        inter_sdir=inter_sdir,
                        status_results=status_results,
                        calc_status=calc_status,
                        preprocess_status=preprocess_status,
                        num_type=num_type,
                        space_wait_timeout=space_wait_timeout,
                        reserve_free_pct=reserve_free_pct,
                    )
                except Exception as e:
                    method_item_tuple = tuple(method_item)
                    method_item_out = (
                        method_item_tuple[1:5] + (method_item_tuple[5].__class__.__name__,) + method_item_tuple[6:8]
                    )
                    raise ValueError(
                        f"\nTest failed for chain: \nChain index: {chain_ind}, \nChain: {chain};\
                            \n\nProcess ID: {step}, \nProcess item: {method_item_out}\n\n"
                    ) from e

            # Update step_procs reference for next iteration
            step_procs = step_procs_local

            # Update result list and key list
            chain_results_per_chain[chain_ind] = chain_result
            step_procs_per_chain[chain_ind] = step_procs

        # Prune results if final_result_only
        if final_result_only and stepi >= 1:

            # Step ID to prune
            step_to_prune = stepi - 1

            for idx, (step_id, step_procs_key, dl_in, dl_out, sample_result) in enumerate(
                status_results[step_to_prune]
            ):
                if sample_result is not None:
                    # Remove the intermediate results
                    if isinstance(sample_result, dict) and sample_result.get("__disk_backed__"):
                        sample_result_path = sample_result["path"]
                        if os.path.exists(sample_result_path):
                            os.remove(sample_result_path)
                    # Remove the intermediate result handle
                    status_results[step_to_prune][idx] = (step_id, step_procs_key, dl_in, dl_out, None)

    return status_results


def _dump_disk_backed_data(
    num_result: object,
    data_path: str,
    num_type: Union[str, type] = np.float32,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
) -> dict:
    """Dump disk backed data according to given path and return the handle. data_path must have no extension."""
    data_path = os.path.splitext(data_path)[0]
    # Dump to npy if array-like
    try:
        num_result = arraylike_validator()(num_result)
        assert isinstance(num_result, np.ndarray)
        num_result = num_result.astype(num_type)
        data_path_ext = unc_path(data_path + ".npy")
        loader_type = "numpy"
        # Disk space safeguard
        _wait_for_free_space(
            obj=num_result,
            path=data_path_ext,
            space_wait_timeout=space_wait_timeout,
            reserve_free_pct=reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
            obj_size_buffer_coeff=1.1,
        )
        np.save(data_path_ext, num_result)
    # Else to dill
    except Exception:
        # TODO: data_path_ext = unc_path(data_path + ".dill")
        loader_type = "dill"
        # Dill dump
        # TODO: dump_dill(data_path_ext, {"data": num_result}, backup=False)
        dump_dill(
            {"data": num_result},
            target_file_path=unc_path(data_path_ext),
            backup=False,
            space_wait_timeout=space_wait_timeout,
            reserve_free_pct=reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )
    # Return handle
    return {"__disk_backed__": True, "path": data_path_ext, "loader": loader_type}


def _load_disk_backed_data(handle_obj: object) -> object:
    """Load disk backed data using defined handle object."""
    if isinstance(handle_obj, dict) and handle_obj.get("__disk_backed__"):
        if "path" in handle_obj.keys():
            if not os.path.exists(handle_obj["path"]):
                raise FileNotFoundError(f"Data file not found, expected path:\n{handle_obj['path']}")
        else:
            raise ValueError("handle 'path' is undefined, cannot load data.")
        if "loader" in handle_obj.keys():
            loader_type = handle_obj.get("loader")
            if loader_type == "numpy":
                return np.load(handle_obj["path"], allow_pickle=False)
            elif loader_type == "dill":
                return load_dill(handle_obj["path"])["data"]
            else:
                raise ValueError(f"handle loader type must be 'numpy' or 'dill', got: {loader_type}")
        else:
            raise ValueError("handle 'loader' is undefined, cannot load data.")
    return handle_obj


# Single process handler
def _single_process_handler(
    sample_data_label: str,
    dl_in: int,
    chain_result: list,
    method_func: Callable,
    step_input_data: object,
    roi_coords: list[list[tuple[Union[int, float], Union[int, float]]]],
    step_procs_key: tuple,
    stepi: int,
    chain_ind: int,
    dl_out: int,
    preprocessed_img_dir: str,
    inter_sdir: str,
    status_results: list[list],
    calc_status: list[list],
    preprocess_status: dict,
    num_type: Union[str, type],
    *,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    # Dependencies for multiprocessing
    copy: ModuleType = copy,
    os: ModuleType = os,
    time: ModuleType = time,
    datetime: type = datetime,
    np: ModuleType = np,
    torch: ModuleType = torch,
) -> tuple[list[list], list[list]]:
    """Single process handler to allocate method wrapper according to data levels."""
    # Status_result: (0 - step_id, 1 step_procs, 2 dl_in, 3 dl_out, 4 sample_result)
    # Step_id as str of procs id
    step_id = ""
    for proc_id in step_procs_key:
        step_id = step_id + "p_" + str(proc_id) + "-"
    # Apply the step process function
    # ============ Image processing ============
    # Create files for image processing computation start and completion status
    if dl_in <= 4:
        step_input_data = str(step_input_data)
        # Image processing
        processed_image_path = _image_processing_step(
            dl_in=dl_in,
            preprocessed_img_dir=preprocessed_img_dir,
            input_image_path=step_input_data,
            method_func=method_func,
            preprocess_status=preprocess_status,
        )
        chain_result.append(processed_image_path)
    # ============ ROI data extraction ============
    elif dl_in == 5:
        # Compute - accepts image path and ROI coords as input
        num_result = method_func(step_input_data, roi_coords)
        # Dump result
        data_path = f"{inter_sdir}Sample_{sample_data_label}_step_{stepi}_chain_{chain_ind}"
        result_data_handle = _dump_disk_backed_data(
            num_result=num_result,
            data_path=data_path,
            num_type=num_type,
            space_wait_timeout=space_wait_timeout,
            reserve_free_pct=reserve_free_pct,
        )
        # Store handle
        chain_result.append(result_data_handle)
    # ============ Extracted data / Sample data processing ============
    elif (dl_in >= 6) & (dl_in <= 7):
        # Load step_input_data
        step_input_data_loaded = _load_disk_backed_data(step_input_data)
        # Compute
        num_result = method_func(step_input_data_loaded)
        # Dump result
        data_path = f"{inter_sdir}Sample_{sample_data_label}_step_{stepi}_chain_{chain_ind}"
        result_data_handle = _dump_disk_backed_data(
            num_result=num_result,
            data_path=data_path,
            num_type=num_type,
            space_wait_timeout=space_wait_timeout,
            reserve_free_pct=reserve_free_pct,
        )
        # Store handle
        chain_result.append(result_data_handle)
    # Save calculated step results
    # Store step result and calculation status
    status_results[stepi].append((step_id, step_procs_key, dl_in, dl_out, chain_result[-1]))
    calc_status[stepi].append(step_procs_key)
    return status_results, calc_status


def _image_processing_step(
    dl_in: int,
    preprocessed_img_dir: str,
    input_image_path: str,
    method_func: Callable,
    preprocess_status: dict,
    *,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    # Dependencies for multiprocessing
    copy: ModuleType = copy,
    os: ModuleType = os,
    time: ModuleType = time,
    datetime: type = datetime,
    np: ModuleType = np,
    torch: ModuleType = torch,
    unc_path: Callable = unc_path,
) -> str:
    """
    Multiprocessing compatible image processing step with shared status for operation control on a same file.
    """
    # Output dst image path
    img_path_val = os.path.splitext(str(input_image_path).replace("\\", "/").replace("//", "/"))
    img_name = img_path_val[0].split("/")[-1]
    if dl_in == 0:
        output_image_path = preprocessed_img_dir + img_name + "_processed_by_" + method_func.__name__ + img_path_val[1]
    else:
        output_image_path = preprocessed_img_dir + img_name + "_px_app_" + method_func.__name__ + img_path_val[1]

    # Extract shared objects - preprocess_status of pathos.helpers.mp.Manager.list and lock in pipeline.py
    start_status = preprocess_status['start_status']
    completion_status = preprocess_status['completion_status']
    processed_image_init = preprocess_status['processed_image_init']
    lock = preprocess_status['lock']

    # Avoid race
    with lock:
        if output_image_path in start_status:
            wait = True
        else:
            wait = False
        # Processing of unprocessed image
        # Write starting status
        start_status.append(output_image_path)
        # Initialize output image - remove if exists before run
        if output_image_path not in processed_image_init:
            if os.path.exists(unc_path(output_image_path)):
                os.remove(unc_path(output_image_path))
            processed_image_init.append(output_image_path)

    # Image processing
    if wait:
        _wait_for_completion(output_image_path, preprocess_status)
        return str(output_image_path)
    else:
        output_image_path = _image_processor(
            input_image_path=input_image_path,
            dl_in=dl_in,
            preprocessed_img_dir=preprocessed_img_dir,
            method_func=method_func,
            output_image_path=output_image_path,
            space_wait_timeout=space_wait_timeout,
            reserve_free_pct=reserve_free_pct,
            preprocess_status=preprocess_status,
        )
        # Write completion status and process results
        with lock:
            completion_status.append(output_image_path)
        return str(output_image_path)


@simple_type_validator
def _wait_for_completion(
    output_image_path: str,
    preprocess_status: dict,
    *,
    max_wait_time: int = 10800,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    # Dependencies for multiprocessing
    time: ModuleType = time,
    np: ModuleType = np,
) -> None:
    """Wait for the completion of started processing of existed image."""
    start_time = time.time()
    # Validate max wait time
    max_wait_time = max(max_wait_time, 1)
    lock = preprocess_status['lock']
    completion_status = preprocess_status['completion_status']
    # TODO: new
    waiting_for_disk_space = preprocess_status['waiting_for_disk_space']
    while True:
        with lock:
            if output_image_path in completion_status:
                break
            # TODO: new
            if output_image_path in waiting_for_disk_space:
                start_time = time.time()
        if time.time() - start_time > max_wait_time:
            raise TimeoutError(f"Image processing timeout, target image:\n{output_image_path}")
        time.sleep(np.random.uniform(4, 6))
    return None


def _image_processor(
    input_image_path: str,
    dl_in: int,
    preprocessed_img_dir: str,
    method_func: Callable,
    output_image_path: str,
    *,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    preprocess_status: Optional[dict[str, Any]] = None,
    # Dependencies for multiprocessing
    copy: ModuleType = copy,
    os: ModuleType = os,
    time: ModuleType = time,
    datetime: type = datetime,
    np: ModuleType = np,
    torch: ModuleType = torch,
) -> str:
    """Processing images and return to specified path"""
    # For data level 1~4 implementing pixel application - choose mode
    if dl_in == 1:
        pix_app_mode = "spec"
    elif dl_in == 2:
        pix_app_mode = "array"
    elif dl_in == 3:
        pix_app_mode = "tensor"
    else:
        pix_app_mode = "tensor_hyper"
    # Process image and return path of processed image
    if dl_in == 0:
        output_image_path = method_func(input_image_path, output_image_path)
    else:
        output_image_path = pixel_apply(
            input_image_path,
            method_func,
            pix_app_mode,
            output_image_path,
            progress=False,
            override=False,
            _space_wait_timeout=space_wait_timeout,
            _reserve_free_pct=reserve_free_pct,
            _preprocess_status=preprocess_status,
        )
    # Return path of processed image
    return output_image_path


# %% TODO: Sample_list assembly tools


# Basic assembly
# TODO: Step results to modeling-ready sample_list data
@simple_type_validator
def _sample_list_constructor(  # noqa: C901
    result_directory: str,
    sample_data: list[dict[str, Any]],  # self._sample_data
    specpipe_process: list[tuple[str, str, str, str, int, Union[Callable, object], int, int]],  # self.process
    process_chains: list[tuple[str, ...]],  # self.process_chains
    to_csv: bool = True,
    show_progress: bool = False,
    backup: bool = False,
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
) -> None:
    """
    Convert Step_result data from file to modeling-ready sample_list data.
    """
    # Validate report directory
    preprocess_result_dir = result_directory + "Preprocessing/"
    os.makedirs(unc_path(preprocess_result_dir), exist_ok=True)

    # Validate preprocessing result file paths
    sd_paths = [
        f"{preprocess_result_dir}Step_results/PreprocessingResult_sample_{sd['label']}.dill" for sd in sample_data
    ]
    for sdp in sd_paths:
        if not os.path.exists(unc_path(sdp)):
            raise ValueError(f"\nPreprocessing step result file path not found : \n{sdp}\n")

    ## Chain results to sample_list data
    # Get preprocess chains of all preprocessing steps

    # TODO: removed
    # pchains = []
    # for pchain in process_chains:
    #     if pchain[:-1] not in pchains:
    #         pchains.append(pchain[:-1])

    # TODO: new
    # Get preprocess chains of all preprocessing steps
    model_ids = [pit[0] for pit in specpipe_process if pit[3] == "model"]
    assembly_ids = [pit[0] for pit in specpipe_process if pit[3] == "assembly"]

    # Get pre-modeling processing chain step numbers
    preprocess_chain_length = len(
        [proc_id for proc_id in process_chains[0] if (proc_id not in assembly_ids) and (proc_id not in model_ids)]
    )
    n_non_preprocess_step = len(process_chains[0]) - preprocess_chain_length

    # Get pre-modeling processing chains
    pchains = []
    for chain in process_chains:
        pchain = chain[:-n_non_preprocess_step]
        if pchain not in pchains:
            pchains.append(pchain)

    ## Loop preprocess chains across all samples and transform to modeling data
    if show_progress:
        print("\nConstruct chain sample list ...\nChain :")
    # pci - process chain id
    pre_results: list[
        tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, Annotated[Any, arraylike_validator(ndim=1)]]
    ]
    finished_paths = []
    for pci in tqdm(range(len(pchains)), total=len(pchains), disable=(not show_progress)):
        pchain = pchains[pci]
        # Preprocessing results
        pre_results = []
        for spath in sd_paths:
            sdata = load_dill(unc_path(spath))
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
        # Typing: list[tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, Annotated[Any,arraylike_validator(ndim=1)]]]  # noqa: E501
        res_path_dill = preprocess_result_dir + chain_name1 + ".dill"
        # TODO: changed
        dump_dill(
            {"chain_ind": str(pci), "chain_procs": pchain, "chain_res": pre_results},
            target_file_path=unc_path(res_path_dill),
            backup=backup,
            space_wait_timeout=space_wait_timeout,
            reserve_free_pct=reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )
        finished_paths.append(res_path_dill)

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

            # TODO: changed
            # coln_chain_res = ["Sample_ID", "Label", "Validation_group", "Test", "Train", "X_shape", "y"] + [
            #     f"x{i}" for i in range(arr_chain_res.shape[1] - 7)
            #     ]
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
            # TODO: df_chain_res.to_csv(unc_path(res_path_csv), index=False)
            df_to_csv(
                dataframe=df_chain_res,
                csv_path=unc_path(res_path_csv),
                index=False,
                space_wait_timeout=space_wait_timeout,
                reserve_free_pct=reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )

    # Dump sample_list file paths for assembly processes
    finished_paths_path_dir = result_directory + "/Assembly/.__swectral_dill_data/"
    os.makedirs(unc_path(finished_paths_path_dir), exist_ok=True)
    path_file_path = finished_paths_path_dir + ".__sample_list_paths_finished.dill"
    # TODO: dump_dill(unc_path(path_file_path), {"finished_paths": finished_paths}, backup=backup)
    # TODO: changed
    dump_dill(
        {"finished_paths": finished_paths},
        target_file_path=unc_path(path_file_path),
        backup=backup,
        space_wait_timeout=space_wait_timeout,
        reserve_free_pct=reserve_free_pct,
        min_sec_random_wait=5.0,
        max_sec_random_wait=5.0,
    )

    # Add line after progress bar
    print("")


# TODO: Additional assembly methods
# TODO: Sample assembly of all samples of a single preprocessing chain
@simple_type_validator
def _single_preprocess_assembly(
    dpath: str,
    assembly_result_dir: str,
    assem_interm_path: str,
    assem_log_dir_path: str,
    assembly_chains: list[tuple[str, ...]],
    specpipe_process: list[tuple[str, str, str, str, int, Union[Callable, object], int, int]],
    n_step_choice_dict: dict[int, int],
    final_result_only: bool,
    backup: bool,
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
) -> None:
    """Apply the entire assembly stage on a preprocessing chain result."""
    # Import dependencies
    import os
    from datetime import datetime
    from swectral.specio import unc_path

    try:
        # Intermediate data dir
        pc_fname_split = os.path.splitext(os.path.basename(dpath))

        for stepi in range(len(assembly_chains[0])):
            step_proc_ids = [chain[stepi] for chain in assembly_chains]
            snum = 0
            for proc_id in step_proc_ids:
                _apply_step_assembly(
                    dpath=dpath,
                    assembly_result_dir=assembly_result_dir,
                    assem_interm_path=assem_interm_path,
                    stepi=stepi,
                    snum=snum,
                    proc_id=proc_id,
                    specpipe_process=specpipe_process,
                    assembly_chains=assembly_chains,
                    n_step_choice_dict=n_step_choice_dict,
                    backup=backup,
                    space_wait_timeout=space_wait_timeout,
                    reserve_free_pct=reserve_free_pct,
                )
                snum += 1
            # Remove used data files after computation of current step result if final_result_only
            if final_result_only and stepi >= 1:
                for psnum in range(n_step_choice_dict[stepi]):
                    # Input path - previous step
                    input_filename = pc_fname_split[0] + f"_a&{stepi-1}&{psnum}" + pc_fname_split[1]
                    input_path = assem_interm_path + input_filename
                    if os.path.exists(input_path):
                        try:
                            os.remove(input_path)
                        except Exception:
                            pass

        # Write a log file for the current progress to mark as finished
        log_path = assem_log_dir_path + os.path.splitext(os.path.basename(dpath))[0]
        with open(log_path, 'w') as f:
            f.write("")
    except Exception as e:
        errorlog_dir_path = assem_interm_path + "Error_logs/"
        os.makedirs(unc_path(errorlog_dir_path), exist_ok=True)
        assert hasattr(datetime, 'now')
        cts: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # PID for multiprocessing
        pid = os.getpid()
        error_log_path = errorlog_dir_path + f"error_{cts}_pid_{pid}.log"
        # Write error log
        err_msg = f"Failed in the assembly of '{dpath}' sample data, error message: \n\n{str(e)}"
        with open(unc_path(error_log_path), "w") as f:
            f.write(err_msg)
        raise ValueError(f"Failed in the assembly of '{dpath}' sample data") from e


# TODO: Apply a single sample assembly step
# Process items: [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
@simple_type_validator
def _apply_step_assembly(
    dpath: str,
    assembly_result_dir: str,
    assem_interm_path: str,
    stepi: int,
    snum: int,
    proc_id: str,
    specpipe_process: list[tuple[str, str, str, str, int, Union[Callable, object], int, int]],
    assembly_chains: list[tuple[str, ...]],
    n_step_choice_dict: dict[int, int],
    backup: bool,
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
) -> None:
    """Apply assembly method of an assembly step."""

    # Imports
    import os
    from swectral.specio import (
        load_dill,
        dump_dill,
        unc_path,
    )

    # Get process method
    method_match = [proc[5] for proc in specpipe_process if proc[0] == proc_id]
    if len(method_match) == 1:
        method = method_match[0]
        if not callable(method):
            raise TypeError(f"Method not callable, process ID '{proc_id}'.")
    else:
        raise ValueError(f"Corrupted process ID '{proc_id}', {len(method_match)} match(es) found.")

    # Intermediate data dir
    pc_fname_split = os.path.splitext(os.path.basename(dpath))

    # Read input sample list and process
    # Primary assembly - in addition to basic default assembly
    if stepi == 0:
        # Input path - previous step
        pc_sample_list_data = load_dill(dpath)
        # Get sample list data
        chain_ind = pc_sample_list_data["chain_ind"]
        chain_procs = pc_sample_list_data["chain_procs"]
        chain_res = pc_sample_list_data["chain_res"]
        # Process sample_list
        chain_ind = chain_ind + f"_a&{stepi}&{snum}"
        chain_procs = chain_procs + (proc_id,)
        chain_res = method(chain_res)
        # Output path - current step
        output_filename = pc_fname_split[0] + f"_a&{stepi}&{snum}" + pc_fname_split[1]
        if stepi == len(assembly_chains[0]) - 1:
            output_path = assembly_result_dir + output_filename
        else:
            output_path = assem_interm_path + output_filename
        # Dump processed sample_list
        # dump_dill(
        #     unc_path(output_path),
        #     {"chain_ind": chain_ind, "chain_procs": chain_procs, "chain_res": chain_res},
        #     backup=backup,
        # )
        # TODO: changed
        dump_dill(
            {"chain_ind": chain_ind, "chain_procs": chain_procs, "chain_res": chain_res},
            target_file_path=unc_path(output_path),
            backup=backup,
            space_wait_timeout=space_wait_timeout,
            reserve_free_pct=reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )

    # Secondary assembly
    else:
        # Loop previous step result and compute result of current step
        for psnum in range(n_step_choice_dict[stepi]):
            # Input path - previous step
            input_filename = pc_fname_split[0] + f"_a&{stepi-1}&{psnum}" + pc_fname_split[1]
            input_path = assem_interm_path + input_filename
            pc_sample_list_data = load_dill(input_path)
            # Get sample list data
            chain_ind = pc_sample_list_data["chain_ind"]
            chain_procs = pc_sample_list_data["chain_procs"]
            chain_res = pc_sample_list_data["chain_res"]
            # Process sample_list
            chain_ind = chain_ind + f"_a&{stepi}&{snum}"
            chain_procs = chain_procs + (proc_id,)
            chain_res = method(chain_res)
            # Output path - current step
            output_filename = pc_fname_split[0] + f"_a&{stepi}&{snum}" + pc_fname_split[1]
            if stepi == len(assembly_chains[0]) - 1:
                output_path = assembly_result_dir + output_filename
            else:
                output_path = assem_interm_path + output_filename
            # Dump processed sample_list
            # TODO: changed
            dump_dill(
                {"chain_ind": chain_ind, "chain_procs": chain_procs, "chain_res": chain_res},
                target_file_path=unc_path(output_path),
                backup=backup,
                space_wait_timeout=space_wait_timeout,
                reserve_free_pct=reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )


# %% Model method wrapper, sample_list as method input


# Model method wrapper, sample_list as method input
class _ModelMethod:
    """
    Model method wrapper to specify modeling and model evaluation parameters.
    """

    def __init__(
        self,
        # Model configuration
        model_label: str,
        is_regression: Optional[bool],
        x_shape: Optional[tuple[int]],
        report_dir: str,
        method: object,
        result_backup: bool,
        # Modeling parameters
        validation_method: str,
        unseen_threshold: Optional[float],
        # Model evaluation reporting parameters
        data_split_config: Union[str, dict[str, Any]],
        validation_config: Union[str, dict[str, Any]],
        metrics_config: Union[str, dict[str, Any], None],
        roc_plot_config: Union[str, dict, None],
        scatter_plot_config: Union[str, dict[str, Any], None],
        residual_config: Union[str, dict[str, Any], None],
        residual_plot_config: Union[str, dict[str, Any], None],
        influence_analysis_config: Union[str, dict[str, Any], None],
        # File dumping parameter
        space_wait_timeout: int = 36000,
        reserve_free_pct: float = 5.0,
    ) -> None:
        self.__name__ = model_label
        self.model_label = model_label
        self.is_regression = is_regression
        self.input_shape = x_shape
        self.report_dir = report_dir
        self.method = method
        self.result_backup = result_backup
        self.validation_method = validation_method
        self.unseen_threshold = unseen_threshold
        self.data_split_config = data_split_config
        self.validation_config = validation_config
        self.metrics_config = metrics_config
        self.roc_plot_config = roc_plot_config
        self.scatter_plot_config = scatter_plot_config
        self.residual_config = residual_config
        self.residual_plot_config = residual_plot_config
        self.influence_analysis_config = influence_analysis_config
        self.space_wait_timeout = space_wait_timeout
        self.reserve_free_pct = reserve_free_pct

    # Sample_list item: (0 - Sample id, 1 - Sample label, 2 - Validation group, 3 - Test mask, 4 - Train mask, 5 - Original shape, 6 - Target value, 7 - Sample predictor values)  # noqa: E501
    @simple_type_validator
    def evaluation(
        self,
        sample_list: list[
            tuple[
                str,
                str,
                str,
                np.int8,
                np.int8,
                tuple[int, ...],
                Any,
                Annotated[Any, arraylike_validator(ndim=1)],
            ]
        ],
        data_label: str,
        report_directory: Optional[str] = None,
        modeleva: type = ModelEva,
        silent_all: bool = False,
    ) -> None:
        """
        Evaluation of specified model. Configured at _ModelMethod instance.

        Parameters
        ----------
        sample_list : list of (str, tuple of int, str or int or bool or float, 1D array-like)
            Standard sample data of SpecPipe for modeling.
        data_label : str
            Label for the specified dataset.
        report_directory : Optional[str], optional
            Report_directory for model evaluation reports. The default is using report_directory of the _ModelMethod instance.
        """  # noqa: E501
        if report_directory is None:
            report_directory = self.report_dir
        # Model Evaluation
        model_eva = modeleva(
            sample_list=sample_list,
            model=self.method,
            validation_method=self.validation_method,
            report_directory=report_directory,
            model_label=self.model_label,
            data_label=data_label,
            is_regression=self.is_regression,
            unseen_threshold=self.unseen_threshold,
            result_backup=self.result_backup,
            silent_all=silent_all,
            space_wait_timeout=self.space_wait_timeout,
            reserve_free_pct=self.reserve_free_pct,
        )
        if self.is_regression:
            model_eva.regressor_evaluation(
                data_split_config=self.data_split_config,
                validation_config=self.validation_config,
                metrics_config=self.metrics_config,
                scatter_plot_config=self.scatter_plot_config,
                residual_config=self.residual_config,
                residual_plot_config=self.residual_plot_config,
                influence_analysis_config=self.influence_analysis_config,
            )
        else:
            model_eva.classifier_evaluation(
                data_split_config=self.data_split_config,
                validation_config=self.validation_config,
                metrics_config=self.metrics_config,
                roc_plot_config=self.roc_plot_config,
                residual_config=self.residual_config,
                influence_analysis_config=self.influence_analysis_config,
            )

    def __call__(self, sample_list: list, data_label: str, report_directory: Optional[str] = None) -> None:
        self.evaluation(sample_list, data_label, report_directory)


# Run modeling on single dataset
# Notes: different from '_test_model',
# Sample_list item: (0 - Sample id, 1 - Sample label, 2 - Validation group, 3 - Test mask, 4 - Train mask, 5 - Original shape, 6 - Target value, 7 - Sample predictor values)  # noqa: E501
@simple_type_validator
def _model_evaluator(  # noqa: C901
    preprocess_result: list[
        tuple[
            str,
            str,
            str,
            np.int8,
            np.int8,
            tuple[int, ...],
            Any,
            Annotated[Any, arraylike_validator(ndim=1)],
        ]
    ],
    preprocess_chain: tuple,
    preprocess_chain_label: str,
    model_processes: list[tuple[str, str, str, str, int, Any, int, int]],
    specpipe_report_directory: str,
    result_directory: str = "",
    lock: object = _DummyLock(),
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    # Import applied functions and modules
    _dl_val: Callable = _dl_val,
    unc_path: Callable = unc_path,
    load_dill: Callable = load_dill,
    dump_dill: Callable = dump_dill,
    _target_type_validation_for_serialization: Callable = _target_type_validation_for_serialization,
    modeleva: type = ModelEva,
    silent_all: bool = False,
) -> None:
    """
    Evaluation of added models on a single sample_list dataset.
    'preprocess_result' must be the standard 'sample_list' output.
    'preprocess_chain_label' serves as the 'data_label' for modeling data.
    'preprocess_chain_label' must be unique across all sample lists for modeling.
    """
    # Import applied modules
    import os
    import time
    import numpy as np

    # Resume testing - initial break status (EnvVar own copy in a subprocess in multiprocessing)
    model_resume_test_num = int(os.getenv("SPECPIPE_MODEL_RESUME_TEST_NUM", "-1"))
    # Resume testing - conditional break
    if model_resume_test_num > 0:
        if model_resume_test_num > 1:
            raise ValueError("Modeling resume test raise")
    # Resume testing - status update (need lock for multiprocessing)
    if model_resume_test_num > 0:
        model_resume_test_num = model_resume_test_num + 1
        os.environ["SPECPIPE_MODEL_RESUME_TEST_NUM"] = str(model_resume_test_num)

    # Name reassignment
    sample_list = preprocess_result
    preprocess_chain = tuple(preprocess_chain)
    sample_list_label = preprocess_chain_label

    # Validate report directory
    if result_directory == "":
        result_directory = specpipe_report_directory
    model_result_dir = result_directory + "Modeling/"
    os.makedirs(unc_path(model_result_dir), exist_ok=True)

    # Validate sample_list_label - Absolutely unique number if label not provided (commonly it should be given)
    if sample_list_label == "":
        if update_progress_log:
            raise ValueError(
                "Consistent and unique preprocessing chain label (sample_list_label) must be provided \
                    if update_progress_log set True for enabling break resuming."
            )
        sample_list_label = str(time.time_ns())[4:-2] + str(os.getpid())

    # Validate model processes
    for procit in model_processes:
        if procit[3] != "model":
            raise ValueError(f"Model process must have output data level of 'model', but got: '{procit[3]}'")
        if not callable(procit[-3]):
            raise ValueError(
                f"Invalid model evaluation method, \
                    given method is not callable : {procit[-3]}, got type : {type(procit[-3])}"
            )

    # Save preprocess chain info for the sample_list
    model_report_dir = model_result_dir + "Model_evaluation_reports/"
    os.makedirs(unc_path(model_report_dir), exist_ok=True)
    # Chain file name
    chain_label_file_path = model_report_dir + f"{preprocess_chain_label}.txt"
    # Save chain process file
    with open(unc_path(chain_label_file_path), "w") as f:
        for pci, pproc in enumerate(preprocess_chain):
            if pci < (len(preprocess_chain) - 1):
                f.write(f"{pproc}\n")
            else:
                f.write(f"{pproc}")

    # Get model processes with input data level index dl_in_ind
    # Process: [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    for modelit in model_processes:
        # Get test model
        model_methodi = modelit[5]
        # Get model input data level
        dl_in_name = _dl_val(modelit[2])[1]
        dl_in_ind = _dl_val(modelit[2])[0]
        # Get model input data shape
        input_dshape = model_methodi.input_shape
        # Validate data shape
        sample_list_shape = (len(sample_list), len(sample_list[0]))
        if input_dshape is None:
            input_dshape = sample_list_shape
        elif np.prod(sample_list_shape) != np.prod(input_dshape):
            raise ValueError(
                f"Cannot reshape sample data with shape {sample_list_shape} \
                    into specified input data shape {input_dshape} of the model.\
                    \nInput step data ID: {sample_list[0]}\nModel label: {modelit[1]}"
            )
        # Modeling
        # Sample_list item: (0 - Sample id, 1 - Sample label, 2 - Validation group, 3 - Test mask, 4 - Train mask, 5 - Original shape, 6 - Target value, 7 - Sample predictor values)  # noqa: E501
        if dl_in_ind == 7 or dl_in_ind == 8:
            # Regression
            if model_methodi.is_regression:
                model_methodi.evaluation(
                    sample_list=sample_list,
                    data_label="chain_" + sample_list_label,
                    report_directory=model_result_dir,
                    modeleva=modeleva,
                    silent_all=silent_all,
                )
            # Classification
            else:
                model_methodi.evaluation(
                    sample_list=sample_list,
                    data_label="chain_" + sample_list_label,
                    report_directory=model_result_dir,
                    modeleva=modeleva,
                    silent_all=silent_all,
                )
        else:
            raise ValueError(
                "Model only accepts input data level 7 ('spec1d') or 8 ('assembly'),"
                + f"but got: {dl_in_ind} ('{dl_in_name}')"
            )

    # Update progress
    log_path = model_report_dir + "modeling_progress_log.dill"
    assert hasattr(lock, "__enter__") and hasattr(lock, "__exit__")
    with lock:
        if os.path.exists(unc_path(log_path)):
            modeling_progress_log = load_dill(log_path)["modeling_progress_log"]
            if preprocess_chain not in modeling_progress_log:
                modeling_progress_log.append(preprocess_chain)
            else:
                warnings.warn(
                    f"Preprocessing chain label (sample_list_label) must be unique, got duplicated label: {sample_list_label}",  # noqa: E501
                    UserWarning,
                    stacklevel=3,
                )
            # TODO: dump_dill({"modeling_progress_log": modeling_progress_log}, log_path, backup=False)
            dump_dill(
                {"modeling_progress_log": modeling_progress_log},
                target_file_path=unc_path(log_path),
                backup=False,
                space_wait_timeout=space_wait_timeout,
                reserve_free_pct=reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )
        else:
            # TODO: dump_dill({"modeling_progress_log": [preprocess_chain]}, log_path, backup=False)
            dump_dill(
                {"modeling_progress_log": [preprocess_chain]},
                target_file_path=unc_path(log_path),
                backup=False,
                space_wait_timeout=space_wait_timeout,
                reserve_free_pct=reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )


@simple_type_validator
def _model_evaluator_mp(
    cdp: str,
    pchains: list[tuple],
    model_processes: list[tuple[str, str, str, str, int, Any, int, int]],
    specpipe_report_directory: str,
    result_directory: str = "",
    lock: object = _DummyLock(),
    # Update progress status, use in a processing loop for resume
    update_progress_log: bool = False,
    # File dumping parameter
    space_wait_timeout: int = 36000,
    reserve_free_pct: float = 5.0,
    # Import applied functions and modules
    _model_evaluator: Callable = _model_evaluator,
    _dl_val: Callable = _dl_val,
    unc_path: Callable = unc_path,
    load_dill: Callable = load_dill,
    dump_dill: Callable = dump_dill,
    _target_type_validation_for_serialization: Callable = _target_type_validation_for_serialization,
    modeleva: type = ModelEva,
    silent_all: bool = True,
) -> None:
    """
    Evaluation of added models on a single sample_list dataset for multiprocessing.
    cdp : chain data path
    pchains : preprocessing chains
    """
    try:
        # Import applied modules
        import os
        from datetime import datetime

        # Load chain data
        pc_it = load_dill(cdp)
        pc_sample_list = pc_it["chain_res"]
        pc_sample_list = _target_type_validation_for_serialization(pc_sample_list)
        pchain = pc_it["chain_procs"]
        # Use preprocess chain ID as chain label
        pproc_chain_label = [f"Preprocessing_#{pci}" for pci, pc in enumerate(pchains) if pc == pchain][0]
        _model_evaluator(
            preprocess_result=pc_sample_list,
            preprocess_chain=pchain,
            preprocess_chain_label=pproc_chain_label,
            model_processes=model_processes,
            specpipe_report_directory=specpipe_report_directory,
            result_directory=result_directory,
            lock=lock,
            update_progress_log=update_progress_log,
            # Import applied functions
            _dl_val=_dl_val,
            load_dill=load_dill,
            dump_dill=dump_dill,
            modeleva=modeleva,
            silent_all=silent_all,
        )

    # Error handling
    except Exception as e:
        # Validate report directory
        if result_directory == "":
            result_directory = specpipe_report_directory
        model_result_dir = result_directory + "Modeling/"
        os.makedirs(unc_path(model_result_dir), exist_ok=True)
        errdir = model_result_dir + "Error_logs/"
        os.makedirs(unc_path(errdir), exist_ok=True)
        cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pid = os.getpid()
        error_log_path = errdir + f"error_{cts}_pid_{pid}.log"
        err_msg = f"\nFailed in the modeling of preprocessing chain from path '{cdp}', error message: \n\n{str(e)}\n"
        with open(unc_path(error_log_path), "w") as f:
            f.write(err_msg)
        raise ValueError(f"\nFailed in the modeling of preprocessing chain from path '{cdp}'") from e
