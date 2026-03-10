# -*- coding: utf-8 -*-
"""
Swectral - Pipeline process meta and process method validators

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os
import warnings
import dill

# Typing
from typing import Annotated, Any, Union, Optional, Callable

# Basic data
import numpy as np
import pandas as pd
from operator import itemgetter

# Raster
import rasterio
from rasterio.windows import Window

# Local
from .rasterop import pixel_apply
from .specio import (
    arraylike_validator,
    simple_type_validator,
    unc_path,
    load_dill,
)
from .assembly import identity_assembly

# For multiprocessing
global ModelEva


# %% Pipeline internal data validators


# Target value type fixing after serialization
# Sample_list item: (0 - Sample id, 1 - Sample label, 2 - Validation group, 3 - Test mask, 4 - Train mask, 5 - Original shape, 6 - Target value, 7 - Sample predictor values)  # noqa: E501
@simple_type_validator
def _target_type_validation_for_serialization(
    pc_sample_list: list[
        tuple[str, str, str, np.int8, np.int8, tuple[int], Any, Annotated[Any, arraylike_validator(ndim=1)]]
    ],
) -> list[
    tuple[
        str,
        str,
        str,
        np.int8,
        np.int8,
        tuple[int],
        Union[str, int, bool, float],
        Annotated[Any, arraylike_validator(ndim=1)],
    ]
]:
    """Fix typing for integer after dill serialization."""
    for loaded_i, loaded_sample in enumerate(pc_sample_list):
        loaded_y = loaded_sample[6]
        # Behavior check to fix type
        test_value_y = loaded_y
        try:
            _ = test_value_y + 1
            if "." in str(loaded_y):
                loaded_y = float(loaded_y)
            elif str(loaded_y) in ["True", "False"]:
                loaded_y = bool(loaded_y)
            else:
                loaded_y = int(loaded_y)
        except Exception:
            loaded_y = str(loaded_y)
        # Update target value
        pc_sample_list[loaded_i] = (
            loaded_sample[0],
            loaded_sample[1],
            loaded_sample[2],
            loaded_sample[3],
            loaded_sample[4],
            loaded_sample[5],
            loaded_y,
            loaded_sample[7],
        )
    return pc_sample_list


# %% Pipeline process meta validators


# Data_level validator
# Data_level: 0 - image (path), \
# 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
# 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
@simple_type_validator
def _dl_val(data_level: Union[str, int]) -> tuple[int, str]:
    """
    Data level validator, input data level name or index number, return (data level index, data level name).
    """
    # Validate data_level
    data_levels = [
        "image",
        "pixel_spec",
        "pixel_specs_array",
        "pixel_specs_tensor",
        "pixel_hyperspecs_tensor",
        "image_roi",
        "roi_specs",
        "spec1d",
        # TODO: new
        "assembly",
        "model",
    ]
    # TODO: data_level_n = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    data_level_n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if type(data_level) is str:
        if data_level.lower() not in data_levels:
            raise ValueError(f"data_level must be one of {data_levels}, but got: {data_level}")
        else:
            dlind = data_levels.index(data_level.lower())
    elif type(data_level) is int:
        if data_level not in data_level_n:
            raise ValueError(f"data_level number must be one of {data_level_n}, but got: {data_level}")
        else:
            dlind = data_level
    return (dlind, data_levels[dlind])


# Process I/O data level and application sequence validator
@simple_type_validator
def _data_level_seq_validator(  # noqa: C901
    input_data_level: Union[str, int],
    output_data_level: Union[str, int],
    application_sequence: int,
    full_application_sequence: int,
    existed_process: list[tuple[str, str, str, str, int, Any, int, int]],
) -> list[tuple[str, str, str, str, int, Any, int, int]]:
    """
    Process input and output data level and application sequence validator.
    Arg 'process' is the SpecPipe.process attribute.
    """
    # Validate Data_level
    dl_in = _dl_val(input_data_level)
    dl_in_name = dl_in[1]
    dl_in_ind = dl_in[0]

    dl_out = _dl_val(output_data_level)
    dl_out_name = dl_out[1]
    dl_out_ind = dl_out[0]

    # Full application sequence
    fapp_seq = full_application_sequence

    # Validate input data level
    # TODO: if dl_in_ind >= 8:
    if dl_in_ind >= 9:
        # TODO: changed err msg
        raise ValueError("Input data level cannot be 'model' or 9 (corresponding index).")

    # Validate output data level
    if dl_out_name == "image_roi":
        raise ValueError(
            f"Output data level '{dl_out_name}' is not supported, "
            + "as dynamic ROIs are not supported currently. "
            + "Please write the generated ROIs to files and start a new SpecPipe "
            + "using SpecExp with the resulting ROI files."
        )
    if dl_out_ind < dl_in_ind:
        raise ValueError(
            "Output_data_level cannot precede the input_data_level in the processing pipeline, "
            + f"got input data level: '{input_data_level}', output data level: '{output_data_level}'"
        )

    # Validate sequence
    if (application_sequence < 0) | (application_sequence > 999999):
        raise ValueError("Application sequence must be within [0, 1,000,000), " + f"got: {application_sequence}")

    # TODO: Validate assembly step input data level
    if dl_out_ind == 8:
        if dl_in_ind < 7 or dl_in_ind > 8:
            raise ValueError(
                f"The input data level of 'assembly' process must be 'spec1d' or 'assembly', got: '{dl_in_name}'"
            )
        # Coerce model dl_in to 8 and update full app sequence if model process added
        coerce_dl8 = False
        for i, proc in enumerate(existed_process):
            if proc[3] == 'model' and proc[2] == 'spec1d':
                existed_process[i] = (
                    proc[0],
                    proc[1],
                    'assembly',
                    'model',
                    proc[4],
                    proc[5],
                    proc[6] + 2000000,
                    proc[7],
                )
                coerce_dl8 = True
        if coerce_dl8:
            warnings.warn(
                "Input data level for newly added model processes has been automatically "
                "set to 'assembly' because an assembly process was added.",
                UserWarning,
                stacklevel=2,
            )

    # TODO: Validate and correct model step input data level
    if dl_out_ind == 9:
        existed_assembly_procs = [proc for proc in existed_process if proc[3] == 'assembly']
        if len(existed_assembly_procs) > 0:
            if dl_in_ind != 8:
                raise ValueError(
                    "Assembly process detected. The modeling step input data level must be 8 ('assembly'), but got: "
                    + f"{dl_in_ind} ('{dl_in_name}')."
                )
        else:
            if dl_in_ind != 7:
                raise ValueError(
                    "No assembly process detected. The modeling step input data level must be 7 ('spec1d'), but got: "
                    + f"{dl_in_ind} ('{dl_in_name}')."
                )

    # Validate existed process labels
    # [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
    f_fapp_seq, l_fapp_seq = -1, np.inf  # fore full application sequence, later full application sequence
    f_out_dl = None  # fore output data level
    l_in_dl = None  # later input data level
    if len(existed_process) > 0:
        # Identify the preceding (fore) and subsequent (later) steps
        for pr in existed_process:
            # Preceding process
            if (pr[6] < fapp_seq) & (pr[6] > f_fapp_seq):
                f_fapp_seq = pr[6]
                f_out_dl = pr[3]
            # Subsequent process
            if (pr[6] > fapp_seq) & (pr[6] < l_fapp_seq):
                l_fapp_seq = pr[6]
                l_in_dl = pr[2]
            # Validate consistency of output with identical input data level and application sequence
            if (dl_in_ind != dl_out_ind) and (pr[6] == fapp_seq):
                if dl_out_name != pr[3]:
                    raise ValueError(
                        "When a processing step has a different output data level with input, "
                        + f"methods with identical input data level (here: '{dl_in_name}') "
                        + f"and application sequence (here: '{application_sequence}') "
                        + "must have identical output data levels. \nGot output data level: "
                        + f"'{dl_out_name}' \nconflicted with process item: \nProcess ID: {pr[0]}"
                        + f"\nOutput data level: '{pr[3]}'"
                    )

    # Validate I/O data level of previous and subsequent processes
    if f_out_dl is not None:
        if dl_in_ind <= 5:
            if _dl_val(f_out_dl)[0] > 4:
                raise ValueError(
                    f"The specified input data level '{dl_in_name}' of added process "
                    + f"is inconsistent with the output data level '{f_out_dl}' of the previous process, "
                    + "the input data level of added process must be image levels (0~4)."
                )
        else:
            if dl_in_ind >= 6 and dl_in_ind != _dl_val(f_out_dl)[0]:
                raise ValueError(
                    f"The specified input data level '{dl_in_name}' of added process "
                    + f"is inconsistent with the output data level '{f_out_dl}' of the previous process, "
                    + "they must be identical."
                )
    if l_in_dl is not None:
        if _dl_val(l_in_dl)[0] <= 5:
            if dl_out_ind > 4:
                raise ValueError(
                    f"The specified output data level '{dl_out_name}' of added process "
                    + f"is inconsistent with the input data level '{l_in_dl}' of the subsequent process, "
                    + "the output data level of added process must be image levels (0~4)."
                )
        if _dl_val(l_in_dl)[0] >= 6 and dl_out_ind != _dl_val(l_in_dl)[0]:
            raise ValueError(
                f"The specified output data level '{dl_out_name}' of added process "
                + f"is inconsistent with the input data level '{l_in_dl}' of the subsequent process, "
                + "they must be identical."
            )

    return existed_process


# %% Pipeline process sequence validator - validate data level and application sequence of all processes of the pipeline


# Validate dl_in and dl_out of all existed processes
# [0 Process_ID, 1 Process_label, 2 Input_data_level, 3 Output_data_level, 4 Application_sequence, 5 Method_callable, 6 _Full_app_seq, 7 _Alternative_number]  # noqa: E501
@simple_type_validator
def _pipeline_process_seq_validator(pipe_processes: list[tuple[str, str, str, str, int, Callable, int, int]]) -> None:
    """Validate data level and application sequence of all processes of the pipeline."""
    # Ensure current process sorted
    processes_sorted = sorted(pipe_processes, key=itemgetter(-2))
    # Apply sequence validator process by process
    for proc in processes_sorted:
        try:
            _ = _data_level_seq_validator(
                input_data_level=proc[2],
                output_data_level=proc[3],
                application_sequence=proc[4],
                full_application_sequence=proc[-2],
                existed_process=processes_sorted,
            )
        except Exception as e:
            raise ValueError(
                "Process configuration is incompatible with its adjacent processes.\n"
                f"Process ID:\n{proc[0]}\n"
                f"Process label: {proc[1]}\n"
            ) from e


# %% Model validators


def _data_transformer_validator(data_transformer: object) -> None:
    """Scikit-learn style transformer validator"""
    if isinstance(data_transformer, type):
        raise TypeError(f"Expected a transformer instance, but got class {data_transformer.__name__}.")
    if callable(data_transformer):
        raise TypeError(f"Expected a transformer instance, but got function {data_transformer.__name__}.")  # type: ignore[attr-defined]
    native_data_type = (int, float, str, bool, list, dict, tuple, set, bytes, type(None))
    if type(data_transformer) in native_data_type:
        raise TypeError(
            f"Expected a transformer instance, but got '{data_transformer} with type '{type(data_transformer)}'."
        )
    if not hasattr(data_transformer, "fit") or not hasattr(data_transformer, "transform"):
        raise ValueError(
            "Expected a transformer instance with 'fit' and 'transform' methods."
            + f"Provided transformer '{type(data_transformer).__name__}' does not have these methods."
        )


def _resampler_validator(resampler: object) -> None:
    """Imbalanced-learn style resampler validator"""
    if isinstance(resampler, type):
        raise TypeError(f"Expected a resampler instance, but got class {resampler.__name__}.")
    if callable(resampler):
        raise TypeError(f"Expected a resampler instance, but got function {resampler.__name__}.")  # type: ignore[attr-defined]
    native_data_types = (int, float, str, bool, list, dict, tuple, set, bytes, type(None))
    if type(resampler) in native_data_types:
        raise TypeError(f"Expected a resampler instance, but got '{resampler}' with type '{type(resampler)}'.")
    if not hasattr(resampler, "fit_resample"):
        raise ValueError(
            "Expected a resampler instance with a 'fit_resample' method."
            + f"Provided resampler '{type(resampler).__name__}' does not have this method."
        )


def _classifier_validator(classifier: object) -> None:
    """Scikit-learn style classifier validator"""
    if isinstance(classifier, type):
        raise TypeError(f"Expected a classifier instance, but got class {classifier.__name__}.")
    if callable(classifier):
        raise TypeError(f"Expected a classifier instance, but got function {classifier.__name__}.")  # type: ignore[attr-defined]
    native_data_type = (int, float, str, bool, list, dict, tuple, set, bytes, type(None))
    if type(classifier) in native_data_type:
        raise TypeError("Expected a classifier instance, but got '{classifier}' with type '{type(classifier)}'.")
    if not hasattr(classifier, "fit") or not hasattr(classifier, "predict") or not hasattr(classifier, "predict_proba"):
        raise ValueError("Expected a classifier instance with 'fit', 'predict' and 'predict_proba' methods.")


def _regressor_validator(regressor: object) -> None:
    """Scikit-learn style regressor validator"""
    if isinstance(regressor, type):
        raise TypeError(f"Expected a regressor instance, but got class {regressor.__name__}.")
    if callable(regressor):
        raise TypeError(f"Expected a regressor instance, but got function {regressor.__name__}.")  # type: ignore[attr-defined]
    native_data_type = (int, float, str, bool, list, dict, tuple, set, bytes, type(None))
    if type(regressor) in native_data_type:
        raise TypeError(f"Expected a regressor instance, but got '{regressor}' with type '{type(regressor)}'.")
    if not hasattr(regressor, "fit") or not hasattr(regressor, "predict"):
        raise ValueError("Expected a regressor instance with 'fit' and 'predict' methods.")


# %% Pipeline process method functionality validator


# Process function validator - pretest validator
# Data levels:
# 0 - image (path), \
# 1 - pixel_spec (1D), 2 - pixel_specs_array (2D), 3 - pixel_specs_tensor (3D), 4 - pixel_hyperspecs_tensor (3D), \
# 5 - image_ROI (img_path + ROI coords), 6 - ROI_specs (2D), 7 - spec1d (1D spec stats)
# Pretest_data: [img_path, test_img_path, roi_coords, test_roi_coords, roitable, spec1d]
@simple_type_validator
def _process_validator(  # noqa: C901
    method: Callable,
    input_data_level: Union[str, int],
    output_data_level: Union[str, int],
    *,
    pretest_data: dict[str, Any],  # swectral._pretest_data
    standalone_specs_sample: list[
        tuple[str, str, str, str, list[Union[float, int]]]
    ],  # swectral.spec_exp.standalone_specs_sample  # noqa: E501
    report_directory: str,  # swectral.report_directory
) -> Callable:
    """
    Validate preprocessing method of specified input data level before execution of entire processing chain.
    """
    # Pretest_data validation for static typing
    if pretest_data is None:
        raise ValueError(
            "Internal Error: 'SpecPipe.pretest_data' is None. "
            + "Pre-execution test data initialization fails. Please report."
        )

    # Applied only for image samples
    if len(standalone_specs_sample) == 0:
        # Validate data_level
        dl_in = _dl_val(input_data_level)[0]
        dl_out = _dl_val(output_data_level)[0]

        # Test image path
        test_img_path = pretest_data["test_img_path"]

        # Test data
        # TODO: if dl_out < 8:
        if dl_out <= 7:
            # Validate function
            if not callable(method):
                raise TypeError(f"Process method must be callable for non-model data levels, got type: {type(method)}")
            if dl_in == 0:
                # Output dst image path
                img_path = os.path.splitext(str(test_img_path).replace("\\", "/").replace("//", "/"))
                img_name = img_path[0].split("/")[-1]
                assert hasattr(method, '__name__')
                output_path = (
                    report_directory
                    + "/Pre_execution_test_data/"
                    + img_name
                    + "_processed_by_"
                    + method.__name__
                    + img_path[1]
                )
                # Process image and return path of processed image
                assert callable(method)
                result = method(test_img_path, output_path)
            elif dl_in == 1:
                # Output dst image path
                img_path = os.path.splitext(str(test_img_path).replace("\\", "/").replace("//", "/"))
                img_name = img_path[0].split("/")[-1]
                assert hasattr(method, '__name__')
                output_path = (
                    report_directory
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
                assert hasattr(method, '__name__')
                output_path = (
                    report_directory
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
                assert hasattr(method, '__name__')
                output_path = (
                    report_directory
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
                assert hasattr(method, '__name__')
                output_path = (
                    report_directory
                    + "/Pre_execution_test_data/"
                    + img_name
                    + "_px_app_"
                    + method.__name__
                    + img_path[1]
                )
                # Process image and return path of processed image
                result = pixel_apply(test_img_path, method, "tensor_hyper", output_path, progress=False)
            elif dl_in == 5:
                assert callable(method)
                result = method(test_img_path, pretest_data["roi_coords"])
            elif dl_in == 6:
                testing_data = pretest_data["roi_specs"]
                assert callable(method)
                result = method(testing_data)
            elif dl_in == 7:
                testing_data = pretest_data["spec1d"]
                assert callable(method)
                result = method(testing_data)
            # TODO: add new
            else:
                # TODO: changed err msg
                raise ValueError("Input data level cannot be 'model' or 9 (corresponding index).")
        else:
            # Model method is not validated here
            return method

        # Output validation
        if result is None:
            assert hasattr(method, '__name__')
            raise ValueError(
                f"Method '{method.__name__}' returns no data. "
                + "The added method must have a return. "
                + "For image processing methods, absolute path of resulting image must be returned."
            )

        # For raster image path and image file output
        if dl_out <= 4:
            # Raster file validation
            if os.path.exists(unc_path(result)):
                # Open raster validation
                try:
                    with rasterio.open(unc_path(result)) as src:
                        # Raster validation
                        if src is None:
                            raise ValueError("Invalid raster: raster is None.")
                        elif (src.width == 0) or (src.height == 0) or (src.count == 0):
                            raise ValueError(
                                "Invalid raster, "
                                + f"got dimensions: {src.width} x {src.height}, got number of bands: {src.count}."
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
                    assert hasattr(method, '__name__')
                    raise ValueError(
                        f"Failed to open resulting raster image of {method.__name__}.\
                            \nGot path:\n{result}\n"
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
                            f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' "
                            + f"must return an 2D array, got array dimension: {result.ndim}"
                        )
                    else:
                        result = np.array(result)
                        if (dl_out == 7) and (result.ndim != 1):
                            raise ValueError(
                                f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' "
                                + f"must return an 1D array-like, got array dimension: {result.ndim}"
                            )
                else:
                    raise ValueError(
                        f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' "
                        + f"must return an array of numbers, got array dtype: {result.dtype}"
                    )
            else:
                raise TypeError(
                    f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' "
                    + f"must return an NumPy array-like, got: {type(result)}"
                )

        return method

    else:
        # Validate data_level
        dl_in = _dl_val(input_data_level)[0]
        dl_out = _dl_val(output_data_level)[0]
        if dl_in != 7:
            raise ValueError(
                "Method for one-dimensional standalone spectra must have input data level of 7 ('spec1d'), "
                + f"but got: {input_data_level}"
            )
        if dl_out < 7:
            raise ValueError(
                "Method for one-dimensional standalone spectra cannot have output data level below 7 ('spec1d'), "
                + f"but got level number: {dl_out}"
            )
        # TODO: if dl_out == 8:
        if dl_out > 7:
            # Model method is not validated here
            return method

        testing_data = pretest_data["spec1d"]
        assert callable(method)
        assert hasattr(method, '__name__')
        result = method(testing_data)

        # Output validation
        if result is None:
            raise ValueError(
                f"Method '{method.__name__}' returns no data. The added method must have a return. "
                + "For image processing methods, absolute path of resulting image must be returned."
            )

        # For array-like output
        result = arraylike_validator()(result)
        if type(result) is np.ndarray:
            if np.issubdtype(result.dtype, np.number):
                result = np.array(result)
                if (dl_out == 7) and (result.ndim != 1):
                    raise ValueError(
                        f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' "
                        + f"must return an 1D array-like, got array dimension: {result.ndim}"
                    )
            else:
                raise ValueError(
                    f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' "
                    + f"must return an array of numbers, got array dtype: {result.dtype}"
                )
        else:
            raise TypeError(
                f"Method with output data level '{dl_out}' or '{_dl_val(dl_out)[1]}' "
                + f"must return an NumPy array-like, got: {type(result)}"
            )

        return method


# %% Assembly method validator


# TODO: _assembly_validator
# Sample_list item: (0 - Sample id, 1 - Sample label, 2 - Validation group, 3 - Test mask, 4 - Train mask, 5 - Original shape, 6 - Target value, 7 - Sample predictor values)  # noqa: E501
# list[tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, Annotated[Any, arraylike_validator(ndim=1)]]]
@simple_type_validator
def _assembly_method_validator(
    method: Callable,
    input_data_level: Union[str, int],
    output_data_level: Union[str, int],
) -> Callable:
    """Validate the basic functionality of provided assembly method."""  # noqa: E501

    # Mock sample list
    mock_sample_list = [
        ("Mock_sample_id0", "sample_id0", "vgroup0", np.int8(1), np.int8(0), (5,), 32, np.array([1, 3, 5, 7, 9])),
        ("Mock_sample_id1", "sample_id1", "vgroup0", np.int8(1), np.int8(1), (5,), 36, np.array([2, 4, 6, 8, 10])),
        ("Mock_sample_id2", "sample_id2", "vgroup1", np.int8(1), np.int8(1), (5,), 24, np.array([0, 1, 0, 3, 5])),
        ("Mock_sample_id3", "sample_id3", "vgroup1", np.int8(1), np.int8(1), (5,), 76, np.array([3, 9, 15, 21, 27])),
        ("Mock_sample_id4", "sample_id4", "vgroup1", np.int8(0), np.int8(1), (5,), 5, np.array([0, 0, 0, 0, 0])),
    ]

    # Validate functionality of provided assembly method
    try:
        test_res = method(mock_sample_list)
    except Exception as e:
        raise ValueError(
            f"Assembly method '{method.__name__}' failed during execution on validation input data."
        ) from e
    try:
        _ = identity_assembly(test_res)
    except Exception as e:
        raise TypeError(
            f"Assembly method '{method.__name__}' returned an invalid data structure.\n"
            "Expected return type:\n"
            "list[tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, 1D array-like]]"
        ) from e

    return method


# %% Validate _sample_list_constructor resulting file integrity


# TODO: _pre_assembly_data_validator
@simple_type_validator
def _pre_assembly_data_validator(
    report_directory: str,  # swectral.report_directory
) -> None:
    """Runtime validate the integrity of sample data produced by _sample_list_constructor for assembly process."""  # noqa: E501

    # Validate constructed (assembled) sample_list data file
    finished_paths_path = report_directory + "Assembly/.__swectral_dill_data/.__sample_list_paths_finished.dill"

    # Validate finish indicator meta file
    if not os.path.exists(unc_path(finished_paths_path)):
        raise FileNotFoundError(
            "Assembled sample metadata file was not found. Preprocessing may not have completed successfully. "
            f"Expected file path:\n{finished_paths_path}"
        )

    # Validate sample list data file
    with open(finished_paths_path, "rb") as f:
        finished_paths = dill.load(f)["finished_paths"]
    for dpath in finished_paths:
        if not os.path.exists(dpath):
            raise FileNotFoundError(
                "Assembled sample data file was not found. "
                "Preprocessing may be incomplete or the output file is corrupted. "
                f"Missing data file path:\n{dpath}"
            )
        # Validate sample list data integrity
        sample_list_i = load_dill(unc_path(dpath))["chain_res"]
        try:
            _ = identity_assembly(sample_list_i)
        except Exception as e:
            raise TypeError(
                "Invalid sample data structure detected in preprocessing output file.\n"
                f"File path:\n{dpath}\n"
                "Expected data type:\n"
                "list[tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, 1D array-like]]"
            ) from e


# %% Helper: Pipeline output size estimation & disk available space validator


@simple_type_validator
def _estimate_output_raster_size(
    image_path: str,
    output_dtype: str = "float32",
    output_shape: Optional[tuple[int, int]] = None,
    output_nbands: int = -1,
) -> int:
    """
    Estimate output raster size from input raster path. Shape uses (height, width).
    """

    with rasterio.open(image_path) as src:
        input_nbands: int = src.count
        input_shape: tuple[int, int] = (src.height, src.width)

    if output_shape is None:
        height: int = input_shape[0]
        width: int = input_shape[1]
    elif output_shape[0] <= 0 or output_shape[1] <= 0:
        raise ValueError(f"output_shape dimensions must be positive integers, got: {output_shape}")
    else:
        height = output_shape[0]
        width = output_shape[1]

    if output_nbands == -1:
        bands: int = input_nbands
    elif output_nbands <= 0:
        raise ValueError(f"output_nbands must be a positive integer, got: {output_nbands}")
    else:
        bands = output_nbands

    bytes_per_pixel: int = np.dtype(output_dtype).itemsize
    total_pixels: int = width * height * bands
    total_bytes: int = total_pixels * bytes_per_pixel

    return total_bytes


# Get chains of image preprocessing
@simple_type_validator
def _num_image_chains(applied_process_chains: pd.DataFrame) -> int:
    """Get number of image preprocessing chains, duplicated image chains are not counted."""
    chains = np.asarray(applied_process_chains)
    nsteps_img = len([step for step in chains[0] if int(step[0]) < 5])
    img_chains = chains[:, :nsteps_img]
    if nsteps_img > 0:
        unique_img_chains = []
        for row in img_chains:
            rowl = list(row)
            if rowl not in unique_img_chains:
                unique_img_chains.append(rowl)
        return len(unique_img_chains)
    else:
        return 0


# Preprocessing step output image size estimator
# (n_images * (n_unique_chains of data level 0~4 + 1 (n_ROI_specs, if exists)))
# self.spec_exp.ls_images(return_dataframe=True)["Path"].tolist()
# n_roispecs_process = len(self.ls_process(input_data_level=5, output_data_level=6, print_result=False, return_result=True))  # noqa: E501
# Add 1 if the n_roispecs_process > 0 - (n_img_chains + int(roispecs))
@simple_type_validator
def _estimate_img_output_size(img_paths: list[str], n_img_chains: int, roispecs: bool) -> int:
    size = 0
    for path in img_paths:
        size = size + _estimate_output_raster_size(path)
    size = size * (n_img_chains + int(roispecs))
    return size


# Report chain plot size estimator (each size: 400 * 1024, regression 2 per chain, classification 1 per chain)
@simple_type_validator
def _estimate_report_plot_size(nchains: int, is_regression: bool) -> int:
    size = 400 * 1024 * (1 + int(is_regression)) * nchains
    return size
