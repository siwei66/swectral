# -*- coding: utf-8 -*-
"""
Group statistics - experiment group and marginal descriptive statistics

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os
import shutil

# Warnings
import warnings

# Basic data
import pandas as pd
import numpy as np

# Typing
from typing import Optional, Any

# Statistics
from scipy.stats import mannwhitneyu

# Local
from .roistats import Stats2d
from .specio import (
    RealNumber,
    simple_type_validator,
    unc_path,
    dump_dill,
    df_to_csv,
    df_from_csv,
)

# Progress
from tqdm import tqdm


# %% Experiment group sample data statistics for a chain


@simple_type_validator
def chain_sample_group_stats(  # noqa: C901
    preprocessing_chain_index: int,
    sample_data_path: str,
    sample_target_path: str,
    output_directory: str,
    is_regression: Optional[bool] = None,
    *,
    _space_wait_timeout: int = 36000,
    _reserve_free_pct: float = 5.0,
) -> None:
    """
    Compute sample X and y overall and group statistics and save to CSV files of specified preprocessing chain.
    Provided data must be "PreprocessingChainResult_chain_ind_<preprocessing_chain_index>.csv" file from SpecPipe.
    Provided targets must be modeling "sample_targets.csv" file from SpecPipe.
    """
    # Validate preprocessing chain index
    if preprocessing_chain_index < 0:
        raise ValueError(f"'preprocessing_chain_index' must be nonnegative, but got: {preprocessing_chain_index}")

    # Validate input and output paths
    # Sample data path
    if not os.path.exists(unc_path(sample_data_path)):
        raise ValueError(f"Invalid path of 'sample_data_path': {sample_data_path}")
    sdata_path_name, sdata_path_ext = os.path.splitext(sample_data_path)
    if str(sdata_path_ext).lower() != ".csv":
        raise ValueError("Sample data CSV file extension '.csv' is missing in the given 'sample_data_path'")
    cind = str(sdata_path_name).split("_")[-1]
    if cind != str(preprocessing_chain_index):
        raise ValueError(
            f"Got inconsistent 'preprocessing_chain_index' in provided data path: {sample_data_path}, "
            f"\nExpected: {preprocessing_chain_index}, Got: {cind}"
        )
    # Sample target path
    if not os.path.exists(unc_path(sample_target_path)):
        raise ValueError(f"Invalid path of 'sample_target_path': {sample_target_path}")
    if str(os.path.splitext(sample_target_path)[1]).lower() != ".csv":
        raise ValueError("Sample data CSV file extension '.csv' is missing in the given 'sample_target_path'")
    # Output dir path
    if not os.path.exists(unc_path(output_directory)):
        raise ValueError(f"Invalid path of 'output_directory': {output_directory}")
    write_dir = output_directory

    # Read preprocessed data
    # TODO: df_preprocessed = pd.read_csv(unc_path(sample_data_path), header=0).iloc[:, 1:]
    df_preprocessed = df_from_csv(csv_path=unc_path(sample_data_path), header=0).iloc[:, 1:]
    # Validate columns
    if len(df_preprocessed.columns) > 7:
        if list(df_preprocessed.columns)[0:7] == [
            "Sample_ID",
            "Label",
            "Validation_group",
            "Test",
            "Train",
            "X_shape",
            "y",
        ]:
            pass
        else:
            raise ValueError(f"Invalid sample data columns: {df_preprocessed.columns}")
    else:
        raise ValueError(f"Invalid sample data columns: {df_preprocessed.columns}")

    # Read sample groups
    # TODO: df_sample_targets = pd.read_csv(unc_path(sample_target_path))
    df_sample_targets = df_from_csv(csv_path=unc_path(sample_target_path))
    # Validate columns
    if list(df_sample_targets.columns) == [
        "Sample_ID",
        "Label",
        "Target_value",
        "Group",
        "Validation_group",
        "Test",
        "Train",
    ]:
        pass
    else:
        raise ValueError(f"Invalid sample data columns: {df_sample_targets.columns}")
    df_group = df_sample_targets.iloc[:, [0, 3]].astype(str)

    # Validate target type if not specified
    # Check numeric-like
    targets = df_preprocessed.iloc[:, 6]
    is_numeric = True
    for yi in targets:
        if not isinstance(yi, RealNumber):
            is_numeric = False
    # Auto check is_regression
    if is_regression is None:
        if is_numeric:
            is_regression = True
        else:
            is_regression = False
    # Forced is_regression
    else:
        if is_regression and (not is_numeric):
            raise ValueError(f"Got categorical target values when is_regression is True: {targets}")
        elif (not is_regression) and is_numeric:
            is_numeric = False
        else:
            pass
    # Force type of target values
    if is_regression:
        df_preprocessed['y'] = df_preprocessed['y'].astype("float")
    else:
        df_preprocessed['y'] = df_preprocessed['y'].astype('str')

    # Match group
    group = []
    for i in range(df_preprocessed.shape[0]):
        group_i = df_group['Group'][df_group['Sample_ID'] == df_preprocessed['Sample_ID'][i]].iloc[0]
        group.append(group_i)
    df_preprocessed['Group'] = group

    # Group stats column names for X and numeric y
    stats_col = ['Group'] + list(df_preprocessed.columns[7:-1])

    # Numeric targets - regression
    if is_regression:
        # Overall stats and default measures
        ostats = Stats2d().summary(df_preprocessed.iloc[:, 6:-1].values)
        # y stats
        df_ystats = pd.DataFrame(np.zeros((1 + len(df_preprocessed['Group'].unique()), 1 + len(ostats.keys()))))
        df_ystats.columns = ['Group'] + list(ostats.keys())
        df_ystats['Group'] = df_ystats['Group'].astype('str')
        ystats_row: list[Any] = ["OVERALL"]
        for m in list(ostats.keys()):
            # Y stats
            ystats_row.append(float(ostats[m][0]))
        df_ystats.iloc[0, :] = ystats_row

        # X stats
        df_xstats_dict: dict = {}
        for m in list(ostats.keys()):
            # X stats
            xstats = ostats[m][1:].tolist()
            xstats_row = [["OVERALL"] + xstats]
            dfm = pd.DataFrame(xstats_row, columns=stats_col)
            df_xstats_dict[m] = dfm

        # Group stats
        for ig, g in enumerate(list(df_preprocessed['Group'].unique())):
            gdata = df_preprocessed[df_preprocessed['Group'] == g]
            gstats = Stats2d().summary(gdata.iloc[:, 6:-1].values)
            ystats_row = [str(g)]
            for m in list(gstats.keys()):
                # Y group stats
                ystats_row.append(float(gstats[m][0]))
                # X group stats
                xstats = gstats[m][1:].tolist()
                xstats_row = [[g] + xstats]
                dfm_new = pd.DataFrame(xstats_row, columns=stats_col)
                # Update X stats dataframe
                dfm = df_xstats_dict[m]
                dfm = pd.concat((dfm, dfm_new), axis=0)
                df_xstats_dict[m] = dfm
            df_ystats.iloc[ig + 1, :] = ystats_row

        # Save results
        # Save target stats
        # TODO: changed
        # df_ystats.to_csv(
        #     unc_path(write_dir + f"PreprocessingChainResult_chain_ind_{preprocessing_chain_index}_y_stats.csv"),
        #     index=False,
        # )
        df_to_csv(
            dataframe=df_ystats,
            csv_path=unc_path(
                write_dir + f"PreprocessingChainResult_chain_ind_{preprocessing_chain_index}_y_stats.csv"
            ),  # noqa: E501
            index=False,
            space_wait_timeout=_space_wait_timeout,
            reserve_free_pct=_reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )
        # Dump y stats dill (swectral private)
        dill_result_path = write_dir + ".__swectral_dill_data/.__swectral_result_summary_sample_targets_stats.dill"
        os.makedirs(unc_path(os.path.dirname(dill_result_path)), exist_ok=True)

        # TODO: changed
        # with open(unc_path(dill_result_path), "wb") as f:
        #     dill.dump(df_ystats, f)
        dump_dill(
            df_ystats,
            target_file_path=unc_path(dill_result_path),
            backup=False,
            space_wait_timeout=_space_wait_timeout,
            reserve_free_pct=_reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )
        # Save X stats
        for m in list(gstats.keys()):
            dfm = df_xstats_dict[m]
            # TODO: changed
            # dfm.to_csv(
            #     unc_path(write_dir + f"PreprocessingChainResult_chain_ind_{preprocessing_chain_index}_X_{m}.csv"),
            #     index=False,
            # )
            df_to_csv(
                dataframe=dfm,
                csv_path=unc_path(
                    write_dir + f"PreprocessingChainResult_chain_ind_{preprocessing_chain_index}_X_{m}.csv"
                ),  # noqa: E501
                index=False,
                space_wait_timeout=_space_wait_timeout,
                reserve_free_pct=_reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )

    # Categorical targets - classification
    else:
        # Overall stats and default measures
        ostats_x = Stats2d().summary(df_preprocessed.iloc[:, 7:-1].values)
        # y stats
        ylabel, ycount = np.unique(df_preprocessed.iloc[:, 6], return_counts=True)
        df_ystats = pd.DataFrame(np.zeros((1 + len(df_preprocessed['Group'].unique()), 1 + len(ylabel))))
        df_ystats.columns = ['Group'] + list(ylabel)
        df_ystats['Group'] = df_ystats['Group'].astype('string')
        ycount_row = ["OVERALL"] + list(ycount)
        df_ystats.iloc[0, :] = ycount_row

        # X stats
        df_xstats_dict = {}
        for m in list(ostats_x.keys()):
            # X stats
            xstats = ostats_x[m].tolist()
            xstats_row = [["OVERALL"] + xstats]
            dfm = pd.DataFrame(xstats_row, columns=stats_col)
            df_xstats_dict[m] = dfm

        # Group stats
        for ig, g in enumerate(list(df_preprocessed['Group'].unique())):
            gdata = df_preprocessed[df_preprocessed['Group'] == g]
            gstats_x = Stats2d().summary(gdata.iloc[:, 7:-1].values)
            ylabel, ycount = np.unique(gdata.iloc[:, 6], return_counts=True)
            ycount_row = list(ycount)
            # Y stats - fill target category counts of the current group
            df_ystats.at[df_ystats.index[ig + 1], 'Group'] = str(g)
            for lb_i, lb in enumerate(ylabel):
                df_ystats.at[df_ystats.index[ig + 1], lb] = ycount[lb_i]
            # X stats
            for m in list(gstats_x.keys()):
                # X group stats
                xstats = gstats_x[m].tolist()
                xstats_row = [[g] + xstats]
                dfm_new = pd.DataFrame(xstats_row, columns=stats_col)
                # Update X stats dataframe
                dfm = df_xstats_dict[m]
                dfm = pd.concat((dfm, dfm_new), axis=0)
                df_xstats_dict[m] = dfm

        # Save results
        # Save target stats
        # TODO: changed
        # df_ystats.to_csv(
        #     unc_path(write_dir + f"PreprocessingChainResult_chain_ind_{preprocessing_chain_index}_y_stats.csv"),
        #     index=False,
        # )
        df_to_csv(
            dataframe=df_ystats,
            csv_path=unc_path(
                write_dir + f"PreprocessingChainResult_chain_ind_{preprocessing_chain_index}_y_stats.csv"
            ),  # noqa: E501
            index=False,
            space_wait_timeout=_space_wait_timeout,
            reserve_free_pct=_reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )
        # Dump y stats dill (swectral private)
        dill_result_path = write_dir + ".__swectral_dill_data/.__swectral_result_summary_sample_targets_stats.dill"
        os.makedirs(unc_path(os.path.dirname(dill_result_path)), exist_ok=True)
        # TODO: changed
        # with open(unc_path(dill_result_path), "wb") as f:
        #     dill.dump(df_ystats, f)
        dump_dill(
            df_ystats,
            target_file_path=unc_path(dill_result_path),
            backup=False,
            space_wait_timeout=_space_wait_timeout,
            reserve_free_pct=_reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )

        # Save X stats
        for m in list(gstats_x.keys()):
            dfm = df_xstats_dict[m]
            # TODO: changed
            # dfm.to_csv(
            #     unc_path(write_dir + f"PreprocessingChainResult_chain_ind_{preprocessing_chain_index}_X_{m}.csv"),
            #     index=False,
            # )
            df_to_csv(
                dataframe=dfm,
                csv_path=unc_path(
                    write_dir + f"PreprocessingChainResult_chain_ind_{preprocessing_chain_index}_X_{m}.csv"
                ),  # noqa: E501
                index=False,
                space_wait_timeout=_space_wait_timeout,
                reserve_free_pct=_reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )


# %% Experiment group sample data statistics for all chains


@simple_type_validator
def sample_group_stats(  # noqa: C901
    report_directory: str,
    output_directory: str = "",
    is_regression: Optional[bool] = None,
    *,
    _space_wait_timeout: int = 36000,
    _reserve_free_pct: float = 5.0,
) -> None:
    """
    Compute the descriptive statistical metrics of the preprocessed sample data and the target values.
    The metrics include mean, standard deviation, skewness, kurtosis, minimum, median, maximum and class counts for categorical target values.

    Parameters
    ----------
    report_directory : str
        report directory of the corresponding SpecPipe pipeline, the running of pipeline should be finished.
    output_directory : str
        Directory for the output of resulting statistics files.
    is_regression : Optional[bool], optional
        Whether the modeling task is regression, if None, the task is determined by the type of target values.
        The default is None.
    """  # noqa: E501
    # Data input and output paths
    sample_data_dir = f"{report_directory}/Preprocessing/".replace("\\", "/").replace("//", "/")
    sample_target_path = f"{report_directory}/Modeling/sample_targets.csv".replace("\\", "/").replace("//", "/")
    if output_directory == "":
        output_directory = sample_data_dir

    # Validate report file and dir paths
    if not os.path.exists(unc_path(sample_data_dir)):
        raise ValueError(f"Missing required file in given pipeline_config_dir: {sample_data_dir}")
    if not os.path.exists(unc_path(sample_target_path)):
        raise ValueError(f"Missing required file in given pipeline_config_dir: {sample_target_path}")
    if not os.path.exists(unc_path(output_directory)):
        raise ValueError(f"Missing required file in given pipeline_config_dir: {output_directory}")

    # Scan preprocessing result files
    preprocessing_fns = [
        str(entry.name) for entry in os.scandir(unc_path(sample_data_dir)) if len(str(entry.name)) > 39
    ]
    chain_result_fns = []
    for fn in preprocessing_fns:
        if (fn[-4:] == ".csv") and (fn[:35] == "PreprocessingChainResult_chain_ind_"):
            try:
                c_num = float(fn.replace(".csv", "").replace("PreprocessingChainResult_chain_ind_", ""))
                chain_id = int(fn.replace(".csv", "").replace("PreprocessingChainResult_chain_ind_", ""))
                if chain_id >= 0 and chain_id == c_num:
                    chain_result_fns.append(fn)
            except Exception:
                pass

    # Validate preprocessing results
    if len(chain_result_fns) < 1:
        raise ValueError(f"No preprocessing result found in the given directory: {sample_data_dir}")

    for fn in chain_result_fns:
        chain_result_path = sample_data_dir + fn
        chain_id = int(fn.replace(".csv", "").replace("PreprocessingChainResult_chain_ind_", ""))
        chain_sample_group_stats(
            preprocessing_chain_index=chain_id,
            sample_data_path=chain_result_path,
            sample_target_path=sample_target_path,
            output_directory=output_directory,
            is_regression=is_regression,
            _space_wait_timeout=_space_wait_timeout,
            _reserve_free_pct=_reserve_free_pct,
        )

    # Add y stats to modeling targets dir
    shutil.copyfile(
        unc_path(f"{output_directory}PreprocessingChainResult_chain_ind_0_y_stats.csv"),
        unc_path(f"{sample_target_path[:-4] + '_stats' + sample_target_path[-4:]}"),
    )


# %% Process ID label converters


def process_id_label_lookup_dict(process_config_df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """
    Generate process ID to label and label to ID lookup dictionaries.
    Prioritize custom process label to method name.
    """

    # Validate ID / label duplication
    df_process = process_config_df
    duplicate_ids = df_process.loc[df_process.duplicated(subset=['ID']), 'ID']
    if len(duplicate_ids) > 0:
        raise ValueError(f"Found duplicated IDs: {duplicate_ids.unique().tolist()}")
    duplicate_labels = df_process.loc[df_process.duplicated(subset=['Process_label']), 'Process_label'].dropna()
    if len(duplicate_labels) > 0:
        raise ValueError(f"Found duplicated labels: {duplicate_labels.unique().tolist()}")

    # Construct label-ID lookup dictionaries
    df_proc = process_config_df.copy()
    df_proc["Final_label"] = df_proc["Process_label"].combine_first(df_proc["Method"])
    duplicate_flabels = df_proc.loc[df_proc.duplicated(subset=['Final_label']), 'Final_label']
    if len(duplicate_flabels) > 0:
        raise ValueError(f"Found duplicated labels: {duplicate_flabels.unique().tolist()}")
    proc_id_to_label: dict = dict(zip(df_proc["ID"], df_proc["Final_label"]))
    proc_label_to_id: dict = dict(zip(df_proc["Final_label"], df_proc["ID"]))

    return (proc_id_to_label, proc_label_to_id)


def process_id_to_label(process_id: str, proc_id_to_label: dict, ignore: bool = False) -> str:
    """
    Convert unique SpecPipe process ID to process label. If ignore True, return input if input is not id.
    "process_config_df" is the SpecPipe_added_process.csv in the configuration subdir.
    """
    if process_id in proc_id_to_label.keys():
        process_label = str(proc_id_to_label[process_id])
        return process_label
    elif not ignore:
        raise ValueError(f"No process label or method name found for given ID: {process_id}")
    else:
        return process_id


def process_label_to_id(process_label: str, proc_label_to_id: dict) -> str:
    """
    Convert unique SpecPipe process label to process ID.
    "process_config_df" is the SpecPipe_added_process.csv in the configuration subdir.
    """
    # Validate whether the process_label is ID, return if it's ID
    if "_%#" in process_label:
        process_label1 = process_label.replace("_%#", "_")
        splited_proc = process_label1.split("_")
        try:
            if (
                int(splited_proc[0]) == float(splited_proc[0])
                and int(splited_proc[1]) == float(splited_proc[1])
                and int(splited_proc[2]) == float(splited_proc[2])
            ):
                return process_label
        except Exception:
            pass

    if process_label in proc_label_to_id.keys():
        process_id = str(proc_label_to_id[process_label])
        return process_id
    else:
        raise ValueError(f"No process ID found for given label: {process_label}")


# %% Model performance summary and marginal performance statistics


# Collect performance metrics
def performance_metrics_summary(  # noqa: C901
    pipeline_config_dir: str,
    model_evaluation_report_dir: str,
    *,
    _space_wait_timeout: int = 36000,
    _reserve_free_pct: float = 5.0,
) -> dict[str, Any]:
    """
    Collect performance metrics from SpecPipe model evaluation reports.

    Parameters
    ----------
    pipeline_config_dir : str
        SpecPipe configuration directory.
    model_evaluation_report_dir : str
        SpecPipe model evaluation reports directory.

    Returns
    -------
    dict
        For classification, returns a dictionary of "is_regression", "chains_in_ID" and "macro_metrics" and "micro_metrics".

        For regression, returns a dictionary of "is_regression", "chains_in_ID" and "regression_metrics" for regression.

    Examples
    --------
    performance_metrics_summary(
        pipeline_config_dir = "report_root/SpecPipe_configuration/",
        model_evaluation_report_dir = "report_root/Modeling/Model_evaluation_reports/",
    )
    """  # noqa: E501

    config_dir = (pipeline_config_dir.replace("\\", "/") + "/").replace("//", "/")
    report_dir = (model_evaluation_report_dir.replace("\\", "/") + "/").replace("//", "/")
    # TODO: process_config_df = pd.read_csv(unc_path(config_dir + "SpecPipe_added_process.csv"))
    process_config_df = df_from_csv(csv_path=unc_path(config_dir + "SpecPipe_added_process.csv"))

    # Construct label-ID lookup dictionaries
    proc_id_to_label: dict[str, str]
    proc_label_to_id: dict[str, str]
    proc_id_to_label, proc_label_to_id = process_id_label_lookup_dict(process_config_df)

    # TODO: changed for compressed scenarios
    # # Chains path
    # chains_id_path = config_dir + "SpecPipe_exec_chains_in_ID.csv"
    # chains_label_path = config_dir + "SpecPipe_exec_chains_in_label.csv"

    # # Validate paths
    # if not os.path.exists(unc_path(chains_id_path)):
    #     raise ValueError(f"Missing required file in given pipeline_config_dir: {chains_id_path}")
    # if not os.path.exists(unc_path(chains_label_path)):
    #     raise ValueError(f"Missing required file in given pipeline_config_dir: {chains_label_path}")

    # Allowed compressed CSV file extensions
    ext_compress_allowed: list = [".gz", ".bz2", ".zip", ".xz", ".zst", ""]

    # Checking for the base CSV or any compressed variant in ext_map
    chains_id_path: Optional[str] = next(
        (
            pathi
            for exti in ext_compress_allowed
            if os.path.exists(unc_path(pathi := f"{config_dir}SpecPipe_exec_chains_in_ID.csv{exti}"))
        ),
        None,
    )
    chains_label_path: Optional[str] = next(
        (
            pathi
            for exti in ext_compress_allowed
            if os.path.exists(unc_path(pathi := f"{config_dir}SpecPipe_exec_chains_in_label.csv{exti}"))
        ),
        None,
    )

    # Validate path found
    if not chains_id_path:
        raise ValueError(f"Missing ID file (CSV/compressed) in: {config_dir}")
    if not chains_label_path:
        raise ValueError(f"Missing Label file (CSV/compressed) in: {config_dir}")

    # Chains
    # TODO: df_cid = pd.read_csv(unc_path(chains_id_path))
    df_cid = df_from_csv(csv_path=unc_path(chains_id_path))
    # TODO: df_clab = pd.read_csv(unc_path(chains_label_path))
    df_clab = df_from_csv(csv_path=unc_path(chains_label_path))

    # Validate results
    # Configuration chains
    df_config_chains = df_cid.copy(deep=True)
    df_config_chains.iloc[:, -1] = df_clab.iloc[:, -1]
    config_chains = []
    for chain in df_config_chains.values.tolist():
        chain = tuple(chain)
        config_chains.append(chain)

    # Report result chains
    # Reconstruct chains
    dir_names = [
        entry.name
        for entry in os.scandir(unc_path(report_dir))
        if entry.is_dir() and "Data_chain_Preprocessing_#" in entry.name and "_Model_" in entry.name
    ]
    if len(dir_names) < 1:
        raise ValueError(f"No model evaluation report found in the given report path: {report_dir}")
    chain_txt_names = [
        entry.name
        for entry in os.scandir(unc_path(report_dir))
        if entry.is_file() and ".txt" in entry.name and "Preprocessing_#" in entry.name
    ]
    if len(chain_txt_names) < 1:
        raise ValueError(f"No preprocessing chain information found in the given report path: {report_dir}")

    # Read chain content
    result_chains_dir_map: dict = {}
    for dir_name in dir_names:
        dn_split = dir_name.replace("Data_chain_Preprocessing_#", "##split#block##").replace(
            "_Model_", "##split#block##"
        )
        # Validate eva report dir name
        dn_splited = dn_split.split("##split#block##")
        validated = False
        if dn_splited[0] == "" and len(dn_splited) == 3:
            try:
                # Preprocessing and assembly numbers
                index_str = dn_splited[1]
                if "_a&" in index_str:
                    pachain_ids = index_str.split("_a&")
                    # Validate preproc chain id
                    preproc_id = pachain_ids[0]
                    assert int(preproc_id) == float(preproc_id)
                    # Validate assem chain ids
                    assembly_ids = pachain_ids[1:]
                    for assembly_id in assembly_ids:
                        assembly_stepn, astepn_id = assembly_id.split("&")
                        assert int(assembly_stepn) == float(assembly_stepn)
                        assert int(astepn_id) == float(astepn_id)
                else:
                    assert int(index_str) == float(index_str)
                validated = True
            except Exception:
                validated = False
        if not validated:
            raise ValueError(f"Invalid directory name format for model evaluation report: {dir_name}")
        # Get model name
        model_name = dn_splited[2]
        # Get chain number
        chain_num = dn_splited[1]
        chain_txt_found = [txt_name for txt_name in chain_txt_names if f"Preprocessing_#{chain_num}.txt" == txt_name]
        if len(chain_txt_found) == 1:
            chain_txt = chain_txt_found[0]
        else:
            raise ValueError(
                f"None or multiple preprocessing chain file found for 'Preprocessing_#{chain_num}', "
                f"got chain file names: {chain_txt_names}"
            )
        # Get preprocessing chain
        with open(unc_path(report_dir + chain_txt), "r", encoding="utf-8") as f:
            steps_list: list = [line.strip() for line in f.readlines()]
        steps: tuple = tuple(steps_list + [model_name])
        # Add to full chain
        result_chains_dir_map[steps] = dir_name

    # Validate results and configuration consistency
    if set(config_chains) != set(result_chains_dir_map.keys()):
        raise ValueError(
            f"Pipeline model evaluation reports imply inconsistent processing chains with pipeline configurations:\n"
            f"Configured chains:\n{config_chains},\nReport implied chains:\n{result_chains_dir_map.keys()}\n"
        )

    # Reorder chains to match configuration
    ordered_result_chains = []
    ordered_dir_names = []
    for chain in config_chains:
        if chain not in result_chains_dir_map:
            raise ValueError(f"Configured chain {chain} not found in report directories")
        ordered_result_chains.append(chain)
        ordered_dir_names.append(result_chains_dir_map[chain])

    # Get chain model performance metrics
    metrics_micro = []
    metrics_macro = []
    metrics_reg = []
    print("\nCollect model performances...")
    for dir_name, result_chain in tqdm(zip(ordered_dir_names, ordered_result_chains), total=len(ordered_dir_names)):
        # Validate result chain - all item to process IDs
        result_chain1 = []
        for proc_item in result_chain:
            result_chain1.append(process_label_to_id(proc_item, proc_label_to_id))
        result_chain = tuple(result_chain1)
        # Metrics directory
        metrics_dir = f"{report_dir}{dir_name}/"
        # Save processes of the full chain
        cprocs_in_id = result_chain1
        cprocs_in_label = [process_id_to_label(proc_id, proc_id_to_label) for proc_id in cprocs_in_id]
        df_cprocs = pd.DataFrame({"Chain_in_process_ID": cprocs_in_id, "Chain_in_process_label": cprocs_in_label})
        # Dump dill (swectral private)
        dill_result_path = metrics_dir + ".__swectral_dill_data/.__swectral_core_result_Chain_process_info.dill"
        os.makedirs(unc_path(os.path.dirname(dill_result_path)), exist_ok=True)
        # TODO: changed
        # with open(unc_path(dill_result_path), "wb") as f:
        #     dill.dump(df_cprocs, f)
        dump_dill(
            df_cprocs,
            target_file_path=unc_path(dill_result_path),
            backup=False,
            space_wait_timeout=_space_wait_timeout,
            reserve_free_pct=_reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )
        # Read performance metrics
        metrics_filename = [
            entry.name
            for entry in os.scandir(unc_path(metrics_dir))
            if f"_performance_{dir_name.split('_Model_')[-1]}.csv" in entry.name
        ][0]
        # TODO: df_metrics = pd.read_csv(unc_path(f"{report_dir}{dir_name}/{metrics_filename}"))
        df_metrics = df_from_csv(csv_path=unc_path(f"{report_dir}{dir_name}/{metrics_filename}"))
        if "Classification_performance_" in metrics_filename:
            # micro metrics
            micro_metrics = (
                df_metrics.loc[
                    df_metrics['Class'] == "Micro_avg", ["Precision", "Recall", "F1_Score", "Accuracy", "AUC"]
                ]
                .to_numpy()
                .tolist()[0]
            )
            metrics_micro.append(tuple(list(result_chain) + cprocs_in_label + micro_metrics))
            # macro metrics
            macro_metrics = (
                df_metrics.loc[
                    df_metrics['Class'] == "Macro_avg", ["Precision", "Recall", "F1_Score", "Accuracy", "AUC"]
                ]
                .to_numpy()
                .tolist()[0]
            )
            metrics_macro.append(tuple(list(result_chain) + cprocs_in_label + macro_metrics))
        elif "Regression_performance_" in metrics_filename:
            # Regression metrics
            reg_metrics = df_metrics.iloc[[0], :].to_numpy().tolist()[0]
            metrics_reg.append(tuple(list(result_chain) + cprocs_in_label + reg_metrics))
        else:
            raise ValueError(f"Invalid performance file name: {dir_name}")

    # Convert to metrics dataframe
    if len(metrics_micro) > 0 and len(metrics_macro) > 0 and len(metrics_reg) == 0:
        df_micro_metrics = pd.DataFrame(metrics_micro)
        df_micro_metrics.columns = (
            list(df_config_chains.columns)
            + [f"{s}_name" for s in list(df_config_chains.columns)]
            + [
                "Precision",
                "Recall",
                "F1_Score",
                "Accuracy",
                "AUC",
            ]
        )
        df_macro_metrics = pd.DataFrame(metrics_macro)
        df_macro_metrics.columns = (
            list(df_config_chains.columns)
            + [f"{s}_name" for s in list(df_config_chains.columns)]
            + [
                "Precision",
                "Recall",
                "F1_Score",
                "Accuracy",
                "AUC",
            ]
        )
        # Output results
        metrics_dict = {
            "is_regression": False,
            "chains_in_ID": df_cid,
            "macro_metrics": df_macro_metrics,
            "micro_metrics": df_micro_metrics,
        }
        return metrics_dict
    elif len(metrics_micro) == 0 and len(metrics_macro) == 0 and len(metrics_reg) > 0:
        df_reg_metrics = pd.DataFrame(metrics_reg)
        df_reg_metrics.columns = (
            list(df_config_chains.columns)
            + [f"{s}_name" for s in list(df_config_chains.columns)]
            + list(df_metrics.columns)
        )
        # Output results
        metrics_dict = {"is_regression": True, "chains_in_ID": df_cid, "regression_metrics": df_reg_metrics}
        return metrics_dict
    else:
        raise ValueError(
            f"Got corrupted performance data:\n\n\
                         metrics_micro: {metrics_micro}\n\n\
                         metrics_macro: {metrics_macro}\n\n\
                         metrics_macro: {metrics_reg}"
        )


# Marginal performance statistics for regression
def regression_performance_marginal_stats(  # noqa: C901
    metrics_dict: dict[str, Any],
    pipeline_config_dir: str,
    model_evaluation_report_dir: str,
    validate_process: bool = True,
    *,
    _space_wait_timeout: int = 36000,
    _reserve_free_pct: float = 5.0,
) -> dict[str, Any]:
    """
    Compute marginal performance statistics using the result dictionary from function 'performance_metrics_summary'.
    """
    # Validate model_evaluation_report_dir
    report_dir = (model_evaluation_report_dir.replace("\\", "/") + "/").replace("//", "/")
    if not os.path.exists(unc_path(report_dir)):
        raise ValueError(f"Invalid 'model_evaluation_report_dir': {report_dir}")

    # Get summarized metrics data and corresponding chains
    df_cid = metrics_dict["chains_in_ID"]
    df_reg_metrics = metrics_dict["regression_metrics"]
    config_dir = pipeline_config_dir

    # Load config and create a lookup dictionary
    # TODO: process_config_df = pd.read_csv(unc_path(config_dir + "SpecPipe_added_process.csv"))
    process_config_df = df_from_csv(csv_path=unc_path(config_dir + "SpecPipe_added_process.csv"))

    # Construct label-ID lookup dictionaries
    proc_id_to_label: dict[str, str]
    proc_label_to_id: dict[str, str]
    proc_id_to_label, proc_label_to_id = process_id_label_lookup_dict(process_config_df)

    # Compute and output marginal perf stats of each step
    marginal_performance_stats: dict = {}
    print("\nAnalyze marginal performance...")
    for step in tqdm(list(df_cid.columns), total=len(list(df_cid.columns))):
        # Step process IDs
        step_process_ids = list(df_cid[step].unique())

        # Group by step method
        grouped_data = df_reg_metrics.groupby(step)["R2"].apply(list).to_dict()

        # Aggregate group of all records
        r2_all = list(df_reg_metrics.loc[:, "R2"])

        all_ids = ['All'] + step_process_ids
        group_r2 = {'All': r2_all}
        group_r2.update(grouped_data)

        # Pre-allocate matrix for faster numpy location
        num_stats_rows = 6
        num_mwu_rows = len(all_ids)
        matrix_r2 = np.empty((num_stats_rows + num_mwu_rows, len(all_ids)), dtype=object)

        for col_idx, pid1 in enumerate(all_ids):
            r2_1 = group_r2[pid1]

            # Label row
            if pid1 == 'All':
                matrix_r2[0, col_idx] = "All"
            else:
                label_val = process_id_to_label(pid1, proc_id_to_label, ignore=(not validate_process))
                matrix_r2[0, col_idx] = label_val

            # Stats rows
            matrix_r2[1, col_idx] = len(r2_1)
            matrix_r2[2, col_idx] = np.nanmean(r2_1)
            matrix_r2[3, col_idx] = np.nanmin(r2_1)
            matrix_r2[4, col_idx] = np.nanmedian(r2_1)
            matrix_r2[5, col_idx] = np.nanmax(r2_1)

            # Stat test rows
            for row_offset, pid2 in enumerate(all_ids):
                r2_2 = group_r2[pid2]
                if len(step_process_ids) > 1:
                    # Identity cases
                    if pid1 == pid2:
                        matrix_r2[row_offset + 6, col_idx] = 1.0
                    else:
                        matrix_r2[row_offset + 6, col_idx] = mannwhitneyu(r2_1, r2_2)[1]
                else:
                    matrix_r2[row_offset + 6, col_idx] = np.nan

        # Construct DataFrames from matrices
        step_gstats_r2 = pd.DataFrame(matrix_r2, columns=all_ids)

        # Add Process_ID column with original string identifiers
        desc_col = ["Process_label", "n_records", "Mean_R2", "Min_R2", "Median_R2", "Max_R2"] + all_ids
        step_gstats_r2.insert(0, "Process_ID", desc_col)

        # Collect step result
        marginal_performance_stats[step] = {"r2": step_gstats_r2, "summary": df_reg_metrics}

        # Save step result
        if len(step_process_ids) > 1:
            # TODO: step_gstats_r2.to_csv(unc_path(report_dir + f"Marginal_R2_stats_{str(step).lower()}.csv"), index=False)  # noqa: E501
            df_to_csv(
                dataframe=step_gstats_r2,
                csv_path=unc_path(report_dir + f"Marginal_R2_stats_{str(step).lower()}.csv"),
                index=False,
                space_wait_timeout=_space_wait_timeout,
                reserve_free_pct=_reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )
            dill_result_path = (
                report_dir
                + f".__swectral_dill_data/.__swectral_result_summary_Marginal_R2_stats_{str(step).lower()}.dill"
            )
            os.makedirs(unc_path(os.path.dirname(dill_result_path)), exist_ok=True)
            # TODO: changed
            # with open(unc_path(dill_result_path), "wb") as f:
            #     dill.dump(step_gstats_r2, f)
            dump_dill(
                step_gstats_r2,
                target_file_path=unc_path(dill_result_path),
                backup=False,
                space_wait_timeout=_space_wait_timeout,
                reserve_free_pct=_reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )

    # Save summaries
    # TODO: df_reg_metrics.to_csv(unc_path(report_dir + "Performance_summary.csv"), index=False)
    df_to_csv(
        dataframe=df_reg_metrics,
        csv_path=unc_path(report_dir + "Performance_summary.csv"),
        index=False,
        space_wait_timeout=_space_wait_timeout,
        reserve_free_pct=_reserve_free_pct,
        min_sec_random_wait=5.0,
        max_sec_random_wait=5.0,
    )
    # TODO: new, new columns
    # Add report subdir names for chain report location
    map_preprocessing_files(
        csv_name="Performance_summary.csv",
        result_directory=report_dir,
        _space_wait_timeout=_space_wait_timeout,
        _reserve_free_pct=_reserve_free_pct,
    )
    # Save dill
    dill_result_path = report_dir + ".__swectral_dill_data/.__swectral_result_summary_Performance_summary.dill"
    os.makedirs(unc_path(os.path.dirname(dill_result_path)), exist_ok=True)
    # TODO: changed
    # with open(unc_path(dill_result_path), "wb") as f:
    #     dill.dump(df_reg_metrics, f)
    dump_dill(
        df_reg_metrics,
        target_file_path=unc_path(dill_result_path),
        backup=False,
        space_wait_timeout=_space_wait_timeout,
        reserve_free_pct=_reserve_free_pct,
        min_sec_random_wait=5.0,
        max_sec_random_wait=5.0,
    )

    return marginal_performance_stats


def classification_performance_marginal_stats(  # noqa: C901
    metrics_dict: dict[str, Any],
    pipeline_config_dir: str,
    model_evaluation_report_dir: str,
    validate_process: bool = True,
    *,
    _space_wait_timeout: int = 36000,
    _reserve_free_pct: float = 5.0,
) -> dict[str, Any]:
    """
    Compute marginal performance statistics using the result dictionary from function 'performance_metrics_summary'.
    """
    # Validate model_evaluation_report_dir
    report_dir = (model_evaluation_report_dir.replace("\\", "/") + "/").replace("//", "/")
    if not os.path.exists(unc_path(report_dir)):
        raise ValueError(f"Invalid 'model_evaluation_report_dir': {report_dir}")

    df_cid = metrics_dict["chains_in_ID"]
    df_macro_metrics = metrics_dict["macro_metrics"]
    df_micro_metrics = metrics_dict["micro_metrics"]
    config_dir = pipeline_config_dir

    # Load config and create a lookup dictionary
    # TODO: process_config_df = pd.read_csv(unc_path(config_dir + "SpecPipe_added_process.csv"))
    process_config_df = df_from_csv(csv_path=unc_path(config_dir + "SpecPipe_added_process.csv"))

    # Construct label-ID lookup dictionaries
    proc_id_to_label: dict[str, str]
    proc_label_to_id: dict[str, str]
    proc_id_to_label, proc_label_to_id = process_id_label_lookup_dict(process_config_df)

    # Compute and output marginal perf stats of each step
    marginal_performance_stats: dict = {}
    print("\nAnalyze marginal performance...")
    for step in tqdm(list(df_cid.columns), total=len(list(df_cid.columns))):
        step_process_ids = list(df_cid[step].unique())

        # Group by step method and convert to dict
        grouped_macro = df_macro_metrics.groupby(step)["AUC"].apply(list).to_dict()
        grouped_micro = df_micro_metrics.groupby(step)["AUC"].apply(list).to_dict()

        all_ids = ['All'] + step_process_ids
        group_macauc = {'All': list(df_macro_metrics["AUC"])}
        group_macauc.update(grouped_macro)
        group_micauc = {'All': list(df_micro_metrics["AUC"])}
        group_micauc.update(grouped_micro)

        # Pre-allocate matrix for faster numpy location
        num_rows = 6 + len(all_ids)
        matrix_macauc = np.empty((num_rows, len(all_ids)), dtype=object)
        matrix_micauc = np.empty((num_rows, len(all_ids)), dtype=object)

        for col_idx, pid1 in enumerate(all_ids):
            mac_1, mic_1 = group_macauc[pid1], group_micauc[pid1]

            # Label row
            if pid1 == 'All':
                matrix_macauc[0, col_idx] = "All"
                matrix_micauc[0, col_idx] = "All"
            else:
                label_val = process_id_to_label(pid1, proc_id_to_label, ignore=(not validate_process))
                matrix_macauc[0, col_idx] = label_val
                matrix_micauc[0, col_idx] = label_val

            # Stats rows
            for m, vals in [(matrix_macauc, mac_1), (matrix_micauc, mic_1)]:
                m[1, col_idx] = len(vals)
                m[2, col_idx] = np.nanmean(vals)
                m[3, col_idx] = np.nanmin(vals)
                m[4, col_idx] = np.nanmedian(vals)
                m[5, col_idx] = np.nanmax(vals)

            # Stat test rows
            for row_offset, pid2 in enumerate(all_ids):
                if len(step_process_ids) > 1:
                    # Identity cases
                    if pid1 == pid2:
                        p_mac, p_mic = 1.0, 1.0
                    else:
                        p_mac = mannwhitneyu(mac_1, group_macauc[pid2])[1]
                        p_mic = mannwhitneyu(mic_1, group_micauc[pid2])[1]
                    matrix_macauc[row_offset + 6, col_idx] = p_mac
                    matrix_micauc[row_offset + 6, col_idx] = p_mic
                else:
                    matrix_macauc[row_offset + 6, col_idx] = np.nan
                    matrix_micauc[row_offset + 6, col_idx] = np.nan

        # Construct DataFrames from matrices
        step_gstats_macauc = pd.DataFrame(matrix_macauc, columns=all_ids)
        step_gstats_micauc = pd.DataFrame(matrix_micauc, columns=all_ids)

        # Add Process_ID column with original string identifiers
        desc_mac = [
            "Process_label",
            "n_records",
            "Mean_AUC_macro",
            "Min_AUC_macro",
            "Median_AUC_macro",
            "Max_AUC_macro",
        ] + all_ids
        desc_mic = [
            "Process_label",
            "n_records",
            "Mean_AUC_micro",
            "Min_AUC_micro",
            "Median_AUC_micro",
            "Max_AUC_micro",
        ] + all_ids

        step_gstats_macauc.insert(0, "Process_ID", desc_mac)
        step_gstats_micauc.insert(0, "Process_ID", desc_mic)

        # Collect and Save logic
        marginal_performance_stats[step] = {"macro_auc": step_gstats_macauc, "micro_auc": step_gstats_micauc}

        if len(step_process_ids) > 1:
            # Macro
            # TODO: changed
            # step_gstats_macauc.to_csv(
            #     unc_path(report_dir + f"Marginal_macro_avg_AUC_stats_{str(step).lower()}.csv"), index=False
            # )
            df_to_csv(
                dataframe=step_gstats_macauc,
                csv_path=unc_path(report_dir + f"Marginal_macro_avg_AUC_stats_{str(step).lower()}.csv"),
                index=False,
                space_wait_timeout=_space_wait_timeout,
                reserve_free_pct=_reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )
            dill_path = unc_path(
                report_dir
                + ".__swectral_dill_data/"
                + f".__swectral_result_summary_Marginal_macro_avg_AUC_stats_{str(step).lower()}.dill"
            )
            os.makedirs(os.path.dirname(dill_path), exist_ok=True)
            # TODO: changed
            # with open(dill_path, "wb") as f:
            #     dill.dump(step_gstats_macauc, f)
            dump_dill(
                step_gstats_macauc,
                target_file_path=unc_path(dill_path),
                backup=False,
                space_wait_timeout=_space_wait_timeout,
                reserve_free_pct=_reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )
            # Micro
            # TODO: changed
            # step_gstats_micauc.to_csv(
            #     unc_path(report_dir + f"Marginal_micro_avg_AUC_stats_{str(step).lower()}.csv"),
            #     index=False,
            # )
            df_to_csv(
                dataframe=step_gstats_micauc,
                csv_path=unc_path(report_dir + f"Marginal_micro_avg_AUC_stats_{str(step).lower()}.csv"),
                index=False,
                space_wait_timeout=_space_wait_timeout,
                reserve_free_pct=_reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )
            dill_path = unc_path(
                report_dir
                + ".__swectral_dill_data/"
                + f".__swectral_result_summary_Marginal_micro_avg_AUC_stats_{str(step).lower()}.dill"
            )
            # TODO: changed
            # with open(dill_path, "wb") as f:
            #     dill.dump(step_gstats_micauc, f)
            dump_dill(
                step_gstats_micauc,
                target_file_path=unc_path(dill_path),
                backup=False,
                space_wait_timeout=_space_wait_timeout,
                reserve_free_pct=_reserve_free_pct,
                min_sec_random_wait=5.0,
                max_sec_random_wait=5.0,
            )

    # Save summaries
    marginal_performance_stats.update({"macro_summary": df_macro_metrics, "micro_summary": df_micro_metrics})
    for df_sum, prefix in [(df_macro_metrics, "Macro"), (df_micro_metrics, "Micro")]:
        # TODO: df_sum.to_csv(unc_path(report_dir + f"{prefix}_avg_performance_summary.csv"), index=False)
        df_to_csv(
            dataframe=df_sum,
            csv_path=unc_path(report_dir + f"{prefix}_avg_performance_summary.csv"),
            index=False,
            space_wait_timeout=_space_wait_timeout,
            reserve_free_pct=_reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )
        # TODO: new, new columns
        # Add report subdir names for chain report location
        map_preprocessing_files(
            csv_name=f"{prefix}_avg_performance_summary.csv",
            result_directory=report_dir,
            _space_wait_timeout=_space_wait_timeout,
            _reserve_free_pct=_reserve_free_pct,
        )
        # Save dill
        dill_path = unc_path(
            report_dir + f".__swectral_dill_data/.__swectral_result_summary_{prefix}_avg_performance_summary.dill"
        )
        os.makedirs(os.path.dirname(dill_path), exist_ok=True)
        # TODO: changed
        # with open(dill_path, "wb") as f:
        #     dill.dump(df_sum, f)
        dump_dill(
            df_sum,
            target_file_path=unc_path(dill_path),
            backup=False,
            space_wait_timeout=_space_wait_timeout,
            reserve_free_pct=_reserve_free_pct,
            min_sec_random_wait=5.0,
            max_sec_random_wait=5.0,
        )

    return marginal_performance_stats


# Marginal performance statistics
@simple_type_validator
def performance_marginal_stats(
    report_directory: str,
    metrics_dict: Optional[dict[str, Any]] = None,
    *,
    _space_wait_timeout: int = 36000,
    _reserve_free_pct: float = 5.0,
) -> dict[str, Any]:
    """
    Compute marginal model performance statistics and summary of model performance metrics from SpecPipe model evaluation reports.

    Parameters
    ----------
    pipeline_config_dir : str
        Root of SpecPipe report directory.
    metrics_dict: dict or None
        Dictionary of performance summary from ``performance_metrics_summary``.
        If provided, the summary is skipped. Default is None.

    Returns
    -------
    dict[str, Any]
        Dictionary of marginal model performance statistics and summary of model performance metrics at each step.
    """  # noqa: E501
    pipeline_config_dir = f"{report_directory}/SpecPipe_configuration/"
    model_evaluation_report_dir = f"{report_directory}/Modeling/Model_evaluation_reports/"

    # Validate metrics_dict
    compute_metrics_dict: bool
    if metrics_dict is None:
        compute_metrics_dict = True
    elif set(metrics_dict.keys()) == {
        "is_regression",
        "chains_in_ID",
        "macro_metrics",
        "micro_metrics",
    }:
        compute_metrics_dict = False
    elif set(metrics_dict.keys()) == {
        "is_regression",
        "chains_in_ID",
        "regression_metrics",
    }:
        compute_metrics_dict = False
    else:
        compute_metrics_dict = True
        warnings.warn(
            f"\n\nCorrupted metrics_dict, got keys: {metrics_dict.keys()}\nThe metrics_dict is recomputed.\n\n",
            stacklevel=2,
        )

    # Summarize performance
    if compute_metrics_dict:
        metrics_dict = performance_metrics_summary(
            pipeline_config_dir=pipeline_config_dir,
            model_evaluation_report_dir=model_evaluation_report_dir,
            _space_wait_timeout=_space_wait_timeout,
            _reserve_free_pct=_reserve_free_pct,
        )

    # Compute stats
    assert metrics_dict is not None
    if metrics_dict["is_regression"]:
        marginal_performance_stats = regression_performance_marginal_stats(
            metrics_dict,
            pipeline_config_dir,
            model_evaluation_report_dir,
            _space_wait_timeout=_space_wait_timeout,
            _reserve_free_pct=_reserve_free_pct,
        )
    else:
        marginal_performance_stats = classification_performance_marginal_stats(
            metrics_dict,
            pipeline_config_dir,
            model_evaluation_report_dir,
            _space_wait_timeout=_space_wait_timeout,
            _reserve_free_pct=_reserve_free_pct,
        )
    return marginal_performance_stats


# %% Supplementary: column for result searching

# TODO: add new columns of report subdir names for conveniently chain report location


def get_step_index(column_name: str) -> int:
    """Extract the numeric index from 'Step_n' for sorting."""
    return int([p for p in column_name.split('_') if p.isdigit()][0])


def match_row_to_file(
    row: pd.Series, match_cols: list[str], model_cols: list[str], txt_lookup: dict[tuple[str, ...], str]
) -> Optional[str]:
    """Compare CSV row values against the pre-loaded text file data."""
    # Get row steps
    current_steps = []
    for col in match_cols:
        current_steps.append(str(row[col]))

    current_steps_tuple = tuple(current_steps)

    # Get result dir preprocessing number
    filename = txt_lookup.get(current_steps_tuple)
    if not filename:
        return None

    prefix = filename.replace(".txt", "")
    model_vals = [str(row[c]).strip() for c in model_cols if str(row[c]).lower() != 'nan']
    return f"Data_chain_{prefix}_Model_{'_'.join(model_vals)}"


# Add a column to performance summary CSV tables for the convenience of chain report dir location
def map_preprocessing_files(
    csv_name: str,
    result_directory: str,
    *,
    _space_wait_timeout: int = 60,
    _reserve_free_pct: float = 3.0,
) -> None:
    """
    Append txt filenames to a CSV based on Step ID matches, changes are inplace.
    """

    from glob import glob

    csv_path = os.path.join(result_directory, csv_name)
    df_summary = df_from_csv(csv_path=unc_path(csv_path))

    # Identify and sort Step columns
    step_id_cols: list[str] = []
    model_step_cols: list[str] = []
    for col in df_summary.columns:
        c_str = str(col)
        parts = c_str.split('_')
        if len(parts) == 2 and parts[0] == 'Step' and parts[1].isdigit():
            step_id_cols.append(c_str)
        elif c_str.startswith('Model_step_'):
            model_step_cols.append(c_str)

    # Sort columns
    step_id_cols.sort(key=get_step_index)
    model_step_cols.sort(key=get_step_index)

    # Exclude the last Step
    match_cols = step_id_cols[:-1]

    # Lookup dictionary from .txt files
    txt_lookup = {}
    txt_pattern = os.path.join(result_directory, "Preprocessing_#*.txt")

    for txt_path in glob(txt_pattern):
        with open(txt_path, 'r') as f:
            # Read lines and clean up whitespace
            lines = []
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    lines.append(clean_line)

            content_key = tuple(lines)
            filename = os.path.basename(txt_path)
            txt_lookup[content_key] = filename

    # Apply the matching
    df_summary['Result_subdirectory'] = df_summary.apply(
        match_row_to_file,
        axis=1,
        args=(match_cols, model_step_cols, txt_lookup),
    )

    # Save result
    output_path = os.path.join(result_directory, csv_name)
    df_to_csv(
        dataframe=df_summary,
        csv_path=unc_path(output_path),
        index=False,
        space_wait_timeout=_space_wait_timeout,
        reserve_free_pct=_reserve_free_pct,
        min_sec_random_wait=5.0,
        max_sec_random_wait=5.0,
    )
