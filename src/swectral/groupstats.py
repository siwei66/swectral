# -*- coding: utf-8 -*-
"""
Group statistics - experiment group and marginal descriptive statistics

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os
import shutil

# Basic data
import pandas as pd
import numpy as np

# Typing
from typing import Optional, Any

# File
import dill

# Statistics
from scipy.stats import mannwhitneyu

# Local
from .roistats import Stats2d
from .specio import RealNumber, simple_type_validator, unc_path


# %% Experiment group sample data statistics for a chain


@simple_type_validator
def chain_sample_group_stats(  # noqa: C901
    preprocessing_chain_index: int,
    sample_data_path: str,
    sample_target_path: str,
    output_directory: str,
    is_regression: Optional[bool] = None,
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
            f"Got inconsistent 'preprocessing_chain_index' in provided data path: {sample_data_path}, \
            \nExpected: {preprocessing_chain_index}, Got: {cind}"
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
    df_preprocessed = pd.read_csv(unc_path(sample_data_path), header=0).iloc[:, 1:]
    # Validate columns
    if len(df_preprocessed.columns) > 5:
        if list(df_preprocessed.columns)[0:5] == ["Sample_ID", "Label", "Validation_group", "X_shape", "y"]:
            pass
        else:
            raise ValueError(f"Invalid sample data columns: {df_preprocessed.columns}")
    else:
        raise ValueError(f"Invalid sample data columns: {df_preprocessed.columns}")

    # Read sample groups
    df_sample_targets = pd.read_csv(unc_path(sample_target_path))
    # Validate columns
    # if list(df_sample_targets.columns) == ['Sample_ID', 'Label', 'Target_value', 'Group']:
    if list(df_sample_targets.columns) == ['Sample_ID', 'Label', 'Target_value', 'Group', 'Validation_group']:
        pass
    else:
        raise ValueError(f"Invalid sample data columns: {df_sample_targets.columns}")
    df_group = df_sample_targets.iloc[:, [0, 3]].astype(str)

    # Validate target type if not specified
    # Check numeric-like
    targets = df_preprocessed.iloc[:, 4]
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
        if is_regression and not is_numeric:
            raise ValueError(f"Got categorical target values when is_regression is True: {targets}")
        elif not is_regression and is_numeric:
            is_numeric = False
        else:
            pass
    # Force type of target values
    if is_regression:
        df_preprocessed['y'] = df_preprocessed['y'].astype("float")
    else:
        df_preprocessed['y'] = df_preprocessed['y'].astype("str")

    # Match group
    group = []
    for i in range(df_preprocessed.shape[0]):
        group_i = df_group['Group'][df_group['Sample_ID'] == df_preprocessed['Sample_ID'][i]].iloc[0]
        group.append(group_i)
    df_preprocessed['Group'] = group

    # Group stats column names for X and numeric y
    stats_col = ['Group'] + list(df_preprocessed.columns[5:-1])

    # Numeric targets - regression
    if is_regression:
        # Overall stats and default measures
        ostats = Stats2d().summary(df_preprocessed.iloc[:, 4:-1].values)
        # y stats
        df_ystats = pd.DataFrame(np.zeros((1 + len(df_preprocessed['Group'].unique()), 1 + len(ostats.keys()))))
        df_ystats.columns = ['Group'] + list(ostats.keys())
        df_ystats['Group'] = df_ystats['Group'].astype('string')
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
            gstats = Stats2d().summary(gdata.iloc[:, 4:-1].values)
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
        df_ystats.to_csv(
            unc_path(write_dir + f"PreprocessingChainResult_chain_ind_{preprocessing_chain_index}_y_stats.csv"),
            index=False,
        )
        # Dump y stats dill (swectral private)
        dill_result_path = write_dir + ".__swectral_dill_data/.__swectral_result_summary_sample_targets_stats.dill"
        os.makedirs(unc_path(os.path.dirname(dill_result_path)), exist_ok=True)
        dill.dump(df_ystats, open(unc_path(dill_result_path), "wb"))
        # Save X stats
        for m in list(gstats.keys()):
            dfm = df_xstats_dict[m]
            dfm.to_csv(
                unc_path(write_dir + f"PreprocessingChainResult_chain_ind_{preprocessing_chain_index}_X_{m}.csv"),
                index=False,
            )

    # Categorical targets - classification
    else:
        # Overall stats and default measures
        ostats_x = Stats2d().summary(df_preprocessed.iloc[:, 5:-1].values)
        # y stats
        ylabel, ycount = np.unique(df_preprocessed.iloc[:, 4], return_counts=True)
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
            gstats_x = Stats2d().summary(gdata.iloc[:, 5:-1].values)
            ylabel, ycount = np.unique(gdata.iloc[:, 4], return_counts=True)
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
        df_ystats.to_csv(
            unc_path(write_dir + f"PreprocessingChainResult_chain_ind_{preprocessing_chain_index}_y_stats.csv"),
            index=False,
        )
        # Dump y stats dill (swectral private)
        dill_result_path = write_dir + ".__swectral_dill_data/.__swectral_result_summary_sample_targets_stats.dill"
        os.makedirs(unc_path(os.path.dirname(dill_result_path)), exist_ok=True)
        dill.dump(df_ystats, open(unc_path(dill_result_path), "wb"))

        # Save X stats
        for m in list(gstats_x.keys()):
            dfm = df_xstats_dict[m]
            dfm.to_csv(
                unc_path(write_dir + f"PreprocessingChainResult_chain_ind_{preprocessing_chain_index}_X_{m}.csv"),
                index=False,
            )


# %% Experiment group sample data statistics for all chains


@simple_type_validator
def sample_group_stats(  # noqa: C901
    report_directory: str,
    output_directory: str = "",
    is_regression: Optional[bool] = None,
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
        )

    # Add y stats to modeling targets dir
    shutil.copyfile(
        unc_path(f"{output_directory}PreprocessingChainResult_chain_ind_0_y_stats.csv"),
        unc_path(f"{sample_target_path[:-4] + '_stats' + sample_target_path[-4:]}"),
    )


# %% Process ID label converters


def process_id_to_label(process_id: str, process_config_df: pd.DataFrame, ignore: bool = False) -> str:
    """
    Convert unique SpecPipe process ID to process label. If ignore True, return input if input is not id.
    "process_config_df" is the SpecPipe_added_process.csv in the configuration subdir.
    """
    # config_dir = (pipeline_config_dir + "/").replace("//", "/")
    # df_proc = pd.read_csv(unc_path(config_dir + "SpecPipe_added_process.csv"))
    df_proc = process_config_df
    process_labels = list(df_proc["Method"][df_proc["ID"] == process_id])
    if not ignore:
        if len(process_labels) < 1:
            raise ValueError(f"No label found for given process ID: {process_id}")
        return str(process_labels[0])
    else:
        if len(process_labels) < 1:
            return process_id
        else:
            return str(process_labels[0])


def process_label_to_id(process_label: str, process_config_df: pd.DataFrame) -> str:
    """
    Convert unique SpecPipe process label to process ID.
    "process_config_df" is the SpecPipe_added_process.csv in the configuration subdir.
    """
    # Validate if ID
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
    # Convert label to ID
    # config_dir = (pipeline_config_dir + "/").replace("//", "/")
    # df_proc = pd.read_csv(unc_path(config_dir + "SpecPipe_added_process.csv"))
    df_proc = process_config_df
    process_ids = list(df_proc["ID"][df_proc["Method"] == process_label])
    if len(process_ids) > 1:
        raise ValueError(
            f"Multiple process IDs for the given label '{process_label}': {process_ids}, \
                         label to convert to ID must be unique."
        )
    if len(process_ids) < 1:
        raise ValueError(f"No process ID found for given label: {process_label}")
    return str(process_ids[0])


# %% Model performance summary and marginal performance statistics


# Collect performance metrics
def performance_metrics_summary(  # noqa: C901
    pipeline_config_dir: str,
    model_evaluation_report_dir: str,
) -> dict[str, Any]:
    """
    Collect performance metrics from SpecPipe model evaluation reports.
    """
    config_dir = (pipeline_config_dir.replace("\\", "/") + "/").replace("//", "/")
    report_dir = (model_evaluation_report_dir.replace("\\", "/") + "/").replace("//", "/")
    process_config_df = pd.read_csv(unc_path(config_dir + "SpecPipe_added_process.csv"))

    # Chains path
    chains_id_path = config_dir + "SpecPipe_exec_chains_in_ID.csv"
    chains_label_path = config_dir + "SpecPipe_exec_chains_in_label.csv"

    # Validate paths
    if not os.path.exists(unc_path(chains_id_path)):
        raise ValueError(f"Missing required file in given pipeline_config_dir: {chains_id_path}")
    if not os.path.exists(unc_path(chains_label_path)):
        raise ValueError(f"Missing required file in given pipeline_config_dir: {chains_label_path}")

    # Chains
    df_cid = pd.read_csv(unc_path(chains_id_path))
    df_clab = pd.read_csv(unc_path(chains_label_path))

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
                assert int(dn_splited[1]) == float(dn_splited[1])
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
                f"None or multiple preprocessing chain file found for 'Preprocessing_#{chain_num}', \
                    got chain file names: {chain_txt_names}"
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
            f"Pipeline model evaluation reports imply inconsistent processing chains with pipeline configurations:\n\
            Configured chains:\n{config_chains},\nReport implied chains:\n{result_chains_dir_map.keys()}\n"
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
    for dir_name, result_chain in zip(ordered_dir_names, ordered_result_chains):
        # Validate result chain - all item to process IDs
        result_chain1 = []
        for proc_item in result_chain:
            result_chain1.append(process_label_to_id(proc_item, process_config_df))
        result_chain = tuple(result_chain1)
        # Metrics directory
        metrics_dir = f"{report_dir}{dir_name}/"
        # Save processes of the full chain
        cprocs_in_id = result_chain1
        cprocs_in_label = [process_id_to_label(proc_id, process_config_df) for proc_id in cprocs_in_id]
        df_cprocs = pd.DataFrame({"Chain_in_process_ID": cprocs_in_id, "Chain_in_process_label": cprocs_in_label})
        # Dump dill (swectral private)
        dill_result_path = metrics_dir + ".__swectral_dill_data/.__swectral_core_result_Chain_process_info.dill"
        os.makedirs(unc_path(os.path.dirname(dill_result_path)), exist_ok=True)
        dill.dump(df_cprocs, open(unc_path(dill_result_path), "wb"))
        # Read performance metrics
        metrics_filename = [
            entry.name
            for entry in os.scandir(unc_path(metrics_dir))
            if f"_performance_{dir_name.split('_Model_')[-1]}.csv" in entry.name
        ][0]
        df_metrics = pd.read_csv(unc_path(f"{report_dir}{dir_name}/{metrics_filename}"))
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
def regression_performance_marginal_stats(
    metrics_dict: dict[str, Any],
    pipeline_config_dir: str,
    model_evaluation_report_dir: str,
    validate_process: bool = True,
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
    process_config_df = pd.read_csv(unc_path(config_dir + "SpecPipe_added_process.csv"))

    # Compute and output marginal perf stats of each step
    marginal_performance_stats: dict = {}
    for step in list(df_cid.columns):
        # Step process IDs
        step_process_ids = list(df_cid[step].unique())
        # For parallel processes
        # Step process grouped performance
        group_r2: dict = {}
        # Resulting df
        step_gstats_r2 = pd.DataFrame(
            np.zeros((len(step_process_ids) + 7, len(step_process_ids) + 2)),
            columns=["Process_ID", "All"] + step_process_ids,
            dtype=object,
        )
        step_gstats_r2["Process_ID"] = [
            "Process_label",
            "n_records",
            "Mean_R2",
            "Min_R2",
            "Median_R2",
            "Max_R2",
            "All",
        ] + step_process_ids
        step_gstats_r2.loc[0, :] = ["Process_label", "All"] + [
            process_id_to_label(proc_id, process_config_df, ignore=(not validate_process))
            for proc_id in step_gstats_r2.columns[2:]
        ]
        # Aggregate group of all records
        r2_all = list(df_reg_metrics.loc[:, "R2"])
        group_r2['All'] = r2_all
        # Get group stats
        for pid in step_process_ids:
            r2 = list(df_reg_metrics.loc[df_reg_metrics[step] == pid, "R2"])
            group_r2[pid] = r2
        # Compute stats and p in test for comparison
        for pid1 in ['All'] + step_process_ids:
            # R2 metrics
            r2_1 = group_r2[pid1]
            step_gstats_r2.loc[1, pid1] = len(r2_1)
            step_gstats_r2.loc[2, pid1] = np.nanmean(r2_1)
            step_gstats_r2.loc[3, pid1] = np.nanmin(r2_1)
            step_gstats_r2.loc[4, pid1] = np.nanmedian(r2_1)
            step_gstats_r2.loc[5, pid1] = np.nanmax(r2_1)
            # Mann-Whitney-U-p-Value
            for i, pid2 in enumerate(['All'] + step_process_ids):
                r2_2 = group_r2[pid2]
                if len(step_process_ids) > 1:
                    step_gstats_r2.loc[i + 6, pid1] = mannwhitneyu(r2_1, r2_2)[1]
                else:
                    step_gstats_r2.loc[i + 6, pid1] = np.nan

        # Collect step result
        marginal_performance_stats[step] = {"r2": step_gstats_r2, "summary": df_reg_metrics}
        # Save step result
        if len(step_process_ids) > 1:
            step_gstats_r2.to_csv(unc_path(report_dir + f"Marginal_R2_stats_{str(step).lower()}.csv"), index=False)
            # Dump dill (swectral private)
            dill_result_path = (
                report_dir
                + f".__swectral_dill_data/.__swectral_result_summary_Marginal_R2_stats_{str(step).lower()}.dill"
            )
            os.makedirs(unc_path(os.path.dirname(dill_result_path)), exist_ok=True)
            dill.dump(step_gstats_r2, open(unc_path(dill_result_path), "wb"))

    # Save summary results used
    df_reg_metrics.to_csv(unc_path(report_dir + "Performance_summary.csv"), index=False)
    # Dump dill (swectral private)
    dill_result_path = report_dir + ".__swectral_dill_data/.__swectral_result_summary_Performance_summary.dill"
    os.makedirs(unc_path(os.path.dirname(dill_result_path)), exist_ok=True)
    dill.dump(df_reg_metrics, open(unc_path(dill_result_path), "wb"))

    return marginal_performance_stats


# Marginal performance statistics for classification
def classification_performance_marginal_stats(  # noqa: C901
    metrics_dict: dict[str, Any],
    pipeline_config_dir: str,
    model_evaluation_report_dir: str,
    validate_process: bool = True,
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
    df_macro_metrics = metrics_dict["macro_metrics"]
    df_micro_metrics = metrics_dict["micro_metrics"]
    config_dir = pipeline_config_dir
    process_config_df = pd.read_csv(unc_path(config_dir + "SpecPipe_added_process.csv"))

    # Compute and output marginal perf stats of each step
    marginal_performance_stats: dict = {}
    for step in list(df_cid.columns):
        # Step process IDs
        step_process_ids = list(df_cid[step].unique())
        # For parallel processes
        # Step process grouped performance
        group_macauc: dict = {}
        group_micauc: dict = {}
        # Resulting df
        step_gstats_macauc = pd.DataFrame(
            np.zeros((len(step_process_ids) + 7, len(step_process_ids) + 2)),
            columns=["Process_ID", "All"] + step_process_ids,
            dtype=object,
        )
        step_gstats_macauc["Process_ID"] = [
            "Process_label",
            "n_records",
            "Mean_AUC_macro",
            "Min_AUC_macro",
            "Median_AUC_macro",
            "Max_AUC_macro",
            "All",
        ] + step_process_ids
        step_gstats_macauc.loc[0, :] = ["Process_label", "All"] + [
            process_id_to_label(proc_id, process_config_df, ignore=(not validate_process))
            for proc_id in step_gstats_macauc.columns[2:]
        ]
        step_gstats_micauc = step_gstats_macauc.copy(deep=True)
        step_gstats_micauc.loc[2:5, "Process_ID"] = [
            "Mean_AUC_micro",
            "Min_AUC_micro",
            "Median_AUC_micro",
            "Max_AUC_micro",
        ]
        # Aggregate group of all records
        macauc_all = list(df_macro_metrics.loc[:, "AUC"])
        group_macauc['All'] = macauc_all
        micauc_all = list(df_micro_metrics.loc[:, "AUC"])
        group_micauc['All'] = micauc_all
        # Get group stats
        for pid in step_process_ids:
            macro_auc = list(df_macro_metrics.loc[df_macro_metrics[step] == pid, "AUC"])
            group_macauc[pid] = macro_auc
            micro_auc = list(df_micro_metrics.loc[df_micro_metrics[step] == pid, "AUC"])
            group_micauc[pid] = micro_auc
        # Compute stats and p in test for comparison
        for pid1 in ['All'] + step_process_ids:
            # Macro avg AUC
            macro_auc_1 = group_macauc[pid1]
            step_gstats_macauc.loc[1, pid1] = len(macro_auc_1)
            step_gstats_macauc.loc[2, pid1] = np.nanmean(macro_auc_1)
            step_gstats_macauc.loc[3, pid1] = np.nanmin(macro_auc_1)
            step_gstats_macauc.loc[4, pid1] = np.nanmedian(macro_auc_1)
            step_gstats_macauc.loc[5, pid1] = np.nanmax(macro_auc_1)
            # Micro avg AUC
            micro_auc_1 = group_micauc[pid1]
            step_gstats_micauc.loc[1, pid1] = len(micro_auc_1)
            step_gstats_micauc.loc[2, pid1] = np.nanmean(micro_auc_1)
            step_gstats_micauc.loc[3, pid1] = np.nanmin(micro_auc_1)
            step_gstats_micauc.loc[4, pid1] = np.nanmedian(micro_auc_1)
            step_gstats_micauc.loc[5, pid1] = np.nanmax(micro_auc_1)
            # Mann-Whiteney U-Test P value
            for i, pid2 in enumerate(['All'] + step_process_ids):
                if len(step_process_ids) > 1:
                    macro_auc_2 = group_macauc[pid2]
                    step_gstats_macauc.loc[i + 6, pid1] = mannwhitneyu(macro_auc_1, macro_auc_2)[1]
                    micro_auc_2 = group_micauc[pid2]
                    step_gstats_micauc.loc[i + 6, pid1] = mannwhitneyu(micro_auc_1, micro_auc_2)[1]
                else:
                    step_gstats_macauc.loc[i + 6, pid1] = np.nan
                    step_gstats_micauc.loc[i + 6, pid1] = np.nan

        # Collect step result
        marginal_performance_stats[step] = {"macro_auc": step_gstats_macauc, "micro_auc": step_gstats_micauc}
        # Save step result
        if len(step_process_ids) > 1:
            # Save macro-avg AUC
            step_gstats_macauc.to_csv(
                unc_path(report_dir + f"Marginal_macro_avg_AUC_stats_{str(step).lower()}.csv"), index=False
            )
            # Dump dill (swectral private)
            dill_result_path = unc_path(
                report_dir
                + ".__swectral_dill_data/"
                + f".__swectral_result_summary_Marginal_macro_avg_AUC_stats_{str(step).lower()}.dill"
            )  # noqa: E501
            os.makedirs(os.path.dirname(dill_result_path), exist_ok=True)
            dill.dump(step_gstats_macauc, open(dill_result_path, "wb"))
            # Save micro-avg AUC
            step_gstats_micauc.to_csv(
                unc_path(report_dir + f"Marginal_micro_avg_AUC_stats_{str(step).lower()}.csv"), index=False
            )
            # Dump dill (swectral private)
            dill_result_path = unc_path(
                report_dir
                + ".__swectral_dill_data/"
                + f".__swectral_result_summary_Marginal_micro_avg_AUC_stats_{str(step).lower()}.dill"
            )  # noqa: E501
            os.makedirs(os.path.dirname(dill_result_path), exist_ok=True)
            dill.dump(step_gstats_micauc, open(dill_result_path, "wb"))

    # Collect performance summary
    marginal_performance_stats["macro_summary"] = df_macro_metrics
    marginal_performance_stats["micro_summary"] = df_micro_metrics
    # Save performance summary
    # Save macro-avg performance
    df_macro_metrics.to_csv(unc_path(report_dir + "Macro_avg_performance_summary.csv"), index=False)
    # Dump dill (swectral private)
    dill_result_path = unc_path(
        report_dir + ".__swectral_dill_data/.__swectral_result_summary_Macro_avg_performance_summary.dill"
    )
    os.makedirs(os.path.dirname(dill_result_path), exist_ok=True)
    dill.dump(df_macro_metrics, open(dill_result_path, "wb"))
    # Save micro-avg performance
    df_micro_metrics.to_csv(unc_path(report_dir + "Micro_avg_performance_summary.csv"), index=False)
    # Dump dill (swectral private)
    dill_result_path = unc_path(
        report_dir + ".__swectral_dill_data/.__swectral_result_summary_Micro_avg_performance_summary.dill"
    )
    os.makedirs(os.path.dirname(dill_result_path), exist_ok=True)
    dill.dump(df_micro_metrics, open(dill_result_path, "wb"))

    return marginal_performance_stats


# Marginal performance statistics
@simple_type_validator
def performance_marginal_stats(report_directory: str) -> dict[str, Any]:
    """
    Compute marginal model performance statistics and summary of model performance metrics from SpecPipe model evaluation reports.

    Parameters
    ----------
    pipeline_config_dir : str
        Root of SpecPipe report directory.

    Returns
    -------
    dict[str, Any]
        Dictionary of marginal model performance statistics and summary of model performance metrics at each step.
    """  # noqa: E501
    pipeline_config_dir = f"{report_directory}/SpecPipe_configuration/"
    model_evaluation_report_dir = f"{report_directory}/Modeling/Model_evaluation_reports/"
    metrics_dict = performance_metrics_summary(pipeline_config_dir, model_evaluation_report_dir)
    if metrics_dict["is_regression"]:
        marginal_performance_stats = regression_performance_marginal_stats(
            metrics_dict, pipeline_config_dir, model_evaluation_report_dir
        )
    else:
        marginal_performance_stats = classification_performance_marginal_stats(
            metrics_dict, pipeline_config_dir, model_evaluation_report_dir
        )
    return marginal_performance_stats
