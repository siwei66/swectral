# -*- coding: utf-8 -*-
"""
Swectral - Basic usage example for SpecPipe - Mock data version, No data downloading needed

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Simple example code

# 1. Prepare mock spectral experiment data
# Create a directory for mock experiment data (The example uses a temporary directory)
import os
import shutil

data_dir = os.getcwd() + "/SpecPipeDemoMock/"

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir)

# Create random mock data
from swectral import create_example_raster, create_example_roi_xml

create_example_raster(f"{data_dir}/example.tif")
create_example_roi_xml(f"{data_dir}/example_roi.xml")


# 2. Configure your experiment data
# 2.1 Create a spectral experiment
# Here we use the same directory as report directory
report_dir = data_dir

# Create a SpecExp instance for experiment data
from swectral import SpecExp

exp = SpecExp(report_dir)

# Check report directory
exp.report_directory


# 2.2. Experiment group management
# Add experiment groups
exp.add_groups(["group_1", "group_2"])

# Check groups
exp.ls_groups()

# Remove a group
exp.rm_group("group_2")


# 2.3. Raster image management
# Add raster images
exp.add_images_by_name(image_name="example", image_directory=data_dir, group="group_1")

# Check added images
exp.ls_images()


# 2.4. Region of interest (ROI) management
# Load image ROIs using suffix to image names
exp.add_rois_by_suffix("_roi.xml", data_dir, "group_1")

# Remove ROIs by name
exp.rm_rois(roi_name="ROI_10")

# Load ROIs to a image using ROI files by paths
exp.add_rois_by_file([f"{data_dir}/example_roi.xml"], image_name="example.tif", group="group_1")

# Check added ROIs
exp.ls_rois()

# Check sample_rois
exp.ls_rois_sample()


# 2.5. Sample labels and target values

# 2.5.1 Set sample labels

# List sample label dataframe
labels = exp.ls_labels()

# Update sample labels
labels.iloc[:, 1] = [f"sample_{str(i + 1)}" for i in range(len(labels))]  # type: ignore

# Set sample labels using the updated label dataframe
exp.sample_labels = labels  # type: ignore

# Check sample labels
exp.ls_labels()

# 2.5.2 Set target values

# List target value dataframe
targets = exp.ls_sample_targets()

# Create mock target values for regression and update target dataframe
targets["Target_value"] = [i for i in range(len(targets))]  # type: ignore

# Load target values from updated target dataframe
exp.sample_targets_from_df(targets)

# Check target values
exp.ls_targets()


# 3. Design testing pipeline
_ = """
The processing functions are wrapped according to different 'data levels'.
A classic data levels in spectral image processing is:

##    raster images -> ROI spectra -> ROI statistics -> traits to model

Data levels for spectroscopy:

##    Standalone 1D spectra of sample -> traits to model

The defined data levels in SpecPipe is:

    Raster images:

        0 - 'image', input image path and output processed image path.
        1 - 'pixel_spec', 1D spectrum of image pixel
        2 - 'pixel_specs_array', 2D spectra array of image pixels
        3 - 'pixel_specs_tensor', 2D spectra tensor of image pixels
        4 - 'pixel_hyperspecs_tensor', 2D hyperspectra tensor of image pixels (optimized for hyperspectral images)

    ROI spectra:

        5 - 'image_roi', get image ROI data, commonly get a unsorted list of ROI spectra
        6 - 'roispecs', ROI spectra in array

    ROI spectral statistics or standalone 1D spectra:

        7 - 'spec1d', arbitray 1D data extracted from roispecs or standalone 1D spectra from spectrascopy

    Sample data:
        8 - "assembly", sample data list for cross-sample interaction

    Models:

        9 - 'model', model evaluation with standard reports in files, only as output level.

A process method is wrapped according to its data level.
Parallel processes with same data levels and application sequences are arranged using full-factorial approach.
"""

# 3.1 Create processing pipeline
from swectral import SpecPipe

pipe = SpecPipe(exp)

# 3.2 Image processing

# Create some image processing functions
import numpy as np


# Standard normal variate
def snv(v):  # type: ignore
    vmean = np.mean(v, axis=1, keepdims=True)
    vstd = np.std(v, axis=1, keepdims=True)
    snv = (v - vmean) / vstd
    return snv


# Compared with raw data
def raw(v):  # type: ignore
    return v


# Add these process to the pipeline
pipe.add_process(
    input_data_level="pixel_specs_array",
    output_data_level="pixel_specs_array",
    application_sequence=0,
    method=snv,
)

# Or specify data level using data level number
pipe.add_process(2, 2, 0, raw)


# 3.3 ROI statistics
# Import some ROI spectral statistic metrics
from swectral import roi_mean, roi_median

# Add these process to the pipeline
pipe.add_process(
    input_data_level="image_roi",
    output_data_level="spec1d",
    application_sequence=0,
    method=roi_mean,
)

# Or specify data level using data level number
pipe.add_process(5, 7, 0, roi_median)


# 3.4 Sample data wrangling
# Create a function to remove nan and inf values
def replace_nan(v: np.ndarray) -> np.ndarray:  # type: ignore
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore


# Add process
pipe.add_process("spec1d", "spec1d", 0, replace_nan)

# Check added process
pipe.ls_process()

# Remove added process
pipe.rm_process(method="raw")

# 3.5 Add models to the pipeline
# Create some models
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore
from sklearn.svm import SVR  # type: ignore

linear_regressor = LinearRegression()
rf_regressor = RandomForestRegressor(n_estimators=10)
knn_regressor = KNeighborsRegressor(n_neighbors=3)
svr = SVR()

# Add model using add_process
pipe.add_process("spec1d", "model", 1, linear_regressor, validation_method="5-fold")
pipe.add_process(7, 9, 1, rf_regressor, validation_method="5-fold")

# Add model using add_model
pipe.add_model(knn_regressor, validation_method="5-fold")
pipe.add_model(svr, validation_method="5-fold")

# Check added models
pipe.ls_model()

# Check all processes including models
pipe.ls_process()


# 4 Run pipeline

# Check processing chains with method id
pipe.ls_chains()

# Run pipeline
pipe.run()

# Set resume True to enable resuming after break
# pipe.run(resume=True)


# 5 Check results

# Retrieve reports in console
result_summary = pipe.report_summary()
chain_results = pipe.report_chains()

# Check summary reports
result_summary.keys()
result_summary['Performance_summary'].columns


# Check processing chain reports
chain_results[0].keys()
chain_results[0]['Scatter_plot']


# %%
# Input data file structure
_ = """
Input data directory

    data_directory/
    ├── SpecExp_configuration/
    │    ├── Loading_history/
    │    │   ├── Loaded_images.csv
    │    │   └── Loaded_ROIs.csv
    │    └── SpecExp_data_configuration.dill
    ├── Your_rasters.tif
    └── Your_ROIs.xml
"""

# Output report file structure for regression task
_ = """
Output report directory

    report_directory/
    ├── Modeling/
    │    ├── sample_targets.csv
    │    ├── sample_targets_stats.csv
    │    └── Model_evaluation_reports/
    │        ├── Data_chain_Preprocessing_#0_Model_(model label 0)/
    │        │   ├── Model_for_application/
    │        │   ├── Model_in_validation/
    │        │   ├── Regression_performance.csv
    │        │   ├── Validation_results.csv
    │        │   ├── Residual_analysis.csv
    │        │   ├── Influence_analysis.csv
    │        │   ├── Scatter_plot.png
    │        │   └── Residual_plot.png
    │        ├── Data_chain_Preprocessing_#0_Model_(model label 1)/
    │        ├── Data_chain_Preprocessing_#1_Model_(model label 0)/
    │        ├── Data_chain_Preprocessing_#1_Model_(model label 1)/
    │        ├── Performance_summary.csv
    │        ├── Marginal_R2_stats_(process step).csv
    │        ├── Preprocessing_#0.txt
    │        ├── Preprocessing_#0.txt
    │        └── Preprocessing_#1.txt
    ├── Pre_execution_test_data/
    ├── Preprocessing/
    │    ├── Step_results/
    │    ├── PreprocessingChainResult_chain_0.csv
    │    ├── PreprocessingChainResult_chain_0_X_(stats metrics).csv
    │    └── PreprocessingChainResult_chain_1.csv
    ├── SpecPipe_configuration/
    └── test_run/
"""

# Output report file structure for classification task
_ = """
Output report directory

    report_directory/
    ├── Modeling/
    │    └── Model_evaluation_reports/
    │        ├── Data_chain_Preprocessing_#0_Model_(model label 0)/
    │        │   ├── Model_for_application/
    │        │   ├── Model_in_validation/
    │        │   ├── Classification_performance.csv
    │        │   ├── Validation_results.csv
    │        │   ├── Residual_analysis.csv
    │        │   ├── Influence_analysis.csv
    │        │   └── ROC_curve.png
    │        ├── Data_chain_Preprocessing_#0_Model_(model label 1)/
    │        ├── Data_chain_Preprocessing_#1_Model_(model label 0)/
    │        ├── Data_chain_Preprocessing_#1_Model_(model label 1)/
    │        ├── Macro_avg_performance_summary.csv
    │        ├── Micro_avg_performance_summary.csv
    │        ├── Marginal_macro_avg_AUC_stats_(process step).csv
    │        ├── Marginal_micro_avg_AUC_stats_(process step).csv
    │        ├── Preprocessing_#0.txt
    │        └── Preprocessing_#1.txt
    ├── Pre_execution_test_data/
    ├── Preprocessing/
    │    ├── Step_results/
    │    ├── PreprocessingChainResult_chain_0.csv
    │    ├── PreprocessingChainResult_chain_0_X_(stats metrics).csv
    │    └── PreprocessingChainResult_chain_1.csv
    ├── SpecPipe_configuration/
    └── test_run/
"""

# Clear the temporary directory after use

# import shutil

# shutil.rmtree(data_dir)
