# -*- coding: utf-8 -*-
"""
Swectral - Simple demonstration from new README.md

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Real-world data demo

# %% Data preparation --------------------------------------------------------------------------------------------------

# 1. Data preparation
# Set data directory path
import os
import shutil

# Setup a directory for demo
demo_dir = os.getcwd() + "/SpecPipeDemoReadme/"

if os.path.exists(demo_dir):
    shutil.rmtree(demo_dir)

# Setup data directory and demo data
data_dir = demo_dir + "demo_data/"

os.makedirs(data_dir)

# Download real-world demo raster image and ROI files
# Demo data URL: https://github.com/siwei66/swectral/tree/master/demo/demo_data/
from swectral import download_demo_data

download_demo_data(data_dir)

# Create a directory for pipeline results
report_dir = demo_dir + "/demo_results_classification/"

os.makedirs(report_dir)


# %% Configure your experiment data ------------------------------------------------------------------------------------

# 2. Configure your experiment data

# 2.1 Create a spectral experiment
# Create a SpecExp instance for experiment data
from swectral import SpecExp

exp = SpecExp(report_dir)

# 2.2. Add experiment groups
exp.add_groups(['group_1', 'group_2'])

# 2.3. Raster image management
# Add raster images
exp.add_images_by_name(image_name="demo.", image_directory=data_dir, group="group_1")
exp.add_images_by_name("demo.", data_dir, "group_2")


# 2.4. Region of interest (ROI) management
# Load image ROIs using suffix to image names
exp.add_rois_by_suffix(roi_filename_suffix="_[12].xml", search_directory=data_dir, group="group_1")
exp.add_rois_by_suffix("_[345].xml", data_dir, "group_2")

# Show raster RGB preview with associated ROIs
exp.show_image("demo.tiff", "group_1", rgb_band_index=(19, 12, 6), output_path=report_dir + "demo_rast_rgb1.png")
exp.show_image("demo.tiff", "group_2", rgb_band_index=(19, 12, 6), output_path=report_dir + "demo_rast_rgb2.png")


# 2.5. Sample labels and target values

# 2.5.1 Set sample labels

# Retrieve original sample label dataframe
labels = exp.ls_labels()

# Update sample labels using sample ROI names
labels.iloc[:, 1] = exp.ls_rois_sample(return_dataframe=True, print_result=False)["ROI_name"]  # type: ignore

# Set sample labels using the updated label dataframe
exp.sample_labels = labels  # type: ignore

# 2.5.2 Set target values

# List target value dataframe
targets = exp.ls_sample_targets()

# Set the leaf sequence number as target values
targets["Target_value"] = [f"leaf_{labl[0]}" for labl in targets['Label']]  # type: ignore

# Load target values from updated target dataframe
exp.sample_targets_from_df(targets)

# Check target values
exp.ls_targets()[["Label", "Target_value"]]


# %% Design testing pipeline -------------------------------------------------------------------------------------------

# 3. Design testing pipeline

# 3.1 Create processing pipeline
from swectral import SpecPipe

pipe = SpecPipe(exp)


# 3.2 Image processing
# Create some image processing functions
# Standard normal variate
from swectral.functions import snv


# Compared with raw data for example
def raw(v):  # type: ignore
    return v


# 3.3 ROI statistics
# Import some ROI spectral statistic metrics
from swectral import roi_mean, roi_median

# 3.4 Add models to the pipeline
# Fittable feature engineering models
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from swectral.modelconnector import IdentityTransformer  # Passthrough transformer for comparison

# Test feature selection
selector1 = SelectKBest(f_classif, k=5)  # Select 5 features
selector2 = IdentityTransformer()  # For passthrough (no selection)

# Create some estimators
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore

rf = RandomForestClassifier(n_estimators=10)
knn = KNeighborsClassifier(n_neighbors=3)

# Compose transformers and estimators to full factorial chains
from swectral import factorial_model_chains

models = factorial_model_chains(
    [StandardScaler()],
    {'feat5': selector1, 'feat_all': selector2},  # Specify custom model labels in dictionary
    estimators=[knn, rf],
    is_regression=False
)

# 3.5 Compose pipelines

pipe.build_pipeline(
    [
        # 1 Image-wide baseline correction
        ((2, 2), [raw, snv]),
        # 2 ROI statistics
        ((5, 7), [roi_mean, roi_median]),
        # 3 Models (Feature selector included)
        ((7, 9), models, {'validation_method': '2-fold'})
    ]
)

# Check processing chains with method id
pipe.ls_chains()


# %% Run pipelines -----------------------------------------------------------------------------------------------------

# 4 Run pipelines
pipe.run()


# %% Check classification results --------------------------------------------------------------------------------------

# 5 Check classification results

# Retrieve reports in console
result_summary = pipe.report_summary()
chain_results = pipe.report_chains()

# Check summary reports
result_summary.keys()
result_summary['Macro_avg_performance_summary']
result_summary['Marginal_macro_avg_AUC_stats_step_0']

# Check processing chain reports
len(chain_results)
chain_results[0].keys()
chain_results[0]['ROC_curve']


# %% Regression Case ---------------------------------------------------------------------------------------------------

# 6 Regression Case
# 6.1 Create a directory for regression results
report_dir_reg = demo_dir + "/demo_results_regression/"
if not os.path.exists(report_dir_reg):
    os.makedirs(report_dir_reg)


# 6.2 Copy and update the pipeline data manager to regression
# Copy SpecExp and SpecPipe
import copy

exp_reg = copy.deepcopy(exp)
pipe_reg = copy.deepcopy(pipe)
targets_reg = copy.deepcopy(targets)

# Update report directory of SpecExp
exp_reg.report_directory = report_dir_reg

# Check report directory
exp_reg.report_directory

# Modify targets to numeric, here the numbers approaximate the age of the leaves
targets_reg["Target_value"] = [(5 - int(labl[0])) for labl in targets['Label']]  # type: ignore

# Set the ROIs within the same leaf to a validation group to prevent data leakage
targets_reg["Validation_group"] = [f"leaf_{labl[0]}" for labl in targets['Label']]

# Update target information using the modified target dataframe
exp_reg.sample_targets_from_df(targets_reg)

# Check target values
exp_reg.ls_targets()[["Label", "Target_value", "Validation_group"]]


# 6.3 Update the pipeline models to regressors
# Check and remove classification models
pipe_reg.ls_model()
pipe_reg.rm_model()

# Update the pipelines
pipe_reg.spec_exp = exp_reg

# Fittable feature engineering models
from sklearn.feature_selection import f_regression  # type: ignore

selector1_reg = SelectKBest(f_regression, k=7)  # Select 7 of 46 features

# Add regressors to the pipelines
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore

knn_reg = KNeighborsRegressor(n_neighbors=3)
rf_reg = RandomForestRegressor(n_estimators=10)

# Compose transformers and estimators to full factorial chains
models_reg = factorial_model_chains(
    [StandardScaler()],
    {'feat5': selector1_reg, 'feat_all': selector2},  # Specify custom model labels in dictionary
    estimators=[knn_reg, rf_reg],
    is_regression=True
)

# Add models
pipe_reg.add_model(models_reg, validation_method="2-fold")


# 6.4 Run new regression pipelines
pipe_reg.run()


# 6.5 Check regression results

# Retrieve reports in console
result_summary_reg = pipe_reg.report_summary()
chain_results_reg = pipe_reg.report_chains()

# Check summary reports
result_summary_reg.keys()
result_summary_reg['Performance_summary'].columns

# Check processing chain reports
chain_results_reg[0].keys()
chain_results_reg[0]['Scatter_plot']
