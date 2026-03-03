# -*- coding: utf-8 -*-
"""
Swectral - Basic usage demonstration from README.md

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Real-world data demo

# %% -------------------------------------------------------------------------------------------------------------------

# 1. Data preparation
# Set data directory path
import os
# import shutil

# Setup a directory for demo
demo_dir = os.getcwd() + "/SpecPipeDemoDataAugmentation/"

# if os.path.exists(demo_dir):
#     shutil.rmtree(demo_dir)

# Setup data directory and demo data
data_dir = demo_dir + "demo_data/"

# os.makedirs(data_dir)

# Download real-world demo raster image and ROI files
# Demo data URL: https://github.com/siwei66/swectral/tree/master/demo/demo_data/
# from swectral import download_demo_data

# download_demo_data(data_dir)

# Create a directory for pipeline results
report_dir = demo_dir


# %% -------------------------------------------------------------------------------------------------------------------

# 2. Configure your experiment data

# 2.1 Create a spectral experiment
# Create a SpecExp instance for experiment data
from swectral import SpecExp

exp = SpecExp(report_dir)

# 2.2. Experiment group management
# Add experiment groups
exp.add_groups(['group_1', 'group_2'])

# 2.3. Raster image management
# Add raster images
exp.add_images_by_name(image_name="demo.", image_directory=data_dir, group="group_1")
exp.add_images_by_name("demo.", data_dir, "group_2")

# 2.4. Region of interest (ROI) management
# Load image ROIs using suffix to image names
exp.add_rois_by_suffix(roi_filename_suffix="_[12].xml", search_directory=data_dir, group="group_1")
exp.add_rois_by_suffix("_[345].xml", data_dir, "group_2")


# 2.5. Sample labels and target values

# 2.5.1 Set sample labels

# Retrieve original sample label dataframe
labels = exp.ls_labels()

# Update sample labels using sample ROI names
labels.iloc[:, 1] = exp.ls_rois_sample(return_dataframe=True, print_result=False)["ROI_name"]  # type: ignore

# Set sample labels using the updated label dataframe
exp.sample_labels = labels  # type: ignore

# Check new sample labels
exp.ls_labels()["Label"]

# 2.5.2 Set target values

# List target value dataframe
targets = exp.ls_sample_targets()

# Set the leaf sequence number as target values
targets["Target_value"] = [(5 - int(labl[0])) for labl in targets['Label']]  # type: ignore

# Set the ROIs within the same leaf to a validation group to prevent data leakage
targets["Validation_group"] = [f"leaf_{labl[0]}" for labl in targets['Label']]

# Update target information using the modified target dataframe
exp.sample_targets_from_df(targets)

# Check target values
exp.ls_targets()[["Label", "Target_value", "Validation_group"]]


# 3. Design testing pipeline -------------------------------------------------------------------------------------------

# 3.1 Create processing pipeline
from swectral import SpecPipe

pipe = SpecPipe(exp)


# 3.2 Image processing

# Standard normal variate
from swectral.functions import snv


# Compared with raw data for example
def raw(v):  # type: ignore
    return v


# 3.3 ROI statistics
# Import some ROI spectral statistic metrics
from swectral import roi_mean, roi_median

# 3.4 Sample validation group-level data augmentation
from swectral import blend_samples

blend = blend_samples(n_samples=25, is_regression=True)

# 3.5 Add models to the pipeline
# Fittable feature engineering models
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression  # type: ignore
from swectral.modelconnector import IdentityTransformer

selector1 = SelectKBest(f_regression, k=7)  # Select 7 of 46 features
selector2 = IdentityTransformer()

# Add regressors to the pipeline
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore

knn = KNeighborsRegressor(n_neighbors=3)
rf = RandomForestRegressor(n_estimators=10)

# Compose transformers and estimators to full factorial chains
from swectral import factorial_transformer_chains

# Compose transformers and estimators to full factorial chains
models = factorial_transformer_chains(
    [StandardScaler()],
    {'feat5': selector1, 'feat_all': selector2},  # Specify custom model labels in dictionary
    estimators=[knn, rf],
    is_regression=True
)

# Build pipelines
pipe.build_pipeline(
    [
        # 1 Baseline correction
        ((2, 2), [raw, snv]),
        # 2 ROI statistics
        ((5, 7), [roi_mean, roi_median]),
        # 3 Data augmentation
        ((7, 8), [raw, blend]),
        # 4 Models (Feature selector included)
        ((8, 9), models, {'validation_method': '2-fold'})
    ]
)


# %% -------------------------------------------------------------------------------------------------------------------

# 4 Run pipeline

# Run pipeline
pipe.run(model_test_coverage=0.01, assembly_test_coverage=0.01)

# Enable resume after interruption
# pipe.run(resume=True)
# If the implementation is interrupted or forcibly terminated,
# running the pipeline again with `resume=True` to continue from last completed step.


# %% -------------------------------------------------------------------------------------------------------------------

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
