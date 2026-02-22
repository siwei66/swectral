# -*- coding: utf-8 -*-
"""
Swectral - Demonstration for parallel processing in windows

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Real-world data demo
# Multiprocessing usage demo on Windows

# Tip 1: For multiprocessing on Windows, all working codes must be written within block "if __name__ == '__main__':"
if __name__ == '__main__':

    # %% ---------------------------------------------------------------------------------------------------------------
    # 1. Data preparation
    # Set data directory path
    import os
    import shutil

    # Setup a directory for demo
    demo_dir = os.getcwd() + "/SpecPipeDemoWinParallel/"

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
    report_dir = demo_dir + "/demo_results_classification_parallel/"

    os.makedirs(report_dir)

    # %% ---------------------------------------------------------------------------------------------------------------
    # 2. Configure your experiment data
    # 2.1 Create a spectral experiment
    # Create a SpecExp instance for experiment data
    from swectral import SpecExp

    exp = SpecExp(report_dir)

    # 2.2. Experiment group management
    # Add experiment groups
    exp.add_groups(["group_1", "group_2", "group_3"])

    # 2.3. Raster image management
    # Add raster images
    exp.add_images_by_name("demo.", data_dir, "group_1")
    exp.add_images_by_name("demo.", data_dir, "group_2")

    # Although use shared raster image is acceptable,
    # it is reccommended to use isolated images for multiprocessing of image processing processes
    # e.g. copy shared images or segment shared large rasters
    # Copy image
    import shutil

    shutil.copy(exp.images[0][-1], exp.images[0][-1].replace("demo.tiff", "demo_copy.tiff"))
    # Add copied image to the pipeline
    exp.add_images_by_name("demo_copy.", data_dir, "group_3")

    # 2.4. Region of interest (ROI) management
    # Load image ROIs using suffix to image names
    exp.add_rois_by_suffix("_[12].xml", data_dir, "group_1")
    exp.add_rois_by_suffix("_[34].xml", data_dir, "group_2")
    # The file 'demo_5.xml' does not match the new image name 'demo_copy.tiff', so uses adding by file list method
    exp.add_rois_by_file(path=[data_dir + "demo_5.xml"], image_name="demo_copy.tiff", group="group_3")

    # Check added ROIs
    exp.ls_rois()

    # 2.5. Sample labels and target values
    # 2.5.1 Set sample labels

    # Retrieve original sample label dataframe
    labels = exp.ls_labels()

    # Update sample labels using sample ROI names
    labels.iloc[:, 1] = exp.ls_rois_sample(return_dataframe=True)["ROI_name"]  # type: ignore

    # Set sample labels using the updated label dataframe
    exp.sample_labels = labels  # type: ignore

    # 2.5.2 Set target values

    # List target value dataframe
    targets = exp.ls_sample_targets()

    # Set the leaf order as target values
    targets["Target_value"] = [f"leaf_{labl[0]}" for labl in targets['Label']]  # type: ignore

    # Load target values from updated target dataframe
    exp.sample_targets_from_df(targets)

    # 3. Design testing pipeline

    # 3.1 Create processing pipeline
    from swectral import SpecPipe

    pipe = SpecPipe(exp)

    # 3.2 Image processing

    # Create some image processing functions

    # Create some image processing functions
    # Tip 2: Import function dependencies inside for multiprocessing
    # Standard normal variate
    def snv(v):  # type: ignore
        import numpy as np

        vmean = np.mean(v, axis=1, keepdims=True)
        vstd = np.std(v, axis=1, keepdims=True)
        vstd[vstd == 0] = 1e-10
        snv = (v - vmean) / vstd
        return snv

    # Tip 2 (cont.): or passing function dependencies as parameters with default values
    import numpy as np

    def snv_pass(v, np=np):  # type: ignore
        vmean = np.mean(v, axis=1, keepdims=True)
        vstd = np.std(v, axis=1, keepdims=True)
        vstd[vstd == 0] = 1e-10
        snv = (v - vmean) / vstd
        return snv

    # Compared with raw data for example
    def raw(v):  # type: ignore
        return v

    # Add these process to the pipeline
    pipe.add_process(2, 2, 0, raw)
    pipe.add_process(2, 2, 0, snv)
    pipe.add_process(2, 2, 0, snv_pass)

    # 3.3 ROI statistics
    # Import some ROI spectral statistic metrics
    from swectral import roi_mean, roi_median

    # Add these process to the pipeline
    pipe.add_process(5, 7, 0, roi_mean)
    pipe.add_process(5, 7, 0, roi_median)

    # 3.4 Add models to the pipeline
    # Create some models
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.neighbors import KNeighborsClassifier  # type: ignore

    rf_classifier = RandomForestClassifier(n_estimators=10)
    knn_classifier = KNeighborsClassifier(n_neighbors=3)

    # Add models
    pipe.add_model(knn_classifier, validation_method="2-fold")
    pipe.add_model(rf_classifier, validation_method="2-fold")

    # 4 Run pipeline

    # Check processing chains with method id
    pipe.ls_chains()

    # Run
    # Tip 3:
    # n_processor = -1 (default): uses maximal available CPUs minus 1 processors (non-Windows) or 1 CPU (Windows)
    # n_processor = -2: uses maximal available CPUs minus 1 processors on all platforms
    pipe.run(n_processor=-2, model_test_coverage=0.1)

    # %% ---------------------------------------------------------------------------------------------------------------
    # 5 Regression Case

    # Create a directory for regression results
    report_dir_reg = demo_dir + "/demo_results_regression_parallel/"
    if not os.path.exists(report_dir_reg):
        os.makedirs(report_dir_reg)

    # Copy SpecExp and SpecPipe
    import copy

    exp_reg = copy.deepcopy(exp)
    pipe_reg = copy.deepcopy(pipe)
    targets_reg = copy.deepcopy(targets)

    # Update report directory of SpecExp
    exp_reg.report_directory = report_dir_reg

    # Modify targets to numeric, here the numbers approaximate the age of the leaves
    targets_reg["Target_value"] = [(5 - int(labl[0])) for labl in targets['Label']]  # type: ignore
    exp_reg.sample_targets_from_df(targets_reg)

    # Remove classification models
    pipe_reg.rm_model()

    # Update the pipeline
    pipe_reg.spec_exp = exp_reg

    # Add regressors to the pipeline
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.neighbors import KNeighborsRegressor  # type: ignore

    rf_regressor = RandomForestRegressor(n_estimators=10)
    knn_regressor = KNeighborsRegressor(n_neighbors=3)

    # Add models
    pipe_reg.add_model(knn_regressor, validation_method="2-fold")
    pipe_reg.add_model(rf_regressor, validation_method="2-fold")

    # Check processing chains and run the pipeline
    pipe_reg.ls_chains()

    # Run regression pipeline
    pipe_reg.run(n_processor=2, model_test_coverage=0.1)
