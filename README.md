<div align="left">
  <img src="https://raw.githubusercontent.com/siwei66/swectral/master/assets/docs/SpecPipeLogo.png" alt="SpecPipeLogo" width="150" height="150">
</div>

# Swectral

[![Tests](https://github.com/siwei66/swectral/actions/workflows/tests.yml/badge.svg)](https://github.com/siwei66/swectral/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/siwei66/swectral/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/swectral.svg)](https://pypi.org/project/swectral/)


## A Python framework for automated batch composition, implementation and method assessment of plant hyperspectral modeling pipelines.

<!-- start-doc -->
Swectral streamlines the batch testing and optimization of plant hyperspectral analysis workflows. It provides a structured and extensible framework to apply and assess various image processing techniques (calibration, baseline correction, denoising, feature engineering, etc.) in combination with various machine learning models. The framework employs a comprehensive full-factorial design to evaluate all method combinations on user spectral dataset and generates standard reports on performance metrics, comparative statistical tests, residual analysis, influence anlaysis and visualizations.


## Core features
- **Batch processing**: Automate numerous data processing and modeling workflows in a single batch operation.
- **File-based**: A resumable, file-based processing pipeline with full-scale auditability and break tolerance.
- **High-performance**: Optimized for hyperspectral images with minimal memory consumption and options of GPU acceleration and pipeline-level multiprocessing.
- **Simple extensible integration**: Intuitive data management and straightforward integration for custom processing functions and Scikit-learn-style models.
<!-- end-doc -->


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Documentation

- [User Guide](https://siwei66.github.io/swectral/index.html)
- [API Reference](https://siwei66.github.io/swectral/api/index.html)
- [Examples](https://siwei66.github.io/swectral/usage.html#tutorials-demos)


## Installation <a name="installation"></a>

Follow these steps to install the project:

1.  **Prerequisites:** Ensure you have Python 3.9 or higher installed.

2.  **Install from PyPI (Recommended):**
    ```python
    pip install swectral
    ```

3.  **Install from source (for development):**
    ```python
    git clone https://github.com/siwei66/SpecPipe.git
    cd SpecPipe
    pip install -e swectral
    ```


## Usage <a name="usage"></a>

### 1.  Data preparation

- Setup a demo directory in current working directory
    ```python
    import os
    demo_dir = os.getcwd() + "/SpecPipeDemo/"
    ```

- Create a data directory and download real-world demo data
    ```python
    data_dir = demo_dir + "demo_data/"
    os.makedirs(data_dir)
    
    from swectral import download_demo_data
    download_demo_data(data_dir)
    ```

- Create a directory for pipeline results
    ```python
    report_dir = demo_dir + "/demo_results_classification/"
    os.makedirs(report_dir)
    ```

### 2. Data configuration

#### 2.1 Create a spectral experiment instance

- Create a SpecExp instance:
    ```python
    from swectral import SpecExp
    exp = SpecExp(report_dir)
    ```
The instance stores and organizes the data loading configurations of an experiment, which faciliates lazy-loading.

- Check report directory:
    ```python
    exp.report_directory
    ```
    Output:
    ```text
    '~/SpecPipeDemo/demo_results_classification/'
    ```

#### 2.2. Experiment group management

- Add experiment groups:
    ```python
    exp.add_groups(['group_1', 'group_2', 'group_3'])
    ```

- Check groups:
    ```python
    exp.groups
    ```
    Output:
    ```text
    ['group_1', 'group_2', 'group_3']
    ```

- Remove a group:
    ```python
    exp.rm_group('group_3')
    ```
    Output:
    ```text
    Following group is removed:
    group_3
    ```


#### 2.3. Raster image management

- Add raster images:

    Use parameter name:
    ```python
    exp.add_images_by_name(image_name="demo.", image_directory=data_dir, group="group_1")
    ```
    Output:
    ```text
    Following image items are added:
        Group    Image    Mask
    0   group_1  demo.tiff
    ```

    Or use parameter position:
    ```python
    exp.add_images_by_name("demo.", data_dir, "group_2")
    ```
    Output:
    ```text
    Following image items are added:
         Group    Image    Mask
    0    group_2  demo.tiff     
    ```

- Check added images:
    ```python
    exp.ls_images()
    ```
    Output:
    ```text
        Group    Image    Mask
    0   group_1  demo.tiff     
    1   group_2  demo.tiff     
    ```


#### 2.4. Region of interest (ROI) management

- Load image ROIs using suffix to image names:
    ```python
    # By parameter name
    exp.add_rois_by_suffix(roi_filename_suffix="_[12].xml", search_directory=data_dir, group="group_1")
    # Or by parameter position
    exp.add_rois_by_suffix("_[345].xml", data_dir, "group_2")
    ```
    Output:
    ```text
    Following ROI items loaded:
       Group    Image    ROI_name    ROI_type    ROI_source_file
    0  group_1  demo.tiff      1-1   sample      demo_1.xml
    1  group_1  demo.tiff      1-2   sample      demo_1.xml
    ...
    9  group_1  demo.tiff      2-5   sample      demo_2.xml
    ```

- Remove ROIs by name:
    ```python
    exp.rm_rois(roi_name='5-5')
    ```

- Remove ROIs by source file name:
    ```python
    exp.rm_rois(roi_source_file_name='demo_5.xml')
    ```

- Load ROIs to a image using ROI files by paths:
    ```python
    exp.add_rois_by_file([f"{data_dir}/demo_5.xml"], image_name="example.tif", group="group_2")
    ```

- Check added ROIs:
    ```python
    exp.ls_rois()
    ```
    ```text
        Group    Image    ROI_name    ROI_type
    0   group_1  demo.tiff      1-1   sample
    1   group_1  demo.tiff      1-2   sample
    ...
    24  group_2  demo.tiff      5-5   sample
    ```

- Show raster RGB preview with associated ROIs:
    ```python
    exp.show_image("demo.tiff", "group_1", rgb_band_index=(19, 12, 6), output_path=report_dir + "demo_rast_rgb1.png")
    ```
    Output:
    <div align="center">
    <img src="https://raw.githubusercontent.com/siwei66/swectral/master/demo/demo_results_classification/demo_rast_rgb1.png"
         alt="SpecPipe SpecExp RGB preview 1"
         width="400"
         style="max-width: 100%;">
    </div>
          
    ```python
    exp.show_image("demo.tiff", "group_2", rgb_band_index=(19, 12, 6), output_path=report_dir + "demo_rast_rgb2.png")
    ```
    Output:
    <div align="center">
    <img src="https://raw.githubusercontent.com/siwei66/swectral/master/demo/demo_results_classification/demo_rast_rgb2.png"
         alt="SpecPipe SpecExp RGB preview 2"
         width="400"
         style="max-width: 100%;">
    </div>


#### 2.5. Sample labels and target values

##### 2.5.1 Set sample labels

- Get current sample label dataframe:
    ```python
    labels = exp.ls_labels()
    ```

- Set new sample labels in the dataframe:

    Here we use sample ROI names as sample labels:

    ```python
    labels.iloc[:, 1] = exp.ls_rois_sample(return_dataframe=True, print_result=False)["ROI_name"]
    ```

- Update sample labels:
    ```python
    exp.sample_labels = labels
    ```

- Check sample labels:
    ```python
    exp.ls_labels()["Label"]
    ```
    Output:
    ```text
    0     1-1
    1     1-2
    ...
    24    5-5
    ```

##### 2.5.2 Set target values

- List target value dataframe:
    ```python
    targets = exp.ls_sample_targets()
    ```

- Create mock target values for regression and update target dataframe:

    Here we use leaf number:

    ```python
    targets["Target_value"] = [f"leaf_{labl[0]}" for labl in targets['Label']]
    ```

- Load target values from updated target dataframe:
    ```python
    exp.sample_targets_from_df(targets)
    ```

- Check target values:
    ```python
    exp.ls_targets()[["Label", "Target_value"]]
    ```
    Output:
    ```text
        Label Target_value
    0    1-1       leaf_1
    1    1-2       leaf_1
    ...
    24   5-5       leaf_5
    ```


### 3. Design testing pipelines

- SpecPipe follows a structured data processing workflow with these sequential data levels:
    ```text
    Raster image data -> ROI spectra -> ROI statistics -> Traits to model
    ```

- The technical data levels in SpecPipe includes:
    ```text
    Raster images:
        0 - "image", input image path and output processed image path.
    
        1 - "pixel_spec", if the process callable is applied to 1D spectrum of image pixel
    
        2 - "pixel_specs_array", if the process callable is applied to 2D spectra array of image pixels
    
        3 - "pixel_specs_tensor", if the process callable is applied to 3D spectra tensor of image pixels
    
        4 - "pixel_hyperspecs_tensor", same as "pixel_specs_tensor" but optimized for hyperspectral images
    
    ROI spectra:
        5 - "image_roi", raster with sample ROIs, for spectrum extraction
    
        6 - "roispecs", 2D array of ROI spectra
    
    ROI statistics:
        7 - "spec1d", arbitrary 1D data of samples, e.g. 1D spectra, flattened spectra statistical metrics
    
    Sample data:
        8 - "assembly", sample data list for cross-sample interaction
    
    Models:
        9 - "model", model evaluation with standard report output as files
    ```

- The corresponding data processing workflow is:
    ```text
    Raster image data:    0 ~ 4
        ↓
    Extract ROI spectra:  5 - "image_roi"
        ↓
    ROI spectra:          6 - "roispecs"
        ↓
    ROI statistics:       7 - "spec1d"
        ↓
    Sample assembly:      8 - "assembly"
        ↓
    Model evaluation:     9 - "model"
    ```
The processing functions are wrapped in the pipeline according to the specified "data levels".
Parallel processes can be added with identical "data level" and "application sequence", and they are arranged using full-factorial approach in the pipeline.


#### 3.1 Create processing pipeline
- Create processing pipeline from SpecExp instance configured above:
    ```python
    from swectral import SpecPipe
    pipe = SpecPipe(exp)
    ```

#### 3.2 Image processing

- Create some image processing functions, such as: 

- Standard normal variate: 
    ```python
    def snv(v):
        import numpy as np
        vmean = np.mean(v, axis=1, keepdims=True)
        vstd = np.std(v, axis=1, keepdims=True)
        snv = (v - vmean) / vstd
        return snv
    ```
    **TIP**: Import working function dependency inside for multiprocessing.

- Raw data for performance comparison:
    ```python
    def raw(v):
        return v
    ```

- Add these processing functions to the pipeline:
    ```python
    pipe.add_process(
        input_data_level="pixel_specs_array",
        output_data_level="pixel_specs_array",
        application_sequence=0,
        method=snv,
    )
    ```

- Or we can specify the data level using the corresponding number:
    ```python
    pipe.add_process(2, 2, 0, raw)
    ```


#### 3.3 ROI statistics

- Import some ROI spectral statistic metrics:
    ```python
    from swectral import roi_mean, roi_median
    ```

- Add these processes to the pipeline:

    Specify data level using name:
    ```python
    pipe.add_process(
        input_data_level='image_roi', 
        output_data_level='spec1d', 
        application_sequence=0, 
        method=roi_mean
        )
    ```

    Or specify data level using number:
    ```python
    pipe.add_process(5, 7, 0, roi_median)
    ```


#### 3.4 Sample data wrangling

- Create a function to remove nan and inf values:
    ```python
    import numpy as np
    def replace_nan(v: np.ndarray, np=np) -> np.ndarray:
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    ```
    **TIP**: Instead of import inside, you can also passing working function dependencies as parameters with default values for multiprocessing.

- Add the process to the pipeline:
    ```python
    pipe.add_process('spec1d', 'spec1d', 0, replace_nan)
    ```

- Check all added preprocessing processes:
    ```python
    pipe.ls_process()
    ```
    Output:
    ```text
       ID        Process_label       Input_data_level   Output_data_level     Application_sequence    Method
    0  2_0_%#1                       pixel_specs_array  pixel_specs_array     0                       snv
    1  2_0_%#2                       pixel_specs_array  pixel_specs_array     0                       raw
    2  5_0_%#1                       image_roi          spec1d                0                       roi_mean
    3  5_0_%#2                       image_roi          spec1d                0                       roi_median
    4  7_0_%#1                       spec1d             spec1d                0                       replace_nan
    ```

- To remove added processes from the pipeline:
    ```python
    pipe.rm_process(method='replace_nan')
    ```
Processes can be removed by various criteria, the example removes the function 'replace_nan' by its name.


#### 3.5 Add models to the pipeline

- Create some models:
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    rf_classifier = RandomForestRegressor(n_estimators=10)
    knn_classifier = KNeighborsRegressor(n_neighbors=3)
    ```

- Add models to the pipeline:
    ```python
    pipe.add_model(knn_classifier, validation_method="2-fold")
    pipe.add_model(rf_classifier, validation_method="2-fold")
    ```

- Check added models:
    ```python
    pipe.ls_model()
    ```

- Check all processes including models:
    ```python
    pipe.ls_process()
    ```
    Output:
    ```text
       ID Process_label    Input_data_level    Output_data_level    Application_sequence    Method
    0  7_0_%#1             spec1d              model                0                       KNeighborsClassifier
    1  7_0_%#2             spec1d              model                0                       RandomForestClassifier
    ```


### 4 Run pipelines

- Check processing chains of the pipeline:
    ```python
    pipe.ls_chains()
    ```
    Output:
    ```text
         Step_0    Step_1         Step_2
    0    snv       roi_mean       KNeighborsClassifier
    1    snv       roi_mean       RandomForestClassifier
    2    snv       roi_median     KNeighborsClassifier
    3    snv       roi_median     RandomForestClassifier
    4    raw       roi_mean       KNeighborsClassifier
    5    raw       roi_mean       RandomForestClassifier
    6    raw       roi_median     KNeighborsClassifier
    7    raw       roi_median     RandomForestClassifier
    ```

- Run pipeline:
    ```python
    pipe.run()
    ```

- Enable resume after interruption:
    ```python
    pipe.run(resume=True)
    ```
If the implementation is interrupted or forcibly terminated, running the pipeline again with `resume=True` to continue from last completed step.


### 5 Running results

- The pipeline produces following results for every processing chain, including:
    ```text
    • Final and intermediate processing results
    • Configurations
    • Validation and application models
    • Model evaluation reports
    • Visualization
    ```

- The resulting file structure is as follows:

- For input data:
    ```text
    report_directory/
    ├── SpecExp_configuration/
    │    ├── Loading_history/
    │    │   ├── Loaded_images.csv
    │    │   └── Loaded_ROIs.csv
    │    └── SpecExp_data_configuration.dill
    └── SpecPipe_configuration/
         ├── SpecPipe_added_process.csv
         ├── SpecPipe_exec_chains_in_ID.csv
         ├── SpecPipe_exec_chains_in_label.csv
         ├── SpecPipe_full_factorial_chains_in_ID.csv
         ├── SpecPipe_full_factorial_chains_in_label.csv
         └── SpecPipe_pipeline_configuration.dill
    ```

- For classification tasks, the pipeline generates:
    ```text
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
    ```

- Retrieve reports in console
    ```python
    result_summary = pipe.report_summary()
    chain_results = pipe.report_chains()
    ```

- Check summary reports
    The summary reports include:
    ```python
    result_summary.keys()
    ```
    Output:
    ```text
    dict_keys([
        'Macro_avg_performance_summary',
        'Marginal_macro_avg_AUC_stats_step_0',
        'Marginal_macro_avg_AUC_stats_step_1',
        'Marginal_macro_avg_AUC_stats_step_2',
        'Marginal_micro_avg_AUC_stats_step_0',
        'Marginal_micro_avg_AUC_stats_step_1',
        'Marginal_micro_avg_AUC_stats_step_2',
        'Micro_avg_performance_summary',
        'sample_targets_stats'])
    ```

    Demonstration of macro-average performance metrics of classification:
    ```python
    result_summary['Macro_avg_performance_summary']
    ```
    Output:
    ```text
        Step_0   Step_1   Step_2  Precision  Recall  F1_Score  Accuracy    AUC
    0  2_0_%#1  5_0_%#1  7_0_%#1   0.860000    0.84  0.842828     0.936  0.947
    ...
    7  2_0_%#2  5_0_%#2  7_0_%#2   0.769524    0.72  0.684242     0.888  0.829
    ```

    Demonstration of marginal macro-average performance metrics of classification:
    ```python
    result_summary['Marginal_macro_avg_AUC_stats_step_0']
    ```
    Output:
    ```text
             Process_ID       All   2_0_%#1   2_0_%#2
    0     Process_label       All       snv       raw
    1         n_records         8         4         4
    2    Mean_AUC_macro   0.85425   0.95275   0.75575
    3     Min_AUC_macro     0.631     0.942     0.631
    4  Median_AUC_macro     0.906    0.9495     0.761
    5     Max_AUC_macro      0.97      0.97      0.87
    6               All       1.0  0.199557  0.199557
    7           2_0_%#1  0.199557       1.0  0.028571
    8           2_0_%#2  0.199557  0.028571       1.0
    ```
    The processes of the step (here raw image and standard normal variates) are compared using non-parametric Mann-Whitney-U test.

- Check processing chain reports
    It's reports of every processing chains:
    ```python
    len(chain_results)
    ```
    Output:
    ```text
    8
    ```

    For each chain, the reports include:
    ```python
    chain_results[0].keys()
    ```
    Output:
    ```text
    dict_keys([
        'Chain_processes',
        'Classification_performance',
        'Influence_analysis',
        'Residual_analysis',
        'ROC_curve',
        'Validation_results'])
    ```

    Demonstration of Receiver-Operating-Characteristic curve:
    ```python
    chain_results[0]['ROC_curve']
    ```
    Output:
    <div align="center">
    <img src="https://raw.githubusercontent.com/siwei66/swectral/master/demo/demo_results_classification/Modeling/Model_evaluation_reports/Data_chain_Preprocessing_%230_Model_StandardScaler_feat_all_KNeighborsClassifier/ROC_curve_StandardScaler_feat_all_KNeighborsClassifier.png"
         alt="Demo receiver operating characteristic curve"
         width="400"
         style="max-width: 100%;">
    </div>


### 6 Regression demonstration

#### 6.1 Create a directory for regression results
- Create a directory for regression results
    ```python
    report_dir_reg = demo_dir + "/demo_results_regression/"
    os.makedirs(report_dir_reg)
    ```


#### 6.2 Copy and update the previous pipelines to regression
- Copy and update SpecExp and SpecPipe instances
    ```python
    import copy

    exp_reg = copy.deepcopy(exp)
    pipe_reg = copy.deepcopy(pipe)
    targets_reg = copy.deepcopy(targets)
    ```

- Update report directory of SpecExp
    ```python
    exp_reg.report_directory = report_dir_reg
    ```

- Modify targets to numeric, here the numbers approaximate the age of the leaves
    ```python
    targets_reg["Target_value"] = [(5 - int(labl[0])) for labl in targets['Label']]
    ```

- Specify the ROIs within a same leaf to a validation group to prevent data leakage
    ```python
    targets_reg["Validation_group"] = [f"leaf_{labl[0]}" for labl in targets['Label']]
    ```

- Update target information using the modified target dataframe
    ```python
    exp_reg.sample_targets_from_df(targets_reg)
    ```

- Check target values and validation groups
    ```python
    exp_reg.ls_targets()[["Label", "Target_value", "Validation_group"]]
    ```


#### 6.3 Update the pipeline models to regressors
- Check and remove classification models
    ```python
    pipe_reg.ls_model()
    pipe_reg.rm_model()
    ```

- Update the data manager
    ```python
    pipe_reg.spec_exp = exp_reg
    ```

- Add regressors to the pipeline
    Create some regressors:
    ```python
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor

    rf_regressor = RandomForestRegressor(n_estimators=10)
    knn_regressor = KNeighborsRegressor(n_neighbors=3)
    ```
    The pipeline supports sklearn-style models, wrap into the style for arbitrary models.

    Let's skip the time-consuming influence analysis:
    ```python
    pipe_reg.add_model(knn_regressor, validation_method="2-fold", influence_analysis_config=None)
    pipe_reg.add_model(rf_regressor, validation_method="2-fold", influence_analysis_config=None)
    ```
    **TIP**: Influence analysis adopts leave-one-out approach, which is often the slowest step of model evaluation.


#### 6.4 Check and run new pipeline
- Check processing chains
    ```python
    pipe_reg.ls_chains()
    ```

- Run regression pipeline
    ```python
    pipe_reg.run()
    ```


#### 6.5 Check results of a regression pipeline

- For regression tasks, the pipeline generates:
    ```text
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
    ```

- Retrieve reports in console
    ```python
    result_summary_reg = pipe_reg.report_summary()
    chain_results_reg = pipe_reg.report_chains()
    ```

- Check summary reports
    The summary reports include:
    ```python
    result_summary_reg.keys()
    ```
    Output:
    ```text
    dict_keys([
        'Marginal_R2_stats_step_0',
        'Marginal_R2_stats_step_1',
        'Marginal_R2_stats_step_2',
        'Performance_summary',
        'sample_targets_stats'])
    ```

    Demonstration of performance summary content:
    ```python
    result_summary_reg['Performance_summary'].columns
    ```
    Output:
    ```text
    Index([
        'Step_0', 'Step_1', 'Step_2',
        'Mean_Error', 'Standard_Deviation_of_Error', 'Mean_Absolute_Error',
        'Normalized_MAE', 'CV_MAE',
        'Mean_Squared_Error', 'Root_Mean_Squared_Error',
        'Normalized_RMSE', 'CV_RMSE',
        'Residual_Prediction_Deviation', 'R2'
    ], dtype='object')
    ```

- Check processing chain reports
    For each chain, the reports include:
    ```python
    chain_results_reg[0].keys()
    ```
    Output:
    ```text
    dict_keys([
        'Chain_processes',
        'Regression_performance',
        'Residual_analysis',
        'Residual_plot',
        'Scatter_plot',
        'Validation_results'])
    ```
    The influence analysis is absent, because we skip it in model addition.

    Demonstration of the scatter plot of the processing chain:
    ```python
    chain_results_reg[0]['Scatter_plot']
    ```
    Output:
    <div align="center">
    <img src="https://raw.githubusercontent.com/siwei66/swectral/master/demo/demo_results_regression/Modeling/Model_evaluation_reports/Data_chain_Preprocessing_%230_Model_StandardScaler_feat_all_KNeighborsRegressor/Scatter_plot_StandardScaler_feat_all_KNeighborsRegressor.png"
         alt="Demo receiver operating characteristic curve"
         width="400"
         style="max-width: 100%;">
    </div>


### 7 Feature engineering fittable tests

Feature engineering fittables (data transformers) are fitted during the model validation process and function as integrated parts of the model. To incorporate these transformers, use the model connector functions 'combine_transformer_classifier' or 'combine_transformer_regressor' (similar to sklearn.pipeline).

The SpecPipe module also includes a composer that generates batchwise combined models using a full factorial design. Each component within these combined models automatically supports all marginal statistics and testing features available in the module.

- For example:
    ```python
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from swectral.modelconnector import IdentityTransformer  # Passthrough transformer for comparison

    selector1 = SelectKBest(f_classif, k=5)  # Select 5 features
    selector2 = IdentityTransformer()  # For passthrough (no selection)

    from swectral import factorial_transformer_chains

    models = factorial_transformer_chains(
        [StandardScaler(), IdentityTransformer()],  # Model step 1: test data scalers
        {'Feat5': selector1, 'FeatAll': selector2},  # Model step 2: test feature selection fittables
        # ...
        estimators={'KNN': knn_classifier, 'RF': rf_classifier},  # Estimators (specify custom labels using dictionary input)
        is_regression=False
    )
    print(models)
    ```
    Output:
    ```text
    [TransClassifier_StandardScaler_Feat5_KNN,
     TransClassifier_StandardScaler_Feat5_RF,
     TransClassifier_StandardScaler_FeatAll_KNN,
     TransClassifier_StandardScaler_FeatAll_RF,
     TransClassifier_IdentityTransformer_Feat5_KNN,
     TransClassifier_IdentityTransformer_Feat5_RF,
     TransClassifier_IdentityTransformer_FeatAll_KNN,
     TransClassifier_IdentityTransformer_FeatAll_RF]
    ```
- Finally, add the generated models to your pipeline:
    ```python
    for model in models:
        pipe.add_model(model, validation_method="2-fold")
    ```


## Contributing <a name="contributing"></a>

This is an initial release of SpecPipe. Your experience applying this toolset in your specialized field is extremely valuable. Any feedback and contributions are highly welcomed!

- **Report bugs**: Found an issue? Please open a [GitHub issue](https://github.com/siwei66/swectral/issues) with details
- **Share your domain expertise**: Tell us how SpecPipe works (or doesn't work) in your specific application area in [discussions](https://github.com/siwei66/swectral/discussions)
- **Suggest features**: Have ideas for improvements? Use the [GitHub discussions](https://github.com/siwei66/swectral/discussions) or issues tab
- **Submit pull requests**: Feel free to fork and submit PRs for bug fixes or small features
- **Test and provide feedback**: Try it out and let us know about your experience in [discussions](https://github.com/siwei66/swectral/discussions)


## License <a name="license"></a>

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.
