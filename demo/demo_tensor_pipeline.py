# -*- coding: utf-8 -*-
"""
Swectral - Usage example for SpecPipeTensor using mock data. No data downloading required.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Simple example code

# 1. Prepare mock spectral experiment data
# Create a directory for mock experiment data (The example uses a temporary directory)
import os
import shutil

data_dir = os.getcwd() + "/SpecPipeDemoMockTensor/"

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir)

# Create random mock data
from swectral import create_example_raster_shaped

create_example_raster_shaped(data_dir, task_type='classification', n_classes=2)


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
exp.add_groups(["exp_group"])


# 2.3. Raster image management
# Add raster images
exp.add_images_by_name(image_name="sample*", image_directory=f"{data_dir}/mock_rasters/", group="exp_group")

# Check added images
exp.ls_images()


# 2.4. Region of interest (ROI) management
# Load image ROIs using suffix to image names
exp.add_rois_from_bbox()

# Check added ROIs
exp.ls_rois()

# Check sample_rois
exp.ls_rois_sample()


# 2.5. Sample labels and target values

# 2.5.1 Set sample labels

# List sample label dataframe
labels = exp.ls_labels()

# Update sample labels
labels.iloc[:, 1] = [f"sample_{ind.split('_')[4]}" for ind in labels['Sample_ID']]  # type: ignore

# Set sample labels using the updated label dataframe
exp.sample_labels = labels  # type: ignore

# Check sample labels
exp.ls_labels()['Label']

# 2.5.2 Set target values

# List target value dataframe
targets = exp.ls_sample_targets()

# Create mock target values for regression and update target dataframe
targets["Target_value"] = [f"c{ind.split('_')[6].split('-')[0]}" for ind in targets['Sample_ID']]  # type: ignore

# Load target values from updated target dataframe
exp.sample_targets_from_df(targets)

# Check target values
exp.ls_targets()[['Label', 'Target_value']]


# 3. Design testing pipeline
_ = """
Processing functions are organized into pipelines according to distinct data levels.
The SpecPipeTensor framework defines the following data levels for image tensor processing and modeling:

    0 - 'function':
        Accepts input image tensors and performs deterministic preprocessing operations.

    1 - 'fittable':
        Accepts tensors for adaptive transformation and outputs transformed tensors.

    2 - 'model':
        Accepts tensors for predictive modeling and produces final model evaluation reports.

Parallel processes sharing the same data level and execution sequence are combined using a full-factorial design.
"""

# 3.1 Create processing pipeline
from swectral import SpecPipeTensor

pipet = SpecPipeTensor(exp)

pipet._sample_targets
pipet._is_target_numeric


# 3.2 Deterministic image processing

from swectral.functions import snv


# Compared with raw data for example
def raw(v):  # type: ignore
    return v


import torch  # noqa: E402
import torch.nn as nn


# 3.3 Adaptive image processing

class Conv1x1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.conv(x)


# Compared with spectral band binning
class Passthrough(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 2, 1, bias=False)

        with torch.no_grad():
            w = torch.zeros(2, 4, 1, 1)
            w[0, 0, 0, 0] = 1
            w[0, 1, 0, 0] = 1
            w[1, 2, 0, 0] = 1
            w[1, 3, 0, 0] = 1
            self.conv.weight.copy_(w)

    def forward(self, x):
        return self.conv(x)


conv1x1 = Conv1x1()
passthrough = Passthrough()
opt_conv1 = torch.optim.Adam(conv1x1.parameters())
opt_pass = torch.optim.Adam(passthrough.parameters())


# 3.4 Add models to the pipeline
class CNNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class MLPClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Global average over H,W
        x = x.mean(dim=(2, 3))
        return self.head(x)


cnn = CNNClassifier(2, num_classes=2)
mlp = MLPClassifier(2, num_classes=2)
opt_cnn = torch.optim.Adam(cnn.parameters())
opt_mlp = torch.optim.Adam(mlp.parameters())

crit = nn.CrossEntropyLoss()


# 3.5 Compose pipelines
pipet.compose_pipeline([
    ((0, 1), [snv, raw]),
    ((1, 1), {"Conv1x1": conv1x1, "Passthrough": passthrough},
     {'batch_size': 16, 'shuffle': True, 'criterion': crit, 'optimizer': [opt_conv1, opt_pass]}),
    ((1, 2), {'CNN': cnn, 'MLP': mlp},
     {'batch_size': 16, 'shuffle': True, 'criterion': crit, 'optimizer': [opt_cnn, opt_mlp]})
])






