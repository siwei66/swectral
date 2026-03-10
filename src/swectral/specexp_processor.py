# -*- coding: utf-8 -*-
"""
Swectral - SpecExp spatial preprocessors

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Typing
from typing import Union, Any

# Basic data
import numpy as np

# Local
from .specio import simple_type_validator
from .sample_aug import resample_roi


# %% ROI subset augmentation processor


@simple_type_validator
def _roi_subset_augmentation_core(
    rois_sample_item: tuple,
    targets_dict: dict[str, Any],
    n_sub: int,
    resolution: Union[int, float],
    coverage_ratio: float,
    random_state: int,
) -> list[tuple[dict[str, Any], tuple, tuple]]:
    """
    ROI subset augmentation processor for multiprocessing.
    Returns list of (sub_ROI_parameters_for_add_rois_by_coords, sub_ROI_label_tuple, sub_ROI_target_tuple).
    """
    roit = rois_sample_item
    # Look for target item
    target_it = targets_dict[roit[0]]
    # Generate subset coordinates
    coords = roit[-1]
    subcoords_list = [
        resample_roi(
            coord_lists=coords,
            resolution=resolution,
            coverage_ratio=coverage_ratio,
            random_state=random_state + k,
        )
        for k in range(n_sub)
    ]
    # Add new ROIs
    sub_rois: list[tuple[dict[str, Any], tuple, tuple]] = []
    for k, subcoords in enumerate(subcoords_list):
        sub_rois_params = {
            "roi_name": f"{roit[3]}_&#aug{k}",
            "coord_lists": subcoords,
            "image_name": roit[2],
            "group": roit[1],
            "as_mask": False,
            "print_update": False,
            "_roi_id": f"{roit[0]}_&#aug{k}",
        }
        # Generate additional sample labels
        label_sub_item = (
            f"{roit[0]}_&#aug{k}",
            f"{target_it[1]}_&#aug{k}",
            target_it[3],
        )
        # Generate additional sample targets
        target_sub_item = (
            f"{roit[0]}_&#aug{k}",
            f"{target_it[1]}_&#aug{k}",
            target_it[2],
            target_it[3],
            target_it[4],
            np.int8(0),
            np.int8(1),
        )
        sub_rois.append((sub_rois_params, label_sub_item, target_sub_item))

    return sub_rois
