# -*- coding: utf-8 -*-
"""
Swectral - SpecExp - Data management module for spectral experiments

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os
from pathlib import Path
import dill
from copy import deepcopy

# Warning
import warnings

# Typing
from typing import Annotated, Any, Callable, Literal, Optional, Union, overload

# Time
from datetime import datetime

# Basic data
import numpy as np
import pandas as pd

# Interface
from tqdm import tqdm

# Local
from .specio import (
    arraylike_validator,
    dataframe_validator,
    envi_roi_coords,
    names_filter,
    search_file,
    shp_roi_coords,
    simple_type_validator,
    unc_path,
)
from .specexp_vis import raster_rgb_preview
from .sample_aug import resample_roi


# %% Spectral Experiment Class - SpecExp


# SpecExp for input file management
class SpecExp:
    """
    SpecExp is the experiment data manager for SpecPipe batch processing, modeling, and model evaluation.
    The SpecPipe pipeline module uses SpecExp for data loading.
    Create and configure a ``SpecExp`` instance before designing SpecPipe testing pipelines.

    Attributes
    ----------
    report_directory : str
        Root directory for reports.

    log_loading : bool
        Whether configuration logging is enabled. Default is True.

    groups : list of str
        Experiment groups. Each item is a group name.

    images : list of tuple
        Raster image loading data.
        Each tuple contains:

        - image_id : str
        - group : str
        - image_name : str
        - mask_of : str
        - image_path : str

    images_data : list of tuple
        Raster images for sample spectral data. Same structure as ``images``.

    images_mask : list of tuple
        Raster images serving as masks. Same structure as ``images``.

    rois_from_file : list of tuple
        Regions of interest (ROIs) loaded from ROI files.
        Each tuple contains:

        - roi_id : str
        - group : str
        - image_name : str
        - roi_name : str
        - roi_type : str
        - coord_lists : list of list of tuple of 2 (float or int)
        - roi_file_name : str
        - roi_file_path : str

    rois_from_coords : list of tuple
        ROIs added from coordinate lists (without file info).
        Each tuple contains:

        - roi_id : str
        - group : str
        - image_name : str
        - roi_name : str
        - roi_type : str
        - coord_lists : list of list of tuple of 2 (float or int)

    rois : list of tuple
        All regions of interest. Same structure as ``rois_from_coords``.

    rois_sample : list of tuple
        ROIs serving as samples. Same structure as ``rois_from_coords``.

    rois_mask : list of tuple
        ROIs serving as masks. Same structure as ``rois_from_coords``.

    standalone_specs : list of tuple
        Standalone spectral data.
        Each tuple contains:

        - spectrum_id : str
        - group : str
        - use_type : str
        - sample_name : str
        - spectral_data : list of (float or int)

    standalone_specs_sample : list of tuple
        Training sample spectral data. Same structure as ``standalone_specs``.

    sample_labels : list of tuple
        Custom sample labels.
        Each tuple contains:

        - sample_id : str
        - label : str
        - group : str

    sample_targets : list of tuple
        Target values of samples for modeling.
        Each tuple contains:

        - sample_id : str
        - label : str
        - target_value : str, bool, int, or float
        - group : str
        - validation_group : str
        - test_mask : int
        - train_mask : int

    create_time : str
        Creation date and time of this SpecExp isntance.

    Methods
    -------
    add_groups
        Add experiment groups by list of group names.

    ls_groups
        List added groups.

    rm_group
        Remove a group, along with associated image paths, ROI files, and ROIs.

    add_images_by_name
        Add raster images to a group by searching name pattern(s) in a directory.

    add_images_by_path
        Add raster images to a group by file path(s).

    ls_images
        List added image items filtered by name and group.

    rm_images
        Remove raster images by name and group; removes associated ROIs.

    add_rois_by_suffix
        Load ROIs from files by suffix patterns in associated images.

    add_rois_by_file
        Load ROIs from ROI files in the path list to an image.

    add_roi_by_coords
        Add a (multi-)polygon ROI from vertex coordinates.

    ls_rois
        List added ROIs filtered by roi_name, group_name, image name, or source ROI file.

    rm_rois
        Remove loaded ROIs by provided values or patterns.

    roi_subset_augmentation
        Perform spatial subset augmentation on the ROI data.

    add_standalone_specs
        Add 1D standalone spectra. Alias: add_specs.

    load_standalone_specs
        Load/reload standalone spectra from CSV file. Alias: load_specs.

    ls_standalone_specs
        List added spectra filtered by group, use_type, and sample_name. Alias: ls_specs.

    rm_standalone_specs
        Remove added spectra by group, use_type, and sample_name. Alias: rm_specs.

    ls_sample_labels
        Retrieve custom sample labels as a dataframe. Alias: ls_labels.

    sample_labels_from_df
        Load custom sample labels from a dataframe. Alias: labels_from_df.

    sample_labels_to_csv
        Save custom sample labels to CSV. Alias: labels_to_csv.

    sample_labels_from_csv
        Load custom sample labels from a CSV. Alias: labels_from_csv.

    ls_sample_targets
        Retrieve sample target values as a dataframe. Alias: ls_targets.

    sample_targets_from_df
        Load target values from a dataframe. Alias: targets_from_df.

    sample_targets_to_csv
        Save target values to CSV. Alias: targets_to_csv.

    sample_targets_from_csv
        Load sample target values from CSV. Alias: targets_from_csv.

    save_data_config
        Save current configurations. Alias: save_config.

    load_data_config
        Reload configurations from saved file. Alias: load_config.

    Examples
    --------
    Create a ``SpecExp`` instance from a directory path::

        >>> exp = SpecExp("/experiment")
    """

    @simple_type_validator
    def __init__(self, report_directory: str, log_loading: bool = False) -> None:
        # log_loading
        self._log_loading: bool = log_loading

        # Report directory
        report_directory = (report_directory.replace("\\", "/") + "/").replace("//", "/")
        report_directory_path = Path(unc_path(report_directory)).absolute()
        report_directory = (str(report_directory_path).replace("\\", "/") + "/").replace("//", "/")
        if report_directory_path.is_dir() is False:
            report_directory_path.mkdir(parents=True, exist_ok=True)
            warnings.warn(
                f"\nThe specified report directory does not exist and is created: \n'{report_directory}'",
                UserWarning,
                stacklevel=2,
            )
        self._report_directory: str = report_directory

        # Experiment group management
        # [0 group]
        self._groups: list[str] = []

        # Image I/O
        # [0 id, 1 group, 2 image_name, 3 mask_of, 4 image_path]
        self._images: list[tuple[str, str, str, str, str]] = []
        self._images_data: list[tuple[str, str, str, str, str]] = []
        self._images_mask: list[tuple[str, str, str, str, str]] = []

        # ROI I/O
        # File-loaded ROIs - ROIs from file
        # [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
        self._rois_from_file: list[
            tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]], str, str]
        ] = []
        # Console-added ROIs - ROIs from coordinates
        # [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
        self._rois_from_coords: list[
            tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]
        ] = []

        # All ROIs - updated in any ROI addtion or loading
        # [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
        self._rois: list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]] = []
        self._rois_sample: list[
            tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]
        ] = []
        self._rois_mask: list[
            tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]
        ] = []

        # Standalone spectra
        # [0 id, 1 group, 2 use_type, 3 sample_name, 4 spectral_data_list]
        self._standalone_specs: list[tuple[str, str, str, str, list[Union[float, int]]]] = []
        self._standalone_specs_sample: list[tuple[str, str, str, str, list[Union[float, int]]]] = []

        # Sample labels
        # sample_labels: [0 fixed sample id, 1 user assinged labels, 2 sample belonging group]
        self._sample_labels: list[tuple[str, str, str]] = []

        # Sample targets
        # sample_targets: [0 fixed sample id, 1 user assinged labels, 2 target values, 3 sample belonging group, 4 validation group, 5 test mask, 6 train mask]  # noqa: E501
        self._sample_targets: list[tuple[str, str, Union[str, bool, int, float], str, str, np.int8, np.int8]] = []

        # Creating time
        self._create_time = datetime.now().strftime("created_at_%Y-%m-%d_%H-%M-%S")

    ### Property methods
    ## Setters with validation
    @property
    def report_directory(self) -> str:
        return self._report_directory

    @report_directory.setter
    @simple_type_validator
    def report_directory(self, value: str) -> None:
        value = (value.replace("\\", "/") + "/").replace("//", "/")
        value_path = Path(unc_path(value)).absolute()
        value = (str(value_path).replace("\\", "/") + "/").replace("//", "/")
        if value_path.is_dir() is False:
            raise ValueError(f"\nreport_directory is invalid: \n'{value}'")
        self._report_directory = value

    @property
    def sample_labels(self) -> list[tuple[str, str, str]]:
        return self._sample_labels

    @sample_labels.setter
    @simple_type_validator
    def sample_labels(self, value: Union[pd.DataFrame, np.ndarray, list[tuple[str, str, str]]]) -> None:
        # Validate shape
        arraylike_validator(ndim=2)(value)
        value_df = pd.DataFrame(value)
        value_df.columns = ["Sample_ID", "Label", "Group"]
        value_df = value_df.astype(str)
        if len(self._sample_labels) == 0:
            raise ValueError("Cannot set sample labels: No samples exist.\nPlease add samples first.")
        else:
            value_df = dataframe_validator(shape=(len(self._sample_labels), 3))(value_df)
        # Validate labels
        lb_list = list(value_df["Label"])
        lbset = set(lb_list)
        lbset.discard("-")
        lb_list_valid = [v for v in lb_list if v != "-"]
        if len(lbset) < len(lb_list_valid):
            duplicates = [v for v in lb_list if lb_list.count(v) > 1]
            raise ValueError(f"Sample labels must be unique, got duplicated label(s) : {duplicates}")
        # Validate IDs
        idv_list = list(value_df["Sample_ID"])
        ids_list = [lbt[0] for lbt in self._sample_labels]
        idv = set(idv_list)
        ids = set(ids_list)
        if idv == ids:
            new_lbs = [
                (sid, value_df["Label"][value_df["Sample_ID"] == sid].values[0], self._sample_id_to_group([sid])[0])
                for sid in ids_list
            ]
        else:
            raise ValueError(
                f"Given sample labels and existed sample labels do not match\n\n\
                    Got sample labels:\n{idv_list},\n\nexisted sample labels:\n{ids_list}"
            )
        self._sample_labels = new_lbs
        # Update labels in sample_targets
        sample_targets = []
        for lbt in new_lbs:
            sid = lbt[0]
            lbi = lbt[1]
            sgroup = lbt[2]
            target, sgroup1, vgroup, mte, mtr = [tgt[2:7] for tgt in self.sample_targets if tgt[0] == sid][0]
            # Validate group consistency
            assert sgroup == sgroup1
            sample_targets.append((sid, lbi, target, sgroup, vgroup, mte, mtr))
        self._sample_targets = sample_targets

    @property
    def sample_targets(self) -> list[tuple[str, str, Union[str, bool, int, float], str, str, np.int8, np.int8]]:
        return self._sample_targets

    @sample_targets.setter
    @simple_type_validator
    def sample_targets(  # noqa: C901
        self,
        value: Annotated[Any, arraylike_validator(shape=(0, 7))],
    ) -> None:

        # Validate DataFrame
        if type(value) is pd.DataFrame:
            if (
                value.columns == ["Sample_ID", "Label", "Target_value", "Group", "Validation_group", "Test", "Train"]
            ).all() & (list(value["Sample_ID"]) == [lbt[0] for lbt in self._sample_labels]):
                value = value.iloc[:, :]

        # Validate Non-DataFrame array-like
        value_df = pd.DataFrame(value)
        del value
        value_df.columns = ["Sample_ID", "Label", "Target_value", "Group", "Validation_group", "Test", "Train"]
        value_df["Label"] = np.array(value_df["Label"]).astype(str)

        # Validate masks
        te_arr = np.array(value_df["Test"]).astype(np.int8)
        tr_arr = np.array(value_df["Train"]).astype(np.int8)
        for te, tr in zip(te_arr, tr_arr):
            if te not in (np.int8(0), np.int8(1)):
                raise ValueError(f"Test mask value must be 0 or 1, got: {te},\nin test mask:\n{te_arr}")
            if tr not in (np.int8(0), np.int8(1)):
                raise ValueError(f"Train mask value must be 0 or 1, got: {tr},\nin train mask:\n{tr_arr}")
            if te == 0 and tr == 0:
                raise ValueError("Test and train mask values cannot both be 0 for the same sample.")
        value_df["Test"] = te_arr
        value_df["Train"] = tr_arr

        # Validate shape
        if len(self._sample_labels) == 0:
            raise ValueError("Cannot set target values: No samples exist.\nPlease add samples first.")
        if len(self._sample_targets) == 0:
            value_df = dataframe_validator(shape=(len(self._sample_labels), 7))(value_df)
        else:
            value_df = dataframe_validator(shape=(len(self._sample_targets), 7))(value_df)

        # Validate labels
        labelv_list = list(value_df["Label"])
        labels_list = [st[1] for st in self._sample_labels]
        labelv = set(labelv_list)
        labels = set(labels_list)
        if ("-" in labels) or ("-" in labelv):
            if "-" in labels:
                warn_msg = f"\nExisted sample labels are not provided or incomplete.\
                    \nGot labels:\n{labels_list}\nThe target values are matched by order of samples.\n"
                warnings.warn(warn_msg, UserWarning, stacklevel=2)
            if "-" in labelv:
                warn_msg = f"\nGiven sample labels of target values are incomplete:\
                    \n{labelv_list}\nThe target values are matched by order of samples.\n"
                warnings.warn(warn_msg, UserWarning, stacklevel=2)
            # Sample values
            # sample_targets: [0 fixed sample id, 1 user assinged labels, 2 target values, 3 sample belonging group, 4 validation group, 5 test mask, 6 train mask]  # noqa: E501
            sample_targets = [
                lt[0:2] + (t,) + lt[2:] + (vg,) + (np.int8(te),) + (np.int8(tr),)
                for lt, t, vg, te, tr in zip(
                    self._sample_labels,
                    value_df["Target_value"],
                    value_df["Validation_group"],
                    value_df["Test"],
                    value_df["Train"],
                )
            ]
        elif labelv == labels:
            sample_targets = [
                lt[0:2]
                + (value_df["Target_value"][value_df["Label"] == lt[1]].values[0],)
                + lt[2:]
                + (value_df["Validation_group"][value_df["Label"] == lt[1]].values[0],)
                + (np.int8(value_df["Test"][value_df["Label"] == lt[1]].values[0]),)
                + (np.int8(value_df["Train"][value_df["Label"] == lt[1]].values[0]),)
                for lt in self._sample_labels
            ]
        else:
            raise ValueError(
                f"Given sample labels and existed sample labels do not match\n\n\
                    Got sample labels:\n{list(labelv)},\n\nexisted sample labels:\n{list(labels)}"
            )

        self._sample_targets = sample_targets

    ## Read only properties
    @property
    def log_loading(self) -> bool:
        return self._log_loading

    @log_loading.setter
    def log_loading(self, value: bool) -> None:
        raise ValueError("log_loading is immutable after creation")

    @property
    def create_time(self) -> str:
        return self._create_time

    @create_time.setter
    def create_time(self, value: str) -> None:
        raise ValueError("create_time is immutable after creation")

    @property
    def groups(self) -> list[str]:
        return self._groups

    @groups.setter
    def groups(self, value: list[str]) -> None:
        raise ValueError("groups cannot be modified directly, use method 'add_groups' and 'rm_group' instead")

    @property
    def images(self) -> list[tuple[str, str, str, str, str]]:
        return self._images

    @images.setter
    def images(self, value: list[tuple[str, str, str, str, str]]) -> None:
        raise ValueError(
            "images cannot be modified directly, \
                use method 'add_images_by_name', 'add_images_by_path' and 'rm_images' instead"
        )

    @property
    def images_data(self) -> list[tuple[str, str, str, str, str]]:
        return self._images_data

    @images_data.setter
    def images_data(self, value: list[tuple[str, str, str, str, str]]) -> None:
        raise ValueError(
            "images_data cannot be modified directly, \
                use method 'add_images_by_name', 'add_images_by_path' and 'rm_images' instead"
        )

    @property
    def images_mask(self) -> list[tuple[str, str, str, str, str]]:
        return self._images_mask

    @images_mask.setter
    def images_mask(self, value: list[tuple[str, str, str, str, str]]) -> None:
        raise ValueError(
            "images_mask cannot be modified directly, \
                use method 'add_images_by_name', 'add_images_by_path' and 'rm_images' instead"
        )

    @property
    def rois_from_file(
        self,
    ) -> list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]], str, str]]:
        return self._rois_from_file

    @rois_from_file.setter
    def rois_from_file(
        self,
        value: list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]], str, str]],
    ) -> None:
        raise ValueError(
            "rois_from_file cannot be modified directly, \
                use method 'add_rois_by_suffix', 'add_rois_by_file' and 'rm_rois' instead"
        )

    @property
    def rois_from_coords(
        self,
    ) -> list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]]:
        return self._rois_from_coords

    @rois_from_coords.setter
    def rois_from_coords(
        self, value: list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]]
    ) -> None:
        raise ValueError(
            "rois_from_coords cannot be modified directly, use method 'add_roi_by_coords' and 'rm_rois' instead"
        )

    @property
    def rois(self) -> list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]]:
        return self._rois

    @rois.setter
    def rois(
        self, value: list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]]
    ) -> None:
        raise ValueError("SpecExp.rois cannot be modified")

    @property
    def rois_sample(
        self,
    ) -> list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]]:
        return self._rois_sample

    @rois_sample.setter
    def rois_sample(
        self, value: list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]]
    ) -> None:
        raise ValueError("SpecExp.rois_sample cannot be modified")

    @property
    def rois_mask(
        self,
    ) -> list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]]:
        return self._rois_mask

    @rois_mask.setter
    def rois_mask(
        self, value: list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]]
    ) -> None:
        raise ValueError("SpecExp.rois_mask cannot be modified")

    @property
    def standalone_specs(self) -> list[tuple[str, str, str, str, list[Union[float, int]]]]:
        return self._standalone_specs

    @standalone_specs.setter
    def standalone_specs(self, value: list[tuple[str, str, str, str, list[Union[float, int]]]]) -> None:
        raise ValueError(
            "SpecExp.standalone_specs cannot be modified directly, \
                use method 'add_standalone_specs' and 'rm_standalone_specs' instead"
        )

    @property
    def standalone_specs_sample(self) -> list[tuple[str, str, str, str, list[Union[float, int]]]]:
        return self._standalone_specs_sample

    @standalone_specs_sample.setter
    def standalone_specs_sample(self, value: list[tuple[str, str, str, str, list[Union[float, int]]]]) -> None:
        raise ValueError(
            "SpecExp.standalone_specs_sample cannot be modified directly, \
                use method 'add_standalone_specs' and 'rm_standalone_specs' instead"
        )

    ### DATA FILE MANAGEMENT METHODS

    ## Group management ====================
    # Add experiment groups by list of group names
    # Format of associated attribute:
    # self._groups: [0 group]
    @simple_type_validator
    def add_groups(self, group_name_list: list[str]) -> None:
        """
        Add experiment groups by list of group names.

        Parameters
        ----------
        group_list : list of str
            List of group names to add. Groups can be experiment groups or groups of certain features to test.

        Examples
        --------
        >>> exp = SpecExp("/experiment")
        >>> exp.add_groups(['group1', 'group2'])
        """
        for group_name in group_name_list:
            if group_name not in self._groups:
                self._groups.append(group_name)
            else:
                print(f"Group name '{group_name}' already exists")

    def ls_groups(self) -> list:
        """
        List added groups.

        Returns
        -------
        list of str
            List of group names.

        Examples
        --------
        >>> exp = SpecExp("/experiment")
        >>> exp.add_groups(['group1', 'group2'])
        >>> exp.ls_groups()

        The result is the same as::

            >>> exp.groups
        """
        return self.groups

    # Remove group and its associated configuration data
    # Format of associated attribute:
    # self._groups: [0 group]
    # self._images: [0 id, 1 group, 2 image_name, 3 mask_of, 4 image_path]
    # self._rois_from_file: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
    # self._rois_from_coords: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    @simple_type_validator
    def rm_group(self, group_name: str) -> None:  # noqa: C901
        """
        Remove a group from this SpecExp instance by name, simultaneously removing associated image paths, ROI files, and added ROIs.

        Parameters
        ----------
        group_name : str
            Group to remove.

        Examples
        --------
        >>> exp = SpecExp("/experiment")
        >>> exp.add_groups(['group1', 'group2'])

        >>> exp.rm_group('group2')
        """  # noqa: E501

        # Validate groups
        if len(self._groups) == 0:
            print("No group added!")
            return

        # Remove group
        if group_name in self._groups:
            self._groups = [group for group in self._groups if group != group_name]
            print(f"\nFollowing group is removed:\n{group_name}\n")
        else:
            print(f"\nGroup '{group_name}' is not found.")
            return

        # Remove images of the group
        result_imgs: list = [[], []]
        for img in self._images:
            if img[1] != group_name:
                result_imgs[0].append(img)
            else:
                result_imgs[1].append(img)
        if len(result_imgs[1]) > 0:
            self._images = result_imgs[0]
            print("\nFollowing associated images are removed:\n")
            self._df_img(result_imgs[1])

        # Update images
        self._update_image_rev()

        # Remove file-loaded ROIs of the group
        result_roif: list = [[], []]
        for roif in self._rois_from_file:
            if roif[1] != group_name:
                result_roif[0].append(roif)
            else:
                result_roif[1].append(roif)
        if len(result_roif[1]) > 0:
            self._rois_from_file = result_roif[0]
            print("\nFollowing associated file-loaded ROIs are removed:\n")
            self._rois_from_file_to_df(result_roif[1])

        # Remove console-added ROIs of the group
        result_roic: list = [[], []]
        for roic in self._rois_from_coords:
            if roic[1] != group_name:
                result_roic[0].append(roic)
            else:
                result_roic[1].append(roic)
        if len(result_roic[1]) > 0:
            self._rois_from_coords = result_roic[0]
            print("\nFollowing associated console-added ROIs are removed:\n")
            self._df_roic(result_roic[1])

        # Update ROIs
        self._update_roi()
        # Update sample labels & targets
        self._update_sample_labels_targets()

    ## Image management ====================

    # Update self._images using self._images_data and self._images_mask
    def _update_image(self) -> None:
        self._images = self._images_data + self._images_mask

    # Reverse update self._images_data and self._images_mask using self._images
    def _update_image_rev(self) -> None:
        self._images_data = [imgt for imgt in self._images if imgt[3] == ""]
        self._images_mask = [imgt for imgt in self._images if imgt[3] != ""]

    # Image list to dataframe
    @simple_type_validator
    def _df_img(
        self,
        image_item_list: list[tuple[str, str, str, str, str]],
        print_simple: bool = True,
        return_df: bool = False,
    ) -> Optional[pd.DataFrame]:
        df_img = pd.DataFrame(image_item_list, columns=["ID", "Group", "Image", "Mask", "Path"])
        if print_simple:
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(df_img.iloc[:, 1:-1])
        if return_df:
            return df_img
        else:
            return None

    # Add image items to attribute self._images
    # Format of associated attribute:
    # self._groups: [0 group]
    # self._images: [0 id, 1 group, 2 image_name, 3 mask_of, 4 image_path]
    @simple_type_validator
    def _add_image_paths(self, group_name: str, mask_of: str, image_path_list: list[str]) -> None:
        """
        Add image items to attribute self._images
        """
        # Get existed img IDs
        if len(self._images) > 0:
            existed_ids = [imgt[0] for imgt in self._images]
        else:
            existed_ids = []

        # Add img paths
        added_items = []
        updated_items = []
        for fpath in image_path_list:
            # Extract img properties from the image path
            img_path = fpath.replace("\\", "/")
            img_name = img_path.split("/")[-1]
            img_id = group_name + "_" + os.path.splitext(img_name)[0] + "_" + mask_of

            # Construct img item
            added_item = (img_id, group_name, img_name, mask_of, img_path)

            # Add / Update img item
            if img_id not in existed_ids:
                self._images.append(added_item)
                added_items.append(added_item)
            else:
                indext = [nid for nid, extid in enumerate(existed_ids) if extid == img_id][0]
                self._images[indext] = added_item
                updated_items.append(added_item)

        # Update images
        self._update_image_rev()

        # Print report
        if len(added_items) > 0:
            print("\nFollowing image items are added:\n")
            self._df_img(added_items)

        if len(updated_items) > 0:
            print("\nFollowing image items already exist and are updated:\n")
            self._df_img(updated_items)

        # Report items
        cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df_all: pd.DataFrame = self._df_img(self._images, False, True)
        df_load_report: pd.DataFrame = self._df_img((added_items + updated_items), False, True)

        # Report directory
        if self.log_loading:
            report_dir = self.report_directory + "SpecExp_configuration/Loading_history/"
            if not os.path.isdir(unc_path(report_dir)):
                os.makedirs(unc_path(report_dir))
            # Save updating reports
            df_all.to_csv(unc_path(report_dir + "All_loaded_images_" + cts + ".csv"), index=False)
            df_load_report.to_csv(unc_path(report_dir + "Loaded_images_" + cts + ".csv"), index=False)

    # Add raster image paths to an experiment group
    # Format of associated attribute:
    # self._groups: [0 group]
    @simple_type_validator
    def _add_images(  # noqa: C901
        self,
        image_name: Union[str, list[str], None] = None,
        image_directory: str = "",
        image_full_path: Union[str, list[str], None] = None,
        group_name: str = "",
        mask_of: str = "",
    ) -> None:
        """
        Add paths of raster images to an experiment group, either by searching name pattern in a directory or by image paths.

        Parameters
        ----------
        image_name : str or list of str or None, optional
            Image name(s) or pattern(s) of image names in the given directory. For multiple patterns or names, please provide them in a list.
            Unix-like filename pattern is supported for image names.
            Image names must be unique within each group for the same type, or the existed duplicated record will be overwritten.
            The default is empty string.

        image_directory : str, optional
            Directory path of the images to add. The default is empty string. The default uses current working directory.

        image_full_path : str or list of str or None, optional
            Absolute path or list of absolute paths of the image files. The default is [].
            The implemention of image_full_path is preferred if image_name and image_directory are simultaneously provided.

        group_name : str
            Group of the added raster images.

        mask_of : str
            If the given image is a binary raster mask of a spectral image, specify the name of the target image it corresponds to.
            If not, this is default to empty string.
        """  # noqa: E501

        # Validate group name
        group_name = str(group_name)
        if group_name == "":
            raise ValueError("missing 1 required argument: 'group_name'")

        if group_name not in self._groups:
            warn_msg = f"\nGroup: {group_name} does not exists, \
                {group_name} is automatically added to experiment groups.\n"
            warnings.warn(warn_msg, UserWarning, stacklevel=2)
            self._groups.append(group_name)

        ## Path list mode
        if image_full_path is not None:
            if image_name is not None:
                warn_msg = "\nDouble definition of name pattern and paths. \
                    The images are added by paths ignoring name pattern.\n"
                warnings.warn(warn_msg, UserWarning, stacklevel=2)
            # Construct path list
            if type(image_full_path) is str:
                image_path_list: list[str] = [image_full_path]
            else:
                assert type(image_full_path) is list
                image_path_list = image_full_path

            # Validate path
            for pathi in image_path_list:
                if not os.path.exists(unc_path(pathi)):
                    raise ValueError(f"Given path does not exist: {pathi}")

            self._add_image_paths(group_name, mask_of, image_path_list)

        ## Name pattern mode
        elif image_name is not None:
            # Construct image name list
            if type(image_name) is str:
                image_name_list: list[str] = [image_name]
            else:
                assert type(image_name) is list
                image_name_list = image_name

            path_total: list = []
            for name_pattern in image_name_list:
                # Default input from current working directory
                if len(image_directory) == 0:
                    image_directory = (os.getcwd().replace("\\", "/") + "/").replace("//", "/")

                # Get paths
                img_paths = search_file(directory_path=image_directory, search_pattern=name_pattern)
                if len(img_paths) > 0:
                    path_total = path_total + img_paths

            # Validate extensions
            img_path_total = []
            img_exts = [
                ".bil",
                ".bsq",
                ".bip",
                ".tif",
                ".tiff",
                ".img",
                ".jp2",
                ".nc",
                ".hdf",
            ]  # Supported img extensions
            for pathi in path_total:
                if os.path.splitext(pathi)[1] in img_exts:
                    img_path_total.append(pathi)

            # Validate path
            for pathi in img_path_total:
                if not os.path.exists(unc_path(pathi)):
                    raise ValueError(f"Given path does not exist: {pathi}")

            # Update images
            if len(img_path_total) > 0:
                self._add_image_paths(group_name, mask_of, img_path_total)
            else:
                print(f"\nNo files found with the given names or name patterns: \n{image_name_list}")

        else:
            raise ValueError("Neither name_pattern nor image_full_path is provided")

    @simple_type_validator
    def add_images_by_name(  # noqa: C901
        self,
        image_name: Union[str, list[str], None] = None,
        image_directory: str = "",
        group: str = "",
        mask_of: str = "",
    ) -> None:
        """
        Add paths of raster images to an experiment group by searching name pattern(s) in the specified directory.

        Parameters
        ----------
        image_name : str or list of str or None, optional
            Image name(s) or pattern(s) of image names in the given directory. For multiple patterns or names, please provide them in a list.
            Unix-like filename pattern is supported for image names.
            Image names must be unique within each group for the same type, or the existed duplicated record will be overwritten.
            The default is None.

        image_directory : str, optional
            Directory path of the images to add.
            If empty string empty string, the current working directory is used. The default is empty string.

        group : str
            Group of the added raster images.

        mask_of : bool
            If the given image is a binary raster mask of a spectral image, specify the name of the target image it corresponds to.
            If not, this is default to empty string.

        Examples
        --------
        Prepare SpecExp instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")

        Add single image::

            >>> exp.add_images_by_name("image.tif", "/rasters", "group1")

        Add single image using default current working directory:

            >>> exp.add_images_by_name("image.tif", group="group1")

        Add multiple images::

            >>> exp.add_images_by_name(["image1.tif", "image2.tif"], "/rasters", "group1")

        Add by pattern::

            >>> exp.add_images_by_name("*.tif", "/rasters", "group1")
            >>> exp.add_images_by_name(["*.tif", "*.tiff"], "/rasters", "group1")

        Add raster mask::

            >>> exp.add_images_by_name("image_mask.tif", "/rasters", "group1", mask_of="image1")
        """  # noqa: E501

        self._add_images(
            image_name=image_name,
            image_directory=image_directory,
            group_name=group,
            mask_of=mask_of,
        )

    @simple_type_validator
    def add_images_by_path(  # noqa: C901
        self,
        path: Union[str, list[str], None] = None,
        group: str = "",
        mask_of: str = "",
    ) -> None:
        """
        Add paths of raster images to an experiment group by image path(s).

        Parameters
        ----------
        path : str or list of str or None, optional
            Absolute path or list of absolute paths of the image files. The default is None.
            The implemention of image_full_path is preferred if image_name and image_directory are simultaneously provided.

        group : str
            Group of the added raster images.

        mask_of : bool
            If the given image is a binary raster mask of a spectral image, specify the name of the target image it corresponds to.
            If not, this is default to empty string.

        Examples
        --------
        Prepare SpecExp instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")

        Add single image::

            >>> exp.add_images_by_path("/image.tif", "group1")

        Add multiple images::

            >>> exp.add_images_by_path(["/image1.tif", "/image2.tif"], "group1")
        """  # noqa: E501

        self._add_images(
            image_full_path=path,
            group_name=group,
            mask_of=mask_of,
        )

    # Get image items by image [0 name list, 1 group]
    # Format of associated attribute:
    # self._images: [0 id, 1 group, 2 image_name, 3 mask_of, 4 image_path]
    @simple_type_validator
    def _get_images(
        self,
        image_name: Union[str, list[str], None] = None,
        group_name: str = "",
        mask_of: Union[str, list[str], None] = None,
    ) -> tuple[list[tuple[str, str, str, str, str]], list[tuple[str, str, str, str, str]]]:
        """
        Retrieve image items from attribute self._images based on image_name and belonging group.
        When not provided, all items are retrieved.

        Returns matched image items and unmatched items.
        """
        # Validate images
        if len(self._images) == 0:
            print("No image added!")
            return ([], [])

        # Retrieve removed and remained image items
        get_items = []
        rest_items = []
        for imgt in self._images:
            # Image name condition
            if image_name is None:
                condi = True
            elif type(image_name) is str:
                condi = imgt[2] == image_name
            else:
                condi = imgt[2] in image_name
            # Belonging group condition
            if group_name == "":
                condg = True
            else:
                condg = imgt[1] == group_name
            # Mask condition
            if mask_of is None:
                condt = True
            elif type(mask_of) is str:
                condt = imgt[3] == mask_of
            else:
                condt = imgt[3] in mask_of
            rcond = condi & condg & condt

            # Filtering condition
            if rcond:
                get_items.append(imgt)
            else:
                rest_items.append(imgt)

        return get_items, rest_items

    # Get image items by image [0 name list, 1 group]
    @overload
    def ls_images(
        self,
        image_name: Union[str, list[str], None] = None,
        group: str = "",
        mask_of: Union[str, list[str], None] = None,
        *,
        print_result: bool = True,
        return_dataframe: Literal[False] = False,
    ) -> None: ...

    @overload
    def ls_images(
        self,
        image_name: Union[str, list[str], None] = None,
        group: str = "",
        mask_of: Union[str, list[str], None] = None,
        *,
        print_result: bool = True,
        return_dataframe: Literal[True] = True,
    ) -> Annotated[
        pd.DataFrame,
        dataframe_validator({"ID": str, "Group": str, "Image": str, "Type": str, "Path": str}),
    ]: ...

    # Get image items by image [0 name list, 1 group]
    # Format of associated attribute:
    # self._images: [0 id, 1 group, 2 image_name, 3 mask_of, 4 image_path]
    @simple_type_validator
    def ls_images(
        self,
        image_name: Union[str, list[str], None] = None,
        group: str = "",
        mask_of: Union[str, list[str], None] = None,
        *,
        print_result: bool = True,
        return_dataframe: bool = False,
    ) -> Optional[
        Annotated[
            pd.DataFrame,
            dataframe_validator({"ID": str, "Group": str, "Image": str, "Type": str, "Path": str}),
        ]
    ]:
        """
        List added image items based on image name and group.
        When a parameter is not specified, the corresponding filter is not applied.

        Parameters
        ----------
        image_name : str or list of str or None, optional
            Name or list of names of the image(s) to filter.

            - If a string, only images with that exact name are matched.
            - If a list of strings, any image matching one of the names is included.
            - If None, no filtering by image name is applied.

            Default is None.

        group : str, optional
            Name of the group to filter images by.

            - If empty string, no filtering by group is applied.

            Default is an empty string.

        mask_of : str or list of str or None, optional
            Name(s) of the target image(s) if filtering for raster masks.

            - For sample images, provide an empty string.
            - For masks of a single image, provide the target image name as a string.
            - For masks of multiple images, provide a list of target image names.
            - If None, no filtering by mask association is applied.

            Default is None.

        print_result : bool, optional
            Whether to print the matched image items to the console.
            Default is True.

        return_dataframe : bool, optional
            Whether to return the results as a pandas DataFrame.
            Default is False.

        Returns
        -------
        pandas.DataFrame or None
            If ``return_dataframe=True``, returns a pandas DataFrame containing matched image items.
            Each row corresponds to one image. Columns:

                ID : str
                    Unique identifier for the image.
                Group : str
                    Group to which the image belongs.
                Image : str
                    Name of the image file.
                Type : str
                    Type of the image item, e.g., ``'sample'`` or ``'mask'``.
                Path : str
                    Full path to the image file.

            If ``return_dataframe=False``, returns None and prints the result to console.

        Examples
        --------
        Prepare a ``SpecExp`` instance and add a group and image::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name("image.tif", "group1")

        List all added images::

            >>> exp.ls_images()

        Filter images by name or group::

            >>> exp.ls_images("image.tif")
            >>> exp.ls_images(group="group1")

        Retrieve image item dataframe::

            >>> df = exp.ls_images(return_dataframe=True)
        """

        group_name: str = group

        if len(self._images) == 0:
            if print_result:
                print("No image added!")
            if return_dataframe:
                return pd.DataFrame(
                    columns=["ID", "Group", "Image", "Type", "Path"],
                    dtype={"ID": str, "Group": str, "Image": str, "Type": str, "Path": str},
                )
            else:
                return None
        else:
            matched_items = self._get_images(image_name, group_name, mask_of)[0]
            df_matched = self._df_img(matched_items, print_result, True)
            if return_dataframe:
                return df_matched
            else:
                return None

    # Display images with ROIs
    @simple_type_validator
    def show_image(
        self,
        image_name: str,
        group: str,
        rgb_band_index: tuple[int, int, int],
        display_size: tuple[Union[int, float], Union[int, float]] = (12, 9),
        roi_name: Optional[str] = None,
        roi_color: Union[str, tuple[float, float, float]] = "red",
        roi_linewidth: Union[int, float] = 3,
        *,
        display_roi: bool = True,
        normalize: bool = True,
        output_path: Optional[str] = None,
        dpi: int = 150,
        show_image: bool = True,
    ) -> None:
        """
        Display a raster image with associated ROIs in RGB preview.

        Parameters
        ----------
        image_name : str
            Name of the image to display.

        group : str
            Group to which the image belongs.

        rgb_band_index : tuple of 3 int
            Indices of the Red, Green, and Blue bands for RGB display.
            Example: (R_index, G_index, B_index)

        display_size : tuple of 2 (float or int), optional
            Size of the preview plot in inches (width, height).
            Default is (12, 9).

        roi_name : str or None, optional
            Name of the ROI to display.
            If None, all ROIs associated with the image are displayed.
            Default is None.

        roi_color : str or tuple of 3 float, optional
            Color of the ROI frame.

            - If str, standard color names (e.g., "red") are accepted.
            - If tuple, specify RGB values as float numbers in the range 0-1, e.g., (1.0, 0.0, 0.0) for red.

            Default is "red".

        roi_linewidth : int or float, optional
            Line width of the ROI frame in the plot.
            Default is 3.

        display_roi : bool, optional
            Whether to display ROI frames on the image.
            Default is True.

        normalize : bool, optional
            Whether to normalize image values for RGB display.
            Default is True.

        output_path : str or None, optional
            Path to save the RGB preview image.
            If None, no file is saved.
            Default is None.

        dpi : int, optional
            Dots per inch (DPI) for the output image.
            Default is 150.

        show_image : bool, optional
            Whether to display the image in the console.
            Default is True.

        Returns
        -------
        None
            This function does not return a value.
            The image is displayed in console and saved to file depending on the parameters.

        Examples
        --------
        Prepare a ``SpecExp`` instance and add images::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name("image.tif", "group1")

        Basic usage::

            >>> exp.show_image("image.tif", "group1", rgb_band_index=(192, 124, 64))

        Save output to a PNG image file::

            >>> exp.show_image(
            ...     "image.tif",
            ...     "group1",
            ...     rgb_band_index=(192, 124, 64),
            ...     output_path="/image_rgb.png"
            ... )
        """

        # Get image
        if roi_name is None:
            roi_name_list: Optional[list] = None
        else:
            roi_name_list = [roi_name]
        image = self.ls_images(
            image_name=image_name, group=group, mask_of=None, print_result=False, return_dataframe=True
        )
        raster_path = image['Path'][0]
        roi_coord_list = []
        if display_roi:
            # Get ROIs
            rois = self.ls_rois_sample(
                roi_name_list=roi_name_list,
                image_name=image_name,
                group=group,
                print_result=False,
                return_dataframe=True,
            )['Coordinates']
            # Aggregate ROIs
            for roi in rois:
                for poly in roi:
                    roi_coord_list.append(poly)
        # Plot RGB
        raster_rgb_preview(
            raster_path=raster_path,
            rgb_band_index=rgb_band_index,
            display_size=display_size,
            roi_coords=roi_coord_list,
            roi_color=roi_color,
            roi_linewidth=roi_linewidth,
            normalize=normalize,
            output_path=output_path,
            dpi=dpi,
            show_image=show_image,
        )

    # Remove images by name pattern
    # Format of associated attribute:
    # self._images: [0 id, 1 group, 2 image_name, 3 mask_of, 4 image_path]
    # self._rois_from_file: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
    # self._rois_from_coords: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    @simple_type_validator
    def rm_images(  # noqa: C901
        self,
        image_name: Union[str, list[str], None] = None,
        group: str = "",
        mask_of: Union[str, list[str], None] = None,
        globbing: bool = True,
    ) -> None:
        """
        Remove raster images from this SpecExp instance by group, image name(s) and mask target(s), simultaneously removing associated ROIs.
        If a criterion is not provided, matched images will be removed regardless of the criterion.

        Parameters
        ----------
        image_name : str or list of str, or None, optional
            Name(s) of the image(s) to remove.

            - If str, remove a single image.
            - If list of str, remove multiple images.
            - If None, all images matching the other criteria are removed.

            Default is None.

        group : str, optional
            Name of the group to which the images belong.
            If empty string or not provided, images are removed regardless of group.
            Default is "".

        mask_of : str or list of str, or None, optional
            Name(s) of the target images if masks should be removed.

            - If str, remove masks associated with a single image.
            - If list of str, remove masks associated with multiple images.
            - If None, no mask filtering is applied.

            Default is None.

        globbing : bool, optional
            Whether Unix-like filename patterns (globbing) are enabled.

            - True (default): patterns like "image*" are allowed.
            - False: filenames must match exactly.

        Examples
        --------
        Prepare a ``SpecExp`` instance and add images::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name(["image1.tif", "image2.tif"], "group1")

        Remove all images::

            >>> exp.rm_images()

        Remove a single image by name::

            >>> exp.rm_images("image1.tif")

        Remove multiple images by name::

            >>> exp.rm_images(["image1.tif", "image2.tif"])

        Remove images by group::

            >>> exp.rm_images(group="group2")

        Remove images by name pattern::

            >>> exp.rm_images("image*")
        """  # noqa: E501

        group_name: str = group

        # Validate images
        if len(self._images) == 0:
            print("No image added!")
            return

        # Validate groups
        if (len(group_name) > 0) & (group_name not in self._groups):
            raise ValueError(f"\nGroup '{group_name}' is not found")

        # Choose image name pattern list
        if image_name is None:
            image_name_list = [imgt[2] for imgt in self._images]
        elif type(image_name) is str:
            image_name_list = [image_name]
        elif type(image_name) is list:
            image_name_list = image_name

        removed_total: dict = {"img": [], "roif": [], "roic": []}
        for image_name_pattern in image_name_list:
            # Construct image name list
            imgn_list0 = [imgt[2] for imgt in self._images]
            imgn_list = []
            for imgn in imgn_list0:
                if imgn not in imgn_list:
                    imgn_list.append(imgn)
            if globbing:
                filt_imgn_list = names_filter(imgn_list, image_name_pattern)[0]
            else:
                filt_imgn_list = [image_name_pattern]

            # Get removed image items
            for imgn in filt_imgn_list:
                get_results = {
                    "img": self._get_images(image_name=[imgn], group_name=group_name, mask_of=mask_of),
                    "roif": self._get_rois(image_name=imgn, group_name=group_name, source_type="file"),
                    "roic": self._get_rois(image_name=imgn, group_name=group_name, source_type="coords"),
                }
                # Get updated images, roif and roic
                for key in removed_total.keys():
                    removed_items = get_results[key][0]
                    assert type(removed_items) is list
                    removed_list = removed_total[key]
                    for ritem in removed_items:
                        if (len(removed_list) == 0) or (ritem[0] not in [imgt[0] for imgt in removed_list]):
                            removed_total[key].append(ritem)

        # Get updates
        to_update: dict[str, list] = {
            "img": self._images,
            "roif": self._rois_from_file,
            "roic": self._rois_from_coords,
        }
        updated: dict = {"img": [], "roif": [], "roic": []}
        for key in removed_total.keys():
            removed_ids = [item[0] for item in removed_total[key]]
            updated[key] = [item for item in to_update[key] if item[0] not in removed_ids]

        # Update
        self._images = updated["img"]
        self._rois_from_file = updated["roif"]
        self._rois_from_coords = updated["roic"]

        # Print report
        # self._images update report
        if len(removed_total["img"]) > 0:
            print("\nFollowing image items are removed:\n")
            self._df_img(removed_total["img"])
        else:
            print("\nNo image item found, no image is removed")

        # self._rois_from_file update report
        if len(removed_total["roif"]) > 0:
            print("\nFollowing associated ROI file items are removed:\n")
            self._rois_from_file_to_df(removed_total["roif"])
        else:
            print("\nNo file-added ROI item found, no file-added ROI is removed")

        # self._rois_from_coords update report
        if len(removed_total["roic"]) > 0:
            print("\nFollowing associated console-added ROI items are removed:\n")
            self._df_roic(removed_total["roic"])
        else:
            print("\nNo console-added ROI item found, no console-added ROI is removed")

        # Update image mask and image sample
        self._update_image_rev()
        # Update ROIs
        self._update_roi()
        # Update sample labels & targets
        self._update_sample_labels_targets()

    ## ROI Management ====================

    # ROI list to dataframe - roif
    @simple_type_validator
    def _rois_from_file_to_df(
        self,
        roi_item_list: list[
            tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]], str, str]
        ],
        print_simple: bool = True,
        return_df: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Convert roif list to pandas dataframe or print simplified roif dataframe.
        """
        df_roif = pd.DataFrame(
            roi_item_list,
            columns=[
                "ID",
                "Group",
                "Image",
                "ROI_name",
                "ROI_type",
                "Coordinates",
                "ROI_source_file",
                "ROI_file_path",
            ],
        )
        if print_simple:
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(df_roif.iloc[:, [1, 2, 3, 4, 6]])
        if return_df:
            return df_roif
        else:
            return None

    # ROI list to dataframe - roic
    @simple_type_validator
    def _df_roic(
        self,
        roi_c_item_list: list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]],
        print_simple: bool = True,
        return_df: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Convert roic list to pandas dataframe or print simplified roic dataframe.
        """
        df_roic = pd.DataFrame(roi_c_item_list, columns=["ID", "Group", "Image", "ROI_name", "ROI_type", "Coordinates"])
        if print_simple:
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(df_roic.iloc[:, 1:-1])
        if return_df:
            return df_roic
        else:
            return None

    # ROI file parser
    @simple_type_validator
    def _roi_parser(
        self, roi_file_path_list: list[str], group_name: str, image_name: str, roi_type: str
    ) -> tuple[list, list, list]:
        """
        Retrieve ROI items from ROI files.

        Returns three lists:
            First list is retrieved ROI items.
            Second list is failed ROI file names.
            Third list is failed ROI file list with paths and error info.
        """
        # Outputs
        new_rois = []
        fail_list = []
        fail_err_list = []

        # Validate image existence in the specified group
        existed_images = [imgt for imgt in self.images if (imgt[2] == image_name and imgt[1] == group_name)]
        if len(existed_images) < 1:
            raise ValueError(f"No image with name '{image_name}' added in given group '{group_name}'.")
        if len(existed_images) > 1:
            raise ValueError(
                f"Duplicated image with name '{image_name}' found in given group '{group_name}': \
                             {existed_images}. Image name must be unique in a group."
            )

        # Parsing ROI files
        for roi_path in roi_file_path_list:
            roi_fn = roi_path.split("/")[-1]
            exti = roi_fn.rpartition(".")[2]

            # Choose parser
            x_roi_coords: Callable[[str], list[dict[str, Any]]]
            if exti == "xml":
                x_roi_coords = envi_roi_coords
            elif exti == "shp":
                x_roi_coords = shp_roi_coords
            else:
                fmsg = f"\nUnkown type of ROI file is ignored: {roi_fn}"
                fail_list.append(roi_fn)
                fail_err_list.append(["Unkown type of ROI file", roi_path, "-", "-"])
                print(fmsg)
                continue

            # Parsing
            result = []
            try:
                result = x_roi_coords(roi_path)
                if len(result) == 0:
                    fmsg = f"\nNo ROI found in the ROI file: {roi_fn}"
                    fail_list.append(roi_fn)
                    fail_err_list.append(["No ROI found in the ROI file", roi_path, "-", "-"])
                    print(fmsg)
            except Exception as e:
                result = []
                fmsg = f"\nParsing error: {roi_fn}"
                fail_list.append(roi_fn)
                fail_err_list.append(["Parsing error", roi_path, type(e).__name__, str(e)])
                print(fmsg)
                continue

            # Add roi item
            if len(result) > 0:
                for roi in result:
                    roi_name = roi["name"]
                    roi_coords = roi["coordinates"]
                    roi_id = (
                        "File-loaded-ROI_"
                        + group_name
                        + "_"
                        + image_name.replace(".", "-")
                        + "_"
                        + roi_fn.replace(".", "-")
                        + "_"
                        + roi_name
                    )
                    roi_item = (
                        roi_id,
                        group_name,
                        image_name,
                        roi_name,
                        roi_type,
                        roi_coords,
                        roi_fn,
                        roi_path,
                    )
                    new_rois.append(roi_item)

        # Output
        return new_rois, fail_list, fail_err_list

    # Receive ROI in self._rois_from_file from parsed ROI items
    # self._rois_from_file: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
    @simple_type_validator
    def _roi_receiver(self, new_roi_list: list, failed_filename_list: list, failed_error_list: list) -> None:
        """
        Add ROIs from parsed lists to self._rois_from_file.

        Loading report is displayed, and error report is stored in csv file in reporting directory.
        """
        new_rois = new_roi_list
        fail_list = failed_filename_list
        fail_err_list = failed_error_list

        # Report dir
        report_dir = self.report_directory + "SpecExp_configuration/Loading_history/"
        if not os.path.isdir(unc_path(report_dir)):
            os.makedirs(unc_path(report_dir))

        # Current time for saving reports
        cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Update self._rois_from_file
        if len(new_rois) > 0:
            # Update
            if len(self._rois_from_file) == 0:
                # Initial addition
                self._rois_from_file = new_rois
            else:
                # Matching update existed ROIs
                for i in range(len(self._rois_from_file)):
                    for roit in new_rois:
                        if self._rois_from_file[i][0] == roit[0]:
                            self._rois_from_file[i] = roit
                # Add new ROIs
                existed_roi_ids = [roit[0] for roit in self._rois_from_file]
                newly_added_rois = [roit for roit in new_rois if roit[0] not in existed_roi_ids]
                if len(newly_added_rois) > 0:
                    self._rois_from_file = self._rois_from_file + newly_added_rois

            # Save & print report: Current all ROIs
            df_rois_from_file = self._rois_from_file_to_df(self._rois_from_file, False, True)
            assert type(df_rois_from_file) is pd.DataFrame
            df_all = df_rois_from_file.iloc[:, [1, 2, 3, 4, 6, 7]]

            # Save & print report: Newly loaded or updated
            print("\nFollowing ROI items loaded:\n")
            df_rois_from_file = self._rois_from_file_to_df(new_rois, True, True)
            assert type(df_rois_from_file) is pd.DataFrame
            df_load_report = df_rois_from_file.iloc[:, [1, 2, 3, 4, 6, 7]]

            # Report directory
            if self.log_loading:
                # Save updating reports
                df_all.to_csv(unc_path(report_dir + "All_loaded_ROIs_" + cts + ".csv"), index=False)
                df_load_report.to_csv(unc_path(report_dir + "Loaded_ROIs_" + cts + ".csv"), index=False)
                # Print saved path
                print("\nROI updating reports saved in: \n", report_dir)

        else:
            print("No ROI added!")

        # Save err report
        if len(fail_err_list) > 0:
            df_err = pd.DataFrame(fail_err_list, columns=["Item", "Path", "Error_line", "Error_message"])
            df_err.to_csv(unc_path(report_dir + "_failed_ROI_loading_" + cts + ".csv"), index=False)
            # Print err
            print("\nLoading from following ROI files failed:\n", fail_list)

    # TODO : Auto-ROI - Not implemented in current version
    # Function : Each image as a sample, generate multipolygon ROIs of all valid regions of the image using a mask value
    # Here for API, function in new file
    # Or each untouched independent valid area in a raster image as a sample, set by 'segment' parameter.
    # self._rois_from_coords: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    # @simple_type_validator
    # def add_rois_auto(
    #         self,
    #         group_name: str,
    #         image_name: str,
    #         mask: bool,
    #         mask_value:Union[int, float],
    #         min_size:Union[int, float]
    #         ) -> None:
    #     # Get image
    #     self._get_images()

    # Add ROI file paths to an image by ROI file name suffix to image names
    # Format of associated attribute:
    # self._groups: [0 group]
    # self._images: [0 id, 1 group, 2 image_name, 3 mask_of, 4 image_path]
    # self._rois_from_file: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
    # self._rois_from_coords: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    @simple_type_validator
    def add_rois_by_suffix(  # noqa: C901
        self,
        roi_filename_suffix: str = "",
        search_directory: str = "",
        group: str = "",
        as_mask: bool = False,
        *,
        include_roiname: Union[str, list[str], None] = None,
        exclude_roiname: Union[str, list[str], None] = None,
    ) -> None:
        """
        Load ROIs from files by searching for a suffix pattern in the names of associated images and add them to this SpecExp instance.

        Parameters
        ----------
        roi_filename_suffix : str, optional
            Suffix pattern to match ROI files to the names of added images.
            If empty string, all ROI files with arbitrary suffixes are considered.
            Default is empty string.

        search_directory : str, optional
            Directory path in which to search for ROI files.
            If empty string, current working directory is used.
            Default is empty string.

        group : str
            Name of the image group to which ROIs will be added.
            Must be one of the groups in ``SpecExp.groups``.

        as_mask : bool, optional
            If True, added ROIs are labeled as masks.
            If False, added ROIs are labeled as sample ROIs.
            Default is False.

        include_roiname : str or list of str or None, optional
            Inclusion filter for ROI names.

            - If str, ROIs containing that substring are included.
            - If list of str, ROIs containing any of the substrings are included.
            - If None, all ROIs are included.

            Default is None.

        exclude_roiname : str or list of str or None, optional
            Exclusion filter for ROI names.

            - If str, ROIs containing that substring are excluded.
            - If list of str, ROIs containing any of the substrings are excluded.
            - If None, no ROIs are excluded.

            Default is None.

        Examples
        --------
        Prepare SpecExp instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name("image1.tif", "group1")

        Add ROIs matching a suffix pattern::

            >>> exp.add_rois_by_suffix("_roi.xml", group="group1")

        Add ROIs using a glob pattern::

            >>> exp.add_rois_by_suffix("*.xml", group="group1")

        Exclude a specific ROI name::

            >>> exp.add_rois_by_suffix("*.xml", group="group1", exclude_roiname='name_to_exclude')

        Exclude multiple ROI names::

            >>> exp.add_rois_by_suffix("*.xml", group="group1", exclude_roiname=['name_to_exclude1', 'name_to_exclude2'])

        Only include ROIs by specific ROI name pattern::

            >>> exp.add_rois_by_suffix("*.xml", group="group1", include_roiname='_sample')

        Add ROI as masks to the associated images::

            >>> exp.add_rois_by_suffix("_mask.xml", group="group1", as_mask=True)
        """  # noqa: E501

        group_name: str = group

        # Validate group name
        group_name = str(group_name)
        if group_name == "":
            raise ValueError("missing 1 required argument: 'group'")

        # Initialize 'include' and 'exclude'
        if include_roiname is None:
            include = []
        elif isinstance(include_roiname, str):
            include = [include_roiname]
        else:
            assert isinstance(include_roiname, list)
            include = include_roiname
        if exclude_roiname is None:
            exclude = []
        elif isinstance(exclude_roiname, str):
            exclude = [exclude_roiname]
        else:
            assert isinstance(exclude_roiname, list)
            exclude = exclude_roiname

        # Validate groups
        if group_name not in self._groups:
            raise ValueError(f"\nGroup '{group_name}' is not found")

        # Validate images
        if len(self._images) == 0:
            raise ValueError("No image added")

        # ROI type
        if as_mask:
            roi_type = "mask"
        else:
            roi_type = "sample"

        # default input from current working directory
        if len(search_directory) == 0:
            search_directory = (os.getcwd().replace("\\", "/") + "/").replace("//", "/")

        # Get image list
        # image list
        imgn_list0 = [imgt[2] for imgt in self._images if imgt[1] == group_name]
        imgn_list = []
        for imgn in imgn_list0:
            if imgn not in imgn_list:
                imgn_list.append(imgn)

        # Retrieve ROIs
        new_rois: list = []
        fail_list: list = []
        fail_err_list: list = []
        for imgn in imgn_list:
            # Construct ROI file names
            roi_pattern = imgn.rpartition(".")[0] + "*" + roi_filename_suffix

            # Search ROI file paths
            roi_pathsi = search_file(directory_path=search_directory, search_pattern=roi_pattern)

            # Parsing result
            # rois_from_file structure: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
            pars_result = self._roi_parser(roi_pathsi, group_name, imgn, roi_type)

            # Combine updated results
            new_rois = new_rois + pars_result[0]

            # Combine failed results
            fail_list = fail_list + pars_result[1]
            fail_err_list = fail_err_list + pars_result[2]

        # Filter parsed new ROI names
        # Includes
        if len(include) > 0:
            new_rois_filtered = []
            for roi_item in new_rois:
                whether_filtered = True
                for pat in include:
                    if pat in roi_item[3]:
                        whether_filtered = False
                if not whether_filtered:
                    new_rois_filtered.append(roi_item)
            new_rois = new_rois_filtered
        # Excludes
        if len(exclude) > 0:
            new_rois_filtered = []
            for roi_item in new_rois:
                whether_filtered = False
                for pat in exclude:
                    if pat in roi_item[3]:
                        whether_filtered = True
                if not whether_filtered:
                    new_rois_filtered.append(roi_item)
            new_rois = new_rois_filtered

        # Update self._rois_from_file and output reports
        self._roi_receiver(new_rois, fail_list, fail_err_list)

        # Update ROIs
        self._update_roi()
        # Update sample labels & targets
        self._update_sample_labels_targets()

    # Add ROI file paths to an image by full ROI file paths
    # Format of associated attribute:
    # self._groups: [0 group]
    # self._images: [0 id, 1 group, 2 image_name, 3 mask_of, 4 image_path]
    # self._rois_from_file: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
    # self._rois_from_coords: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    @simple_type_validator
    def add_rois_by_file(  # noqa: C901
        self,
        path: Union[str, list[str]],
        image_name: str,
        group: str,
        as_mask: bool = False,
        *,
        include_roiname: Union[str, list[str], None] = None,
        exclude_roiname: Union[str, list[str], None] = None,
    ) -> None:
        """
        Load ROIs from specified files, such as ENVI '.xml' ROI files or shapefiles ('.shp') from QGIS/ArcGIS, and add them to this SpecExp instance.

        Parameters
        ----------
        path : str or list of str
            Path or list of paths to ROI files to load.

        image_name : str
            Name of the belonging raster image for the ROI(s).
            If empty, ROIs are associated automatically with matching images.

        group : str
            Name of the experiment group to which the ROIs belong.
            Must be one of the groups in ``SpecExp.groups``.

        as_mask : bool, optional
            If True, added ROIs are labeled as masks.
            If False, added ROIs are labeled as sample ROIs.
            Default is False.

        include_roiname : str or list of str or None, optional
            Inclusion filter for ROI names:

            - If str, ROIs containing that substring are included.
            - If list of str, ROIs containing any of the substrings are included.
            - If None, all ROIs are included.

            Default is None.

        exclude_roiname : str or list of str or None, optional
            Exclusion filter for ROI names:

            - If str, ROIs containing that substring are excluded.
            - If list of str, ROIs containing any of the substrings are excluded.
            - If None, no ROIs are excluded.

            Default is None.

        Examples
        --------
        Prepare SpecExp instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name("image1.tif", "group1")

        Add a single ROI file::

            >>> exp.add_rois_by_file("/image1_roi.xml", group="group1")

        Exclude a single ROI name::

            >>> exp.add_rois_by_file("/image1_roi.xml", group="group1", exclude_roiname='name_to_exclude')

        Exclude multiple ROI names::

            >>> exp.add_rois_by_file("/image1_roi.xml", group="group1", exclude_roiname=['name_to_exclude1', 'name_to_exclude2'])

        Include ROIs by specific ROI name pattern::

            >>> exp.add_rois_by_file("/image1_roi.xml", group="group1", include_roiname='_sample')

        Add ROI as mask of the corresponding image::

            >>> exp.add_rois_by_file("/image1_roi_mask.xml", group="group1", as_mask=True)
        """  # noqa: E501

        group_name: str = group

        # Validate group name
        group_name = str(group_name)
        if group_name == "":
            raise ValueError("missing 1 required argument: 'group'")

        # Validate groups
        if group_name not in self._groups:
            raise ValueError(f"\nGroup '{group_name}' is not found")

        # Validate image_name
        image_name_list = [imgt[2] for imgt in self._images]
        if image_name == "" or image_name not in image_name_list:
            raise ValueError(f"raster image name '{image_name}' is not found")

        # ROI type
        if as_mask:
            roi_type = "mask"
        else:
            roi_type = "sample"

        # Initialize 'include' and 'exclude'
        if include_roiname is None:
            include = []
        elif isinstance(include_roiname, str):
            include = [include_roiname]
        else:
            assert isinstance(include_roiname, list)
            include = include_roiname
        if exclude_roiname is None:
            exclude = []
        elif isinstance(exclude_roiname, str):
            exclude = [exclude_roiname]
        else:
            assert isinstance(exclude_roiname, list)
            exclude = exclude_roiname

        # Add roi path list to image
        if isinstance(path, str):
            path_list = [path]
        else:
            path_list = path
        roi_pathsi = [os.path.abspath(os.path.normpath(fpath)).replace("\\", "/") for fpath in path_list]

        # Validate file structure of paths
        for fpath in roi_pathsi:
            roi_ext = fpath.split(".")[-1]
            if (roi_ext != "xml") & (roi_ext != "shp"):
                raise ValueError(f"ROI file format '{roi_ext}' is not supported")

        # Validate existence of given paths
        existing_paths = []
        non_existing_paths = []
        for roi_path in roi_pathsi:
            if os.path.exists(unc_path(roi_path)):
                existing_paths.append(roi_path)
            else:
                non_existing_paths.append(roi_path)
        if len(existing_paths) == 0:
            warn_msg = "\nAll ROI file paths in the given lists are invalid. No ROI file is added.\n"
            warnings.warn(warn_msg, UserWarning, stacklevel=2)
            return
        if len(non_existing_paths) > 0:
            warn_msg = f"\nFollowing ROI file paths is not valid:\n{non_existing_paths}\nThese paths are not added.\n"
            warnings.warn(warn_msg, UserWarning, stacklevel=2)
        roi_pathsi = existing_paths

        # Parsing result
        # rois_from_file structure: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
        pars_result = self._roi_parser(roi_pathsi, group_name, image_name, roi_type)

        # Combine updated results
        new_rois = pars_result[0]

        # Filter parsed new ROI names
        # Includes
        if len(include) > 0:
            new_rois_filtered = []
            for roi_item in new_rois:
                whether_filtered = True
                for pat in include:
                    if pat in roi_item[3]:
                        whether_filtered = False
                if not whether_filtered:
                    new_rois_filtered.append(roi_item)
            new_rois = new_rois_filtered
        # Excludes
        if len(exclude) > 0:
            new_rois_filtered = []
            for roi_item in new_rois:
                whether_filtered = False
                for pat in exclude:
                    if pat in roi_item[3]:
                        whether_filtered = True
                if not whether_filtered:
                    new_rois_filtered.append(roi_item)
            new_rois = new_rois_filtered

        # Combine failed results
        fail_list = pars_result[1]
        fail_err_list = pars_result[2]

        # Update self._rois_from_file and output reports
        self._roi_receiver(new_rois, fail_list, fail_err_list)

        # Update ROIs
        self._update_roi()
        # Update sample labels & targets
        self._update_sample_labels_targets()

    # Add a ROI to an image by ROI name and vertices of ROI polygons
    # Format of associated attribute:
    # self._groups: [0 group]
    # self._images: [0 id, 1 group, 2 image_name, 3 mask_of, 4 image_path]
    # self._rois_from_coords: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    @simple_type_validator
    def add_roi_by_coords(  # noqa: C901
        self,
        roi_name: str,
        coord_lists: list[list[tuple[Union[int, float], Union[int, float]]]],
        image_name: str,
        group: str,
        as_mask: bool = False,
        print_update: bool = True,
        *,
        _roi_id: Optional[str] = None,  # Specify ROI ID
    ) -> None:
        """
        Add a (multi-)polygon ROI to a raster image using vertex coordinates.
        The ROI can consist of multiple polygons.
        Each polygon is defined as a list of (x, y) vertex coordinate pairs and must be closed.

        Parameters
        ----------
        roi_name : str
            Name of the ROI to add.

        coord_lists : list of list of tuple of 2 (float or int)
            Coordinates of the ROI polygons.
            Structure::

                [
                    [ (x1, y1), (x2, y2), ..., (xn, yn), (x1, y1) ],  # Polygon 1
                    [ (x1, y1), (x2, y2), ..., (xm, ym), (x1, y1) ],  # Polygon 2
                    ...
                ]

            Each inner list represents a polygon (for multipart geometries), and each tuple is a vertex coordinate.

        image_name : str
            Name of the raster image to which the ROI belongs.

        group : str
            Name of the experiment group of the ROI. Must be one of the existing ``SpecExp.groups``.

        as_mask : bool, optional
            If True, the added ROI is labeled as a mask.
            If False, the added ROI is labeled as a sample.
            Default is False.

        print_update : bool, optional
            If True, prints the ROI update report. Set to False to suppress the update report.
            Defaults to True.

        Examples
        --------
        Prepare SpecExp instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name("image1.tif", "group1")

        Add ROI by vertex coordinates::

            >>> coordinates = [[(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]]
            >>> exp.add_roi_by_coords(
            ...     roi_name="test_roi",
            ...     group="group1",
            ...     image_name="image1.tif",
            ...     coord_lists=coordinates
            ... )
        """  # noqa: E501

        group_name: str = group

        # Validate groups
        if group_name not in self._groups:
            raise ValueError(f"\nGroup '{group_name}' is not found")

        # Validate image_name
        image_name_list = [imgt[2] for imgt in self._images]
        if image_name not in image_name_list:
            raise ValueError(f"raster image name '{image_name}' is not found")

        # Validate image existence in the specified group
        existed_images = [imgt for imgt in self.images if (imgt[2] == image_name and imgt[1] == group_name)]
        if len(existed_images) < 1:
            raise ValueError(f"No image with name '{image_name}' added in given group '{group_name}'.")
        if len(existed_images) > 1:
            raise ValueError(
                f"Duplicated image with name '{image_name}' found in given group '{group_name}': \
                             {existed_images}. Image name must be unique in a group."
            )

        # ROI type
        if as_mask:
            roi_type = "mask"
        else:
            roi_type = "sample"

        # Validate ROI coordinate integrity
        vertex_coordinate_pair_lists1 = []
        for coordlist in coord_lists:
            # Validate integrity of vertex coordinate list of every polygon
            if len(coordlist) < 4:
                raise ValueError("number of vertex coordinate pairs should be at least 4 for a polygon")
            # Convert coordinate numeric types to float
            coordlist1 = [(float(coordpair[0]), float(coordpair[1])) for coordpair in coordlist]
            # Force side vector integrity - last coordinates return to start to form a closed shape
            if coordlist1[-1] != coordlist1[0]:
                coordlist1.append(coordlist1[0])
            # Get coordinate list for update
            vertex_coordinate_pair_lists1.append(coordlist1)

        # Construct ROI item
        if _roi_id is None:
            roi_id = "Console-added-ROI_" + group_name + "_" + image_name.replace(".", "-") + "_" + roi_name
        else:
            roi_id = _roi_id

        # Update rois_from_coords
        new_roic_item = (
            roi_id,
            group_name,
            image_name,
            roi_name,
            roi_type,
            vertex_coordinate_pair_lists1,
        )
        existed_roi_ids = [roit[0] for roit in self._rois_from_coords]
        if roi_id not in existed_roi_ids:
            self._rois_from_coords.append(new_roic_item)
        else:
            for i in range(len(self._rois_from_coords)):
                if self._rois_from_coords[i][0] == roi_id:
                    self._rois_from_coords[i] = new_roic_item

        # Print report
        if print_update:
            print("\nFollowing ROI item added or updated:\n")
            self._df_roic([new_roic_item])

        # Update ROIs
        self._update_roi()
        # Update sample labels & targets
        self._update_sample_labels_targets()

    # self._rois updater
    # Format of associated attribute:
    # self._rois_from_file: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
    # self._rois_from_coords: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    # self._rois: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    # self._rois_sample: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    # self._rois_mask: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    # self._sample_labels: [0 fixed_label, 1 user-assigned label]
    def _update_roi(self) -> None:
        """
        Update self._rois, self._rois_sample and self._rois_mask based on self._rois_from_file and self._rois_from_coords
        """  # noqa: E501
        # Combine ROI from files and ROI from console, ignoring ROI file names and paths
        rois_f_sub = [roit[0:6] for roit in self._rois_from_file]
        self._rois = rois_f_sub + self._rois_from_coords

        # Divide into sample ROIs and mask ROIs, and update their attributes
        rois_sample = []
        rois_mask = []
        for roit in self._rois:
            if roit[4] == "sample":
                rois_sample.append(roit)
            elif roit[4] == "mask":
                rois_mask.append(roit)
            else:
                raise ValueError(
                    f"\nUnrecognized ROI type '{roit[4]}', ROI type must be 'sample' or 'mask'\nError item: {roit}"
                )
        self._rois_sample = rois_sample
        self._rois_mask = rois_mask

        # Add back standalone sample labels
        self._update_sample_sspecs()
        # Update sample labels & targets
        self._update_sample_labels_targets()

    # Get ROI items by [0 ROI name list, 1 group, 2 image_name, 3 names of source ROI file]
    # Format of associated attribute:
    # self._rois_from_file: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
    # self._rois_from_coords: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    # self._rois: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    @simple_type_validator
    def _get_rois(  # noqa: C901
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group_name: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        source_type: str = "",
    ) -> tuple[list, list]:
        """
        Retrieve ROI items based on roi_name, belonging group, image_name and ROI_file_name.
        If a parameter is not provided, it is ignored in retrieval.

        Returns matched and unmatched ROI items.
        """

        # Initialize name lists
        if roi_name_list is None:
            roi_name_list_init = []
        else:
            roi_name_list_init = roi_name_list
        if roi_file_name_list is None:
            roi_file_name_list_init = []
        else:
            roi_file_name_list_init = roi_file_name_list

        # Update ROIs
        if source_type == "":
            self._update_roi()

        # Source_type
        roit_list: Union[
            list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]],
            list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]], str, str]],
        ]
        if source_type == "":
            roit_list = self._rois
        elif source_type == "file":
            roit_list = self._rois_from_file
        elif source_type == "coords":
            roit_list = self._rois_from_coords

        # Validate ROIs
        if len(roit_list) == 0:
            print("No ROIs found!")
            return ([], [])

        # Retrieve removed and remained image items
        matched_items = []
        unmatched_items = []
        for roit in roit_list:
            # ROI name condition
            if len(roi_name_list_init) == 0:
                condrn = True
            else:
                condrn = roit[3] in roi_name_list_init
            # ROI type condition
            if roi_type == "":
                condt = True
            else:
                condt = roit[4] == roi_type
            # Group condition
            if group_name == "":
                condg = True
            else:
                condg = roit[1] == group_name
            # Image name condition
            if image_name == "":
                condi = True
            else:
                condi = roit[2] == image_name
            # ROI file name condition
            if len(roi_file_name_list_init) == 0:
                condrfn = True
            else:
                if len(roit) == 8:
                    condrfn = roit[6] in roi_file_name_list_init
                else:
                    warnings.warn(
                        "Only ROIs from file support retrieving by ROI file names, the filename criterion is ignored.",
                        UserWarning,
                        stacklevel=3,
                    )
                    condrfn = True
            # All conditions
            rcond = condrn & condt & condg & condi & condrfn
            # Retrieval
            if rcond:
                matched_items.append(roit)
            else:
                unmatched_items.append(roit)

        return matched_items, unmatched_items

    # List ROI items by [0 ROI name list, 1 group, 2 image_name, 3 names of source ROI file]
    @overload
    def ls_rois(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        source_type: str = "",
        image_name: str = "",
        group: str = "",
        *,
        print_result: bool = True,
        return_dataframe: Literal[False] = False,
    ) -> None: ...

    @overload
    def ls_rois(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        source_type: str = "",
        image_name: str = "",
        group: str = "",
        *,
        print_result: bool = True,
        return_dataframe: Literal[True] = True,
    ) -> Union[
        Annotated[
            pd.DataFrame,
            dataframe_validator(
                {
                    "ID": str,
                    "Group": str,
                    "Image": str,
                    "ROI_name": str,
                    "ROI_type": str,
                    "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                }
            ),
        ],
        Annotated[
            pd.DataFrame,
            dataframe_validator(
                {
                    "ID": str,
                    "Group": str,
                    "Image": str,
                    "ROI_name": str,
                    "ROI_type": str,
                    "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                    "ROI_source_file": str,
                    "ROI_file_path": str,
                }
            ),
        ],
    ]: ...

    # List ROI items by [0 ROI name list, 1 group, 2 image_name, 3 names of source ROI file]
    # Format of associated attribute:
    # self._rois_from_file: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
    # self._rois_from_coords: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    # self._rois: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    @simple_type_validator
    def ls_rois(  # noqa: C901
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        source_type: str = "",
        image_name: str = "",
        group: str = "",
        *,
        print_result: bool = True,
        return_dataframe: bool = False,
    ) -> Optional[
        Union[
            Annotated[
                pd.DataFrame,
                dataframe_validator(
                    {
                        "ID": str,
                        "Group": str,
                        "Image": str,
                        "ROI_name": str,
                        "ROI_type": str,
                        "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                    }
                ),
            ],
            Annotated[
                pd.DataFrame,
                dataframe_validator(
                    {
                        "ID": str,
                        "Group": str,
                        "Image": str,
                        "ROI_name": str,
                        "ROI_type": str,
                        "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                        "ROI_source_file": str,
                        "ROI_file_path": str,
                    }
                ),
            ],
        ]
    ]:
        """
        List added ROI items based on filtering conditions.
        If a filter criterion is not provided, the corresponding filter is not applied.

        Parameters
        ----------
        roi_name_list : list of str or None, optional
            List of ROI names to filter by.
            If None, no filtering by ROI name is applied.
            Default is None.

        roi_type : str, optional
            Use type of ROI.
            Accepted values are:

            - ``"sample"``
            - ``"mask"``

            If empty, no filtering by ROI type is applied.
            Default is "".

        roi_file_name_list : list of str or None, optional
            List of ROI source file names to filter by.
            If None, no filtering by ROI file name is applied.
            Default is None.

        source_type : str, optional
            Source type of ROI addition.
            Accepted values are:

            - ``"file"`` : ROIs loaded from files
            - ``"coords"`` : ROIs added from coordinate lists

            If empty, no filtering by source type is applied.
            Default is "".

        image_name : str, optional
            Name of the associated raster image.
            If empty, no filtering by image name is applied.
            Default is "".

        group : str, optional
            Name of the experiment group.
            If empty, no filtering by group is applied.
            Default is "".

        print_result : bool, optional
            Whether to print the matched ROI items to the console.
            Default is True.

        return_dataframe : bool, optional
            Whether to return the matched ROI items as a pandas DataFrame.
            Default is False.

        Returns
        -------
        pandas.DataFrame or None
            If ``return_dataframe=True``, returns a pandas DataFrame containing matched ROI items.

            The DataFrame contains the following columns:

                ID : str
                    unique identifier of the ROI.
                Group : str
                    group to which the ROI belongs.
                Image : str
                    name of the associated image.
                ROI_name : str
                    name of the ROI.
                ROI_type : str
                    use type of the ROI ("sample" or "mask").
                Coordinates : list of list of tuple of 2 (float or int)
                    vertex coordinates of the ROI polygons.

            If ROIs were loaded from files, the following columns are additionally included:

                ROI_source_file : str
                    name of the ROI source file.
                ROI_file_path : str
                    full path to the ROI source file.

            If ``return_dataframe=False``, returns None.

        Examples
        --------
        Prepare SpecExp instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name("image1.tif", "group1")
            >>> exp.add_rois_by_suffix("_roi.xml", group="group1")

        List all ROIs::

            >>> exp.ls_rois()

        List ROIs with filtering conditions::

            >>> exp.ls_rois(roi_type="sample")
            >>> exp.ls_rois(roi_name_list=["ROI1", "ROI2"])
            >>> exp.ls_rois(source_type="coords")
            >>> exp.ls_rois(group="group1")
            >>> exp.ls_rois(group="group1", roi_type="sample")

        Return results as a DataFrame::

            >>> df = exp.ls_rois(return_dataframe=True)
        """  # noqa: E501

        group_name: str = group

        # Update ROIs
        if source_type == "":
            self._update_roi()

        # ROIs to search according to source_type
        roit_list: Union[
            list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]]]],
            list[tuple[str, str, str, str, str, list[list[tuple[Union[int, float], Union[int, float]]]], str, str]],
        ]
        if source_type == "":
            roit_list = self._rois
        elif source_type == "file":
            roit_list = self._rois_from_file
        elif source_type == "coords":
            roit_list = self._rois_from_coords
        else:
            raise ValueError("Given source_type must be 'file' or 'coords'")

        # Validate ROI type
        if (roi_type != "") & (roi_type != "sample") & (roi_type != "mask"):
            raise ValueError("Given roi_type must be 'sample' or 'mask'")

        # Validate existence of ROIs - No ROIs found
        if len(roit_list) == 0:
            if print_result:
                print("No ROIs found!")
            if return_dataframe:
                if source_type == "file":
                    return pd.DataFrame(
                        columns=[
                            "ID",
                            "Group",
                            "Image",
                            "ROI_name",
                            "ROI_type",
                            "Coordinates",
                            "ROI_source_file",
                            "ROI_file_path",
                        ]
                    )
                else:
                    return pd.DataFrame(columns=["ID", "Group", "Image", "ROI_name", "ROI_type", "Coordinates"])

        # Search ROIs
        matched_items = self._get_rois(
            roi_name_list, roi_type, group_name, image_name, roi_file_name_list, source_type
        )[0]
        if source_type == "file":
            df_matched = self._rois_from_file_to_df(matched_items, print_result, True)
        else:
            df_matched = self._df_roic(matched_items, print_result, True)

        # Return result
        if return_dataframe:
            return df_matched
        else:
            return None

    # List file-added rois
    @overload
    def ls_rois_from_file(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        *,
        print_result: bool = True,
        return_dataframe: Literal[False] = False,
    ) -> None: ...

    @overload
    def ls_rois_from_file(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        *,
        print_result: bool = True,
        return_dataframe: Literal[True] = True,
    ) -> Annotated[
        pd.DataFrame,
        dataframe_validator(
            {
                "ID": str,
                "Group": str,
                "Image": str,
                "ROI_name": str,
                "ROI_type": str,
                "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                "ROI_source_file": str,
                "ROI_file_path": str,
            }
        ),
    ]: ...

    # List file-added rois
    @simple_type_validator
    def ls_rois_from_file(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        *,
        print_result: bool = True,
        return_dataframe: bool = False,
    ) -> Optional[
        Annotated[
            pd.DataFrame,
            dataframe_validator(
                {
                    "ID": str,
                    "Group": str,
                    "Image": str,
                    "ROI_name": str,
                    "ROI_type": str,
                    "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                    "ROI_source_file": str,
                    "ROI_file_path": str,
                }
            ),
        ]
    ]:
        """
        List file-added Regions of Interest. For detailed use, see 'ls_rois'.

        See Also
        --------
        ls_rois
        """

        if return_dataframe:
            return self.ls_rois(
                roi_name_list=roi_name_list,
                roi_type=roi_type,
                group=group,
                image_name=image_name,
                roi_file_name_list=roi_file_name_list,
                print_result=print_result,
                source_type="file",
                return_dataframe=True,
            )
        else:
            self.ls_rois(
                roi_name_list=roi_name_list,
                roi_type=roi_type,
                group=group,
                image_name=image_name,
                roi_file_name_list=roi_file_name_list,
                print_result=print_result,
                source_type="file",
                return_dataframe=False,
            )
            return None

    # List console-added rois
    @overload
    def ls_rois_from_coords(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        *,
        print_result: bool = True,
        return_dataframe: Literal[False] = False,
    ) -> None: ...

    # List console-added rois
    @overload
    def ls_rois_from_coords(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        *,
        print_result: bool = True,
        return_dataframe: Literal[True] = True,
    ) -> Annotated[
        pd.DataFrame,
        dataframe_validator(
            {
                "ID": str,
                "Group": str,
                "Image": str,
                "ROI_name": str,
                "ROI_type": str,
                "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
            }
        ),
    ]: ...

    # List console-added rois
    @simple_type_validator
    def ls_rois_from_coords(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        *,
        print_result: bool = True,
        return_dataframe: bool = False,
    ) -> Optional[
        Annotated[
            pd.DataFrame,
            dataframe_validator(
                {
                    "ID": str,
                    "Group": str,
                    "Image": str,
                    "ROI_name": str,
                    "ROI_type": str,
                    "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                }
            ),
        ]
    ]:
        """
        List Regions of Interest added using vertex coordinates. For detailed use, see ``ls_rois``.

        See Also
        --------
        ls_rois
        """
        if return_dataframe:
            return self.ls_rois(
                roi_name_list=roi_name_list,
                roi_type=roi_type,
                group=group,
                image_name=image_name,
                roi_file_name_list=roi_file_name_list,
                print_result=print_result,
                source_type="coords",
                return_dataframe=True,
            )
        else:
            self.ls_rois(
                roi_name_list=roi_name_list,
                roi_type=roi_type,
                group=group,
                image_name=image_name,
                roi_file_name_list=roi_file_name_list,
                print_result=print_result,
                source_type="coords",
                return_dataframe=False,
            )
            return None

    # List sample rois
    @overload
    def ls_rois_sample(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        source_type: str = "",
        *,
        print_result: bool = True,
        return_dataframe: Literal[False] = False,
    ) -> None: ...

    # List sample rois
    @overload
    def ls_rois_sample(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        source_type: str = "",
        *,
        print_result: bool = True,
        return_dataframe: Literal[True] = True,
    ) -> Union[
        Annotated[
            pd.DataFrame,
            dataframe_validator(
                {
                    "ID": str,
                    "Group": str,
                    "Image": str,
                    "ROI_name": str,
                    "ROI_type": str,
                    "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                }
            ),
        ],
        Annotated[
            pd.DataFrame,
            dataframe_validator(
                {
                    "ID": str,
                    "Group": str,
                    "Image": str,
                    "ROI_name": str,
                    "ROI_type": str,
                    "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                    "ROI_source_file": str,
                    "ROI_file_path": str,
                }
            ),
        ],
    ]: ...

    # List sample rois
    @simple_type_validator
    def ls_rois_sample(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        source_type: str = "",
        *,
        print_result: bool = True,
        return_dataframe: bool = False,
    ) -> Optional[
        Union[
            Annotated[
                pd.DataFrame,
                dataframe_validator(
                    {
                        "ID": str,
                        "Group": str,
                        "Image": str,
                        "ROI_name": str,
                        "ROI_type": str,
                        "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                    }
                ),
            ],
            Annotated[
                pd.DataFrame,
                dataframe_validator(
                    {
                        "ID": str,
                        "Group": str,
                        "Image": str,
                        "ROI_name": str,
                        "ROI_type": str,
                        "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                        "ROI_source_file": str,
                        "ROI_file_path": str,
                    }
                ),
            ],
        ]
    ]:
        """
        List Regions of Interest as samples. For detailed use, see ``ls_rois``.

        See Also
        --------
        ls_rois
        """
        if return_dataframe:
            return self.ls_rois(
                roi_name_list=roi_name_list,
                group=group,
                image_name=image_name,
                roi_file_name_list=roi_file_name_list,
                source_type=source_type,
                print_result=print_result,
                roi_type="sample",
                return_dataframe=True,
            )
        else:
            self.ls_rois(
                roi_name_list=roi_name_list,
                group=group,
                image_name=image_name,
                roi_file_name_list=roi_file_name_list,
                source_type=source_type,
                print_result=print_result,
                roi_type="sample",
                return_dataframe=False,
            )
            return None

    # List mask rois
    @overload
    def ls_rois_mask(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        source_type: str = "",
        *,
        print_result: bool = True,
        return_dataframe: Literal[False] = False,
    ) -> None: ...

    @overload
    def ls_rois_mask(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        source_type: str = "",
        *,
        print_result: bool = True,
        return_dataframe: Literal[True] = True,
    ) -> Union[
        Annotated[
            pd.DataFrame,
            dataframe_validator(
                {
                    "ID": str,
                    "Group": str,
                    "Image": str,
                    "ROI_name": str,
                    "ROI_type": str,
                    "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                }
            ),
        ],
        Annotated[
            pd.DataFrame,
            dataframe_validator(
                {
                    "ID": str,
                    "Group": str,
                    "Image": str,
                    "ROI_name": str,
                    "ROI_type": str,
                    "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                    "ROI_source_file": str,
                    "ROI_file_path": str,
                }
            ),
        ],
    ]: ...

    # List mask rois
    @simple_type_validator
    def ls_rois_mask(
        self,
        roi_name_list: Optional[list[str]] = None,
        roi_type: str = "",
        group: str = "",
        image_name: str = "",
        roi_file_name_list: Optional[list[str]] = None,
        source_type: str = "",
        *,
        print_result: bool = True,
        return_dataframe: bool = False,
    ) -> Optional[
        Union[
            Annotated[
                pd.DataFrame,
                dataframe_validator(
                    {
                        "ID": str,
                        "Group": str,
                        "Image": str,
                        "ROI_name": str,
                        "ROI_type": str,
                        "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                    }
                ),
            ],
            Annotated[
                pd.DataFrame,
                dataframe_validator(
                    {
                        "ID": str,
                        "Group": str,
                        "Image": str,
                        "ROI_name": str,
                        "ROI_type": str,
                        "Coordinates": list[list[tuple[Union[int, float], Union[int, float]]]],
                        "ROI_source_file": str,
                        "ROI_file_path": str,
                    }
                ),
            ],
        ]
    ]:
        """
        List Regions of Interest as masks. For detailed use, see ``ls_rois``.

        See Also
        --------
        ls_rois
        """
        if return_dataframe:
            return self.ls_rois(
                roi_name_list=roi_name_list,
                group=group,
                image_name=image_name,
                roi_file_name_list=roi_file_name_list,
                source_type=source_type,
                print_result=print_result,
                roi_type="mask",
                return_dataframe=True,
            )
        else:
            self.ls_rois(
                roi_name_list=roi_name_list,
                group=group,
                image_name=image_name,
                roi_file_name_list=roi_file_name_list,
                source_type=source_type,
                print_result=print_result,
                roi_type="mask",
                return_dataframe=False,
            )
            return None

    # Remove ROIs from self._rois_from_file or self._rois_from_coords
    # Format of associated attribute:
    # self._rois: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    # self._rois_from_file: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
    @simple_type_validator
    def _rm_rois_x(  # noqa: C901
        self,
        rm_target: str = "file",
        group_name: str = "",
        image_name: str = "",
        roi_name: str = "",
        roi_type: str = "",
        roi_file_name: str = "",
        roi_file_path: str = "",
    ) -> None:
        """
        Remove loaded ROIs by given parameters, Unix-like patterns is supported (using fnmatch).

        A parameter will be ignored if not provided.
        """
        # Validate ROI type
        if (roi_type.lower() != "mask") & (roi_type.lower() != "sample") & (roi_type.lower() != ""):
            raise ValueError("ROI_type must be 'sample' or 'mask'")

        # ID indices
        item_pattern = {
            "group": group_name,
            "image_name": image_name,
            "ROI_name": roi_name,
            "ROI_type": roi_type,
            "ROI_file_name": roi_file_name,
            "ROI_file_path": roi_file_path,
        }
        item_id = {
            "group": 1,
            "image_name": 2,
            "ROI_name": 3,
            "ROI_type": 4,
            "ROI_file_name": 6,
            "ROI_file_path": 7,
        }

        # Mode
        if rm_target == "file":
            item_list_original = self._rois_from_file[:]
        elif rm_target == "coords":
            item_list_original = self._rois_from_coords[:]  # type: ignore[assignment]
            # mypy type error ignored because of condition-dependent typing ('file' / 'coords')
            del item_pattern["ROI_file_name"], item_pattern["ROI_file_path"]
            del item_id["ROI_file_name"], item_id["ROI_file_path"]
        else:
            raise ValueError("rm_target must be 'file' or 'coords'.")

        # Validate ROI data
        if len(item_list_original) == 0:
            print(f"\nNo ROIs_from_{rm_target} added.")
            return

        # Validate pattern (raise error if no criterion is given)
        # val_value = 0
        # for item_pat in item_pattern.values():
        #     val_value = val_value + len(item_pat)
        # if val_value == 0:
        #     raise ValueError("\nNo criterion is provided for removal")

        removed_rois = item_list_original
        remained_rois = []
        value_list: list[str]
        for iid, ipat in zip(item_id.values(), item_pattern.values()):
            # Value list for item i
            if len(ipat) > 0:
                value_list = []
                for roii in item_list_original:
                    if roii[iid] not in value_list:
                        roii_value_iid = roii[iid]
                        assert type(roii_value_iid) is str
                        value_list.append(roii_value_iid)

                # Get matched and unmatched exact ROI file names
                matched_list, unmatched_list = names_filter(value_list, ipat)

                # Filter
                if len(matched_list) > 0:
                    new_roif = []
                    for roii in removed_rois:
                        if roii[iid] in matched_list:
                            new_roif.append(roii)
                        else:
                            remained_rois.append(roii)
                    removed_rois = new_roif

        # Update ROIs and print report
        if (rm_target == "file") & (len(removed_rois) > 0):
            # Update
            self._rois_from_file = remained_rois
            self._update_roi()
            # Report
            print("\nFollowing file-loaded ROI items are removed:\n")
            self._rois_from_file_to_df(removed_rois)

        elif (rm_target == "coords") & (len(removed_rois) > 0):
            # Update
            self._rois_from_coords = remained_rois  # type: ignore[assignment]
            # mypy type error ignored because of condition-dependent typing ('file' / 'coords')
            self._update_roi()
            # Report
            print("\nFollowing console-added ROI items are removed:\n")
            self._df_roic(removed_rois)  # type: ignore[arg-type]
            # mypy type error ignored because of condition-dependent typing ('file' / 'coords')

        else:
            print(f"\nNo ROI_{rm_target} removed with given filtering parameters")

    # Remove ROIs - rois_from_file and rois_from_coords based on group, img_name, roi_name and roi_type
    # Format of associated attribute:
    # self._rois_from_file: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs, 6 ROI file name, 7 ROI file path]  # noqa: E501
    # self._rois_from_coords: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    # self._rois: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    @simple_type_validator
    def rm_rois(
        self,
        roi_name: str = "",
        roi_type: str = "",
        roi_source_file_name: str = "",
        roi_source_file_path: str = "",
        image_name: str = "",
        group: str = "",
    ) -> None:
        """
        Remove Regions of Interest (ROIs) from this SpecExp instance based on provided filters.
        All parameters act as filters. If a parameter is not specified or left empty, the corresponding filter criterion is not applied.
        Unix-like wildcard patterns like ``*`` and ``?`` are supported for string parameters.

        Parameters
        ----------
        roi_name : str, optional
            Name or name pattern of the ROI(s) to remove.

        roi_type : str, optional
            Type of ROI to remove.
            Allowed values are ``"sample"`` and ``"mask"``.

        roi_source_file_name : str, optional
            Name or name pattern of the source ROI file(s).

        roi_source_file_path : str, optional
            Path or path pattern of the source ROI file(s).

        image_name : str, optional
            Name or name pattern of the associated image(s).

        group : str, optional
            Name or name pattern of the belonging experiment group(s).

        Examples
        --------
        Prepare a ``SpecExp`` instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name("image1.tif", "group1")
            >>> exp.add_rois_by_suffix("_roi.xml", group="group1")

        Remove all ROIs::

            >>> exp.rm_rois()

        Remove ROIs by condition::

            >>> exp.rm_rois(roi_type="mask")
            >>> exp.rm_rois(group="group1")
            >>> exp.rm_rois(image_name="image1.tif")
            >>> exp.rm_rois(roi_source_file_name="image1_roi.xml")
        """  # noqa: E501

        group_name: str = group

        self._rm_rois_x(
            rm_target="file",
            group_name=group_name,
            image_name=image_name,
            roi_name=roi_name,
            roi_type=roi_type,
            roi_file_name=roi_source_file_name,
            roi_file_path=roi_source_file_path,
        )

        self._rm_rois_x(
            rm_target="coords",
            group_name=group_name,
            image_name=image_name,
            roi_name=roi_name,
            roi_type=roi_type,
            roi_file_name=roi_source_file_name,
            roi_file_path=roi_source_file_path,
        )

        # Update sample labels & targets
        self._update_sample_labels_targets()

    # TODO: ROI subset augmentation
    # rois_sample: [0 id, 1 group, 2 image_name, 3 ROI_name, 4 ROI_type, 5 list of lists of coordinate pairs]
    # sample_targets: [0 fixed sample id, 1 user assinged labels, 2 target values, 3 sample belonging group, 4 validation group, 5 test mask, 6 train mask]  # noqa: E501
    @simple_type_validator
    def roi_subset_augmentation(
        self,
        n_sub: int,
        resolution: Union[int, float],
        coverage_ratio: float,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Perform spatial subset augmentation on the ROI data.

        This method can only be applied on completely configured SpecExp instance.

        The method generates multiple synthetic spatial subsets from the original ROIs and automatically updates the related data.
        The generated sub-ROIs shares the same target value as the original sample ROI.

        Parameters
        ----------
        n_sub : int
            The number of augmented synthetic data points (subsets) to generate for each sample ROI.
        resolution : int or float
            The side length of the square grid cells used for sampling.
            This effectively defines the spatial grain of the augmentation.
        coverage_ratio : float
            The target fraction of the total original ROI area to retain in each augmented sample.
            Must be a value between 0.0 and 1.0.
        random_state : int, optional
            Seed for the internal NumPy random number generator to ensure reproducibility.
            Default is None.

        Examples
        --------
        >>> exp.roi_subset_augmentation(
        ...     n_sub=5,
        ...     resolution=2,
        ...     coverage_ratio=0.3,
        ...     random_state=42
        ... )
        """  # noqa: E501

        # Validate random state
        if random_state is None:
            random_state = np.random.randint(0, np.iinfo(np.int32).max)

        # Validate n_sub
        if n_sub < 1:
            raise ValueError(f"n_sub must be at least 1, got: {n_sub}")

        # Validate self configured
        try:
            _spec_exp_validator(self)
        except Exception as e:
            raise ValueError(
                "SpecExp configuration is incomplete. "
                + "Complete the configuration before performing ROI subset augmentation."
            ) from e

        # Loop existed ROIs
        print("\n\nApply ROI stohastic spatial subset augmentation...")
        for roit in tqdm(self.rois_sample, total=len(self.rois_sample)):
            # Look for target item
            target_it = [tit for tit in self.sample_targets if tit[0] == roit[0]][0]
            # Copy old targets
            targets_old = deepcopy(self.sample_targets)
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
            for k, subcoords in enumerate(subcoords_list):
                self.add_roi_by_coords(
                    roi_name=f"{roit[3]}_&#aug{k}",
                    coord_lists=subcoords,
                    image_name=roit[2],
                    group=roit[1],
                    as_mask=False,
                    print_update=False,
                    _roi_id=f"{roit[0]}_&#aug{k}",
                )
            # Generate additional sample targets
            target_sub_list = [
                (
                    f"{roit[0]}_&#aug{k}",
                    f"{target_it[1]}_&#aug{k}",
                    target_it[2],
                    target_it[3],
                    target_it[4],
                    np.int8(0),
                    np.int8(1),
                )
                for k in range(len(subcoords_list))
            ]
            # Validate IDs
            existed_ids = [tit[0] for tit in self.sample_targets]
            for tsub in target_sub_list:
                if tsub[0] not in existed_ids:
                    raise ValueError(
                        f"Sample ID mismatch, expected ID from target: '{tsub[0]}'\n\navailable IDs: {existed_ids}"
                    )
            # Update sample targets
            self.sample_targets = targets_old + target_sub_list

    ## Standalone 1D spectrum samples
    @overload
    def add_standalone_specs(
        self,
        spec_data: Union[
            list[list[Union[int, float]]],
            Annotated[Any, arraylike_validator(ndim=2, as_type="float")],
        ],
        group: str = "",
        use_type: str = "sample",
        sample_name_list: Optional[list[str]] = None,
        *,
        silent_run: bool = False,
        save_backup: bool = True,
        return_updates: Literal[False] = False,
    ) -> None: ...

    @overload
    def add_standalone_specs(
        self,
        spec_data: Union[
            list[list[Union[int, float]]],
            Annotated[Any, arraylike_validator(ndim=2, as_type="float")],
        ],
        group: str = "",
        use_type: str = "sample",
        sample_name_list: Optional[list[str]] = None,
        *,
        silent_run: bool = False,
        save_backup: bool = True,
        return_updates: Literal[True] = True,
    ) -> dict[str, pd.DataFrame]: ...

    # Add standalone spectrum samples
    # Format of associated attribute:
    # [0 id, 1 group, 2 use_type, 3 sample_name, 4 spectral_data_list]
    @simple_type_validator
    def add_standalone_specs(  # noqa: C901
        self,
        spec_data: Union[
            list[list[Union[int, float]]],
            Annotated[Any, arraylike_validator(ndim=2, as_type="float")],
        ],
        group: str,
        use_type: str = "sample",
        sample_name_list: Optional[list[str]] = None,
        *,
        silent_run: bool = False,
        save_backup: bool = True,
        return_updates: bool = False,
    ) -> Optional[dict[str, pd.DataFrame]]:
        """
        Add one-dimensional standalone spectra to a group.

        Parameters
        ----------
        spec_data : array-like (n_samples, n_bands)
            Spectral values of one-dimensional spectra.

            Can be provided as a list of lists, where each inner list contains the spectral values of one spectrum.

        group : str
            Name of the belonging group.

        use_type : str, optional
            Use-type label of the spectra.
            Set to ``"sample"`` to treat the spectra as samples.
            The default is ``"sample"``.

        sample_name_list : list of str, optional
            Names of the spectra.
            If not provided, spectra are named automatically.

        silent_run : bool, optional
            Whether to suppress console output.
            The default is False.

        save_backup : bool, optional
            Whether to save backup files.
            If False, data are saved without creating a backup.

        return_updates : bool, optional
            Whether to return an update report.
            If True, a summary of the updates is returned.

        Returns
        -------
        dict of pandas.DataFrame or None
            If ``return_updates=True``, returns a dictionary containing update reports:

            updated_items : pandas.DataFrame
                DataFrame containing the updated spectral items.
            added_items : pandas.DataFrame
                DataFrame containing the newly added spectral items.

            If ``return_updates=False``, returns None.

        Notes
        -----
        This method is also available as ``add_specs``.

        Examples
        --------
        >>> exp = SpecExp("/experiment")
        >>> exp.add_groups("group1")

        >>> exp.add_standalone_specs(
        ...     [[1, 2, 3, 4], [1, 3, 3, 3]],
        ...     group="group1",
        ...     sample_name_list=["sample1", "sample2"]
        ... )
        """  # noqa: E501

        group_name: str = group

        # Initialize sample_names
        if sample_name_list is None:
            sample_names = []
        else:
            sample_names = sample_name_list

        # Validate groups
        if group_name not in self._groups:
            raise ValueError(f"\nGroup '{group_name}' is not found")

        # Convert values in array-like to list[list]
        if type(spec_data) is not list:
            spec_data = np.array(spec_data)
            spec_data_new = []
            for drow in spec_data:
                spec_data_new.append(list(drow))
            spec_data = spec_data_new

        # Validate sample names
        if len(sample_names) > 0:
            # Validate length
            if len(sample_names) != len(spec_data):
                raise ValueError(
                    f"Inconsistent sample numbers in spec_data and sample_names,\
                        \nsample number in spec_data: {len(spec_data)},\
                        \nsample number in sample_names: {len(spec_data)}"
                )
            else:
                # Validate uniqueness
                if len(set(sample_names)) < len(sample_names):
                    raise ValueError("Duplicate sample names are not allowed, please provide unique names")
                # Validate naming format
                for sn in sample_names:
                    if "dalone_spec_sample_%#" in sn:
                        if len(sn.split("dalone_spec_sample_%#")) > 1:
                            if not sn.split("dalone_spec_sample_%#")[1].isdigit():
                                raise ValueError(f"Given sample conflicts with default name format: '{sn}'")
                sample_names1 = sample_names
        else:
            # Generate default sample names
            # Prevent duplication with existed
            existed_name_numbers = []
            if len(self._standalone_specs) > 0:
                for stspect in self._standalone_specs:
                    sn = stspect[3]
                    if "Standalone_spec_sample_%#" in sn:
                        if len(sn.split("dalone_spec_sample_%#")) > 1:
                            snnum = sn.split("dalone_spec_sample_%#")[1]
                            if snnum.isdigit():
                                snnum_int = int(snnum)
                                existed_name_numbers.append(snnum_int)
            # Continue labling
            maxsnn: int
            if len(existed_name_numbers) > 0:
                maxsnn = max(existed_name_numbers)
            else:
                maxsnn = 0
            # Default sample names
            sample_names1 = [("Standalone_spec_sample_%#" + str(i + 1 + maxsnn)) for i in range(len(spec_data))]

        # Validate input data
        splen = len(spec_data[0])
        for sp in spec_data:
            if len(sp) != splen:
                raise ValueError(
                    f"Inconsistent number of bands in given spectral data, \ngot: {len(sp)}, \nexpected: {splen}"
                )
        if len(self._standalone_specs) > 0:
            if splen != len(self._standalone_specs[0][4]):
                raise ValueError(
                    f"Given spectral data has inconsistent number of bands with existed standalone spectra,\
                        \ngot: {splen}, \nexisted: {len(self._standalone_specs[0][4])}"
                )

        # Added spec samples
        added_sspecs = []
        added_sspec_names = []
        for sname, sdata in zip(sample_names1, spec_data):
            sid = group_name + "_" + use_type + "_" + str(sname)  # ID
            added_sspec_names.append(sname)
            added_sspecs.append((sid, group_name, use_type, sname, sdata))

        # Remove existed items for update
        if len(self._standalone_specs) > 0:
            not_updated_sspecs = [spect for spect in self._standalone_specs if spect[3] not in added_sspec_names]
            updated_sspecs = [spect for spect in self._standalone_specs if spect[3] in added_sspec_names]
        else:
            updated_sspecs = []
            not_updated_sspecs = []

        # New items
        if len(updated_sspecs) > 0:
            updated_sspec_names = [spect[3] for spect in updated_sspecs]
        else:
            updated_sspec_names = []
        new_sspecs = [spect for spect in added_sspecs if spect[3] not in updated_sspec_names]

        # Update standalone spectrum items
        self._standalone_specs = not_updated_sspecs + added_sspecs
        self._update_sample_sspecs()
        # Update sample labels & targets
        self._update_sample_labels_targets()

        # Print report
        df_updated = pd.DataFrame(updated_sspecs, columns=["ID", "Group", "Use_type", "Sample_name", "Spectral_data"])
        df_added = pd.DataFrame(new_sspecs, columns=["ID", "Group", "Use_type", "Sample_name", "Spectral_data"])
        if not silent_run:
            if len(updated_sspecs) > 0:
                print("\nFollowing standalone spectra are updated:\n")
                with pd.option_context("display.max_rows", None, "display.max_columns", None):
                    print(df_updated.iloc[:, 1:-1])
                print("\n\n")
            if len(new_sspecs) > 0:
                print("\nFollowing standalone spectra are added:\n")
                with pd.option_context("display.max_rows", None, "display.max_columns", None):
                    print(df_added.iloc[:, 1:-1])

        if return_updates:
            return {
                "Updated_items": df_updated.iloc[:, 1:-1],
                "Added_items": df_added.iloc[:, 1:-1],
            }
        else:
            return None

        # Update sample labels & targets
        self._update_sample_labels_targets()

        # Updated saved file
        self._update_sspecs_file(silent_run, save_backup)

    # Alias
    add_specs = add_standalone_specs

    # Update sample standalone spectra and sample labels
    # Format of associated attribute:
    # [0 id, 1 group, 2 use_type, 3 sample_name, 4 spectral_data_list]
    # sample_labels: [0 fixed sample id, 1 user assinged labels, 2 sample belonging group]
    def _update_sample_sspecs(self) -> None:
        self._standalone_specs_sample = self._get_sspecs(use_type="sample")[0]

    # Update sspecs file in report dir
    @simple_type_validator
    def _update_sspecs_file(self, silent_run: bool = False, save_backup: bool = True) -> None:
        # Validate output directory
        wpath = self._report_directory + "Standalone_spectral_data/"
        os.makedirs(unc_path(wpath), exist_ok=True)

        # Result to dataframes
        df_sspecs = pd.DataFrame(
            self._standalone_specs,
            columns=["ID", "Group", "Use_type", "Sample_name", "Spectral_data"],
        )
        df_sspecs_meta = df_sspecs.iloc[:, :-1]
        df_sspecs_data = pd.DataFrame([spect[4] for spect in self._standalone_specs])
        df_sspecs_data.columns = [("Band_" + str(i + 1)) for i in range(len(df_sspecs_data.columns))]
        df_sspecs_out = pd.concat([df_sspecs_meta, df_sspecs_data], axis=1)

        # Write to csv
        # Write current
        df_sspecs_out.to_csv(unc_path(wpath + f"Standalone_spectra_{self._create_time}.csv"), index=False)

        # Write backup
        # Current time for saving backups
        cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if save_backup:
            df_sspecs_out.to_csv(
                unc_path(wpath + f"Standalone_spectra_{self._create_time}_backup_{cts}.csv"), index=False
            )

        # Print update reports
        if not silent_run:
            print(f"\nStandalone spectra are saved in: \n\n{wpath}Standalone_spectra_{self._create_time}.csv\n")
            if save_backup:
                print(
                    f"\nStandalone spectra backup are saved as:\
                        \n\n{wpath}Standalone_spectra_{self._create_time}_backup_{cts}.csv\n"
                )

    # Add standalone spectrum samples
    # Format of associated attribute:
    # [0 id, 1 group, 2 use_type, 3 sample_name, 4 spectral_data_list]
    @simple_type_validator
    def load_standalone_specs(self, csv_file_path: str) -> None:  # noqa: C901
        """
        Load or reload standalone spectra from a CSV file.

        The CSV file should contain a table of spectral data with metadata.
        Expected table columns:

        ['ID', 'Group', 'Use_type', 'Sample_name', 'Band_1', 'Band_2', ...].

        Both 'ID' and 'Sample_name' values must be unique.

        Parameters
        ----------
        csv_file_path : str
            Path to the CSV file containing standalone spectra.

        Notes
        -----
        This method is also available as ``load_specs``.

        Examples
        --------
        Prepare SpecExp instance and load spectra::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.load_standalone_specs("/specs.csv")
        """

        # Validate path
        dpath = os.path.abspath(os.path.normpath(csv_file_path)).replace("\\", "/")

        # Validate extension
        ext = dpath.split(".")[-1]
        if ext.lower() != "csv":
            raise ValueError(f"\nThe file path must contain .csv extension: \n{dpath}")
        # Validate file existence
        if not os.path.exists(unc_path(dpath)):
            raise ValueError(f"\nInvalid file path: \n{dpath}")

        # Validate structure
        try:
            df_sspecs = pd.read_csv(unc_path(dpath))
        except Exception as e:
            raise ValueError("\nUnable to read the provided file: \ndpath\n\n", e) from e
        coln_d = [("Band_" + str(i + 1)) for i in range(len(df_sspecs.columns) - 4)]
        coln_n = ["ID", "Group", "Use_type", "Sample_name"]
        coln = coln_n + coln_d
        if list(df_sspecs.columns) != coln:
            raise ValueError(
                f"Column names must be ['ID', 'Group', 'Use_type', 'Sample_name', 'Band_1', 'Band_2',...].\
                    \n\nExpected: {coln} \n\nGot: \n{list(df_sspecs.columns)}"
            )

        # Validate datatype
        df_sspecs[df_sspecs.columns[:4]] = df_sspecs.iloc[:, :4].astype(str)
        try:
            df_sspecs[df_sspecs.columns[4:]] = df_sspecs[df_sspecs.columns[4:]].astype(float)
        except Exception as e:
            raise ValueError(f"\nBand values must be numeric: \n{e}") from e
        col_dtype = {}
        for cn in coln:
            if cn in coln_n:
                col_dtype[cn] = str
            else:
                col_dtype[cn] = float  # type: ignore[assignment]
                # Condition-dependent typing
        df_sspecs = dataframe_validator(colname_dtypes_dict=col_dtype)(df_sspecs)

        # Update
        updated = pd.DataFrame([], columns=["Group", "Use_type", "Sample_name"])
        added = pd.DataFrame([], columns=["Group", "Use_type", "Sample_name"])
        for rown in range(df_sspecs.shape[0]):
            rowl = list(df_sspecs.iloc[rown, :])
            # Update standalone specs by row
            updates: dict[str, pd.DataFrame]
            updates = self.add_standalone_specs(
                group=rowl[1],
                spec_data=[rowl[4:]],
                use_type=rowl[2],
                sample_name_list=[rowl[3]],
                silent_run=True,
                return_updates=True,
            )
            self._update_sample_sspecs()
            # Update sample labels & targets
            self._update_sample_labels_targets()
            # Save updated item for report
            updated = pd.concat([updated, updates["Updated_items"]], ignore_index=True)
            added = pd.concat([added, updates["Added_items"]], ignore_index=True)

        # Report
        # Updated
        if len(updated) > 0:
            print("\nFollowing standalone spectra are updated:\n")
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(updated)
        # Added
        if len(added) > 0:
            print("\nFollowing standalone spectra are added:\n")
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(added)
        # No item changed
        if (len(updated) == 0) & (len(added) == 0):
            print("No spectra found in the given CSV file")

        # Update sample labels & targets
        self._update_sample_labels_targets()

    # Alias
    load_specs = load_standalone_specs

    # Get standalone spectra
    # Format of associated attribute:
    # [0 id, 1 group, 2 use_type, 3 sample_name, 4 spectral_data_list]
    @simple_type_validator
    def _get_sspecs(  # noqa: C901
        self,
        group_name: str = "",
        use_type: str = "",
        sample_name: str = "",
        sample_name_exact_match: bool = True,
    ) -> tuple[
        list[tuple[str, str, str, str, list[Union[float, int]]]],
        list[tuple[str, str, str, str, list[Union[float, int]]]],
    ]:
        """
        Filter standalone spectra with group, use_type, sample_name, return matched and unmatched items.
        Users can use approximate match by set sample_name_exact_match=False, it will return items with sample_names containing the given name value.
        """  # noqa: E501
        # Loop standalone spectral items
        matched = []
        unmatched = []
        for spect in self._standalone_specs:
            # Filter group name
            condg = False
            if len(group_name) > 0:
                if spect[1] == group_name:
                    condg = True
            else:
                condg = True
            # Filter use_type
            condt = False
            if len(use_type) > 0:
                if spect[2] == use_type:
                    condt = True
            else:
                condt = True
            # Filter sample_name
            condn = False
            if len(sample_name) > 0:
                if sample_name_exact_match:
                    if sample_name == spect[3]:
                        condn = True
                else:
                    if sample_name in spect[3]:
                        condn = True
            else:
                condn = True
            # Get filtered result
            if condg & condt & condn:
                matched.append(spect)
            else:
                unmatched.append(spect)
        # Return results
        return matched, unmatched

    # List standalone spectra
    @overload
    def ls_standalone_specs(
        self,
        sample_name: str = "",
        group: str = "",
        use_type: str = "",
        *,
        sample_name_exact_match: bool = True,
        print_result: bool = True,
        return_dataframe: Literal[False] = False,
    ) -> None: ...

    @overload
    def ls_standalone_specs(
        self,
        sample_name: str = "",
        group: str = "",
        use_type: str = "",
        *,
        sample_name_exact_match: bool = True,
        print_result: bool = True,
        return_dataframe: Literal[True] = True,
    ) -> pd.DataFrame: ...

    # List standalone spectra
    # Format of associated attribute:
    # [0 id, 1 group, 2 use_type, 3 sample_name, 4 spectral_data_list]
    @simple_type_validator
    def ls_standalone_specs(
        self,
        sample_name: str = "",
        group: str = "",
        use_type: str = "",
        *,
        exact_match: bool = True,
        print_result: bool = True,
        return_dataframe: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        List added standalone spectra based on filtering conditions.
        If a filter criterion is not provided, the corresponding filter is not applied.

        Parameters
        ----------
        sample_name : str, optional
            Name of the sample to filter by.
            If None, no filtering by sample name is applied.
            Default is None.

        group : str, optional
            Name of the group to filter by.
            If None, no filtering by group is applied.
            Default is None.

        use_type : str, optional
            Use type of the spectra to filter by.
            If None, no filtering by use type is applied.
            Default is None.

        exact_match : bool, optional
            If False, all items with sample names containing the given string will be included.
            Default is True.

        print_result : bool, optional
            If True, prints a simplified table of results to the console. Default is True.

        return_dataframe : bool, optional
            If True, returns a pandas DataFrame of the filtered spectra. Default is False.

        Returns
        -------
        pandas.DataFrame or None
            If ``return_dataframe=True``, returns a pandas DataFrame of matched standalone spectra.

            If ``return_dataframe=False``, returns None.

        Notes
        -----
        This method is also available as ``ls_specs``.

        Examples
        --------
        Prepare a ``SpecExp`` instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_standalone_specs(
            ...     [[1, 2, 3, 4], [1, 3, 3, 3]],
            ...     group='group1',
            ...     sample_name_list=['sample1', 'sample2']
            ... )

        List all standalone spectra::

            >>> exp.ls_standalone_specs()

        Filter by group::

            >>> exp.ls_standalone_specs(group='group1')

        Filter by partial sample name::

            >>> exp.ls_standalone_specs(sample_name='1', exact_match=False)

        Return results as a DataFrame::

            >>> df = exp.ls_standalone_specs(return_dataframe=True)
        """  # noqa: E501

        group_name: str = group
        sample_name_exact_match: bool = exact_match

        # Get result
        result = self._get_sspecs(
            group_name=group_name,
            use_type=use_type,
            sample_name=sample_name,
            sample_name_exact_match=sample_name_exact_match,
        )[0]

        if len(result) > 0:
            # Convert result to dataframe
            df_sspec = pd.DataFrame(result, columns=["ID", "Group", "Use_type", "Sample_name", "Spectral_data"])
            # Print report
            if print_result:
                with pd.option_context("display.max_rows", None, "display.max_columns", None):
                    print(df_sspec.iloc[:, 1:-1])
            # Return result
            if return_dataframe:
                return df_sspec
        else:
            if print_result:
                print("\nNo matched standalone spectrum item found")
            if return_dataframe:
                return pd.DataFrame(columns=["ID", "Group", "Use_type", "Sample_name", "Spectral_data"])
        return None

    # Alias
    ls_specs = ls_standalone_specs

    # Remove standalone spectra
    # Format of associated attribute:
    # [0 id, 1 group, 2 use_type, 3 sample_name, 4 spectral_data_list]
    @simple_type_validator
    def rm_standalone_specs(
        self,
        sample_name: str = "",
        group: str = "",
        use_type: str = "",
        sample_name_exact_match: bool = True,
        silent_run: bool = False,
    ) -> None:
        """
        Remove added standalone spectra from this SpecExp instance.

        The removal can be filtered by sample name, group, and use-type.
        If a filter parameter is not specified or left empty, the corresponding filter criterion is not applied.

        Parameters
        ----------
        sample_name : str, optional
            Name of the spectra to remove. Default is empty string.

        group : str, optional
            Group name of the spectra to remove. Default is empty string.

        use_type : str, optional
            Use type of the spectra to remove (e.g., 'sample'). Default is empty string.

        sample_name_exact_match : bool, optional
            If False, all spectra with names containing the given string are removed. Default is True.

        silent_run : bool, optional
            If True, suppresses printing the removal report to the console. Default is False.

        Notes
        -----
        This method is also available as ``rm_specs``.

        Examples
        --------
        Prepare a ``SpecExp`` instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_standalone_specs(
            ...     [[1, 2, 3, 4], [1, 3, 3, 3]],
            ...     group='group1',
            ...     sample_name_list=['sample1', 'sample2']
            ... )

        Remove all standalone spectra::

            >>> exp.rm_standalone_specs()

        Remove spectra by group::

            >>> exp.rm_standalone_specs(group='group1')
        """

        group_name: str = group

        # Get result
        result = self._get_sspecs(
            group_name=group_name,
            use_type=use_type,
            sample_name=sample_name,
            sample_name_exact_match=sample_name_exact_match,
        )
        matched = result[0]
        unmatched = result[1]

        # Update
        if len(matched) > 0:
            if len(unmatched) > 0:
                # Update data
                self._standalone_specs = unmatched
                self._update_sample_sspecs()
                # Update sample labels & targets
                self._update_sample_labels_targets()
                # Print report
                if not silent_run:
                    df_matched = pd.DataFrame(
                        matched, columns=["ID", "Group", "Use_type", "Sample_name", "Spectral_data"]
                    )
                    print("\nFollowing standalone spectra are removed:\n")
                    with pd.option_context("display.max_rows", None, "display.max_columns", None):
                        print(df_matched.iloc[:, 1:-1])
            else:
                self._standalone_specs = []
                self._update_sample_sspecs()
                # Update sample labels & targets
                self._update_sample_labels_targets()
                if not silent_run:
                    print("All standalone spectra are removed")
        else:
            if not silent_run:
                print("\nNo matched standalone spectrum item found. No standalone spectra are removed")

        # Update sample labels & targets
        self._update_sample_labels_targets()

        # Updated saved file
        self._update_sspecs_file(silent_run=True)

    # Alias
    rm_specs = rm_standalone_specs

    ## Sample label management ====================
    # Helper: Sample label - group identifier
    @simple_type_validator
    def _sample_id_to_group(self, sample_ids: list[str]) -> list[str]:
        """
        Find sample groups by sample IDs.
        """
        # For image ROI as sample
        if len(self.rois_sample) > 0 and len(self.standalone_specs_sample) == 0:
            sid_group_arr = np.array([roit[0:2] for roit in self.rois_sample])
        # For standalone spectra spectrum as sample
        elif len(self.rois_sample) == 0 and len(self.standalone_specs_sample) > 0:
            sid_group_arr = np.array([roit[0:2] for roit in self.standalone_specs_sample])
        elif len(self.rois_sample) == 0 and len(self.standalone_specs_sample) == 0:
            raise ValueError(
                "No sample is added. \
                    At least one of sample ROI and sample standalone spectra must be added."
            )
        else:
            sid_group_arr = np.array(
                [roit[0:2] for roit in self.rois_sample] + [roit[0:2] for roit in self.standalone_specs_sample]
            )
        # Find sample belonging group by sample IDs
        sample_id_groups: list[str] = []
        for sid in sample_ids:
            id_group = sid_group_arr[sid_group_arr[:, 0] == sid]
            if id_group.shape == (1, 2):
                group_i = str(id_group[0, 1])
                sample_id_groups.append(group_i)
            elif id_group.shape[0] < 1:
                raise ValueError(f"Invalid sample ID: {sid}")
            else:
                raise ValueError(f"Sample ID duplication: {sid}. \nDuplicated item: \n{id_group}\n")
        return sample_id_groups

    # Save labels dataframe to csv file
    # sample_labels: [0 fixed sample id, 1 user assinged labels, 2 sample belonging group]
    @simple_type_validator
    def sample_labels_to_csv(self, path: str = "") -> None:
        """
        Save sample labels to a CSV file.

        Parameters
        ----------
        path : str, optional
            Output CSV file path.
            If not provided, the labels are saved as 'Sample_labels.csv' in the report directory of this SpecExp instance.

        Notes
        -----
        This method is also available as ``labels_to_csv``.

        Examples
        --------
        Prepare a ``SpecExp`` instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name("image1.tif", "group1")
            >>> exp.add_rois_by_suffix("_roi.xml", group="group1")

        Save labels to the default path::

            >>> exp.sample_labels_to_csv()

        Save labels to a custom path::

            >>> exp.sample_labels_to_csv("/labels.csv")
        """  # noqa: E501

        # Default save path
        if path == "":
            save_path = self.report_directory + "Sample_labels.csv"
        elif ("/" not in path) & ("\\" not in path):
            save_path = self.report_directory + path
        else:
            save_path = path

        # Validate output dir path
        dir_path = os.path.dirname(save_path)
        if not (os.path.exists(unc_path(dir_path)) and os.path.isdir(unc_path(dir_path))):
            raise ValueError(f"Output directory is invalid: {dir_path}")

        # Labels as df
        df_lb = pd.DataFrame(self._sample_labels, columns=["Sample_ID", "Label", "Group"])
        df_lb = df_lb.astype("object")

        # Save to file
        df_lb.to_csv(unc_path(save_path), index=False)

    # Alias
    labels_to_csv = sample_labels_to_csv

    # Retrieve labels as dataframe
    @overload
    def ls_sample_labels(
        self, return_dataframe: Literal[True] = True
    ) -> Annotated[pd.DataFrame, dataframe_validator({"Sample_ID": str, "Label": str, "Group": str})]: ...

    @overload
    def ls_sample_labels(self, return_dataframe: Literal[False] = False) -> None: ...

    # Retrieve labels as dataframe
    # sample_labels: [0 fixed sample id, 1 user assinged labels, 2 sample belonging group]
    @simple_type_validator
    def ls_sample_labels(
        self, return_dataframe: bool = True
    ) -> Optional[Annotated[pd.DataFrame, dataframe_validator({"Sample_ID": str, "Label": str, "Group": str})]]:
        """
        List added sample labels.

        Parameters
        ----------
        return_dataframe : bool, optional
            If True, returns the sample labels as a pandas DataFrame.

            If False, prints the labels to the console and returns None.

            Default is True.

        Returns
        -------
        pandas.DataFrame or None
            If ``return_dataframe=True``, returns a pandas DataFrame of sample labels.

            If ``return_dataframe=False``, returns None.

        Notes
        -----
        This method is also available as ``ls_labels``.

        Examples
        --------
        >>> exp = SpecExp("/experiment")
        >>> exp.add_groups("group1")
        >>> exp.add_images_by_name("image1.tif", "group1")
        >>> exp.add_rois_by_suffix("_roi.xml", group="group1")

        >>> exp.ls_sample_labels()
        """

        df_lb = pd.DataFrame(self._sample_labels, columns=["Sample_ID", "Label", "Group"])
        df_lb = df_lb.astype("object")
        # Return results
        if return_dataframe:
            return df_lb
        else:
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(df_lb)
            return None

    # Alias
    ls_labels = ls_sample_labels

    # Read sample labels from a dataframe
    # sample_labels: [0 fixed sample id, 1 user assinged labels, 2 sample belonging group]
    @simple_type_validator
    def sample_labels_from_df(self, labels_dataframe: Annotated[Any, dataframe_validator(ncol=3)]) -> None:
        """
        Load sample labels from a pandas DataFrame into this SpecExp instance.

        Parameters
        ----------
        labels_dataframe : pandas.DataFrame
            DataFrame containing sample labels.

            Expected columns are ['Sample_ID', 'Label', 'Group'].
            Each combination of sample ID and group must be unique.

        Notes
        -----
        This method is also available as ``labels_from_df``.

        Examples
        --------
        >>> exp = SpecExp("/experiment")
        >>> exp.add_groups("group1")
        >>> exp.add_images_by_name("image1.tif", "group1")
        >>> exp.add_rois_by_suffix("_roi.xml", group="group1")

        >>> df = exp.ls_sample_labels()
        >>> exp.sample_labels_from_df(df)
        """

        # Update labels - formatting with property formatting function
        self.sample_labels = labels_dataframe

    # Alias
    labels_from_df = sample_labels_from_df

    # Load labels from csv
    # sample_labels: [0 fixed sample id, 1 user assinged labels, 2 sample belonging group]
    @simple_type_validator
    def sample_labels_from_csv(self, label_csv_path: str = "") -> None:
        """
        Load sample labels from a CSV file into the current SpecExp instance.
        The initial or current label CSV file can be created using SpecExp.sample_labels_to_csv().

        Parameters
        ----------
        label_csv_path : str, optional
            Path to the CSV file containing sample labels.

            Expected columns are ['Sample_ID', 'Label', 'Group'].
            If not provided, the file ``Sample_labels.csv`` in the report directory of this SpecExp instance is used.

        Notes
        -----
        This method is also available as ``labels_from_csv``.

        Examples
        --------
        Prepare a ``SpecExp`` instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name("image1.tif", "group1")
            >>> exp.add_rois_by_suffix("_roi.xml", group="group1")

        Load labels from the default path::

            >>> exp.sample_labels_to_csv()
            >>> exp.sample_labels_from_csv()

        Load labels from a custom path::

            >>> exp.sample_labels_to_csv("/labels.csv")
            >>> exp.sample_labels_from_csv("/labels.csv")
        """

        # Default save path
        if label_csv_path == "":
            read_path = self.report_directory + "Sample_labels.csv"
        elif ("/" not in label_csv_path) & ("\\" not in label_csv_path):
            read_path = self.report_directory + label_csv_path
        else:
            read_path = label_csv_path

        # Validate path existence
        if os.path.exists(unc_path(read_path)) is False:
            raise ValueError(f"The provided path is invalid: {read_path}")

        # Validate path extension
        if os.path.splitext(read_path)[1] != ".csv":
            raise ValueError(f"File format is not '.csv': {read_path}")

        # Load dataframe - table format validated in self.sample_labels_from_df
        df_label = pd.read_csv(unc_path(read_path))
        df_label = df_label.astype("object")

        # Update labels - formatting with property formatting function
        self.sample_labels_from_df(df_label)

    # Alias
    labels_from_csv = sample_labels_from_csv

    # Save existed or example of sample targets to csv
    # sample_targets: [0 fixed sample id, 1 user assinged labels, 2 target values, 3 sample belonging group, 4 validation group, 5 test mask, 6 train mask]  # noqa: E501
    @simple_type_validator
    def sample_targets_to_csv(self, path: str = "", include_header: bool = True) -> None:
        """
        Save sample target values to a CSV file.

        If target values have not been specified, the target value column contains blank entries.

        Parameters
        ----------
        path : str, optional
            Output CSV file path.
            If not provided, the file "Sample_target_values.csv" in the report directory of this SpecExp instance is used.

        include_header : bool, optional
            If True, column headers are written as the first row of the CSV file.
            Default is True.

        Notes
        -----
        This method is also available as ``targets_to_csv``.

        Examples
        --------
        Prepare a ``SpecExp`` instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name("image1.tif", "group1")
            >>> exp.add_rois_by_suffix("_roi.xml", group="group1")

        Save target values to the default path::

            >>> exp.sample_targets_to_csv()

        Save target values to a custom path::

            >>> exp.sample_targets_to_csv("/target.csv")
        """  # noqa: E501

        # Validate labels
        if len(self.sample_labels) == 0:
            raise ValueError("No samples exist.\nPlease add samples first.")

        # Default path
        if len(path) == 0:
            path = self.report_directory + "Sample_target_values.csv"

        # Convert sample_targets to dataframe
        if len(self.sample_targets) > 0:
            dft = pd.DataFrame(
                self.sample_targets,
                # columns=["Sample_ID", "Label", "Target_value", "Group", "Validation_group"],
                columns=["Sample_ID", "Label", "Target_value", "Group", "Validation_group", "Test", "Train"],
            )
        else:
            dft = pd.DataFrame(
                [lbt[:2] + ("",) + lbt[2:] + (lbt[0],) for lbt in self.sample_labels],
                # columns=["Sample_ID", "Label", "Target_value", "Group", "Validation_group"],
                columns=["Sample_ID", "Label", "Target_value", "Group", "Validation_group", "Test", "Train"],
            )

        # Write to csv
        if include_header:
            dft.to_csv(unc_path(path), index=False)
        else:
            dft.to_csv(unc_path(path), index=False, header=False)

    # Alias
    targets_to_csv = sample_targets_to_csv

    # Retrieve existed or example of sample targets as df
    @overload
    def ls_sample_targets(self, return_dataframe: Literal[True] = True) -> pd.DataFrame: ...

    @overload
    def ls_sample_targets(self, return_dataframe: Literal[False] = False) -> None: ...

    # Retrieve existed or example of sample targets as df
    # sample_targets: [0 fixed sample id, 1 user assinged labels, 2 target values, 3 sample belonging group, 4 validation group, 5 test mask, 6 train mask]  # noqa: E501
    # Type: list[tuple[str,str,Union[str,bool,int,float]]]
    @simple_type_validator
    def ls_sample_targets(self, return_dataframe: bool = True) -> Optional[pd.DataFrame]:
        """
        List sample target values.

        If no sample targets have been specified, a blank template dataframe is returned that can be filled and assigned back to the SpecExp instance.

        Parameters
        ----------
        return_dataframe : bool, optional
            Whether to return the dataframe.

            If False, a simplified view is printed to the console.
            Default is True.

        Returns
        -------
        pandas.DataFrame or None
            If ``return_dataframe=True``, returns a pandas DataFrame of sample target values.

            If ``return_dataframe=False``, returns None.

        Notes
        -----
        This method is also available as ``ls_targets``.

        Examples
        --------
        >>> exp = SpecExp("/experiment")
        >>> exp.add_groups("group1")
        >>> exp.add_images_by_name("image1.tif", "group1")
        >>> exp.add_rois_by_suffix("_roi.xml", group="group1")

        >>> exp.ls_sample_targets()
        """  # noqa: E501

        if len(self.sample_targets) > 0:
            dft = pd.DataFrame(
                self.sample_targets,
                columns=["Sample_ID", "Label", "Target_value", "Group", "Validation_group", "Test", "Train"],
            )
        else:
            tvs = [
                (st[0], st[1], np.nan, self._sample_id_to_group([st[0]])[0], st[0], np.int8(1), np.int8(1))
                for st in self.sample_labels
            ]
            dft = pd.DataFrame(
                tvs,
                columns=["Sample_ID", "Label", "Target_value", "Group", "Validation_group", "Test", "Train"],
            )
        dft.iloc[:, :2] = np.array(dft.iloc[:, :2]).astype("object")
        dft.iloc[:, 3:5] = np.array(dft.iloc[:, 3:5]).astype("object")
        dft.iloc[:, 5:7] = np.array(dft.iloc[:, 5:7]).astype("int8")

        # Return results
        if return_dataframe:
            return dft
        else:
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(dft)
            return None

    # Alias
    ls_targets = ls_sample_targets

    # Read sample target values from dataframe
    # sample_labels: [0 fixed sample id, 1 user assinged labels, 2 sample belonging group]
    @simple_type_validator
    def sample_targets_from_df(self, target_value_dataframe: Annotated[Any, dataframe_validator()]) -> None:
        """
        Load sample target values from a dataframe.

        The current target dataframe can be retrieved using ``SpecExp.ls_sample_targets``.
        A blank template dataframe can also be obtained using the same method, which can be filled and assigned back to the SpecExp instance using this method.

        Parameters
        ----------
        target_value_dataframe : pd.DataFrame
            Dataframe containing sample target values. Expected columns:

                ["Sample_ID", "Label", "Target_value", "Group", "Validation_group", "Test", "Train"]

            And each row corresponds to a single sample.

            The dataframe can be obtained from ``SpecExp.ls_sample_targets`` for editing or reloading.

        Notes
        -----
        This method is also available as ``targets_from_df``.

        Examples
        --------
        Prepare SpecExp instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name("image1.tif", "group1")
            >>> exp.add_rois_by_suffix("_roi.xml", group="group1")

        Load or update target values from dataframe::

            >>> df = exp.ls_sample_targets()
            >>> exp.sample_targets_from_df(df)
        """  # noqa: E501
        # Update targets - formatting with property formatting function
        self.sample_targets = target_value_dataframe

    # Alias
    targets_from_df = sample_targets_from_df

    # Read target values from csv file
    # sample_targets: [0 fixed sample id, 1 user assinged labels, 2 target values, 3 sample belonging group, 4 validation group, 5 test mask, 6 train mask]  # noqa: E501
    @simple_type_validator
    def sample_targets_from_csv(
        self,
        path: str = "",
        target_dtype: Union[type, str, None] = None,
        include_header: bool = True,
    ) -> None:
        """
        Load sample target values from a CSV file.

        The current target CSV file can be retrieved using ``SpecExp.sample_targets_to_csv``.
        A blank template CSV can also be obtained using the same method, which can be edited and loaded back using this method.

        Parameters
        ----------
        path : str
            CSV file path.

            Expected columns:

                ["Sample_ID", "Label", "Target_value", "Group", "Validation_group", "Test", "Train"]

            If not provided, the file ``Sample_target_values.csv`` in the report directory of this SpecExp instance is used.

        target_dtype : type, str, or None, optional
            Data type of the target values.

            If None, pandas will infer the type automatically (see ``pandas.read_csv`` documentation).
            Default is None.

        include_header : bool, optional
            Whether the first row of the CSV file is treated as headers.
            Default is True.

        Notes
        -----
        This method is also available as ``targets_from_csv``.

        Examples
        --------
        Prepare SpecExp instance::

            >>> exp = SpecExp("/experiment")
            >>> exp.add_groups("group1")
            >>> exp.add_images_by_name("image1.tif", "group1")
            >>> exp.add_rois_by_suffix("_roi.xml", group="group1")

        Use default path::

            >>> exp.sample_targets_to_csv()
            >>> exp.sample_targets_from_csv()

        Use custom path::

            >>> exp.sample_targets_to_csv("/target.csv")
            >>> exp.sample_targets_from_csv("/target.csv")
        """  # noqa: E501

        # Default path
        if len(path) == 0:
            path = self.report_directory + "Sample_target_values.csv"

        # Default dtype
        if target_dtype is None:
            # dtp = {0: str, 1: str, 3: str, 4: str}
            dtp = {0: str, 1: str, 3: str, 4: str, 5: np.int8, 6: np.int8}
        else:
            # dtp = {0: str, 1: str, 2: target_dtype, 3: str, 4: str}  # type: ignore[dict-item]
            dtp = {0: str, 1: str, 2: target_dtype, 3: str, 4: str, 5: np.int8, 6: np.int8}  # type: ignore[dict-item]

        # Read df
        if include_header:
            dft = pd.read_csv(unc_path(path), dtype=dtp)
        else:
            dft = pd.read_csv(unc_path(path), dtype=dtp, header=None)
        dft.fillna("-", inplace=True)
        str_cols = list(dft.columns[:2]) + list(dft.columns[3:])
        dft[str_cols] = dft[str_cols].astype("object")

        # Update targets - formatting with property formatting function
        self.sample_targets = dft

    # Alias
    targets_from_csv = sample_targets_from_csv

    # Update_sample_labels_targets
    # sample_labels: [0 fixed sample id, 1 user assinged labels, 2 sample belonging group]
    # sample_targets: [0 fixed sample id, 1 user assinged labels, 2 target values, 3 sample belonging group, 4 validation group, 5 test mask, 6 train mask]  # noqa: E501
    def _update_sample_labels_targets(self) -> None:
        # Get all sample labels
        new_ids: list[str] = [roit[0] for roit in self.rois_sample] + [
            spect[0] for spect in self.standalone_specs_sample
        ]

        # Add new labels - new label default to sample ID
        old_ids: list[str] = [lt[0] for lt in self._sample_labels]
        new_labels: list = []
        for nid in new_ids:
            if nid in old_ids:
                label: tuple[str, str, str] = [lt for lt in self._sample_labels if lt[0] == nid][0]
                label = (nid, label[1], self._sample_id_to_group([nid])[0])
            else:
                group: str = self._sample_id_to_group([nid])[0]
                label = (nid, nid, group)
            new_labels.append(label)
        # Update labels - keep old labeled data unchange
        self._sample_labels = new_labels

        # Add new targets - new label default to sample ID
        old_ids = [lt[0] for lt in self._sample_targets]
        new_targets: list = []
        target: tuple[str, str, Any, str, str, np.int8, np.int8]
        for nid in new_ids:
            if nid in old_ids:
                target = [tt for tt in self._sample_targets if tt[0] == nid][0]
                labl: str = [lt[1] for lt in self._sample_labels if lt[0] == nid][0]
                target = (nid, labl, target[2], self._sample_id_to_group([nid])[0], target[4], target[5], target[6])
            else:
                labl = [lt[1] for lt in self._sample_labels if lt[0] == nid][0]
                group = self._sample_id_to_group([nid])[0]
                target = (nid, labl, None, group, nid, np.int8(1), np.int8(1))
            new_targets.append(target)
        # Update targets - keep old labeled data unchange
        self._sample_targets = new_targets

    ## Save data configurations to file and reload ====================

    # Auto-save with copy before each run of testing
    # Save data configurations to file
    @simple_type_validator
    def save_data_config(self, copy: bool = True) -> None:
        """
        Save the current configuration of this SpecExp instance to a file in the root of the report directory.

        The file path is ``(SpecExp.report_directory)/SpecExp_configuration/SpecExp_data_configuration_(creating-time).dill``.

        The copy name is ``SpecExp_data_configuration_(creating-time)_copy_at_(saving-time).dill``

        Parameters
        ----------
        copy : bool, optional
            Whether to create a simultaneous copy of the configuration file.
            Default is True.

        Notes
        -----
        This method is also available as ``save_config``.

        Examples
        --------
        Prepare SpecExp instance::

            >>> exp = SpecExp("/experiment")

        Save configuration with default behavior (copy enabled)::

            >>> exp.save_data_config()

        Save configuration without creating a copy::

            >>> exp.save_data_config(copy=False)
        """  # noqa: E501

        # Current time for saving
        cts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Dump directory
        dump_dir = self.report_directory + "SpecExp_configuration/"
        if not os.path.exists(unc_path(dump_dir)):
            os.mkdir(unc_path(dump_dir))

        # Dump data
        dump_path0 = dump_dir + f"SpecExp_data_configuration_{self.create_time}.dill"
        with open(unc_path(dump_path0), 'wb') as f:
            dill.dump(self, f)

        # Dump copy
        if copy:
            # Validate copy path
            dump_path1 = dump_dir + f"SpecExp_data_configuration_{self.create_time}_copy_at_{cts}.dill"
            dump_path1_sp = os.path.splitext(dump_path1)
            ci = 0
            while os.path.exists(unc_path(dump_path1)):
                ci = ci + 1
                cs = "_" + str(ci)
                dump_path1 = dump_path1_sp[0] + cs + dump_path1_sp[1]
                if ci > 99:
                    raise ValueError(
                        "Too many file copies are being created for ExpSpec configuration data, \
                            copy file creation rate limited to 100 per second."
                    )
            # Dump
            with open(unc_path(dump_path1), 'wb') as f:
                dill.dump(self, f)

            # Print output path
            print("\nSpecExp configurations saved to: \n", dump_path0)

    # Alias
    save_config = save_data_config

    # Load data configurations from saved file
    @simple_type_validator
    def load_data_config(self, config_file_path: str = "") -> None:
        """
        Load SpecExp configurations from a dill file.

        Parameters
        ----------
        config_file_path : str, optional
            Path to the configuration dill file.
            Can be a file path or the file name in the report directory of this SpecExp instance.

            If not provided or empty, the path will be:

                ``(SpecExp.report_directory)/SpecExp_configuration/SpecExp_data_configuration_(SpecExp.create_time).dill``

            Default is empty string.

        Notes
        -----
        This method is also available as ``load_config``.

        Examples
        --------
        Prepare SpecExp instance::

            >>> exp = SpecExp("/experiment")

        Load configuration using default path::

            >>> exp.save_data_config()
            >>> exp.load_data_config()

        Load configuration from a custom path::

            >>> exp.load_data_config("/spec_exp_config.dill")
        """  # noqa: E501

        # Load path
        if config_file_path == "":
            dump_path0 = (
                self.report_directory + "SpecExp_configuration/" + f"SpecExp_data_configuration_{self.create_time}.dill"
            )
        elif ("/" not in config_file_path) & ("\\" not in config_file_path):
            dump_path0 = self.report_directory + "SpecExp_configuration/" + config_file_path
        else:
            dump_path0 = config_file_path

        # Load to instance
        with open(unc_path(dump_path0), 'rb') as f:
            loaded_instance = dill.load(f)
        # self.__dict__.update(loaded_instance.__dict__)
        for key, value in loaded_instance.__dict__.items():
            object.__setattr__(self, key, value)

    # Alias
    load_config = load_data_config


# %% SpecExp validator for SpecPipe


# SpecExp validation
@simple_type_validator
def _spec_exp_validator(spec_exp: SpecExp) -> None:  # noqa: C901
    """Validate whether SpecExp instance is completely and correctly configured."""
    # Validate SpecExp
    if type(spec_exp) is not SpecExp:
        raise TypeError(f"spec_exp must be a SpecExp object, but got: {type(spec_exp)}")

    # Validate report diretory
    if not os.path.isdir(unc_path(spec_exp._report_directory)):
        raise ValueError(f"\nReport directory of given SpecExp is invalid: \n'{spec_exp._report_directory}'")

    # Validate group
    if len(spec_exp.groups) == 0:
        raise ValueError("No group is found in given SpecExp")

    # Validate sample data configs
    if len(spec_exp.standalone_specs_sample) == 0:
        if len(spec_exp.images) == 0:
            raise ValueError("Neither image path nor standalone spectrum is found in given SpecExp")
        elif len(spec_exp.rois_sample) == 0:
            raise ValueError("Neither sample ROI nor standalone spectrum is found in given SpecExp")
        for g in spec_exp.groups:
            group_images = spec_exp.ls_images(group=g, return_dataframe=True, print_result=False)
            group_rois = spec_exp.ls_rois(group=g, roi_type="sample", return_dataframe=True, print_result=False)
            if len(group_images) == 0:
                raise ValueError(f"Neither image nor standalone spectrum is found in group: '{g}'")
            elif len(group_rois) == 0:
                raise ValueError(f"Neither image sample ROI nor standalone spectrum is found in group: '{g}'")
    else:
        if len(spec_exp.images) > 0 or len(spec_exp.rois_sample) > 0:
            raise ValueError(
                "Hybrid samples from both standalone spectra and spectral images "
                + "is not allowed by SpecPipe pipeline."
                + "\nPlease provide either pure image samples or standalone spectrum samples"
            )
        for g in spec_exp.groups:
            if (
                len(
                    spec_exp.ls_standalone_specs(
                        group=g,
                        use_type="sample",
                        print_result=False,
                        return_dataframe=True,
                    )
                )
                == 0
            ):
                raise ValueError(f"No spectrum is found in group: '{g}'")

    # Validate sample target values
    sample_target_values = [spt[2] for spt in spec_exp.sample_targets]
    if len(spec_exp.sample_targets) == 0 or sample_target_values == [None] * len(sample_target_values):
        raise ValueError("No sample target value is found in given SpecExp")
    for stt in spec_exp.sample_targets:
        if stt[2] is None or stt[2] == np.nan:
            raise ValueError(
                "Sample target value with ID '{stt[0]}', label '{stt[1]}' and group '{stt[3]}'\
                is missing. Got sample target value: {stt[2]}"
            )
    return None
