# -*- coding: utf-8 -*-
"""
Tests for data management module of spectral experiment (SpecExp)

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# OS
import os
import sys
import shutil
import tempfile

# Test
import pytest
import unittest

# Data basics
import pandas as pd

# Local
from swectral.specio import silent

# Functions to test
from swectral.specexp import SpecExp, _spec_exp_validator


# %% Helper functions for testing


# Create mock files for testing
def create_test_directory_structure(base_dir: str) -> tuple[str, str]:
    """Create test directories and files for testing"""
    # Create test images directory
    images_dir = os.path.join(base_dir, "test_images")
    os.makedirs(images_dir, exist_ok=True)

    # Create test ROI files directory
    rois_dir = os.path.join(base_dir, "test_rois")
    os.makedirs(rois_dir, exist_ok=True)

    # Create some dummy image files
    image_files = ["image1.tif", "image2.tif", "image3.tif", "mask1.tif", "mask2.tif"]

    for img_file in image_files:
        with open(os.path.join(images_dir, img_file), "w") as f:
            f.write("dummy image content")

    # Create some dummy ROI files (simplified versions)
    roi_files = ["image1_rois.xml", "image2_rois.xml", "image1_mask.shp"]

    for roi_file in roi_files:
        with open(os.path.join(rois_dir, roi_file), "w") as f:
            if roi_file.endswith(".xml"):
                f.write(
                    """<?xml version="1.0" encoding="UTF-8"?>
<RegionsOfInterest version="1.1">
  <Region name="TestROI1" color="255,0,0">
    <GeometryDef>
      <CoordSysStr>none</CoordSysStr>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates>100.0 200.0 300.0 400.0 100.0 600.0 100.0 200.0</Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
    </GeometryDef>
  </Region>
  <Region name="TestROI2" color="0,255,0">
    <GeometryDef>
      <CoordSysStr>none</CoordSysStr>
      <Polygon>
        <Exterior>
          <LinearRing>
            <Coordinates>200.0 200.0 300.0 300.0 500.0 100.0 200.0 200.0</Coordinates>
          </LinearRing>
        </Exterior>
      </Polygon>
    </GeometryDef>
  </Region>
</RegionsOfInterest>"""
                )
            elif roi_file.endswith(".shp"):
                # Simplified shapefile content
                f.write("dummy shapefile content")

    return images_dir, rois_dir


# %% test functions : SpecExp


class TestSpecExp(unittest.TestCase):
    """Test class for SpecExp functionality (static methods + class setup)"""

    test_dir: str
    report_dir: str
    images_dir: str
    rois_dir: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_dir = (str(tempfile.mkdtemp()).replace("\\", "/") + "/").replace("//", "/")
        cls.report_dir = (str(os.path.join(cls.test_dir, "reports")).replace("\\", "/") + "/").replace("//", "/")
        os.makedirs(cls.report_dir, exist_ok=True)

        cls.images_dir, cls.rois_dir = create_test_directory_structure(cls.test_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "test_dir") and os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    @staticmethod
    def spec_exp_init() -> "SpecExp":
        """Create a fresh SpecExp instance for each test"""
        return SpecExp(
            report_directory=TestSpecExp.report_dir,
            log_loading=False,
        )

    @staticmethod
    @silent
    def test_initialization() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        assert spec_exp.report_directory == TestSpecExp.report_dir
        assert spec_exp.log_loading is False
        assert len(spec_exp.groups) == 0
        assert len(spec_exp.images) == 0
        assert len(spec_exp.rois) == 0

        dir_path = TestSpecExp.test_dir + "invalid_path_that_does_not_exist/"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        with pytest.warns(UserWarning, match=dir_path):
            SpecExp(report_directory=dir_path)

    @staticmethod
    @silent
    def test_add_groups() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["group1"])
        assert "group1" in spec_exp.groups

        spec_exp.add_groups(["group2", "group3"])
        assert "group2" in spec_exp.groups
        assert "group3" in spec_exp.groups

        spec_exp.add_groups(["group1"])
        assert spec_exp.groups.count("group1") == 1

    @staticmethod
    @silent
    def test_rm_group() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["group1", "group2"])
        spec_exp.add_images_by_name(
            group="group1",
            image_name="image1.tif",
            image_directory=TestSpecExp.images_dir,
        )

        spec_exp.rm_group("group1")
        assert "group1" not in spec_exp.groups
        assert len(spec_exp.images) == 0

    @staticmethod
    @silent
    def test_add_images() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image*.tif",
            image_directory=TestSpecExp.images_dir,
        )

        images = spec_exp.ls_images(return_dataframe=True)
        assert len(images) == 3

        image_path = os.path.join(TestSpecExp.images_dir, "mask1.tif")
        spec_exp.add_images_by_path(
            group="test_group",
            path=image_path,
            mask_of="image1.tif",
        )

        masks = spec_exp.ls_images(
            mask_of="image1.tif",
            return_dataframe=True,
        )
        assert len(masks) == 1

        img_path = TestSpecExp.test_dir + "invalid_path_that_does_not_exist/image1.tif"
        with pytest.raises(ValueError, match=img_path):
            spec_exp.add_images_by_path(
                group="test_group",
                path=[img_path],
            )

    @staticmethod
    @silent
    def test_ls_images() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image*.tif",
            image_directory=TestSpecExp.images_dir,
        )

        all_images = spec_exp.ls_images(
            return_dataframe=True,
            print_result=False,
        )
        assert len(all_images) == 3

        specific_image = spec_exp.ls_images(
            image_name="image1.tif",
            return_dataframe=True,
            print_result=False,
        )
        assert len(specific_image) == 1

    @staticmethod
    @silent
    def test_rm_images() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image*.tif",
            image_directory=TestSpecExp.images_dir,
        )

        spec_exp.rm_images(image_name="image1.tif")

        remaining_images = spec_exp.ls_images(
            return_dataframe=True,
            print_result=False,
            mask_of="",
        )
        assert len(remaining_images) == 2

    @staticmethod
    @silent
    def test_add_rois_by_suffix_basic() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image*.tif",
            image_directory=TestSpecExp.images_dir,
        )

        # Test basic
        spec_exp.add_rois_by_suffix(
            group="test_group",
            roi_filename_suffix="_rois.xml",
            search_directory=TestSpecExp.rois_dir,
        )

        rois = spec_exp.ls_rois_sample(
            return_dataframe=True,
            print_result=False,
        )
        assert len(rois) == 4

    @staticmethod
    @silent
    def test_add_rois_by_suffix_exclude_roiname() -> None:
        # Test exclude using str
        spec_exp = TestSpecExp.spec_exp_init()
        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image*.tif",
            image_directory=TestSpecExp.images_dir,
        )
        spec_exp.add_rois_by_suffix(
            group="test_group",
            roi_filename_suffix="_rois.xml",
            exclude_roiname='ROI1',
            search_directory=TestSpecExp.rois_dir,
        )
        rois = spec_exp.ls_rois_sample(
            return_dataframe=True,
            print_result=False,
        )
        assert list(rois['ROI_name']) == ['TestROI2', 'TestROI2']
        # Test exclude using list
        spec_exp = TestSpecExp.spec_exp_init()
        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image*.tif",
            image_directory=TestSpecExp.images_dir,
        )
        spec_exp.add_rois_by_suffix(
            group="test_group",
            roi_filename_suffix="_rois.xml",
            exclude_roiname=['ROI1'],
            search_directory=TestSpecExp.rois_dir,
        )
        rois = spec_exp.ls_rois_sample(
            return_dataframe=True,
            print_result=False,
        )
        assert list(rois['ROI_name']) == ['TestROI2', 'TestROI2']

    @staticmethod
    @silent
    def test_add_rois_by_suffix_include_roiname() -> None:
        # Test include using str
        spec_exp = TestSpecExp.spec_exp_init()
        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image*.tif",
            image_directory=TestSpecExp.images_dir,
        )
        spec_exp.add_rois_by_suffix(
            group="test_group",
            roi_filename_suffix="_rois.xml",
            include_roiname='ROI2',
            search_directory=TestSpecExp.rois_dir,
        )
        rois = spec_exp.ls_rois_sample(
            return_dataframe=True,
            print_result=False,
        )
        assert list(rois['ROI_name']) == ['TestROI2', 'TestROI2']
        # Test include using list of str
        spec_exp = TestSpecExp.spec_exp_init()
        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image*.tif",
            image_directory=TestSpecExp.images_dir,
        )
        spec_exp.add_rois_by_suffix(
            group="test_group",
            roi_filename_suffix="_rois.xml",
            include_roiname=['ROI2'],
            search_directory=TestSpecExp.rois_dir,
        )
        rois = spec_exp.ls_rois_sample(
            return_dataframe=True,
            print_result=False,
        )
        assert list(rois['ROI_name']) == ['TestROI2', 'TestROI2']

    @staticmethod
    @silent
    def test_add_rois_by_suffix_edge_cases() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        group = "group_that_doex_not_exist"
        with pytest.raises(ValueError, match=group):
            spec_exp.add_rois_by_suffix(
                group=group,
                roi_filename_suffix="_rois.xml",
                search_directory=TestSpecExp.rois_dir,
            )

        spec_exp = TestSpecExp.spec_exp_init()
        spec_exp.add_groups(["test_group"])
        with pytest.raises(ValueError, match="No image added"):
            spec_exp.add_rois_by_suffix(
                group="test_group",
                roi_filename_suffix="_rois.xml",
                search_directory=TestSpecExp.rois_dir,
            )

    @staticmethod
    @silent
    def test_add_rois_by_file_basic() -> None:

        # Test list of file path
        spec_exp = TestSpecExp.spec_exp_init()
        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image1.tif",
            image_directory=TestSpecExp.images_dir,
        )

        roi_file = os.path.join(TestSpecExp.rois_dir, "image1_rois.xml")
        spec_exp.add_rois_by_file(
            group="test_group",
            path=[roi_file],
            image_name="image1.tif",
        )

        rois = spec_exp.ls_rois_from_file(
            return_dataframe=True,
            print_result=False,
        )
        assert len(rois) == 2

        # Test using file path in string
        spec_exp = TestSpecExp.spec_exp_init()
        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image1.tif",
            image_directory=TestSpecExp.images_dir,
        )

        roi_file = os.path.join(TestSpecExp.rois_dir, "image1_rois.xml")
        spec_exp.add_rois_by_file(
            group="test_group",
            path=roi_file,
            image_name="image1.tif",
        )

        rois = spec_exp.ls_rois_from_file(
            return_dataframe=True,
            print_result=False,
        )
        assert len(rois) == 2

    @staticmethod
    @silent
    def test_add_rois_by_file_edge_cases() -> None:
        spec_exp = TestSpecExp.spec_exp_init()
        roi_file = os.path.join(TestSpecExp.rois_dir, "image1_rois.xml")
        spec_exp.add_groups(["test_group"])

        group = "group_that_doex_not_exist"
        with pytest.raises(ValueError, match=group):
            spec_exp.add_rois_by_file(
                group=group,
                path=[roi_file],
                image_name="image1.tif",
            )

        img_name = "image_name_that_doex_not_exist"
        with pytest.raises(ValueError, match=img_name):
            spec_exp.add_rois_by_file(
                group="test_group",
                path=[roi_file],
                image_name=img_name,
            )

    @staticmethod
    @silent
    def test_add_rois_by_file_include_roiname() -> None:
        spec_exp = TestSpecExp.spec_exp_init()
        roi_file = os.path.join(TestSpecExp.rois_dir, "image1_rois.xml")

        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image1.tif",
            image_directory=TestSpecExp.images_dir,
        )

        spec_exp.add_rois_by_file(
            group="test_group",
            path=[roi_file],
            image_name="image1.tif",
            include_roiname=["1"],
        )

        rois = spec_exp.ls_rois_sample(return_dataframe=True, print_result=False)
        assert len(rois) == 1

    @staticmethod
    @silent
    def test_add_rois_by_file_exclude_roiname() -> None:
        spec_exp = TestSpecExp.spec_exp_init()
        roi_file = os.path.join(TestSpecExp.rois_dir, "image1_rois.xml")

        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image1.tif",
            image_directory=TestSpecExp.images_dir,
        )

        spec_exp.add_rois_by_file(
            group="test_group",
            path=[roi_file],
            image_name="image1.tif",
            exclude_roiname=["1"],
        )

        rois = spec_exp.ls_rois_sample(return_dataframe=True, print_result=False)
        assert len(rois) == 1

    @staticmethod
    @silent
    def test_add_roi_by_coords() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image1.tif",
            image_directory=TestSpecExp.images_dir,
        )

        coordinates = [[(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]]

        spec_exp.add_roi_by_coords(
            roi_name="test_roi",
            group="test_group",
            image_name="image1.tif",
            coord_lists=coordinates,
        )

        rois = spec_exp.ls_rois_from_coords(
            return_dataframe=True,
            print_result=False,
        )
        assert len(rois) == 1

    @staticmethod
    @silent
    def test_ls_rois() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image1.tif",
            image_directory=TestSpecExp.images_dir,
        )

        coordinates = [[(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]]

        spec_exp.add_roi_by_coords(
            roi_name="test_roi",
            group="test_group",
            image_name="image1.tif",
            coord_lists=coordinates,
        )

        all_rois = spec_exp.ls_rois(return_dataframe=True)
        assert len(all_rois) == 1

        specific_roi = spec_exp.ls_rois(
            roi_name_list=["test_roi"],
            return_dataframe=True,
        )
        assert len(specific_roi) == 1

    @staticmethod
    @silent
    def test_rm_rois() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image1.tif",
            image_directory=TestSpecExp.images_dir,
        )

        coordinates = [[(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]]

        spec_exp.add_roi_by_coords(
            roi_name="test_roi",
            group="test_group",
            image_name="image1.tif",
            coord_lists=coordinates,
        )

        spec_exp.rm_rois(roi_name="test_roi")

        remaining_rois = spec_exp.ls_rois(return_dataframe=True)
        assert len(remaining_rois) == 0

    @staticmethod
    @silent
    def test_roi_subset_augmentation() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        # Add group and images
        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image*.tif",
            image_directory=TestSpecExp.images_dir,
        )
        # Add ROIs
        spec_exp.add_rois_by_suffix(
            group="test_group",
            roi_filename_suffix="_rois.xml",
            search_directory=TestSpecExp.rois_dir,
        )
        targets = spec_exp.ls_sample_targets(return_dataframe=True)
        assert len(targets) > 0
        # Add target values
        new_targets = pd.DataFrame(
            {
                "Sample_ID": targets["Sample_ID"],
                "Label": targets["Label"],
                "Target_value": [1.0] * len(targets),
                "Group": ["test_group"] * len(targets),
                "Validation_group": targets["Sample_ID"],
                "Test": [1] * len(targets),
                "Train": [1] * len(targets),
            }
        )
        spec_exp.sample_targets_from_df(new_targets)

        n_roi = len(spec_exp.rois_sample)

        # Resampling augmentation
        num_sub = 3
        spec_exp.roi_subset_augmentation(n_sub=num_sub, resolution=10, coverage_ratio=0.33)
        # Assert ROIs
        origin_roi_ids = [roit[0] for roit in spec_exp.rois_sample if "_&#aug" not in roit[0]]
        aug_roi_ids = [roit[0] for roit in spec_exp.rois_sample if "_&#aug" in roit[0]]
        assert len(origin_roi_ids) == n_roi
        assert len(aug_roi_ids) == n_roi * num_sub
        # Assert SpecExp configured
        _spec_exp_validator(spec_exp)
        # Assert Sample target correctly configured
        for targ in spec_exp.sample_targets:
            if "_&#aug" in targ[0]:
                assert targ[-1] == 1
                assert targ[-2] == 0

    @staticmethod
    @silent
    def test_add_standalone_specs() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])

        spectral_data = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        ]

        spec_exp.add_standalone_specs(
            group="test_group",
            spec_data=spectral_data,
            sample_name_list=["spec1", "spec2"],
        )

        specs = spec_exp.ls_standalone_specs(return_dataframe=True)
        assert len(specs) == 2

    @staticmethod
    @silent
    def test_ls_standalone_specs() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])

        spectral_data = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        ]

        spec_exp.add_standalone_specs(
            group="test_group",
            spec_data=spectral_data,
            sample_name_list=["spec1", "spec2"],
        )

        all_specs = spec_exp.ls_standalone_specs(return_dataframe=True)
        assert len(all_specs) == 2

        specific_spec = spec_exp.ls_standalone_specs(
            sample_name="spec1",
            return_dataframe=True,
        )
        assert len(specific_spec) == 1

    @staticmethod
    @silent
    def test_rm_standalone_specs() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])

        spectral_data = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        ]

        spec_exp.add_standalone_specs(
            group="test_group",
            spec_data=spectral_data,
            sample_name_list=["spec1", "spec2"],
        )

        spec_exp.rm_standalone_specs(sample_name="spec1")

        remaining_specs = spec_exp.ls_standalone_specs(return_dataframe=True)
        assert len(remaining_specs) == 1

    @staticmethod
    @silent
    def test_sample_labels_management() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image1.tif",
            image_directory=TestSpecExp.images_dir,
        )

        coordinates = [[(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]]

        spec_exp.add_roi_by_coords(
            roi_name="test_roi",
            group="test_group",
            image_name="image1.tif",
            coord_lists=coordinates,
        )

        labels = spec_exp.ls_sample_labels(return_dataframe=True)
        assert len(labels) > 0

        new_labels = pd.DataFrame(
            {
                "Sample_ID": labels["Sample_ID"],
                "Label": ["label1"] * len(labels),
                "Group": ["test_group"] * len(labels),
            }
        )

        spec_exp.sample_labels_from_df(new_labels)

        csv_path = os.path.join(TestSpecExp.test_dir, "test_labels.csv")
        spec_exp.sample_labels_to_csv(csv_path)
        assert os.path.exists(csv_path)

        spec_exp.sample_labels_from_csv(csv_path)

        updated_labels = spec_exp.ls_sample_labels(return_dataframe=True)
        assert len(updated_labels) == len(labels)

    @staticmethod
    @silent
    def test_sample_targets_management() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image1.tif",
            image_directory=TestSpecExp.images_dir,
        )

        coordinates = [[(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]]

        spec_exp.add_roi_by_coords(
            roi_name="test_roi",
            group="test_group",
            image_name="image1.tif",
            coord_lists=coordinates,
        )

        labels = spec_exp.ls_sample_labels(return_dataframe=True)
        labels["Label"] = [str(i) for i in range(len(labels))]
        spec_exp.sample_labels_from_df(labels)

        targets = spec_exp.ls_sample_targets(return_dataframe=True)
        assert len(targets) > 0

        new_targets = pd.DataFrame(
            {
                "Sample_ID": targets["Sample_ID"],
                "Label": targets["Label"],
                "Target_value": [1.0] * len(targets),
                "Group": ["test_group"] * len(targets),
                "Validation_group": targets["Sample_ID"],
                "Test": [1] * len(targets),
                "Train": [1] * len(targets),
            }
        )

        spec_exp.sample_targets_from_df(new_targets)

        csv_path = os.path.join(TestSpecExp.test_dir, "test_targets.csv")
        spec_exp.sample_targets_to_csv(csv_path)
        assert os.path.exists(csv_path)

        spec_exp.sample_targets_from_csv(csv_path)

        updated_targets = spec_exp.ls_sample_targets(return_dataframe=True)
        assert len(updated_targets) == len(targets)

    @staticmethod
    @silent
    def test_save_and_load_config() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        spec_exp.add_groups(["test_group"])
        spec_exp.add_images_by_name(
            group="test_group",
            image_name="image1.tif",
            image_directory=TestSpecExp.images_dir,
        )

        coordinates = [[(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]]

        spec_exp.add_roi_by_coords(
            roi_name="test_roi",
            group="test_group",
            image_name="image1.tif",
            coord_lists=coordinates,
        )

        spec_exp.save_data_config(copy=False)
        spec_exp.load_data_config()

        new_spec_exp = SpecExp(
            report_directory=TestSpecExp.report_dir,
            log_loading=False,
        )

        new_spec_exp.load_data_config(f"SpecExp_data_configuration_{spec_exp.create_time}.dill")

        assert len(new_spec_exp.groups) == 1
        assert len(new_spec_exp.images) == 1
        assert len(new_spec_exp.rois) == 1

    @staticmethod
    def test_property_access() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        with pytest.raises(ValueError):
            spec_exp.groups = ["new_group"]

        with pytest.raises(ValueError):
            spec_exp.images = []

        with pytest.raises(ValueError):
            spec_exp.rois = []

        assert isinstance(spec_exp.groups, list)
        assert isinstance(spec_exp.images, list)
        assert isinstance(spec_exp.rois, list)

    @staticmethod
    def test_method_alias() -> None:
        spec_exp = TestSpecExp.spec_exp_init()

        assert spec_exp.add_specs == spec_exp.add_standalone_specs
        assert spec_exp.load_specs == spec_exp.load_standalone_specs
        assert spec_exp.ls_specs == spec_exp.ls_standalone_specs
        assert spec_exp.labels_to_csv == spec_exp.sample_labels_to_csv
        assert spec_exp.ls_labels == spec_exp.ls_sample_labels
        assert spec_exp.labels_from_df == spec_exp.sample_labels_from_df
        assert spec_exp.labels_from_csv == spec_exp.sample_labels_from_csv
        assert spec_exp.targets_from_csv == spec_exp.sample_targets_from_csv
        assert spec_exp.targets_to_csv == spec_exp.sample_targets_to_csv
        assert spec_exp.ls_targets == spec_exp.ls_sample_targets
        assert spec_exp.targets_from_df == spec_exp.sample_targets_from_df
        assert spec_exp.save_config == spec_exp.save_data_config
        assert spec_exp.load_config == spec_exp.load_data_config


# %% Test - SpecExp

# test_specexp = TestSpecExp()
# test_specexp.setUpClass()

# test_specexp.test_initialization()

# test_specexp.test_add_groups()
# test_specexp.test_rm_group()

# test_specexp.test_add_images()
# test_specexp.test_ls_images()
# test_specexp.test_rm_images()

# test_specexp.test_add_rois_by_suffix_basic()
# test_specexp.test_add_rois_by_suffix_exclude_roiname()
# test_specexp.test_add_rois_by_suffix_include_roiname()
# test_specexp.test_add_rois_by_suffix_edge_cases()

# test_specexp.test_add_rois_by_file_basic()
# test_specexp.test_add_rois_by_file_exclude_roiname()
# test_specexp.test_add_rois_by_file_include_roiname()
# test_specexp.test_add_rois_by_file_edge_cases()

# test_specexp.test_add_roi_by_coords()
# test_specexp.test_ls_rois()
# test_specexp.test_rm_rois()

# test_specexp.test_add_standalone_specs()
# test_specexp.test_ls_standalone_specs()
# test_specexp.test_rm_standalone_specs()

# test_specexp.test_sample_labels_management()
# test_specexp.test_sample_targets_management()

# test_specexp.test_save_and_load_config()

# test_specexp.test_property_access()

# test_specexp.test_method_alias()

# test_specexp.tearDownClass()


# %% Test main

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
