# -*- coding: utf-8 -*-
"""
Swectral
A Python framework for automated batch composition, implementation and method assessment of plant hyperspectral modeling pipelines
"""  # noqa: E501

# Package meta
__version__ = "0.6.4"
__author__ = "Siwei Luo"
__license__ = "MIT"

# Imports
__all__ = [
    ## Core pipeline tools
    ### Core modules
    "SpecExp",
    "SpecPipe",
    "raster_rgb_preview",
    ### Model combiners
    "combine_classifier",
    "combine_regressor",
    "factorial_model_chains",
    "IdentityTransformer",
    "IdentityResampler",
    ### Raster operation
    "croproi",
    "resample_roi",
    "pixel_apply",
    ### Assembly tools
    "blend_samples",
    ## Spectral statistics tools
    ### ROI statistics
    "round_digit",
    "make_img_func",
    "make_roi_func",
    "make_array_func",
    "roispec",
    "pixcount",
    "nderiv",
    "moment2d",
    "bandquant",
    "Stats2d",
    "roi_mean",
    "roi_std",
    "roi_median",
    "spectral_angle",
    "spectral_angle_arr",
    ### SpecIO functions
    "search_file",
    "envi_roi_coords",
    "shp_roi_coords",
    "roi_to_envi",
    "roi_to_shp",
    ## Example data
    "create_example_raster",
    "create_example_roi_xml",
    "create_example_spec_exp",
    "download_demo_data",
    ## Submodules
    "denoiser",
    "functions",
    "vegeind",
]

# Components
## Core pipeline tools
## Example data
from .example_data import create_example_raster, create_example_roi_xml, create_example_spec_exp, download_demo_data

## Model combiners
from .modelconnector import (
    factorial_model_chains,
    combine_classifier,
    combine_regressor,
    IdentityTransformer,
    IdentityResampler,
)

## Raster operation
from .rasterop import croproi, pixel_apply

## Spectral statistics tools
from .roistats import (
    round_digit,
    make_img_func,
    make_roi_func,
    make_array_func,
    roispec,
    moment2d,
    nderiv,
    pixcount,
    Stats2d,
    roi_mean,
    roi_median,
    roi_std,
    bandquant,
    spectral_angle,
    spectral_angle_arr,
)

## Cross-sample integration
from .sample_aug import resample_roi, blend_samples

## Spectral experiment data management
from .specexp import SpecExp
from .specexp_vis import raster_rgb_preview

## SpecPipe IO tools
from .specio import (
    envi_roi_coords,
    search_file,
    shp_roi_coords,
    roi_to_envi,
    roi_to_shp,
)

## SpecPipe main pipeline
from .pipeline import SpecPipe

## Submodules
from . import denoiser
from . import functions
from . import vegeind
