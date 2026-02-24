# -*- coding: utf-8 -*-
"""
Swectral - Pipeline sample augmentation methods / helpers

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Typing
from typing import Optional, Union, Annotated, Any

# Basic data
import numpy as np

# Raster
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

# Local
from .specio import simple_type_validator, arraylike_validator


# %% Resample ROI


@simple_type_validator
def resample_roi(  # noqa: C901
    coord_lists: list[list[tuple[Union[int, float], Union[int, float]]]],
    resolution: Union[int, float],
    coverage_ratio: float,
    random_state: Optional[int] = None,
) -> list[list[tuple[Union[int, float], Union[int, float]]]]:
    """
    Randomly resample a multi-part ROI into a list of new multi-part ROIs.

    Each generated sub-ROI is represented as a multi-part structure (list of polygons).
    The resampling follows a grid-based approach where squares must be strictly contained within the original multi-part ROI.

    Parameters
    ----------
    coord_lists : list of list of tuple of 2 (float or int)
        Coordinates of the input multi-part ROI.
        Structure: [[(x1, y1), ...], [Polygon 2 coords], ...]
    resolution : float or int
        The side length of the square grid cells used for resampling.
    coverage_ratio : float
        The target fraction of the total ROI area to resample (0.0 to 1.0).
    random_state : int, optional
        Random state for reproducibility.
        Defaults to None.

    Returns
    -------
    list of list of tuple of 2 (float or int)
        The coordinates of the resampled square sub-ROIs in the same structure as the input ``coord_lists``.

    Examples
    --------
    >>> roi = [[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]]
    >>> samples = resample_roi(roi, resolution=2, coverage_ratio=0.3)
    """  # noqa: E501
    if not (0 <= coverage_ratio <= 1):
        raise ValueError(f"coverage_ratio must be between 0 and 1, got: {coverage_ratio}")

    # Validate random state
    if random_state is None:
        random_state = np.random.randint(0, np.iinfo(np.int32).max)

    rng = np.random.default_rng(random_state)

    # Create the master ROI geometry
    polygons = [Polygon(coords) for coords in coord_lists]
    master_roi = MultiPolygon(polygons)
    if not master_roi.is_valid:
        master_roi = master_roi.buffer(0)

    # Generate candidate squares strictly within the ROI
    min_x, min_y, max_x, max_y = master_roi.bounds

    # Stop boundaries
    stop_x = max_x - resolution + 1e-15
    stop_y = max_y - resolution + 1e-15

    # Compute ranges
    x_coords = np.arange(min_x, stop_x, resolution)
    y_coords = np.arange(min_y, stop_y, resolution)

    candidate_squares = []
    for x in x_coords:
        for y in y_coords:
            square = box(x, y, x + resolution, y + resolution)
            if master_roi.contains(square):
                candidate_squares.append(square)

    if len(candidate_squares) < 1:
        raise ValueError(f"Resolution {resolution} is too large, no square fit strictly within the provided ROI.")

    # Calculate num of resampled squares for the coverage ratio
    total_area = master_roi.area
    target_area = total_area * coverage_ratio
    square_area = resolution**2

    num_to_sample = min(max(1, int(round(target_area / square_area))), len(candidate_squares))

    # Resampling and Merging
    indices = rng.choice(len(candidate_squares), size=num_to_sample, replace=False)
    sampled_geoms = [candidate_squares[i] for i in indices]
    merged_geom = unary_union(sampled_geoms)

    # Output
    output_coords = []
    if isinstance(merged_geom, Polygon):
        output_coords.append(list(merged_geom.exterior.coords))
    elif isinstance(merged_geom, MultiPolygon):
        for poly in merged_geom.geoms:
            output_coords.append(list(poly.exterior.coords))

    return output_coords


# %% In-validation group & cross-sample synthetic sample generator


@simple_type_validator
def _remix_samples(  # noqa: C901
    sample_data: list[
        tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, Annotated[Any, arraylike_validator(ndim=1)]]
    ],
    n_samples: int,
    is_regression: bool,
    use_validation_group: bool = True,
    abs_tol: Union[float, int, None] = None,
    rel_tol: Optional[float] = None,
    random_state: Optional[int] = None,
) -> list[
    tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, Annotated[np.ndarray, arraylike_validator(ndim=1)]]
]:
    """
    Remix synthetic samples for data augmentation.
    Samples are generated for groups with non-lonely samples or globally.
    The generation uses a randomly weighted avg for predictors and numeric targets.
    """
    # Validate random state
    if random_state is None:
        random_state = np.random.randint(0, np.iinfo(np.int32).max)

    rng = np.random.default_rng(random_state)
    sync_sample_data = []

    # Validate predictor type
    sample_data = [d[:-1] + (np.asarray(d[-1]),) for d in sample_data]

    # Filter for valid training samples (Train mask == 1 at index 4)
    train_pool = [s for s in sample_data if s[4] == 1]
    val_groups = sorted({s[2] for s in sample_data})

    # Number of samples per validation group
    n_per_group = max(round(n_samples / len(val_groups)) + int(not use_validation_group), 1)

    # Generate samples per validation group
    if use_validation_group:
        group_samples: list
    else:
        group_samples = train_pool
    n_sync_sample = 0
    for g, vgroup in enumerate(val_groups):
        if use_validation_group:
            group_samples = [s for s in train_pool if s[2] == vgroup]
        if len(group_samples) < 1:
            continue

        # Find neighbors
        if use_validation_group or g == 0:
            valid_anchors_and_neighbors = []

            for anchor in group_samples:
                anchor_target = anchor[-2]
                neighbors = []

                for potential_neighbor in group_samples:
                    # Exclude self as neighbor
                    if potential_neighbor[0] == anchor[0]:
                        continue

                    target = potential_neighbor[-2]

                    if not is_regression:
                        # Classification - find identical
                        if target == anchor_target:
                            neighbors.append(potential_neighbor)
                    else:
                        # Regression - find within thresholds
                        diff = abs(target - anchor_target)
                        denom = max(abs(anchor_target), abs(target))

                        is_within_abs = (abs_tol is None) or (diff <= abs_tol)
                        # Validate target == 0
                        is_within_rel = (rel_tol is None) or (denom == 0) or (diff / denom <= rel_tol)

                        if is_within_abs and is_within_rel:
                            neighbors.append(potential_neighbor)

                if len(neighbors) > 0:
                    valid_anchors_and_neighbors.append((anchor, neighbors))

            if len(valid_anchors_and_neighbors) == 0:
                continue

        # Generate synthetic samples
        k = 0
        while k < n_per_group and (n_sync_sample < n_samples or use_validation_group):

            anchor_idx = rng.integers(len(valid_anchors_and_neighbors))
            anchor, neighbors = valid_anchors_and_neighbors[anchor_idx]

            num_to_pick = rng.integers(1, len(neighbors) + 1)
            neighbor_indices = rng.choice(len(neighbors), size=num_to_pick, replace=False)

            # Samples to blend
            to_blend = [anchor] + [neighbors[i] for i in neighbor_indices]

            weights = rng.dirichlet(np.ones(len(to_blend)))

            # Predictor Weighted Average
            new_predictors = np.zeros_like(to_blend[0][-1], dtype=np.float64)
            for i, s in enumerate(to_blend):
                new_predictors += s[-1] * weights[i]

            # Target Value Calculation
            if is_regression:
                new_target = sum(s[-2] * weights[i] for i, s in enumerate(to_blend))
            else:
                new_target = anchor[-2]

            # Metadata Generation
            if use_validation_group:
                new_id = f"{anchor[0]}_&#syn{g}-{k}"
                new_label = f"{anchor[1]}_&#syn{g}-{k}"
                vgroup_display = vgroup
            else:
                new_id = f"{anchor[0]}_&#syn{n_sync_sample}"
                new_label = f"{anchor[1]}_&#syn{n_sync_sample}"
                vgroup_display = f"&#syn{n_sync_sample}"

            synthetic_item = (
                new_id,
                new_label,
                vgroup_display,
                np.int8(0),
                np.int8(1),
                anchor[-3],
                new_target,
                new_predictors,
            )

            sync_sample_data.append(synthetic_item)
            k += 1
            n_sync_sample += 1

    return sync_sample_data
