# -*- coding: utf-8 -*-
"""
Swectral - Pipeline sample augmentation methods / helpers

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Basics
import warnings

# Typing
from typing import Optional, Union, Annotated, Any, Callable

# Basic data
import numpy as np

# Raster
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

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

    # Check cell counts for mosaic resampling
    n_cells = master_roi.area / resolution**2
    if n_cells > 1000000:
        warnings.warn(
            f"High grid density detected ({int(n_cells)} cells). The computation could be slow."
            "Consider increasing the resolution parameter if statistically unnecessary",
            UserWarning,
            stacklevel=2,
        )

    # Pre-compiling the geometry for spatial queries
    prepared_roi = prep(master_roi)

    # Generate candidate squares strictly within the ROI
    min_x, min_y, max_x, max_y = master_roi.bounds

    # Stop boundaries
    stop_x = max_x - resolution + 1e-15
    stop_y = max_y - resolution + 1e-15

    # Compute ranges
    x_coords = np.arange(min_x, stop_x, resolution)
    y_coords = np.arange(min_y, stop_y, resolution)

    # Vectorized computation
    xv, yv = np.meshgrid(x_coords, y_coords)
    potential_origins = np.stack([xv.ravel(), yv.ravel()], axis=-1)
    candidate_squares = []
    for x, y in potential_origins:
        square = box(x, y, x + resolution, y + resolution)
        if prepared_roi.contains(square):
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


# Blend samples generator
@simple_type_validator
def blend_samples(  # noqa: C901
    n_samples: int,
    is_regression: bool,
    use_validation_group: bool = True,
    abs_tol: Union[float, int, None] = None,
    rel_tol: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Callable:
    """
    Generator for creating a sample blending process using convex combinations.

    The generator returns a callable that accepts and returns a list of tuple with the same structure::

            (
                sample_id : str,
                sample_label : str,
                validation_group : str,
                test_mask : np.int8,
                train_mask : np.int8,
                original_shape : tuple of int,
                target_value : Any,
                predictors : array-like of shape (n_features,)
            )

    Synthetic predictors are computed as Dirichlet-weighted averages of an anchor sample and one or more valid neighbors.
    For regression, targets are blended using the same weights; for classification, the anchor target is retained.

    Samples are generated either per validation group (restricted to groups with at least one non-lonely sample) or globally across the training pool.

    It can be registered using ``add_process`` or used within ``build_pipelines`` with::

        - ``input_data_level`` set to either ``7`` (``"spec1d"``) or ``8`` (``"assembly"``)
        - ``output_data_level`` set to ``8`` (``"assembly"``)

    Parameters
    ----------
    n_samples : int
        Total number of synthetic samples to generate.
        When ``use_validation_group=True``, this value is distributed approximately evenly across eligible validation groups.

    is_regression : bool
        If ``True``, regression mode is used and targets are blended numerically.

        If ``False``, classification mode is used and the synthetic target equals to the anchor target.

    use_validation_group : bool, optional
        If ``True``, synthetic samples are generated independently within each validation group, restricted to groups containing at least one anchor with valid neighbors.

        If ``False``, the full training pool is used globally.

        Default is ``True``.

    abs_tol : float or int or None, optional
        Absolute tolerance for regression neighbor selection.

        If ``None``, no absolute tolerance constraint is applied.
        Default is ``None``.

    rel_tol : float or None, optional
        Relative tolerance for regression neighbor selection.

        If ``None``, no relative tolerance constraint is applied.
        Default is ``None``.

    random_state : int or None, optional
        Seed used to initialize the NumPy random number generator for reproducibility.

        If ``None``, a random seed is used. Default is ``None``.

    See Also
    --------
    SpecPipe.add_process
    SpecPipe.build_pipeline

    Returns
    -------
    Callable
        A pipeline-compatible blending process callable.

    Examples
    --------
    Incorporation into pipeline, for SpecPipe instance ``pipe``::

        >>> blend = blend_samples(n_samples=100, is_regression=False)
        >>> pipe.add_process(7, 8, 0, blend)
    """  # noqa: E501
    return _BlendSamples(
        n_samples=n_samples,
        is_regression=is_regression,
        use_validation_group=use_validation_group,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        random_state=random_state,
    ).blend_samples


# Blend samples
class _BlendSamples:
    """Pipeline wrapper for ``_blend_samples``."""

    @simple_type_validator
    def __init__(
        self,
        n_samples: int,
        is_regression: bool,
        use_validation_group: bool = True,
        abs_tol: Union[float, int, None] = None,
        rel_tol: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_samples: int = n_samples
        self.is_regression: bool = is_regression
        self.use_validation_group: bool = use_validation_group
        self.abs_tol: Union[float, int, None] = abs_tol
        self.rel_tol: Optional[float] = rel_tol
        self.random_state: Optional[int] = random_state

    def blend_samples(
        self,
        sample_data: list[
            tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, Annotated[Any, arraylike_validator(ndim=1)]]
        ],
    ) -> list[
        tuple[str, str, str, np.int8, np.int8, tuple[int, ...], Any, Annotated[np.ndarray, arraylike_validator(ndim=1)]]
    ]:
        result: list[
            tuple[
                str,
                str,
                str,
                np.int8,
                np.int8,
                tuple[int, ...],
                Any,
                Annotated[np.ndarray, arraylike_validator(ndim=1)],
            ]
        ] = _blend_samples(
            sample_data=sample_data,
            n_samples=self.n_samples,
            is_regression=self.is_regression,
            use_validation_group=self.use_validation_group,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
            random_state=self.random_state,
        )
        return result


# Blend samples
# Sample_list item: (0 - Sample id, 1 - Sample label, 2 - Validation group, 3 - Test mask, 4 - Train mask, 5 - Original shape, 6 - Target value, 7 - Sample predictor values)  # noqa: E501
@simple_type_validator
def _blend_samples(  # noqa: C901
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
    Generate synthetic samples by blending training instances via convex combinations.

    Synthetic predictors are computed as Dirichlet-weighted averages of an anchor sample and one or more valid neighbors.
    For regression, targets are blended using the same weights; for classification, the anchor target is retained.

    Samples are generated either per validation group (restricted to groups with at least one non-lonely sample) or globally across the training pool.
    """  # noqa: E501

    # Import dependencies
    import numpy as np

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
                        denom = max(abs(anchor_target), abs(target), 1e-12)

                        is_within_abs = (abs_tol is None) or (diff <= abs_tol)
                        # Validate target == 0
                        is_within_rel = (rel_tol is None) or (denom == 1e-12) or (diff / denom <= rel_tol)

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

    return sample_data + sync_sample_data
