# -*- coding: utf-8 -*-
"""
Swectral case for high-concurrency server deployment: Bak choi Cadmium modeling

The dataset used in this work is not included in this repository and will be made available upon publication.

Validated on: Ubuntu 22.04 (128 vCPU / 128 GB RAM / 400 GB storage)
"""

import os
import shutil  # noqa

# Basic compute
import numpy as np  # noqa

# Swectral
from swectral import SpecExp, SpecPipe


# %% A. Experiment data configuration

# Windows iPython multiprocessing protector
if __name__ == '__main__':

    #----------------------------------------------
    # A. Configure experiment data
    #----------------------------------------------
    # Spectral experiment data directory
    datadir = "/home/user/data/"

    # Create report directory
    repdir = "/case/cls/"
    os.makedirs(repdir, exist_ok=True)

    # Initialize SpecExp
    exp = SpecExp(repdir)

    # Add experiment groups
    exp.add_groups(['Control', 'Cd_10', 'Cd_40', 'Cd_160'])

    # Add images
    exp.add_images_by_name('1-b.bil', datadir, 'Control')
    exp.add_images_by_name('2-3*.bil', datadir, 'Cd_10')
    exp.add_images_by_name(['2-3*.bil', '3-4*.bil'], datadir, 'Cd_40')
    exp.add_images_by_name('3-4*.bil', datadir, 'Cd_160')

    # Add ROIs
    exp.add_rois_by_suffix('-[1b]?.xml', datadir, 'Control')
    exp.add_rois_by_suffix('-[2]?.xml', datadir, 'Cd_10')
    exp.add_rois_by_suffix('-[3]?.xml', datadir, 'Cd_40')
    exp.add_rois_by_suffix('-[4]?.xml', datadir, 'Cd_160')

    # Set labels
    labels = exp.ls_labels()
    labels["Label"] = [(labl[-9:-7] + labl[-3:]) for labl in labels["Sample_ID"]]
    exp.labels_from_df(labels)

    # Set targets
    targets = exp.ls_targets()
    targets["Target_value"] = targets["Group"]
    # Set validation group
    targets["Validation_group"] = [f"plant_{labl[0:2]}" for labl in targets["Label"]]
    # Apply target settings
    exp.targets_from_df(targets)

    # ROI resampling
    exp.roi_subset_augmentation(n_sub=10, resolution=10, coverage_ratio=0.2, n_processor=-2)

    # Set original ROI to test-only
    targets = exp.ls_targets()
    for irow in range(targets.shape[0]):
        if "_&#aug" not in targets['Sample_ID'][irow]:
            targets.loc[irow, 'Train'] = np.int8(0)


# %% B. Pipeline processing unit configuration

# Windows iPython multiprocessing protector
if __name__ == '__main__':

    #----------------------------------------------
    # B. Construct pipeline processing units
    #----------------------------------------------
    # 1 ROI spectra extraction
    from swectral import roispec

    # 2 Image baseline correction - only for ROI spectra
    # 2.1 Standardization, Normalization and AUC normalization
    from swectral.functions import snv, aucnorm, minmax

    # Passthrough
    def raw(X): return X

    # 2.2 Derivatives
    def derv1(X):
        from swectral import nderiv  # noqa
        return nderiv(X, n=1, padding="edge")

    def derv2(X):
        from swectral import nderiv  # noqa
        return nderiv(X, n=2, padding="edge")

    # 3 Denoising
    from swectral.denoiser import LocalPolynomial  # noqa

    # Savitzky-Golay filter - defined in GridSearch-like
    sg_filters: dict = {}
    for window_size in [None, 5, 11, 21]:
        if window_size is not None:
            for n_order in [1, 2]:
                def sg_filter(X, w=window_size, n=n_order):
                    from swectral.denoiser import LocalPolynomial  # noqa
                    return LocalPolynomial(w, n).savitzky_golay_filter(X)
                sg_filters[f"sg_w{window_size}_n{n_order}"] = sg_filter
    sg_filters["no_sg"] = raw

    # 4 ROI statistics
    def mean_spec(X):
        import numpy as np  # noqa
        return np.nanmean(X, axis=0)

    def mom_1_2(X):
        import numpy as np  # noqa
        X = np.asarray(X)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        if mean.ndim == 1:
            result = np.concatenate((mean, std))
        else:
            result = np.concatenate((mean, std), axis=1)
        return result

    def mom_1_4(X):
        import numpy as np  # noqa
        from swectral import moment2d
        X = np.asarray(X)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        skew = np.asarray(moment2d(X, n=3, standardized=True, axis=0))
        kurt = np.asarray(moment2d(X, n=4, standardized=True, axis=0))
        if mean.ndim == 1:
            result = np.concatenate((mean, std, kurt))
        else:
            result = np.concatenate((mean, std, skew, kurt), axis=1)
        return result

    # 5 Scaling and centering
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # 6 Feature selection
    from skrebate.relieff import ReliefF
    from swectral import IdentityTransformer

    # Pass-through feature selector
    feat_all = IdentityTransformer()

    # Construct GridSearch-like
    relieff_models: dict = {}
    for nfeat in [round((1 * 462) ** 0.5), round((2 * 462) ** 0.5), round((4 * 462) ** 0.5), None]:
        if nfeat is not None:
            for n_neighbors in [5, 10, 20, 40]:
                relieff_models[f"feat{nfeat}_n{n_neighbors}"] = ReliefF(
                    n_features_to_select=nfeat,
                    n_neighbors=n_neighbors,
                )
        else:
            relieff_models["feat_all"] = feat_all

    # 7 Models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.cross_decomposition import PLSRegression
    from swectral.modelcombiners import regressor_to_classifier, create_bagging_model

    # Model - RF
    rf = RandomForestClassifier(n_estimators=20)

    # Model - bagging SVC
    bsvc = create_bagging_model(SVC(probability=True), n_estimators=20, feature_subset="sqrt", oversampling=True)

    # Model - bagging PLS classification
    plsc = regressor_to_classifier(PLSRegression())
    bplsc = create_bagging_model(plsc, n_estimators=20, feature_subset="sqrt", oversampling=True)

    # Combine models

    from swectral import factorial_model_chains

    models = factorial_model_chains(
        [scaler],
        relieff_models,
        estimators={
            "RF": rf,
            "BSVC": bsvc,
            "BPLSC": bplsc,
        },
        is_regression=False,
    )


# %% C. Pipeline construction configuration


# Windows iPython multiprocessing protector
if __name__ == '__main__':

    #----------------------------------------------
    # C. Construct pipeline constructor
    #----------------------------------------------
    pipe = SpecPipe(exp)

    # Build pipelines
    pipe.compose_pipeline(
        [
            # 1 ROI spectrum extraction (Apply first to reduce computation)
            ((5, 6), roispec),
            # 2 Baseline correction
            # 2.1 Standardization, Normalization and AUC Normalization
            ((6, 6), [raw, snv, minmax, aucnorm]),
            # 2.2 Derivatives
            ((6, 6), {"orgin": raw, "derv1": derv1, "derv2": derv2}),
            # 3 Denoising
            ((6, 6), sg_filters),
            # 4 ROI statistics
            ((6, 7), [mean_spec, mom_1_2, mom_1_4]),
            # 5 Models (Feature selector included)
            ((7, 9), models, {
                'validation_method': 'loo',
                'validation_config':
                    {
                        'save_fold_model': False,
                        'save_fold_data': False,
                    },
                'influence_analysis_config': None,
                'save_application_model': False,
            })
        ]
    )


# %% D. Check pipelines

# Windows iPython multiprocessing protector
if __name__ == '__main__':

    #----------------------------------------------
    # D. Check pipelines
    #----------------------------------------------
    # Check added processing units
    pipe.ls_process()

    # Check generated processing chains
    pipe.ls_chains(print_label=False)

    # Test run
    # pipe.test_run(model_test_coverage=0.003)


# %% E. Run pipelines

# Windows iPython multiprocessing protector
if __name__ == '__main__':

    pipe.run(n_processor=-1, step_result=False, resume=True, model_test_coverage=0.001, test_model=False)
