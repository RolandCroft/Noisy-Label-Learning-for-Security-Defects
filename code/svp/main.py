"""
Create a ML Pipeline. Load, embed, and model data. Tests various different
techniques for SVP.

Examples
--------
>>> python code/svp/main.py -r 63 -m rf -f bert -t baseline
"""
import argparse
import time
import numpy as np
from data_utils import Data
from pu_estimator import PU_Estimator
from ml_utils import (
    resampling, get_model, tuning_bayes, tuning_confident_learning,
    get_threshold, evaluate, get_pu_classifier, tuning_ocsvm,
)


def main(release, model_type, feature_type, technique):
    # releases for validation and testing
    release = int(release)
    val_release = release + 1
    test_release = release + 2 if release != 78 else 81
    # Release 80 has no testing data so skip
    if release == 79:
        val_release, test_release = 81, 82
    if release == 80:
        exit()

    # Load data
    print('Loading Data.')
    train_data = Data(dataset='Mozilla').set_release(release)
    x_train = train_data.get_features(feature_type, release)
    if feature_type == 'bow':
        val_data = Data(dataset='Mozilla').set_release(val_release)
        x_val = val_data.get_features(feature_type, release, '_val')

        test_data = Data(dataset='Mozilla').set_release(test_release)
        x_test = test_data.get_features(feature_type, release, '_test')
    else:
        val_data = Data(dataset='Mozilla').set_release(val_release)
        x_val = val_data.get_features(feature_type, val_release)

        test_data = Data(dataset='Mozilla').set_release(test_release)
        x_test = test_data.get_features(feature_type, test_release)

    # Load labels
    if technique == 'manual_cleaning':
        train_data = train_data.get_silent(
            apply_labels=1).get_latent(apply_labels=1)
    y_train = train_data.get_dataset().vulnerable.to_numpy()
    y_val = val_data.get_dataset().vulnerable.to_numpy()
    y_test = test_data.get_dataset().vulnerable.to_numpy()

    # Perform resampling
    # x_train, y_train = resampling(x_train, y_train, 'over')
    pu = True if technique == 'adaptation' else False
    oc = True if technique == 'oneclass' else False

    if technique != 'oneclass':
        # Get and optimise model
        print('Tuning.')
        clf, clf_params = get_model(model_type)
        if technique == 'regularize':
            clf, clf_params, val_time, val_score = tuning_bayes(
                model_type, clf_params, x_train, y_train, x_val, y_val, regularize=True)
        else:
            clf, clf_params, val_time, val_score = tuning_bayes(
                model_type, clf_params, x_train, y_train, x_val, y_val)

        # Train a model, dependent on the technique
        print('Training.')
        # if technique == 'baseline' or technique == 'manual_cleaning':
        t_start = time.time()
        clf.fit(x_train, y_train)
        train_time = time.time() - t_start
        threshold = get_threshold(clf, x_val, y_val)
    else:
        # One-Class Learning
        model_type = 'ocsvm'
        # Get the reliable positives
        x_train = x_train[y_train == 1]

        print('Tuning.')
        clf, clf_params, val_time, val_score = tuning_ocsvm(
            model_type, {'nu': (0.001, 0.5)}, x_train, x_val, y_val)
        pu = False

        print('Training.')
        t_start = time.time()
        clf.fit(x_train)
        train_time = time.time() - t_start
        threshold = 0.5

    # Confident Learning
    if technique == 'pruning' or technique == 'flipping':
        # Label flipping vs Pruning
        label_flipping = True if technique == 'flipping' else False

        extra_args = {'release': release, 'technique': technique,
                      'feature': feature_type, 'identifier': '_cl_tuning'}
        # Tune different parameters
        clf, clf_params, train_time = tuning_confident_learning(
                clf, x_train, y_train, x_val, y_val,
                label_flipping=label_flipping, eval=True,
                **extra_args)
        threshold = get_threshold(clf, x_val, y_val)

    # Noise-adaptation based PU-Learning
    elif technique == 'adaptation' or technique == 'biased':
        clf = get_pu_classifier(clf, technique)
        # PU Methods do not accept sparse arrays
        if feature_type == 'bow':
            x_train = np.array(x_train.todense())
            x_test = np.array(x_test.todense())
        # Train
        t_start = time.time()
        clf.fit(x_train, y_train)
        train_time = time.time() - t_start
        threshold = get_threshold(clf, x_val, y_val, pu=pu)

    # Label propagation based PU Learning
    # Propagation1 = One-Stage PU Learning
    # Propagation2 = Two-stage PU Learning
    elif technique == 'propagation1' or technique == 'propagation2':
        # Determine distance measure for calculating reliable negatives
        pu_metric = 'cosine' if feature_type != 'sm' else 'euclidean'
        extra_args = {'release': release, 'technique': technique,
                      'feature': feature_type, 'identifier': '_pu_tuning'}
        pu_model = PU_Estimator(pu_metric)
        # Train Two-stage PU model
        t_start = time.time()
        if technique == 'propagation1':
            clf, clf_params = pu_model.two_stage_pu(
                clf, x_train, y_train, x_val, y_val, **extra_args)
        elif technique == 'propagation2':
            clf, clf_params = pu_model.label_propagation(
                clf, x_train, y_train, x_val, y_val, threshold, **extra_args)
        train_time = time.time() - t_start
        threshold = get_threshold(clf, x_val, y_val)

    # Evaluate
    print('Evaluating.')
    extra_args = {'train_time': train_time, 'model': model_type, 'val_score': val_score,
                  'val_time': val_time, 'release': release, 'technique': technique,
                  'feature': feature_type, 'ml_identifier': '',
                  'original_df': test_data}
    evaluate(clf, clf_params, x_test, y_test, threshold, pu=pu, oneclass=oc, **extra_args)

    return 'Success!'


if __name__ == '__main__':
    # Set arguments
    parser = argparse.ArgumentParser(description='Build SVP Pipeline.')
    parser.add_argument('-r', type=str, required=True, metavar='Release',
                        choices=[str(i) for i in range(63, 83)] +
                        ['chrome', 'linux', 'android', 'imagemagick', 'php-src'],
                        help='Release version or dataset.')
    parser.add_argument('-m', type=str, required=True, metavar='Model',
                        choices=['rf', 'brf', 'xgb'], help='Classifier model.')
    parser.add_argument('-f', type=str, required=True, metavar='Feature',
                        choices=['sm', 'bow', 'bert'], help='Feature type.')
    parser.add_argument('-t', type=str, required=True, metavar='Technique',
                        choices=['baseline', 'manual_cleaning', 'pruning',
                                 'flipping', 'adaptation', 'biased', 'regularize',
                                 'propagation1', 'propagation2', 'oneclass'],
                        help='Learning technique.')
    args = parser.parse_args()
    print(main(args.r, args.m, args.f, args.t))
