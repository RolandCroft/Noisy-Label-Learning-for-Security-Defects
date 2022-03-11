"""Helper machine learning functions for the SVP pipeline"""
import warnings
import pickle
import math
import time
import os
import pandas as pd
import numpy as np
from statistics import mean
from collections import Counter
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, roc_curve
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities
from pulearn import ElkanotoPuClassifier, BaggingPuClassifier
from data_utils import Data
warnings.filterwarnings('ignore')	 # Ignore warnings
n_jobs = 2


def resampling(x_train, y_train, method):
    """
    Oversample a training dataset.
    @param: method - ['over', 'smote']
    """
    if method == 'over':
        sampler = RandomOverSampler(random_state=42)
    if method == 'smote':
        # K_neighbors must be smaller than number of samples for smallest class.
        sampler = SMOTE(k_neighbors=2, random_state=42)
    return sampler.fit_resample(x_train, y_train)


def get_model(model_type):
    """Return a classifier and paramater grid"""
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'brf':
        model = BalancedRandomForestClassifier(random_state=42)
    elif model_type == 'xgb':
        model = XGBClassifier(random_state=42)
    else:
        return -1
    model_parameters = {'max_depth': (2, 15), 'max_leaf_nodes': (2, 500),
                        'n_estimators': (50, 500), 'max_features': (0.01, 0.7)}
    return model, model_parameters


def tuning_bayes(model_type, param_grid, x_train, y_train, x_val, y_val, regularize=False, testing=False):
    """Tune model hyperparameters using bayesian optimisation"""
    # Rebalance class weights. Reference:
    # https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
    counter = Counter(y_train)
    pos_weight = counter[0] / counter[1]

    def model_evaluation(max_depth, max_leaf_nodes, n_estimators, max_features):
        params = {'max_depth': int(max_depth),
                  'max_leaf_nodes': int(max_leaf_nodes),
                  'n_estimators': int(n_estimators),
                  'max_features': max_features
                  }
        if model_type == 'rf':
            clf = RandomForestClassifier(
                class_weight='balanced', random_state=42, **params)
        elif model_type == 'brf':
            clf = BalancedRandomForestClassifier(
                class_weight='balanced', random_state=42, **params)
        elif model_type == 'xgb':
            del params['max_leaf_nodes']
            clf = XGBClassifier(
                random_state=42, tree_method='approx', eval_metric='auc',
                scale_pos_weight=pos_weight, **params)
        clf.fit(x_train, y_train)
        if regularize:
            recall = recall_score(y_val, clf.predict(x_val))
            gmean = (len(y_val) * (recall**2)) / ((y_val == 1).sum())
            print(recall, len(y_val), (y_val == 1).sum(), gmean)
            return gmean
        else:
            return roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1])

    # Bayesian Optimization
    v_start = time.time()
    bayesOpt = BayesianOptimization(model_evaluation, param_grid, random_state=42)

    if not testing:
        try:
            bayesOpt.maximize(init_points=25, n_iter=25)
            # Get optimal model
            val_score = bayesOpt.max['target']
            max_params = bayesOpt.max['params']
            params = {}
            for i in max_params:
                if i != 'max_features':
                    params[i] = int(max_params[i])
                else:
                    params[i] = max_params[i]
        except ValueError:
            # If failed to optimize
            testing = True
    if testing:
        # A testing option for increased speed whilst fiddling
        val_score = -1
        params = {'max_depth': 3, 'max_leaf_nodes': 400,
        'n_estimators': 400, 'max_features': 'auto'}
    val_time = time.time() - v_start
    if model_type == 'rf':
        clf = RandomForestClassifier(class_weight='balanced', random_state=42,
                                     **params)
    if model_type == 'brf':
        clf = BalancedRandomForestClassifier(class_weight='balanced',
                                             random_state=42, **params)
    elif model_type == 'xgb':
        del params['max_leaf_nodes']
        clf = XGBClassifier(
            random_state=42, tree_method='approx', eval_metric='auc',
            scale_pos_weight=pos_weight, **params)

    return clf, params, val_time, val_score


def tuning_ocsvm(model_type, param_grid, x_train, x_val, y_val):
    """Tune a One-Class SVM Classifier using Bayesian Optimization."""

    # Eval function
    def model_evaluation(nu):
        params = {'nu': nu}
        clf = OneClassSVM(**params)
        clf.fit(x_train)
        # Evaluate on MCC as AUC is unsupported for ocsvm
        return matthews_corrcoef(y_val, np.where((clf.predict(x_val) == 1), 1, 0))

    # Bayesian Optimization
    v_start = time.time()
    bayesOpt = BayesianOptimization(model_evaluation, param_grid, random_state=42)
    val_time = time.time() - v_start
    bayesOpt.maximize(init_points=15, n_iter=15)
    # Get optimal model
    val_score = bayesOpt.max['target']
    params = bayesOpt.max['params']
    params = {i: params[i] for i in params}
    clf = OneClassSVM(**params)

    return clf, params, val_time, val_score


def tuning_confident_learning(clf, x_train, y_train, x_val=None, y_val=None,
                              label_flipping=False, eval=True, **kwargs):
    """
    Tune confident learning parameters.
    - prune_by_noise_rate vs prune_by_class
        - num_to_remove_per_class (for by_class)
        - thresholds (for by_noise_rate)
    """
    noise_matrices = []
    scores = []

    # for prune_method in ['prune_by_noise_rate', 'prune_by_class']:
    for prune_method in ['prune_by_noise_rate']:
        # Make a noisy label model
        cl_clf = LearningWithNoisyLabels(clf=clf, seed=42, prune_method=prune_method, pulearning=1)
        if prune_method == 'prune_by_class':
            # Number of samples to remove from the negative class
            params = [50, 100, 500, 1000]
            for p in params:
                cl_clf.fit(
                    x_train, y_train, num_to_remove_per_class=[p, 0],
                    label_flipping=label_flipping)
                if eval:
                    scores.append([
                        f'{prune_method}_{str(p)}',
                        evaluate(cl_clf, f'{prune_method}: {p}', x_val, y_val, write=False, **kwargs),
                    ])
                noise_matrices.append([f'{prune_method}_{str(p)}', cl_clf.noise_mask])
        elif prune_method == 'prune_by_noise_rate':
            # Calculate as a percentage of the default threshold.
            psx = estimate_cv_predicted_probabilities(
                X=x_train, labels=y_train, clf=cl_clf, cv_n_folds=2, seed=42)
            thresholds = [mean(psx[:, k][y_train == k]) for k in [0,1]]
            # Lower noise threshold for labels to be considered vulnerable
            # A sample with probability greater than the threshold will be counted
            #   as having that hidden label.
            # params = [thresholds[1]-(i*0.05) for i in range(0,7)]
            params = [thresholds[1]-(i*0.05) for i in range(2,7)]
            for p in params:
                p = 0 if p < 0 else p   # Check greater than zero
                cl_clf.fit(
                    x_train, y_train, thresholds=[thresholds[0], p],
                    label_flipping=label_flipping)
                if eval:
                    scores.append([
                        f'{prune_method}_{str(p)}',
                        evaluate(cl_clf, f'{prune_method}: {p}', x_val, y_val, write=False, **kwargs),
                    ])
                noise_matrices.append([f'{prune_method}_{str(p)}', cl_clf.noise_mask])

    if eval:
        # If evaluating performance, return best classifier
        best_score = max([i[1] for i in scores])
        best_param = [i[0] for i in scores if i[1] == best_score][0]

        # Train the best classifier
        t_start = time.time()
        if 'prune_by_class' in best_param:
            best_clf = LearningWithNoisyLabels(clf=clf, seed=42, prune_method='prune_by_class', pulearning=1)
            best_clf.fit(
                x_train, y_train,
                num_to_remove_per_class=[int(best_param.rsplit('_')[-1]), 0],
                label_flipping=label_flipping)
        if 'prune_by_noise_rate' in best_param:
            best_clf = LearningWithNoisyLabels(clf=clf, seed=42, prune_method='prune_by_noise_rate', pulearning=1)
            best_clf.fit(
                x_train, y_train,
                thresholds=[thresholds[0], float(best_param.rsplit('_')[-1])],
                label_flipping=label_flipping)
        train_time = time.time() - t_start

        return best_clf, best_param, train_time
    else:
        # Else return noise masks for each parameter
        return noise_matrices


def get_pu_classifier(clf, technique):
    """
    Return a pu classifier depending on the specified technique

    Parameters
    ----------
    clf : :obj:sklearn.model
    The base classifier.

    technique : :obj:['adaptation', 'biased']
      Adaptation returns a pu classifier that aims to fix model predictions
      through an adaptation layer that estimates the class prior.
      Biased return a biased pu learning classifier which attempts to make
      predictions more robust through an ensemble bagging method.
    """
    if technique == 'adaptation':
        return ElkanotoPuClassifier(estimator=clf, hold_out_ratio=0.2)
    elif technique == 'biased':
        return BaggingPuClassifier(
            base_estimator=clf, n_estimators=15, max_samples=1/15,
            n_jobs=n_jobs, random_state=42)


def get_threshold(clf, x_val, y_val, pu=False):
    """
    Determine optimal classification threshold.
    Balance sensitivity and specificity (Gmean).
    """

    if pu:
        proba = clf.predict_proba(x_val)
    else:
        proba = clf.predict_proba(x_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, proba)
    J = tpr - fpr    # Youden's J Statistic
    return thresholds[np.argmax(J)]


def evaluate(clf, clf_params, x_test, y_test, threshold=0.5, write=True, pu=False, oneclass=False, **kwargs):
    """Train, Evaluate and Save a Classifier"""
    # Unpack extra arguments
    identifier = kwargs['identifier'] if 'identifier' in kwargs else ''
    ml_identifier = kwargs['ml_identifier'] if 'ml_identifier' in kwargs else ''
    release = kwargs['release'] if 'release' in kwargs else ''
    technique = kwargs['technique'] if 'technique' in kwargs else ''
    model_type = kwargs['model'] if 'model' in kwargs else ''
    feature_type = kwargs['feature'] if 'feature' in kwargs else ''
    train_time = kwargs['train_time'] if 'train_time' in kwargs else 0
    val_time = kwargs['val_time'] if 'val_time' in kwargs else 0
    val_score = kwargs['val_score'] if 'val_score' in kwargs else 0

    # Open the results file
    if not os.path.exists(f"results/ml_results{identifier}.csv"):
        outfile = open(f"results/ml_results{identifier}.csv", 'w')
        outfile.write("release,classifier,parameters,feature_type,technique,val_auc,accuracy,precision,recall,gmean,auc,f1,mcc,recall_u,gmean_pu,val_time,train_time,pred_time\n")
    else:
        outfile = open(f"results/ml_results{identifier}.csv", 'a')

    # Predict (using a pre-determined threshold)
    p_start = time.time()
    if pu:
        y_pred = np.where((clf.predict_proba(x_test) >= threshold), 1, 0)
    elif oneclass:
        y_pred = np.where((clf.predict(x_test) == 1), 1, 0)
    else:
        y_pred = np.where((clf.predict_proba(x_test)[:, 1] >= threshold), 1, 0)
    pred_time = time.time() - p_start
    # Evaluate traditional metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    gmean = math.sqrt(recall*precision)
    # Calculate AUC
    if oneclass:
        auc = 0
    else:
        if pu:
            proba = clf.predict_proba(x_test)
        else:
            proba = clf.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
    # Evaluate PU metrics
    f1_pu = (len(y_pred) * (recall**2)) / ((y_pred == 1).sum())
    if 'original_df' in kwargs:
        # Get vulnerabilities from the unlabelled set
        data = kwargs['original_df'].get_silent(apply_labels=2).get_latent(apply_labels=2)
        unlabelled_vulns = data.get_dataset().vulnerable.tolist()
        unlabelled_vulns = [1 if i == 2 else 0 for i in unlabelled_vulns]
        recall_u = recall_score(unlabelled_vulns, y_pred)
    else:
        recall_u = -1

    output = f"{release},{model_type},\"{clf_params}\",{feature_type}," + \
             f"{technique},{round(val_score,3)},{round(accuracy_score(y_test, y_pred),3)}," + \
             f"{round(precision,3)},{round(recall,3)},{round(gmean,3)}," + \
             f"{round(auc,3)},{round(f1,3)}," + \
             f"{round(matthews_corrcoef(y_test, y_pred),3)}," + \
             f"{round(recall_u,3)},{round(f1_pu,3)}," + \
             f"{round(val_time,3)},{round(train_time,3)},{round(pred_time,3)}\n"
    print('AUC: ', auc)
    print('MCC: ', matthews_corrcoef(y_test, y_pred))
    print('F1_PU: ', f1_pu)
    # Save results
    outfile.write(output)
    # Save model
    if 'ml_identifier' in kwargs:
        filename = f"r{release}_{model_type}_{feature_type}_{technique}{ml_identifier}"
        pickle.dump(clf, open(f"code/svp/models/{filename}.model", "wb"))
    # Save predictions
    if 'original_df' in kwargs:
        test_labels = kwargs['original_df'].get_dataset()
        test_labels['predicted'] = y_pred
        test_labels.to_csv(f"results/prediction_outputs/{filename}.csv", index=False)
    return auc


def get_overall_recall():
    """Append overall recall to the results file in post."""
    # Load the results file.
    df = pd.read_csv('results/ml_results.csv')
    df['recall_o'] = ''

    for index, row in df.iterrows():
        print(index)
        # Load the data
        test_release = row['release'] + 3 if row['release'] == 78 or row['release'] == 79 else row['release'] + 2
        data = Data(dataset='Mozilla').set_release(test_release)
        data = data.get_silent(apply_labels=1).get_latent(apply_labels=1)
        vulns = data.get_dataset().vulnerable.tolist()

        # Get the y_pred
        pred = pd.read_csv(f'results/prediction_outputs/r{row["release"]}_{row["classifier"]}_{row["feature_type"]}_{row["technique"]}.csv')
        y_pred = pred['predicted'].tolist()

        df.at[index, 'recall_o'] = recall_score(vulns, y_pred)
    df.to_csv('results/ml_results.csv', index=False)


if __name__ == '__main__':
    get_overall_recall()
    exit()
