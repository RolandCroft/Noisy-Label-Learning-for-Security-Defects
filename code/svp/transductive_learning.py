"""
Evaluate PU models on their ability to classify points within the training set.
Performance values are an approximation:
    Recall - The number of holdout SVs predicted.
    Precision - The ratio of SVs predicted in the noisy set.
                (Assumes files are negative)
    G-Mean PU - Based on the recall.
"""
import pandas as pd
import numpy as np
import pickle
from data_utils import Data
from pu_estimator import PU_Estimator

classifier = 'rf'
feature_type = 'bert'

# Open the results file
outfile = open("results/transductive_results.csv", 'w')
outfile.write("release,feature_type,technique,precision,recall,gmean_pu\n")

# For each release
for release in range(63, 83):
    print(release)
    if release == 80:
        continue
    # Get the data labels
    data = Data('Mozilla').set_release(release)
    orig_labels = data.get_dataset().vulnerable.tolist()
    data.get_silent(apply_labels=2).get_latent(apply_labels=2)

    # Load the embeddings
    embeddings = data.get_features(feature_type, release)
    vul_labels = data.get_dataset().vulnerable.tolist()

    # For the three transductive approaches
    for technique in ['pruning', 'propagation1', 'oneclass']:
        if technique == 'propagation1':
            pu = PU_Estimator('cosine')
            x_train, y_train = pu.one_stage_pu(
                embeddings, np.array(orig_labels), 1.0)
            rn = pu.rn
        elif technique == 'pruning':
            clf = pickle.load(open(f"code/svp/models/r{release}_{classifier}_{feature_type}_{technique}.model", "rb"))
            rn = ~clf.noise_mask
        elif technique == 'oneclass':
            clf = pickle.load(open(f"code/svp/models/r{release}_ocsvm_{feature_type}_{technique}.model", "rb"))
            y_pred = clf.predict(embeddings)
            rn = np.where((y_pred == -1), 1, 0)

        # Calculate metrics
        tp, fp, fn, tn = 0, 0, 0, 0
        count, pos_count = 0, 0
        for j in range(len(vul_labels)):
            if vul_labels[j] == 1:
                continue
            count += 1
            if vul_labels[j] == 2:
                pos_count += 1
            # True positive
            if (not rn[j]) and vul_labels[j] == 2:
                tp += 1
            # False positive
            if (not rn[j]) and vul_labels[j] == 0:
                fp += 1
            # False negative
            if rn[j] and vul_labels[j] == 2:
                fn += 1
            # True negative
            if rn[j] and vul_labels[j] == 0:
                tn += 1
        recall = tp / (tp+fn)
        precision = tp / (tp+fp) if tp + fp > 0 else 0
        f1_pu = (count * (recall**2)) / (pos_count)

        output = f"{release},{feature_type},{technique},{round(precision, 3)},{round(recall, 3)},{round(f1_pu, 3)}\n"
        outfile.write(output)
