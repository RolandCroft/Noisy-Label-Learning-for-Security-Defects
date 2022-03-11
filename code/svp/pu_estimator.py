import numpy as np
from sklearn.neighbors import NearestCentroid
from scipy.spatial.distance import cosine
from ml_utils import evaluate


class PU_Estimator:
    """
    One and Two Stage PU Learning Estimators.
    Uses Nearest-Centroids to identify reliable negatives (context-aware).
    """

    def __init__(self, metric):
        self.model = NearestCentroid()
        self.metric = metric

    def one_stage_pu(self, X, y, alpha=1.0):
        """
        Identify reliable negatives of a dataset.
        Use nearest centroid, as the majority of unlabelled posts are expected
        to be negative. A point that is closer to the unlabelled centroid
        than the positive centroid will be considered a reliable negative.

        Parameters
        ----------
        metric: :obj:['cosine', 'euclidean']
        The measure of distance between two centroids.
        Use cosine similarity for NLP features, otherwise euclidean.
        """

        # Identify positive and unlabeled centroids
        self.rn = []
        centroids = self.model.fit(X, y).centroids_
        # Calculate centroid distance
        for c, i in enumerate(X):
            # Convert sparse matrix to dense
            try:
                i = i.todense()
            except AttributeError:
                pass
            # Calculate distance
            if self.metric == 'cosine':
                dist = cosine(i, centroids[0]) < (alpha * cosine(i, centroids[1]))
            elif self.metric == 'euclidean':
                dist = np.linalg.norm(i - centroids[0]) < (alpha * np.linalg.norm(i - centroids[1]))
            if dist or y[c] == 1:
                self.rn.append(True)
            else:
                self.rn.append(False)
        self.rn = np.array(self.rn)

        # Only keep reliable positives and negatives.
        return X[self.rn], y[self.rn]

    def two_stage_pu(self, clf, X, y, x_val, y_val, **kwargs):
        """
        Train a classifier using only the reliable positives and negatives.
        Tune the pu_model using different values of alpha.
        """

        # Tune the value of alpha
        scores = []
        for alpha in [0.8, 0.9, 1.0, 1.1, 1.2]:
            x_train, y_train = self.one_stage_pu(X, y, alpha)
            clf.fit(x_train, y_train)
            scores.append([
                f'alpha_{alpha}',
                evaluate(clf, f'alpha: {alpha}', x_val, y_val, write=True, **kwargs),
            ])

        best_score = max([i[1] for i in scores])
        best_param = [i[0] for i in scores if i[1] == best_score][0]
        x_train, y_train = self.one_stage_pu(x_train, y_train, float(best_param.split('_')[1]))
        return clf.fit(x_train, y_train), best_param

    def label_propagation(self, clf, x_train, y_train, x_val, y_val, threshold=0.5, **kwargs):
        """
        Perform label propagation on the remaining unlabelled samples
        using the 2 stage pu model.
        """

        # Perform 2-stage pu learning
        pu_model, param = self.two_stage_pu(clf, x_train, y_train, x_val, y_val, **kwargs)

        # Propagate labels using the 2-stage pu model.
        predicted = np.where((pu_model.predict_proba(x_train)[:, 1] >= threshold), 1, 0)

        # Update propagated labels
        for c, p in enumerate(predicted):
            if p == 1:
                y_train[c] = p

        # Train a new model using the propagated labels
        return clf.fit(x_train, y_train), param
