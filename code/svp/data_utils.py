import pandas as pd
import numpy as np
from scipy.sparse import vstack


class Data:
    """
    Load and manipulate a dataset

    Example:
    df = Data('Mozilla').get_dataset()
    """
    def __init__(self, dataset: str):
        """
        Load a dataset.
        Supported options:
        - 'Mozilla': The Mozilla dataset.
        - 'Mozilla_Code': The Mozilla dataset w/ code.
        """

        # Load the data
        if dataset == 'Mozilla':
            self.data_name = 'Mozilla'
            self.df = pd.read_csv('data/mozilla/file_labels_63-84.csv')
            self.df['vulnerable'] = np.where(self.df.vulnerable, 1, 0)
        elif dataset == 'Mozilla_Code':
            self.data_name = 'Mozilla'
            self.df = pd.read_csv('data/mozilla/code_labels_63-84.csv')

    def set_release(self, release):
        """Set the release of a datataset (Project for non-Mozilla data)."""
        self.release = release
        self.df = self.df[self.df.release == release]
        return self

    def get_dataset(self):
        """Return loaded dataset."""
        return self.df

    def get_silent(self, apply_labels=False):
        """Return silent vulnerability labels (Only for Mozilla)."""
        if not hasattr(self, 'release') or self.data_name != 'Mozilla':
            return -1
        silent = pd.read_csv('data/mozilla/silent_labels_63-84.csv')
        silent['vulnerable'] = np.where(silent.vulnerable, 1, 0)
        silent = silent[silent.release == self.release]

        # Apply the additional labels to the base dataset
        def extra_label(row):
            if row['file'] in silent.file.tolist():
                return apply_labels
            else:
                return row['vulnerable']
        if apply_labels:
            self.df['vulnerable'] = self.df.apply(lambda x: extra_label(x), axis=1)
            return self
        else:
            return silent

    def get_latent(self, apply_labels=False):
        """Return latent vulnerability labels (Only for Mozilla)."""
        if not hasattr(self, 'release') or self.data_name != 'Mozilla':
            return -1
        latent = pd.read_csv('data/mozilla/latent_labels_63-84.csv')

        # Check if a release contains a particular latent vulnerability
        def check_latent(row):
            if row['first_appear'] <= self.release and self.release <= row['last_appear']:
                return 1
            return 0
        latent['vulnerable'] = latent.apply(lambda x: check_latent(x), axis=1)
        latent = latent[latent.vulnerable == 1]

        # Apply the additional labels to the base dataset
        def extra_label(row):
            if row['file'] in latent.file.tolist():
                return apply_labels
            else:
                return row['vulnerable']
        if apply_labels:
            self.df['vulnerable'] = self.df.apply(lambda x: extra_label(x), axis=1)
            return self
        else:
            return latent

    def get_features(self, feature_type, release, partition=''):
        """
        Return specified features for the loaded dataset.
        @param: feature_type - ['sm', 'bow', 'bert']
        @param: partition - ['', '_val', '_test']
        """

        feature_path = 'code/svp/feature_sets/'
        if feature_type == 'sm':
            features = pd.read_csv(f"{feature_path}software_metrics_release_{release}.csv")
            features['file'] = '-'
            features = features.fillna(0)
            features = features.drop(columns='file').values
        elif feature_type == 'bow':
            features = pd.read_pickle(f"{feature_path}bow_release_{release}{partition}.pkl")
            features['file'] = '-'
            features = vstack(features.drop(columns='file').squeeze().values)
        elif feature_type == 'bert':
            features = pd.read_pickle(f"{feature_path}bert_release_{release}{partition}.pkl")
            features['file'] = '-'
            features = features.drop(columns='file').values
            features = np.asarray([x[0] for x in features])
        return features


if __name__ == '__main__':
    data = Data(dataset='Mozilla').set_release(63)
    print(data.get_features('bow'))
    df = data.get_dataset()
    print(df)
