"""
Created 2 July 2019 by mckaydjensen

Finds likely matches between red books and class reports
"""

import re
import pandas as pd
import numpy as np
import recordlinkage
import joblib

from recordlinkage.index import SortedNeighbourhood
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import is_classifier

# TODO: Retrain classifier
DEFAULT_CLASSIFIER = 'classifier.joblib'


class MatchFinder(object):
    """Includes methods for comparing entries in class reports & red books
    and determining which entries likely refer to the same people.
    """

    def __init__(self, class_report_file, red_books_file):
        """

        :param class_report_file: filename of CSV containing class report, e.g. from update_and_merge.py
        :param red_books_file: filename of master redbook document (from create_master.py)
        """
        # Load class report (cr) and red book (rb) data
        self.cr = pd.read_csv(class_report_file, index_col='PID')
        self.cr = self.cr[~self.cr.index.duplicated(keep='first')]
        self.cr = self.cr[pd.notna(self.cr.index)]
        # TODO: deal more gracefully with duplicated and absent PIDs
        self.rb = pd.read_csv(red_books_file, index_col='index')
        # Compile factors for identifying possible matches
        self.factors_cr = self.get_factors_cr()
        self.factors_rb = self.get_factors_rb()
        self.candidate_pairs = None
        self.features = None
        self.clf = self.load_classifier()

    @staticmethod
    def get_last_cr(name):
        search = re.search(r'[A-Za-z\'\-]+(?=$|,.{0,4}$)', name)
        if search:
            return search.group()
        else:
            try:
                return name.split()[-1]
            except IndexError:
                return ''

    @staticmethod
    def get_first_cr(name):
        try:
            return name.split()[0]
        except IndexError:
            return ''

    def get_factors_cr(self):
        """Assembles variables from class report that will be used for matching
        """
        names = self.cr['name'].apply(lambda x: x.upper() if isinstance(x, str) else '')
        factors_cr = pd.DataFrame(index=self.cr.index)
        factors_cr['last'] = names.apply(self.get_last_cr)
        factors_cr['first'] = names.apply(self.get_first_cr)
        factors_cr['full'] = names
        factors_cr['year'] = self.cr['year']
        factors_cr['high_school'] = self.cr['high_school_name']
        return factors_cr

    @staticmethod
    def get_first_last_rb(name):
        name = re.sub(r',?\s?\(?e\. ?s\.\)?', '', name, flags=re.IGNORECASE).rstrip(',')
        name = re.sub(r', (?:JR\.?|\d\S{0,3}|[A-Z]{1,3})$', '', name.strip(), flags=re.IGNORECASE)  # Remove suffixes
        # Look for name of form LAST, FIRST(, JR.)
        search = re.search(r'^([^,]+), ?(\S.*?)$', name)
        if search is not None:
            return search.group(2).strip(), search.group(1).strip()
        # Look for name of form FIRST LAST(, JR.)
        search = re.search(r'^(.*?)\s([^\s,]+)$', name)
        if search is not None:
            return search.group(1).strip(), search.group(2)
        # Give up
        return '', ''

    @staticmethod
    def get_full_rb(name):
        parts = name.split(', ')
        if len(parts) > 1:
            return ', '.join(parts[1:]) + ' ' + parts[0]
        else:
            return name

    def get_factors_rb(self):
        """Assembles variables from red book that will be used for matching
        """
        names = self.rb['name'].apply(lambda x: x.upper() if isinstance(x, str) else '')
        factors_rb = pd.DataFrame(index=self.rb.index)
        factors_rb['first'], factors_rb['last'] = tuple(zip(*names.apply(self.get_first_last_rb)))
        factors_rb['full'] = names.apply(self.get_full_rb)
        factors_rb['year'] = self.rb['year']
        factors_rb['high_school'] = self.rb['high_school']
        return factors_rb

    def set_candidate_pairs(self):
        """Determine which records are possible matches
        The indices of the possible matches are saved in a Pandas MultiIndex as self.candidate_pairs
        """
        # Find similar names
        name_indexer = recordlinkage.Index()
        name_indexer.add(SortedNeighbourhood(left_on='first', window=7))
        name_indexer.add(SortedNeighbourhood(left_on='last', window=5))
        name_pairs = name_indexer.index(self.factors_cr, self.factors_rb)
        # Find similar years +- 2
        year_indexer = SortedNeighbourhood(left_on='year', window=5)
        year_pairs = year_indexer.index(self.factors_cr, self.factors_rb)
        # Take intersection (names *and* years must be similar)
        candidate_pairs = name_pairs.intersection(year_pairs)
        print('Number of candidate pairs is {}'.format(len(candidate_pairs)))
        self.candidate_pairs = candidate_pairs

    def get_age_score(self, pid, idx, failure_value=2):
        """Estimates the age difference between individuals in two records.

        :param pid: index of the entry from class report
        :param idx: index of the entry from red book
        :param failure_value: value to return if something goes wrong
        :return: estimated distance between ages
        """
        birth_date = self.cr.loc[pid]['birthDate']
        try:
            birth_year = int(re.search(r'\d{4}$', birth_date).group())
        except (TypeError, AttributeError):
            return failure_value
        try:
            rb_year = int(self.rb.loc[idx]['year'])
            rb_age = int(self.rb.loc[idx]['age'])
        except (TypeError, ValueError):
            return failure_value
        est_birth_year = rb_year - rb_age - 3  # 3 since books published at end of freshman year
        return abs(birth_year - est_birth_year)

    def get_age_scores(self, multi_index=None, normalize=False):
        """Gets scores for the similarity of ages of candidate pairs

        :param multi_index: corresponds to pairs of indices in cr and rb data. Default is self.candidate_pairs
        :param normalize: If True, will transform all scores to be between 0 and 1
        :return: a Pandas Series of scores, one for each pair in the given multi_index
        """
        if multi_index is None:
            if self.candidate_pairs is not None:
                multi_index = self.candidate_pairs
            else:
                raise ValueError('No MultiIndex provided')
        scores = pd.Series([self.get_age_score(pid, idx) for pid, idx in multi_index],
                           index=multi_index)
        if normalize:
            scores = 1 - np.exp(-scores/4)
        return scores

    def assemble_features(self, radius=0.4):
        """Create a Pandas DataFrame of features, for machine learning to identify
        which candidate pairs are actually matches.

        :param radius Candidates whose (first_score, last_score) vector is more than radius from (1,1) will be dropped
        :return: A DataFrame of features with columns first_score, last_score, full_score, year_score, age_score
        """
        if self.candidate_pairs is None:
            self.set_candidate_pairs()
        # Compute similarity scores for the included factors
        comparer = recordlinkage.Compare()
        comparer.string(left_on='first', right_on='first',
                        method='jarowinkler', label='first_score')
        comparer.string(left_on='last', right_on='last',
                        method='jarowinkler', label='last_score')
        comparer.string(left_on='full', right_on='full', label='full_score')
        comparer.numeric(left_on='year', right_on='year', label='year_score')
        comparer.string(left_on='high_school', right_on='high_school', label='high_school_score')
        # TODO: Add more features?
        print('Computing similarity scores...')
        features = comparer.compute(self.candidate_pairs,
                                    self.factors_cr, self.factors_rb)
        # Remove obviously bad matches
        features = features[(features.first_score - 1)**2 + (features.last_score - 1)**2 < radius**2]
        # Compile age comparison scores
        features['age_score'] = self.get_age_scores()
        self.features = features
        return features

    @staticmethod
    def cluster(features):
        """Uses KMeans clustering to try to separate features into two groups.
        This doesn't work very well for record linking.
        """
        kmeans = KMeans(n_clusters=2, random_state=0)
        labeled = kmeans.fit_predict(features)
        return pd.Series(labeled, index=features.index)

    def build_training_data(self, features=None, max_size=1000):
        """Generates a set of data that you can manually label and use to train a classifier."""
        if features is None:
            if self.features is None:
                features = self.assemble_features()
            else:
                features = self.features
        if len(features) > max_size:
            features = features.sample(max_size)
        cr_sample = self.cr.loc[features.index.get_level_values(0)]
        rb_sample = self.rb.loc[features.index.get_level_values(1)]
        data = pd.concat((features.reset_index(drop=True),
                          cr_sample.reset_index(drop=True),
                          rb_sample.reset_index(drop=True)),
                         axis=1)
        data.index = features.index
        return data

    @staticmethod
    def train_classifier(features, labels, saveas=None):
        """Trains a RandomForestClassifier that you can use to evaluate candidate pairs."""
        clf = RandomForestClassifier(50, max_depth=5, random_state=0, class_weight='balanced')
        clf.fit(features, labels)
        if saveas is not None:
            joblib.dump(clf, saveas)
        return clf

    @staticmethod
    def load_classifier(source=DEFAULT_CLASSIFIER):
        """Load an sklearn classifier from a joblib file."""
        try:
            clf = joblib.load(source)
            assert(is_classifier(clf))
        except:
            print('Classifier load unsuccessful. Setting to None.')
            clf = None
        return clf

    def find_likely_matches(self, threshold=0.4, only_best=True, append_names=False, save_as=None):
        """Finds likely matches between class reports and red book data.
        Will automatically find candidate pairs and compute similarity features before using
        the provided sklearn classifier to narrow down likely matches.

        :param threshold: Pairs with match scores (predicted probabilities) less than this will be excluded.
        :param only_best: If True, will only include the best match if more than one match is found for a cr entry
        :param append_names: If True, will add columns to the output with the names of the predicted matches.
        :return: A Pandas DataFrame with columns for the class report & red book indices and their match scores
        """
        if self.features is None:
            self.assemble_features()
        probas = self.clf.predict_proba(self.features)[:, 1]
        matches = pd.DataFrame({'PID': self.features.index.get_level_values(0),
                                'index': self.features.index.get_level_values(1),
                                'confidence': probas}
                               )[probas > threshold]
        if only_best:
            # TODO: Figure out how to do this in a reasonable fashion
            matches.sort_values(by='confidence', ascending=False, inplace=True)
            matches.drop_duplicates(subset='PID', keep='first', inplace=True)
        if append_names:
            matches['cr_name'] = [self.cr.loc[pid]['name'] for pid in matches['PID']]
            matches['rb_name'] = [self.rb.loc[idx]['name'] for idx in matches['index']]
        matches.reset_index(drop=True, inplace=True)
        if save_as is not None:
            matches.to_csv(save_as, index=False)
        return matches


def match(class_report_file, red_books_file, output_file=None):
    mf = MatchFinder(class_report_file, red_books_file)
    matches = mf.find_likely_matches(save_as=output_file)
    return matches


def fancy_drop(df):
    """This doesn't work right now! Does not retain any entries where both indices are duplicates!"""
    # Identify duplicated PIDs
    pid_groups = df.groupby('PID')
    pid_counts = pid_groups.size()
    duplicated_pids = pid_counts.index[pid_counts > 1]
    # Identify duplicated redbook indices
    idx_groups = df.groupby('index')
    idx_counts = idx_groups.size()
    duplicated_idxs = idx_counts.index[idx_counts > 1]
    # Retain entries that have either a unique PID or a unique redbook index
    return pd.DataFrame([r for i, r in df.iterrows() if (i not in duplicated_pids) or (i not in duplicated_idxs)])


def merge_match_files(filenames, drop='strict', save_as=None):
    df = pd.concat((pd.read_csv(f) for f in filenames))
    df.sort_values(by='confidence', ascending=False, inplace=True)
    if drop == 'strict':
        df.drop_duplicates(subset='PID', keep='first', inplace=True)
        df.drop_duplicates(subset='index', keep='first', inplace=True)
    elif drop == 'fancy':
        df = fancy_drop(df)
    if save_as is not None:
        df.to_csv(save_as, index=False)
    return df
