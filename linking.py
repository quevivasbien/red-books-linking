# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:55:52 2019

@author: mckaydjensen
"""

import sys
sys.path.append(r'R:\JoePriceResearch\Python\Anaconda3\Lib\site-packages')

import pandas as pd
import numpy as np
import recordlinkage
import re

from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

STATE_FILES = 'R:/JoePriceResearch/record_linking/data/census_1920/data/state_files/'
COLUMNS = ['ark1920', 'event_township', 'pr_age', 'pr_name_gn', 'pr_name_surn']



def last_first_names(names):
    '''Takes names from red book data and splits into first and last names
    
    names is an iterable of strings formatted like "SURNAME, FIRST NAMES"
    returns a list of surnames and a list of given names, both with the same
     length as the input names
    '''
    surnames = []
    given_names = []
    for name in names:
        if type(name) is float:
            surnames.append('')
            given_names.append('')
            continue
        n = name.split(', ')
        if len(n) >= 2:
            surnames.append(n[0])
            gn = n[1].rstrip('.').split()
            if len(gn) > 1:
                # If there's a middle name, just take its first initial
                given_names.append('{} {}'.format(gn[0], gn[1][0]))
            else:
                given_names.append(gn[0])
        else:
            surnames.append(name)
            given_names.append('')
    return surnames, given_names


def cities_states(addresses):
    '''Takes a list of (home) addresses from red book data and tries to extract
    city and state names for each address.
    
    addresses is an iterable of strings
    returns a list of cities and a list of states
    '''
    cities = []
    states = []
    for address in addresses:
        if type(address) is float: # np.nan is a float type
            cities.append('')
            states.append('')
            continue
        chunks = re.split(r',\.? ', address)
        if len(chunks) == 1:
            if chunks[0][0] in '0123456789': # First chunk is a street number
                city = ''
                state = ''
            else:
                city = chunks[0].rstrip('.')
                state = ''
        elif len(chunks) == 2:
            if chunks[0][0] in '0123456789':
                city = chunks[1].rstrip('.')
                state = '' # ! Note to self: Come back to this
                # When state is excluded, it's usually Massachusetts
            else:
                city = chunks[0]
                state = chunks[1]
        else:
            state = chunks[-1]
            if 'Co' in chunks[-2] and chunks[0][1] not in '0123456789':
                city = chunks[-3] # Second-to-last chunk is a county
            else:
                city = chunks[-2]
        city = city.rstrip('.')
        cities.append(city)
        state = state.replace('.', '').strip()
        states.append(state)
    return cities, states


def census_age(years, ages, delta_year=3):
    '''Determines what census year to match the red book data to. Also guesses
    the age of the people in the data during that census year.
    
    Params:
    years is an iterable of ints, corresponds to graduation yr for each record
    ages is an iterable of ints, is the age of each person when the red book
     was published
    delta_year is the difference in years between when the red book was
     published and when the people in it would graduate. The books were
     published at the end of freshman year, so delta_year should be 3.
    
    Returns:
    census_years is a list of census years to try to match in
    year_diffs is the difference in years between the previous census year and
     the year the red book was published
    est_ages is a list of estimates for how old each person was during the
     census year
    '''
    census_years = []
    year_diffs = []
    est_ages = []
    for (year, age) in zip(years, ages):
        year -= delta_year
        year_difference = year % 10
        year_diffs.append(year_difference)
        census_years.append(year - year_difference) # Rounds down to nearest 10
        if pd.notna(age):
            age = int(re.sub('\D', '', age))
            est_ages.append(age - year_difference)
        else:
            est_ages.append(age)
    return census_years, year_diffs, est_ages

        
def format_doc(filename):
    '''Imports a CSV of red book data and puts it in a format that will be used
    for matching.
    '''
    df = pd.read_csv(filename, usecols=['Year', 'Name', 'Age', 'Home Address'])
    # Split name column into first & last names
    surname, given_name = last_first_names(df['Name'])
    # Try to get city & state from Home Address column
    city, state = cities_states(df['Home Address'])
    # Figure out closest census year and age in that year
    # Also determine years since most recent census
    census_year, years_since_census, age = census_age(df['Year'], df['Age'])
    return pd.DataFrame({'surname': surname,
                         'given_name': given_name,
                         'city': city,
                         'state': state,
                         'census_year': census_year,
                         'years_since_census': years_since_census,
                         'est_age': age})
    


# Expect this to take about 1.5 mins for each GB of the file you're importing
def import_census(filename):
    '''Imports census data in .dta format (e.g. 'massachusetts.dta'), drops
    columns not used for matching, changes a bit of formatting, and returns
    a Pandas data frame.
    '''
    # Import in chunks of 500000 at a time so we don't eat up all the memory
    reader = pd.read_stata(STATE_FILES + filename, chunksize=500000,
                           columns=COLUMNS)
    df = pd.DataFrame()
    for chunk in reader:
        print('Read chunk from {}...'.format(filename))
        # Cast ages as numeric, not as str
        chunk.pr_age = pd.to_numeric(chunk.pr_age)
        # Remove individuals older than 30
        df = df.append(chunk[chunk.pr_age <= 30])
    # Capitalize names to match format in Red Books
    df.pr_name_gn = df.pr_name_gn.apply(
            lambda x: x.upper() if type(x) is str else np.nan)
    df.pr_name_surn = df.pr_name_surn.apply(
            lambda x: x.upper() if type(x) is str else np.nan)
    # Remove ward numbers from township column
    df.event_township = df.event_township.apply(
            lambda x: re.sub(r' Ward \d+', '', x) if type(x) is str else np.nan)
    return df


def assemble_features(doc, census):
    '''Takes red book and census data, identifies likely matches, and
    creates features for those likely pairs that can be used for machine
    learning to figure out if the pairs are in fact matches.
    
    doc and census are Pandas data frames returned from format_doc() and
     import_census(), respectively
    '''
    # Find candidate matches based on similar surnames
    indexer = recordlinkage.Index()
    indexer.sortedneighbourhood(left_on='surname', right_on='pr_name_surn',
                                window=5)
    candidate_pairs = indexer.index(doc, census)
    print('Number of candidate pairs is {}.'.format(len(candidate_pairs)))
    # Assign match scores based on name, age, and city similarity
    compare_cl = recordlinkage.Compare()
    compare_cl.string('surname', 'pr_name_surn', label='surname_score')
    compare_cl.string('given_name', 'pr_name_gn', label='given_name_score')
    compare_cl.numeric('est_age', 'pr_age', label='age_score', scale=2)
    compare_cl.string('city', 'event_township', method='jarowinkler',
                      label='city_score')
    print('Computing similarity scores...')
    features = compare_cl.compute(candidate_pairs, doc, census)
    # Start by removing possible matches w/average score < 0.5
    print('Removing unlikely matches...')
    features = features[(features.surname_score + features.given_name_score +
                         features.age_score + features.city_score)/4 > 0.5]
    print('Adding features...')
    # Add years since last census as a feature
    features['years_since_census'] = [doc['years_since_census'].loc[i[0]] \
                                                     for i in features.index]
    # Is the city on the census Cambridge?
    features['is_cambridge'] = [int(census['event_township'].loc[i[1]] \
                                     == 'Cambridge') for i in features.index]
    # Does the city start with North/South/East/West?
    features['cardinal_prefix'] = [int(doc['city'].loc[i[0]][:4] in \
                 ['Nort', 'Sout', 'East', 'West']) for i in features.index]
    return features


def match_doc_with_census(doc, census):
    '''Attempts to find matches between red book data and census data.
    This doesn't work very well. Prefer classify() function below.
    '''
    # Assemble features useful for determining if records are a match
    features = assemble_features(doc, census)
    # Use unsupervised Expectation/Conditional Maximization algorithm
    # to evaluate possible matches.
    clf = recordlinkage.ECMClassifier(binarize=0.5)
    matches = clf.fit_predict(features)
    return list(matches)


def build_training_set(features, doc, census):
    features_random = features.sample(1000)
    features_good = features[(features.surname_score + features.given_name_score +
                              features.age_score + features.city_score)/4 > 0.70]
    features_sample = pd.concat((features_good, features_random)).drop_duplicates()
    candidates = pd.DataFrame([pd.concat((doc.loc[i[0]], census.loc[i[1]])) \
                               for i, r in features_sample.iterrows()])
    candidates.index = features_sample.index
    return pd.concat((features_sample, candidates), axis=1)

'''Notes for record linking strategy
According to GridSearchCV,
RandomForestClassifier(class_weight={0:0.25, 1:0.75}, max_depth=10, n_estimators=50)
works pretty well with balanced accuracy score of 0.866
Outperforms logistic regression
Outperforms simple neural net as well
'''

def train_classifier(features, labels, saveas=None):
    '''Trains an sklearn Random Forest Classifier for determining matches
    between red book and census data.
    
    features is a Pandas DataFrame formatted like the one returned from
     assemble_features()
    labels is a Pandas DataFrame (or Series) with only one column of 0's (not a
     match) and 1's (match). The index on labels should be the same as the one
     on features.
    saveas can be set as a filename string, in which case the trained model
     will be saved in joblib format to that file location.
    returns an sklearn classifier
    '''
    clf = RandomForestClassifier(class_weight='balanced',
                                 max_depth=10, n_estimators=50)
    clf.fit(features, labels)
    if saveas is not None:
        dump(clf, saveas)
    return clf

def classify(features, clf='classifier.joblib', cutoff=0.25):
    '''Makes predictions about whether candidate pairs are matches.
    
    features is a DataFrame returned from assemble_features()
    clf is either a filename of a joblib file for an sklearn classifier or
     an sklearn classifier object
    if the classifier predicts a probability for a given sample greater than
     the cutoff value, that sample will be classified as a match. Lower cutoffs
     will give higher recall but lower precision.
    '''
    if type(clf) is str:
        clf = load(clf)
    # Otherwise assume clf is sklearn classifier
    probas = clf.predict_proba(features)[:,1]
    predictions = (probas > cutoff).astype('int')
    return pd.DataFrame(data={'prediction': predictions, 'confidence': probas},
                        index=features.index)
    
def get_likely_matches(features, just_indices=False):
    results = classify(features)
    positive_results = results[results.prediction == 1]
    if just_indices:
        return list(positive_results.index)
    else:
        return positive_results.confidence
    