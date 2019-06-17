# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:10:37 2019

@author: mckaydjensen
"""

import re
import os
import pandas as pd
import json

import linking

# File containing dictionary of state aliases
STATE_EQUIVS = 'state_equivs.json'
# File containing master set of red book data
MASTER_DATA = 'master_data.csv'

def process_blurb(blurb, filename):
    record = {}
    filename_parse = re.search(r'^DSC_(\d{4})_(\d)\.txt$', filename)
    record['Picture'] = int(filename_parse.group(1))
    record['Entry'] = int(filename_parse.group(2))
    name_search = re.search(r'(?:^|\n)([^,]+,[^,]+?(?:, J\S)?)(?:\.|, Age)', blurb)
    if name_search:
        record['Name'] = name_search.group(1)
    age_search = re.search(r'Age:\s?(\d\d)', blurb)
    if age_search:
        record['Age'] = int(age_search.group(1))
    seco_add_search = re.search(r'Age: ?\S+\.?\s+(.{1,20}?)\n\S+ Address', blurb)
    if seco_add_search:
        record['Secondary Address'] = seco_add_search.group(1)
    home_add_search = re.search(r'Home Address(?:\s\S+)?\s?: ?([^\n]+)', blurb)
    if home_add_search:
        record['Home Address'] = home_add_search.group(1)
    coll_add_search = re.search(r'ollege Address(?:\s\S+)?\s?: ?([^\n]+)', blurb)
    if coll_add_search:
        record['College Address'] = coll_add_search.group(1)
    prep_at_search = re.search(r'[Pp]repared at.? ?([^\n]+)', blurb)
    if prep_at_search:
        record['Prepared at'] = prep_at_search.group(1)
    activities_search = re.search(r'Activities: (.*?)(?:Service|$)', blurb, flags=re.DOTALL)
    if activities_search:
        record['Activities'] = activities_search.group(1).strip().replace('\n', ' ')
    service_search = re.search(r'[Ss]ervice [Rr]ecord: (.*?)$', blurb, flags=re.DOTALL)
    if service_search:
        record['Service Record'] = service_search.group(1).strip().replace('\n', ' ')
    return record

def get_red_book_data(folder_path, year):
    filenames = [f for f in os.listdir(folder_path) if f[:3] == 'DSC']
    records = []
    for f in filenames:
        with open(os.path.join(folder_path, f), 'r', encoding='utf-8') as fh:
            blurb = fh.read()
        record = process_blurb(blurb, f)
        records.append(record)
    df = pd.DataFrame(records)
    df['Year'] = [year] * len(df)
    # Set index as YEAR_PICTURE_ENTRY so it is unique for each record
    df.index = ['{}_{}_{}'.format(r['Year'], r['Picture'], r['Entry']) \
                for i, r in df.iterrows()]
    return df

def add_data_to_master(data):
    # Format data to contain only features useful for matching to census
    doc = linking.format_doc(data)
    # Replace state column with standardized state names
    with open(STATE_EQUIVS, 'r', encoding='utf-8') as fh:
        state_equivs = json.load(fh)
    states = []
    for s in doc.state: # Lookup alias
        state = state_equivs.get(s)
        if state is None:
            state = input('State {} not recognized. Type the name of the state it corresponds to (all lowercase): '.format(s))
            state_equivs[s] = state
        states.append(state)
    doc.state = states
    with open(STATE_EQUIVS, 'w', encoding='utf-8') as fh: # Save aliases
        json.dump(state_equivs, fh, ensure_ascii=False)
    # Append to master data set
    with open(MASTER_DATA, 'a', encoding='utf-8') as fh:
        doc.to_csv(fh, header=False)

def match_from_master(census_year, state, census_filename):
    # Select records with given year and state from master data
    # Master data should already be in correct format
    doc = pd.read_csv('master_data.csv', index_col=0)
    doc = doc[(doc.census_year == census_year) & (doc.state == state)]
    print('Importing census data from {}...'.format(census_filename))
    census = linking.import_census(census_filename)
    print('Building features for most likely matches...')
    features = linking.assemble_features(doc, census)
    print('Calculating likely matches...')
    matches = linking.get_likely_matches(features)
    return matches