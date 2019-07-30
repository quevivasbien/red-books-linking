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
    '''Takes the OCR'd text from one red book entry and uses regex to extract
    identifiers from it.
     
    blurb is a string that is the OCR'd text.
    filename is a string that is the name of the file the text came from.
    '''
    record = {}
    filename_parse = re.search(r'^DSC_(\d{4})_(\d{1,2})\.txt$', filename)
    try:
        record['Picture'] = int(filename_parse.group(1))
        record['Entry'] = int(filename_parse.group(2))
    except AttributeError:
        print(filename)
        raise
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
    '''Creates a pandas DataFrame with data from OCR'd red book entries.
     
    folder_path is the location of the folder that contains the data you want
     to process. This folder should contain a bunch of .txt files, each one
     containing a single record, and their filenames should be formatted like
     DSC_####_#.txt.
    year is the graduating year of the people in the redbooks you're processing
    '''
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
    for i, r in doc.iterrows():
        s = r['state']
        state = state_equivs.get(s)
        if state is None:
            print('State "{}" not recognized. (City is {})'.format(s, r['city']))
            state = input('Type the name of the state it corresponds to (all lowercase):\n')
            state_equivs[s] = state
        states.append(state)
    doc.state = states
    with open(STATE_EQUIVS, 'w', encoding='utf-8') as fh: # Save aliases
        json.dump(state_equivs, fh, ensure_ascii=False)
    # Append to master data set
    with open(MASTER_DATA, 'a', encoding='utf-8') as fh:
        doc.to_csv(fh, header=False)

def get_and_add(folder_path, year):
    '''Combines previous two functions together.
    '''
    df = get_red_book_data(folder_path, year)
    add_data_to_master(df)

def match_from_master(census_year, state, census_filename):
    '''Finds matches from master data matching the given year and state
    
    state is either a string or list of strings
    '''
    if type(state) is str:
        state = [state]
    # Select records with given year and state from master data
    # Master data should already be in correct format
    doc = pd.read_csv('master_data.csv', index_col=0)
    doc = doc[doc.census_year == census_year]
    doc = pd.concat([doc[doc.state == s] for s in state])
    print('Importing census data from {}...'.format(census_filename))
    census = linking.import_census(census_filename)
    print('Building features for most likely matches...')
    features = linking.assemble_features(doc, census)
    print('Calculating likely matches...')
    matches = linking.get_likely_matches(features)
    return matches