# -*- coding: utf-8 -*-

import pandas as pd

# read in the data files
all_ages = pd.read_csv("all-ages.csv")
recent_grads = pd.read_csv("recent-grads.csv")

# find the unique major categories in each dataset (assumed to be the same 
# between files)
categories = all_ages['Major_category'].unique()

# create dictionaries to store the totals per major category
aa_cat_counts = dict()
rg_cat_counts = dict()

# loop through the categories, create a boolean list to distinguish which 
# rows in the dataset contain the category, use the boolean list to select 
# the subset of rows, sum the 'Total' values, and store them in the 
# appropriate dictionary
for cat in categories:
    aa_has_cat = (all_ages['Major_category'] == cat)
    aa_in_cat = all_ages.loc[aa_has_cat, 'Total']
    aa_total = aa_in_cat.sum()
    aa_cat_counts[cat] = aa_total
    
    rg_has_cat = (recent_grads['Major_category'] == cat)
    rg_in_cat = recent_grads.loc[rg_has_cat, 'Total']
    rg_total = rg_in_cat.sum()
    rg_cat_counts[cat] = rg_total
    
# this calculates the percentage of recent grads that held low wage jobs
low_wage_percent = recent_grads['Low_wage_jobs'].sum() / recent_grads['Total'].sum()

# all majors, common to both DataFrames
majors = recent_grads['Major'].unique()
rg_lower_count = 0

# this iterates over the list of majors and counts when recent grads
# have a lower unemployment rate
for major in majors:
    aa_has_major = (all_ages['Major'] == major)
    aa_in_major = all_ages.loc[aa_has_major]
    
    rg_has_major = (recent_grads['Major'] == major)
    rg_in_major = recent_grads.loc[rg_has_major]
    
    diff = rg_in_major['Unemployment_rate'].sum() - aa_in_major['Unemployment_rate'].sum()
    if diff < 0:
        rg_lower_count += 1
        
print(rg_lower_count)

