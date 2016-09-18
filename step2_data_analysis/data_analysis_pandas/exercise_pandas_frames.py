# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:25:05 2016

@author: Brian
"""

# 1. Shared index
import pandas as pd

fandango = pd.read_csv("fandango_score_comparison.csv")

print(fandango.head(2))
print(fandango.index)

# 2. Selecting using integer index
first_last = fandango.iloc[[0,len(fandango.index)-1]]

# 3. Custom index
fandango_films = fandango.set_index('FILM',drop=False)
print(fandango_films.index)

# 4. Selection using custom index
movies = ['The Lazarus Effect (2015)', 'Gett: The Trial of Viviane Amsalem (2015)', 'Mr. Holmes (2015)']
best_movies_ever = fandango_films.loc[movies]

# 5. Apply() over columns
# DEMO PROVIDED
import numpy as np

types = fandango_films.dtypes
float_columns = types[types.values == 'float64'].index
float_df = fandango_films[float_columns]
deviations = float_df.apply(lambda x: np.std(x))
print(deviations)

# 6. Apply() over columns, practice
double_df = float_df.apply(lambda x: x*2)
print(double_df.head(1))

halved_df = float_df.apply(lambda x: x/2)
print(halved_df.head(1))

# 7. Apply() over rows
rt_mt_user = float_df[['RT_user_norm', 'Metacritic_user_nom']]
rt_mt_deviations = rt_mt_user.apply(lambda x: np.std(x), axis=1)
print(rt_mt_deviations[0:5])

rt_mt_means = rt_mt_user.apply(lambda x: np.mean(x), axis=1)
print(rt_mt_means[0:5])