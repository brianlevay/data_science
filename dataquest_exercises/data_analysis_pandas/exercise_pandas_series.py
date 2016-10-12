# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 10:43:23 2016

@author: Brian
"""

# 1. Import data
import pandas as pd

fandango = pd.read_csv("fandango_score_comparison.csv")
print(fandango.head(2))

# 2. Integer index
series_film = fandango['FILM']
series_rt = fandango['RottenTomatoes']

print(series_film.head(5))
print(series_film.head(5))

# 3. Custom index
from pandas import Series

film_names = series_film.values
rt_scores = series_rt.values

series_custom = Series(rt_scores, index=film_names)

# 4. Integer index preserved
fiveten = series_custom[5:10]
print(fiveten)

# 5. Reindexing
original_index = series_custom.index.tolist()
sorted_index = sorted(original_index)
sorted_by_index = series_custom.reindex(index=sorted_index)

# 6. Sorting
sc2 = series_custom.sort_index()
sc3 = series_custom.sort_values()
print(sc2.head(10))
print(sc3.head(10))

# 7. Vectorized operations
series_normalized = series_custom / 20

# 8. Comparing and filtering
criteria_one = series_custom > 50
criteria_two = series_custom < 75

both_criteria = series_custom[criteria_one & criteria_two]

# 9. Alignment
rt_critics = Series(fandango['RottenTomatoes'].values, index=fandango['FILM'])
rt_users = Series(fandango['RottenTomatoes_User'].values, index=fandango['FILM'])

rt_mean = (rt_critics + rt_users) / 2