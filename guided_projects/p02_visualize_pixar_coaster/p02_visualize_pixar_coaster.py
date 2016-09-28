
# coding: utf-8

# **Project: Analysing Pixar Movie Performance**
# 
# This project is meant to explore different data visualization techniques. The dataset contains information about the financial performance and critical receptions of different Pixar films.

# **Part 1: Data Import**

# In[64]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

pixar_movies = pd.read_csv('data/PixarMovies.csv')
pixar_movies


# **Part 2: Basic Data Exploration**

# In[65]:

print('DATASET SHAPE')
print('The dataset has {0} rows and {1} columns'.format(pixar_movies.shape[0], pixar_movies.shape[1]))
print('\n')
print('COLUMN DATA TYPES')
print(pixar_movies.dtypes)
print('\n')
print('SUMMARY STATISTICS')
selections = ['Length', 'Metacritic Score', 'Opening Weekend', 'Worldwide Gross', 'Domestic Gross']
subset = pixar_movies[selections].loc[0:len(pixar_movies.index)-2]
print(subset.describe(percentiles=[]))


# **Part 3: Data Cleaning**

# In[66]:

new_domestic_per = pixar_movies['Domestic %'].str.rstrip('%').astype(float)
new_internat_per = pixar_movies['International %'].str.rstrip('%').astype(float)

pixar_movies['Domestic %'] = new_domestic_per
pixar_movies['International %'] = new_internat_per
pixar_movies['IMDB Score'] = pixar_movies['IMDB Score'] * 10

filtered_pixar = pixar_movies.loc[0:len(pixar_movies.index)-2]

filtered_pixar.set_index(keys='Movie', inplace=True)
pixar_movies.set_index(keys='Movie', inplace=True)


# **Part 4: Data Visualization, Line Plots**

# In[69]:

critics_reviews = pixar_movies[['RT Score', 'IMDB Score', 'Metacritic Score']]
critics_reviews.plot(figsize=(10,6))
plt.show()


# **Part 5: Data Visualization, Box Plot**

# In[70]:

critics_reviews.plot(kind='box', figsize=(9,5))
plt.show()


# **Part 6: Data Visualization, Stacked Bar Plot**

# In[72]:

revenue_proportions = filtered_pixar[['Domestic %', 'International %']]
revenue_proportions.plot(kind='bar', stacked=True, figsize=(10,6))
plt.show()


# **Additional Suggestions:**
# 1. Create grouped bar plot to see if there's a correlation between number of Oscar nominations and number of wins
# 2. What correlates with Adjusted Domestic Gross?

# **Additional Question 1:** Is there a relationship between the number of Oscar nominations and the number of wins? (use grouped bar plot)

# In[80]:

filtered_pixar.plot(y=['Oscars Nominated', 'Oscars Won'], kind='bar', figsize=(10,6))
sns.plt.ylabel('Number of Oscars')
plt.show()

filtered_pixar.plot(x='Oscars Nominated',y='Oscars Won', kind='scatter', figsize=(8,8))
sns.plt.xlabel('Number of Oscar Nominations')
sns.plt.ylabel('Number of Oscar Wins')
plt.show()


# **Answer 1:** The grouped bar plot is not the best figure for this (hence the scatterplot), but in general, there appears to be somewhat of a relationship between number of Oscar nominations and wins.

# **Question 2:** What columns correlate with Adjusted Domestic Gross?

# In[91]:

for column in filtered_pixar:
    if column != 'Adjusted Domestic Gross':
        filtered_pixar.plot(x='Adjusted Domestic Gross',y=column, kind='scatter', figsize=(4,4))
        plt.show()


# **Answer 2:** The primary columns that correlate with Adjusted Domestic Gross are critic scores. There's also a correlation with Domestic Gross, but that is to be expected and it's not that interesting.
