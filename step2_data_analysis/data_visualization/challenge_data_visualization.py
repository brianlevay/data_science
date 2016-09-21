# -*- coding: utf-8 -*-

# 1. Introduction to the data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

hollywood_movies = pd.read_csv('hollywood_movies.csv')
print(hollywood_movies.head(5))
print(hollywood_movies['exclude'].value_counts())

hollywood_movies = hollywood_movies.drop('exclude', axis=1)

# 2. Scatter Plots - Profitability And Audience Ratings
figure = plt.figure(figsize=(6,10))
ax_top = figure.add_subplot(2,1,1)
ax_bot = figure.add_subplot(2,1,2)

ax_top.scatter(hollywood_movies['Profitability'], hollywood_movies['Audience Rating'])
ax_top.set_xlabel('Profitability')
ax_top.set_ylabel('Audience Rating')
ax_top.set_title('Hollywood Movies, 2007-2011')

ax_bot.scatter(hollywood_movies['Audience Rating'], hollywood_movies['Profitability'])
ax_bot.set_xlabel('Audience Rating')
ax_bot.set_ylabel('Profitability')
ax_bot.set_title('Hollywood Movies, 2007-2011')

plt.show()

# 3: Scatter Matrix - Profitability And Critic Ratings
normal_movies = hollywood_movies[hollywood_movies['Film'] != 'Paranormal Activity']

pd.scatter_matrix(normal_movies[['Profitability','Audience Rating']], figsize=(6,6))
plt.show()

# 4: Box Plot - Audience And Critic Ratings
normal_movies[['Critic Rating','Audience Rating']].plot(kind='box')
plt.show()

# 5: Box Plot - Critic Vs Audience Ratings Per Year
normal_movies = normal_movies.sort_values(by='Year')

fig = plt.figure(figsize=(8,4))
ax_lt = fig.add_subplot(1,2,1)
ax_rt = fig.add_subplot(1,2,2)

sns.boxplot(x='Year',y='Critic Rating',data=normal_movies, ax=ax_lt)
sns.boxplot(x='Year',y='Audience Rating',data=normal_movies, ax=ax_rt)

plt.show()

# 6: Box Plots - Profitable Vs Unprofitable Movies
def is_profitable(row):
    if row["Profitability"] <= 1.0:
        return False
    return True
makes_money = normal_movies.apply(is_profitable, axis=1)
normal_movies = normal_movies.assign(Profitable = makes_money)
print(normal_movies["Profitable"].value_counts())

fig = plt.figure(figsize=(12,6))
ax_lt = fig.add_subplot(1,2,1)
ax_rt = fig.add_subplot(1,2,2)

sns.boxplot(x='Profitable',y='Audience Rating',data=normal_movies, ax=ax_lt)
sns.boxplot(x='Profitable',y='Critic Rating',data=normal_movies, ax=ax_rt)

plt.show()