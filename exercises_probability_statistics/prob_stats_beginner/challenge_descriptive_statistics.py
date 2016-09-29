## 1. Introduction ##

import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
movie_reviews = pd.read_csv("fandango_score_comparison.csv")

fig = plt.figure(figsize=(4,8))
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)

ax1.set_xlim([0,5.0])
ax2.set_xlim([0,5.0])
ax3.set_xlim([0,5.0])
ax4.set_xlim([0,5.0])

ax1.hist(movie_reviews["RT_user_norm"])
ax2.hist(movie_reviews["Metacritic_user_nom"])
ax3.hist(movie_reviews["Fandango_Ratingvalue"])
ax4.hist(movie_reviews["IMDB_norm"])

plt.show()

## 2. Mean ##

def calc_mean(series):
    return series.mean()

user_reviews = movie_reviews[["RT_user_norm","Metacritic_user_nom","Fandango_Ratingvalue","IMDB_norm"]]

avg_reviews = user_reviews.apply(calc_mean, axis=0)
rt_mean = avg_reviews.loc["RT_user_norm"]
mc_mean = avg_reviews.loc["Metacritic_user_nom"]
fg_mean = avg_reviews.loc["Fandango_Ratingvalue"]
id_mean = avg_reviews.loc["IMDB_norm"]

## 3. Variance and standard deviation ##

def calc_mean(series):
    vals = series.values
    mean = sum(vals) / float(len(vals))
    return mean

def calc_variance(series):
    vals = series.values
    mean = sum(vals) / len(vals)
    variance = sum([(x-mean)**2 for x in vals]) / float(len(vals))
    return variance

variances = user_reviews.apply(calc_variance, axis=0)

rt_var = variances.loc["RT_user_norm"]
mc_var = variances.loc["Metacritic_user_nom"]
fg_var = variances.loc["Fandango_Ratingvalue"]
id_var = variances.loc["IMDB_norm"]

rt_stdev = rt_var ** (1/2)
mc_stdev = mc_var ** (1/2)
fg_stdev = fg_var ** (1/2)
id_stdev = id_var ** (1/2)

## 4. Scatter plots ##

fig = plt.figure(figsize=(4,8))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

ax1.set_xlim([0,5.0])
ax2.set_xlim([0,5.0])
ax3.set_xlim([0,5.0])

ax1.scatter(movie_reviews["Fandango_Ratingvalue"], movie_reviews["RT_user_norm"])
ax2.scatter(movie_reviews["Fandango_Ratingvalue"], movie_reviews["Metacritic_user_nom"])
ax3.scatter(movie_reviews["Fandango_Ratingvalue"], movie_reviews["IMDB_norm"])

plt.show()

## 5. Covariance ##

def calc_mean(series):
    vals = series.values
    mean = sum(vals) / float(len(vals))
    return mean

def calc_covariance(series1,series2):
    x = series1.values
    y = series2.values
    x_mean = calc_mean(series1)
    y_mean = calc_mean(series2)
    cov_xy = 0
    for i in range(0,len(x)):
        cov_xy += (x[i] - x_mean) * (y[i] - y_mean)
    cov_xy = cov_xy / float(len(x))
    return cov_xy

rt_fg_covar = calc_covariance(movie_reviews["Fandango_Ratingvalue"],movie_reviews["RT_user_norm"])
mc_fg_covar = calc_covariance(movie_reviews["Fandango_Ratingvalue"],movie_reviews["Metacritic_user_nom"])
id_fg_covar = calc_covariance(movie_reviews["Fandango_Ratingvalue"],movie_reviews["IMDB_norm"])

## 6. Correlation ##

def calc_mean(series):
    vals = series.values
    mean = sum(vals) / len(vals)
    return mean

def calc_variance(series):
    mean = calc_mean(series)
    squared_deviations = (series - mean)**2
    mean_squared_deviations = calc_mean(squared_deviations)
    return mean_squared_deviations

def calc_covariance(series_one, series_two):
    x = series_one.values
    y = series_two.values
    x_mean = calc_mean(series_one)
    y_mean = calc_mean(series_two)
    x_diffs = [i - x_mean for i in x]
    y_diffs = [i - y_mean for i in y]
    codeviates = [x_diffs[i] * y_diffs[i] for i in range(len(x))]
    return sum(codeviates) / len(codeviates)

def calc_correlation(series_one, series_two):
    stdev1 = calc_variance(series_one) ** (1/2)
    stdev2 = calc_variance(series_two) ** (1/2)
    cov_12 = calc_covariance(series_one, series_two)
    return cov_12 / (stdev1 * stdev2)

rt_fg_corr = calc_correlation(movie_reviews["RT_user_norm"], movie_reviews["Fandango_Ratingvalue"])
mc_fg_corr = calc_correlation(movie_reviews["Metacritic_user_nom"], movie_reviews["Fandango_Ratingvalue"])
id_fg_corr = calc_correlation(movie_reviews["IMDB_norm"], movie_reviews["Fandango_Ratingvalue"])