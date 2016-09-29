
# coding: utf-8

# Guided Project
# ---
# Analyzing Movie Reviews

# **Part 1:** Introduction and Setup

# In[34]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy
get_ipython().magic('matplotlib inline')

movies = pd.read_csv("fandango_score_comparison.csv")
movies


# **Part 2:** Histograms

# In[35]:

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

bins = [-0.25,0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.25]

ax1.hist(movies["Metacritic_norm_round"], bins=bins, align="mid")
ax2.hist(movies["Fandango_Stars"], bins=bins, align="mid")
ax1.set_xlim([-0.25,5.25])
ax2.set_xlim([-0.25,5.25])
ax1.set_xticks([0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
ax2.set_xticks([0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
ax1.set_xlabel("Metacritic Scores, Normalized")
ax2.set_xlabel("Fandango Stars")

plt.show()


# **Discussion:** Fandango scores are significantly higher, on average, than metacritic scores. In fact, on a scale of 0-5, Fandango scores are almost never less than 3, suggesting that the Fandango reviewers are really easy to please (compared to Metacritic reviewers), the scoring systems are not directly comparable (i.e., a 1 does not mean the same thing on each scale), or there is some kind of bias in reporting for one or both of the systems.

# **Part 3:** Mean, Median, and Standard Deviation

# In[36]:

mean_fs = movies["Fandango_Stars"].mean()
mean_mc = movies["Metacritic_norm_round"].mean()
median_fs = movies["Fandango_Stars"].median()
median_mc = movies["Metacritic_norm_round"].median()
std_fs = numpy.std(list(movies["Fandango_Stars"]))
std_mc = numpy.std(list(movies["Metacritic_norm_round"]))

print("Mean    Metacritic Scores: {:.2f}".format(mean_mc))
print("Median  Metacritic Scores: {0}".format(median_mc))
print("Stdev   Metacritic Scores: {:.2f}".format(std_mc))
print("\n")
print("Mean    Fandango Stars:    {:.2f}".format(mean_fs))
print("Median  Fandango Stars:    {0}".format(median_fs))
print("Stdev   Fandango Stars:    {:.2f}".format(std_fs))

# Double checking numbers
vals_mc = sorted(list(movies["Metacritic_norm_round"]))
mean_test = sum(vals_mc) / float(len(vals_mc))
median_test = vals_mc[int(len(vals_mc)/2)]
print("\n")
print("DOUBLE CHECKING METACRITIC MEAN AND MEDIAN")
print(mean_test)
print(median_test)

vals_fs = sorted(list(movies["Fandango_Stars"]))
mean_test = sum(vals_fs) / float(len(vals_fs))
median_test = vals_fs[int(len(vals_fs)/2)]
print("\n")
print("DOUBLE CHECKING FANDANGO MEAN AND MEDIAN")
print(mean_test)
print(median_test)

print("\n")
print("METACRITIC SCORES")
print(vals_mc)
print("\n")
print("FANDANGO STARS")
print(vals_fs)


# **Discussion:**
# 
# *Methods:* Metacritic does a decent job of describing their rating methodologies on their website. Films are rated on a number of different scales by critics (0-4, A-F, etc.) and Metacritic converts each scale to a 100 point equivalent. Then, each critic is given a relative weight based on his or her perceived quality, and those weights are used to calculate an overall average. The scores are then normalized, but there is no description of the normalization process. In this dataset, the scores were then normalized back to a 5 point scale and were rounded to the nearest half point. Fandango, on the other hand, does not describe its methodology. Or, if they do, they hide it very well from Google.
# 
# *Answers:* The mean is very close to the median within each dataset, and given the irregular distributions, it isn't worth trying to examine minuscule differences much further. In fact, my calculations (seen above, and double checked in Excel) show different mean / median shifts relative to what is discussed in the Dataquest Q&A, suggesting the dataset may have changed since writing the questions or the team used a different methodology.

# **Part 4:** Scatter Plots

# In[40]:

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(1,1,1)
ax1.scatter(movies["Metacritic_norm_round"], movies["Fandango_Stars"])
ax1.set_xlabel("Metacritic Scores, Normalized")
ax1.set_ylabel("Fandango Stars")
plt.show()

diff = movies["Metacritic_norm_round"] - movies["Fandango_Stars"]
abs_diff = [abs(x) for x in diff]
movies = movies.assign(fm_diff=abs_diff)
movies.sort_values("fm_diff", ascending=False, inplace=True)
movies[["FILM","Metacritic_norm_round","Fandango_Stars","fm_diff"]].head(5)


# **Part 5:** Correlations

# In[52]:

import scipy.stats as stats

x = list(movies["Metacritic_norm_round"])
y_act = list(movies["Fandango_Stars"])

r, p_val = stats.pearsonr(x,y_act)
print("\n")
print("FANDANGO STARS vs METACRITIC SCORES (NORM)")
print("r-value:     {0:.2f}".format(r))
print("p-value:     {0:.2f}".format(p_val))

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y_act)
predicted = slope*(3.0) + intercept
print("\n")
print("LINEAR REGRESSION RESULTS")
print("slope:       {0:.2f}".format(slope))
print("intercept:   {0:.2f}".format(intercept))
print("\n")
print("PREDICTED FANDANGO SCORE")
print("Metacritic:  3.0")
print("Fandango:    {0:.1f}".format(predicted))


# **Discussion:** The poor correlation between Metacritic and Fandango scores means that the two groups of critics are using very different criteria to rate movies.

# **Part 6:** Finding Residuals

# In[53]:

x_reg = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
y_reg = [slope*i + intercept for i in x_reg]

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(1,1,1)
ax1.scatter(x, y_act)
ax1.plot(x_reg, y_reg)
ax1.set_xlim([0,5])
ax1.set_ylim([0,5])
ax1.set_xlabel("Metacritic Score, Normalized")
ax1.set_ylabel("Fandango Stars")

plt.show()


# **Part 7:** Next Steps
# 
# Suggestions for additional work:
# * Look at the relationships between other ratings
# * Compare user scores to critic scores
# * Try to figure out why some movies had such variable scores
# 
# For now, though, I'm going to park this project.
