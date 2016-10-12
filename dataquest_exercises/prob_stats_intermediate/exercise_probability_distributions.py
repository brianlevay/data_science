## 3. Bikesharing distribution ##

import pandas
bikes = pandas.read_csv("bike_rental_day.csv")

prob_over_5000 = bikes.loc[bikes["cnt"]>5000].shape[0] / bikes.shape[0]

## 4. Computing the distribution ##

import math

# Each item in this list represents one k, starting from 0 and going up to and including 30.
outcome_counts = list(range(31))

def prob_of_outcome(p,q,N,k):
    p_each = (p**k)*(q**(N-k))
    num_each = math.factorial(N) / (math.factorial(k) * math.factorial(N-k))
    return p_each * num_each

p = 0.39
q = 1 - p
outcome_probs = [prob_of_outcome(p,q,30,k) for k in outcome_counts]

## 5. Plotting the distribution ##

import matplotlib.pyplot as plt

# The most likely number of days is between 10 and 15.
plt.bar(outcome_counts, outcome_probs)
plt.show()

## 6. Simplifying the computation ##

import scipy
from scipy import linspace
from scipy.stats import binom

# Create a range of numbers from 0 to 30, with 31 elements (each number has one entry).
outcome_counts = linspace(0,30,31)
dist = binom.pmf(outcome_counts,30,0.39)
plt.bar(outcome_counts,dist)
plt.show()

## 8. Computing the mean of a probability distribution ##

dist_mean = 0.39 * 30

## 9. Computing the standard deviation ##

dist_stdev = (30 * 0.39 * (1-0.39)) ** (1/2)

## 10. A different plot ##

# Enter your answer here.
dist = binom.pmf(outcome_counts,10,0.39)
plt.bar(outcome_counts,dist)
plt.show()

dist = binom.pmf(outcome_counts,100,0.39)
plt.bar(outcome_counts,dist)
plt.show()

## 11. The normal distribution ##

# Create a range of numbers from 0 to 100, with 101 elements (each number has one entry).
outcome_counts = scipy.linspace(0,100,101)

# Create a probability mass function along the outcome_counts.
outcome_probs = binom.pmf(outcome_counts,100,0.39)

# Plot a line, not a bar chart.
plt.plot(outcome_counts, outcome_probs)
plt.show()

## 12. Cumulative density function ##

outcome_counts = linspace(0,30,31)
outcome_probs = binom.cdf(outcome_counts,30,0.39)
plt.plot(outcome_counts, outcome_probs)
plt.show()

## 14. Faster way to calculate likelihood ##

left_16 = binom.cdf(16,30,0.39)
right_16 = 1 - left_16