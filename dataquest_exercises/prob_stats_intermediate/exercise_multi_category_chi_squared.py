## 2. Calculating expected values ##

male_prop = 0.669
female_prop = 1-male_prop
over50k_prop = 0.241
under50k_prop = 1-over50k_prop
tot_people = 32561

males_over50k = male_prop * over50k_prop * tot_people
males_under50k = male_prop * under50k_prop * tot_people
females_over50k = female_prop * over50k_prop * tot_people
females_under50k = female_prop * under50k_prop * tot_people

## 3. Calculating chi-squared ##

observed = [6662, 1179, 15128, 9592]
expected = [5249.8, 2597.4, 16533.5, 8180.3]
values = []

for i, obs in enumerate(observed):
    exp = expected[i]
    value = (obs - exp) ** 2 / exp
    values.append(value)

chisq_gender_income = sum(values)

## 4. Finding statistical significance ##

import numpy as np
from scipy.stats import chisquare

chisquare_value, pvalue_gender_income = chisquare(observed, expected)

## 5. Cross tables ##

import pandas

table = pandas.crosstab(income["sex"], [income["race"]])
print(table)

## 6. Finding expected values ##

import numpy as np
from scipy.stats import chi2_contingency

table = pandas.crosstab(income["sex"], [income["race"]])
chisq_value, pvalue_gender_race, df, expected = chi2_contingency(table)