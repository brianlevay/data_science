## 3. Exploring the data ##

import pandas as pd

avengers = pd.read_csv("avengers.csv")
avengers.head(5)

## 4. Filter out the bad years ##

import matplotlib.pyplot as plt
%matplotlib inline

avengers['Year'].hist()

is_after_1960 = (avengers['Year'] > 1960)
true_avengers = avengers.loc[is_after_1960]


## 5. Consolidating deaths ##

def deaths_total(row):
    total = 0
    for d in range(5):
        death = 'Death' + str(d+1)
        if row[death] == 'YES':
            total += 1
    return total


tot_deaths = true_avengers.apply(deaths_total, axis=1)
true_avengers = true_avengers.assign(Deaths = tot_deaths)

## 6. Years since joining ##

def years_since_correct(row):
    diff = 2015 - row['Year']
    reported = row['Years since joining']
    delta = diff - reported
    if delta != 0:
        return 0
    else:
        return 1

correct_years = true_avengers.apply(years_since_correct, axis=1)
joined_accuracy_count  = correct_years.sum()