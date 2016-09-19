## 1. Seaborn ##

import pandas as pd
births = pd.read_csv('births.csv')
print(births.head(50))

## 2. Histogram: distplot() ##

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.distplot(births['prglngth'], kde=False)
sns.plt.show()

## 3. Seaborn styling ##

import seaborn as sns

plt.hist(births['agepreg'])
plt.show()

## 5. Customizing histogram: distplot() ##

sns.distplot(births['prglngth'], kde=False, axlabel='Pregnancy Length, weeks')
sns.plt.show()

## 6. Practice: customizing distplot() ##

sns.set_style(style='dark')
sns.distplot(births['birthord'], kde=False, axlabel='Birth number')
sns.plt.show()

## 7. Boxplots: boxplot() ##

births = pd.read_csv('births.csv')

sns.boxplot(x=births['birthord'], y=births['agepreg'])
sns.plt.show()

## 8. Pair plot: pairplot() ##

sns.pairplot(data=births[['agepreg','prglngth','birthord']])
sns.plt.show()