## 2. Calculating differences ##

female_diff = (10771 - 16280.5)/16280.5
male_diff = (21790 - 16280.5)/16280.5

## 3. Updating the formula ##

female_diff = ((10771 - 16280.5)**2)/16280.5
male_diff = ((21790 - 16280.5)**2)/16280.5
gender_chisq = female_diff + male_diff

## 4. Generating a distribution ##

import numpy as np
import matplotlib.pyplot as plt

chi_squared_values = []
for i in range(1000):
    fractions = np.random.random(32561,)
    binaries = []
    for num in fractions:
        if num < 0.5:
            binaries.append(0)
        else:
            binaries.append(1)
    male_ct = sum(binaries)
    female_ct = 32561 - male_ct
    male_diff = ((male_ct - 16280.5)**2)/16280.5
    female_diff = ((female_ct - 16280.5)**2)/16280.5
    gender_chisq = female_diff + male_diff
    chi_squared_values.append(gender_chisq)

plt.hist(chi_squared_values)
plt.show()

## 6. Smaller samples ##

female_diff = ((107.71 - 162.805)**2)/162.805
male_diff = ((217.90 - 162.805)**2)/162.805
gender_chisq = female_diff + male_diff

## 7. Sampling distribution equality ##

chi_squared_values = []
for i in range(1000):
    fractions = np.random.random(300,)
    binaries = []
    for num in fractions:
        if num < 0.5:
            binaries.append(0)
        else:
            binaries.append(1)
    male_ct = sum(binaries)
    female_ct = 300 - male_ct
    male_diff = ((male_ct - 150)**2)/150
    female_diff = ((female_ct - 150)**2)/150
    gender_chisq = female_diff + male_diff
    chi_squared_values.append(gender_chisq)

plt.hist(chi_squared_values)
plt.show()

## 9. Increasing degrees of freedom ##

observed = {"white":27816,"black":3124,"asian":1039,"amerindian":311,"other":271}
expected = {"white":26146.5,"black":3939.9,"asian":944.3,"amerindian":260.5,"other":1269.8}
total = 32561

race_chisq = 0
for key in observed:
    diff = ((observed[key] - expected[key])**2)/expected[key]
    race_chisq += diff

## 10. Using SciPy ##

from scipy.stats import chisquare
import numpy as np

observed = [27816,3124,1039,311,271]
expected = [26146.5,3939.9,944.3,260.5,1269.8]

race_chisquare_value, race_pvalue = chisquare(observed, expected)