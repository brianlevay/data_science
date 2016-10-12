## 1. Introduction to the Data ##

import pandas
import matplotlib.pyplot as plt
%matplotlib inline

pisa = pandas.DataFrame({"year": range(1975, 1988), 
                         "lean": [2.9642, 2.9644, 2.9656, 2.9667, 2.9673, 2.9688, 2.9696, 
                                  2.9698, 2.9713, 2.9717, 2.9725, 2.9742, 2.9757]})

print(pisa)

plt.scatter(pisa['year'],pisa['lean'])
plt.show()

## 2. Fit the Linear Model ##

import statsmodels.api as sm

y = pisa.lean # target
X = pisa.year  # features
X = sm.add_constant(X)  # add a column of 1's as the constant term

# OLS -- Ordinary Least Squares Fit
linear = sm.OLS(y, X)
# fit model
linearfit = linear.fit()

print(linearfit.summary())

## 3. Define a Basic Linear Model ##

# Our predicted values of y
yhat = linearfit.predict(X)
print(yhat)

residuals = y - yhat
print(residuals)

## 4. Histogram of Residuals ##

# The variable residuals is in memory
plt.hist(residuals,bins=5)
plt.show()

## 6. Sum of Squares ##

import numpy as np

# sum the (predicted - observed) squared
SSE = np.sum((y.values-yhat)**2)
RSS = np.sum((y.values.mean()-yhat)**2)
TSS = np.sum((y.values-y.values.mean())**2)

## 7. R-Squared ##

SSE = np.sum((y.values-yhat)**2)
ybar = np.mean(y.values)
RSS = np.sum((ybar-yhat)**2)
TSS = np.sum((y.values-ybar)**2)
R2 = RSS / TSS

## 9. Coefficients of the Linear Model ##

# Print the models summary
#print(linearfit.summary())

#The models parameters
print("\n",linearfit.params)

delta = linearfit.params.year*15
print(delta)

## 10. Variance of Coefficients ##

# Enter your code here.
n = len(y)
x_bar = np.mean(pisa['year'])
sum_delX = np.sum((pisa['year']-x_bar)**2)
s2b1 = SSE / ((n-2)*sum_delX)

## 11. T-Distribution ##

from scipy.stats import t

# 100 values between -3 and 3
x = np.linspace(-3,3,100)

# Compute the pdf with 3 degrees of freedom
print(t.pdf(x=x, df=3))

## 12. Statistical Significance of Coefficients ##

# The variable s2b1 is in memory.  The variance of beta_1
tstat = abs(linearfit.params.year - 0) / (s2b1 ** (1/2))

## 13. The P-Value ##

# At the 95% confidence interval for a two-sided t-test we must use a p-value of 0.975
pval = 0.975

# The degrees of freedom
df = pisa.shape[0] - 2

# The probability to test against
p = t.cdf(tstat, df=df)

beta1_test = True