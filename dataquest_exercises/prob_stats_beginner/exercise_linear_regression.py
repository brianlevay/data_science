## 2. Drawing lines ##

import matplotlib.pyplot as plt
import numpy as np

x = [0, 1, 2, 3, 4, 5]
# Going by our formula, every y value at a position is the same as the x-value in the same position.
# We could write y = x, but let's write them all out to make this more clear.
y = [0, 1, 2, 3, 4, 5]

# As you can see, this is a straight line that passes through the points (0,0), (1,1), (2,2), and so on.
plt.plot(x, y)
plt.show()

# Let's try a slightly more ambitious line.
# What if we did y = x + 1?
# We'll make x an array now, so we can add 1 to every element more easily.
x = np.asarray([0, 1, 2, 3, 4, 5])
y = x + 1

# y is the same as x, but every element has 1 added to it.
print(y)

# This plot passes through (0,1), (1,2), and so on.
# It's the same line as before, but shifted up 1 on the y-axis.
plt.plot(x, y)
plt.show()

# By adding 1 to the line, we moved what's called the y-intercept -- where the line intersects with the y-axis.
# Moving the intercept can shift the whole line up (or down when we subtract).

y = x - 1
plt.plot(x,y)
plt.show()

y = x + 10
plt.plot(x,y)
plt.show()

## 3. Working with slope ##

import matplotlib.pyplot as plt
import numpy as np

x = np.asarray([0, 1, 2, 3, 4, 5])
# Let's set the slope of the line to 2.
y = 2 * x

# See how this line is "steeper" than before?  The larger the slope is, the steeper the line becomes.
# On the flipside, fractional slopes will create a "shallower" line.
# Negative slopes will create a line where y values decrease as x values increase.
plt.plot(x, y)
plt.show()

y = 4*x
plt.plot(x,y)
plt.show()

y = 0.5*x
plt.plot(x,y)
plt.show()

y = -2*x
plt.plot(x,y)
plt.show()

## 4. Starting out with linear regression ##

# The wine quality data is loaded into wine_quality
from numpy import cov
slope_density = cov(wine_quality["density"],wine_quality["quality"])[0,1] / wine_quality["density"].var()

## 5. Finishing linear regression ##

from numpy import cov

# This function will take in two columns of data, and return the slope of the linear regression line.
def calc_slope(x, y):
    return cov(x, y)[0, 1] / x.var()

m = calc_slope(wine_quality["density"], wine_quality["quality"])
x_ave = wine_quality["density"].mean()
y_ave = wine_quality["quality"].mean()
intercept_density = y_ave - m*x_ave

## 6. Making predictions ##

from numpy import cov

def calc_slope(x, y):
    return cov(x, y)[0, 1] / x.var()

# Calculate the intercept given the x column, y column, and the slope
def calc_intercept(x, y, slope):
    return y.mean() - (slope * x.mean())

m = calc_slope(wine_quality["density"], wine_quality["quality"])
b = calc_intercept(wine_quality["density"], wine_quality["quality"], m)

def predict_y(x):
    y = m*x + b
    return y

predicted_quality = wine_quality["density"].apply(predict_y)

## 7. Finding error ##

from scipy.stats import linregress

# We've seen the r_value before -- we'll get to what p_value and stderr_slope are soon -- for now, don't worry about them.
slope, intercept, r_value, p_value, stderr_slope = linregress(wine_quality["density"], wine_quality["quality"])

# As you can see, these are the same values we calculated (except for slight rounding differences)
print(slope)
print(intercept)

x = list(wine_quality["density"])
y_act = list(wine_quality["quality"])
y_pre = [i*slope + intercept for i in x]
rss = sum( [(y_act[i] - y_pre[i])**2 for i in range(len(x))] )

## 8. Standard error ##

from scipy.stats import linregress
import numpy as np

# We can do our linear regression
# Sadly, the stderr_slope isn't the standard error, but it is the standard error of the slope fitting only
# We'll need to calculate the standard error of the equation ourselves
slope, intercept, r_value, p_value, stderr_slope = linregress(wine_quality["density"], wine_quality["quality"])

predicted_y = np.asarray([slope * x + intercept for x in wine_quality["density"]])
residuals = (wine_quality["quality"] - predicted_y) ** 2
rss = sum(residuals)
stderr = (rss / (len(residuals)-2))**(1/2)

def percent_error_dist(actual_y, predicted_y, stderr, num_std):
    count = 0
    for i in range(len(actual_y)):
        err = abs(actual_y[i] - predicted_y[i])
        rel = err / stderr
        if rel <= num_std:
            count += 1
    return count / float(len(actual_y))

within_one = percent_error_dist(list(wine_quality["quality"]), predicted_y, stderr, 1)
within_two = percent_error_dist(list(wine_quality["quality"]), predicted_y, stderr, 2)
within_three = percent_error_dist(list(wine_quality["quality"]), predicted_y, stderr, 3)