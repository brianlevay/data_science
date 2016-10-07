## 2. Data cleaning ##

import pandas as pd
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
cars = pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)

no_qm = (cars["horsepower"] != "?")
filtered_cars = cars.loc[no_qm]
hp_float = filtered_cars["horsepower"].astype("float")
filtered_cars["horsepower"] = hp_float

## 3. Data Exploration ##

%matplotlib inline
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(filtered_cars["horsepower"],filtered_cars["mpg"])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(filtered_cars["weight"],filtered_cars["mpg"])
plt.show()

## 4. Fitting a model ##

import sklearn
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(filtered_cars[["horsepower"]],filtered_cars[["mpg"]])
predictions = lr.predict(filtered_cars[["horsepower"]])
actual = list(filtered_cars["mpg"])

for i in range(5):
    print("Actual: {0}, Predicted: {1}".format(actual[i],predictions[i][0]))

## 5. Plotting the predictions ##

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(filtered_cars["horsepower"],predictions, color="blue")
ax.scatter(filtered_cars["horsepower"],filtered_cars["mpg"], color="red")
plt.show()

## 6. Error metrics ##

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(actual,predictions)
rmse = mse ** (1/2)