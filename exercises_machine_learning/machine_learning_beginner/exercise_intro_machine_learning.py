## 2. Introduction to the data ##

cols = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
cars = pd.read_table("auto-mpg.data", delim_whitespace=True, names=cols)
print(cars.head(5))

## 3. Exploratory data analysis ##

import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.scatter(cars['weight'],cars['mpg'])
ax2.scatter(cars['acceleration'],cars['mpg'])
plt.show()

## 5. Scikit-learn ##

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

weight = cars[["weight"]].values
mpg = cars["mpg"].values

lr.fit(weight, mpg)

## 6. Making predictions ##

import sklearn
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=True)
lr.fit(cars[["weight"]], cars[["mpg"]])
predictions = lr.predict(cars[["weight"]])

for i in range(5):
    print("Actual: {0}, Predicted: {1}".format(predictions[i],mpg[i]))

## 7. Plotting the model ##

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(cars["weight"],cars["mpg"],color="red")
ax.scatter(cars["weight"],predictions,color="blue")
plt.show()

## 8. Error metrics ##

from sklearn.metrics import mean_squared_error

lr = LinearRegression()#fit_intercept=True)
lr.fit(cars[["weight"]], cars[["mpg"]])
predictions = lr.predict(cars[["weight"]])
mse = mean_squared_error(cars[["mpg"]], predictions)

## 9. Root mean squared error ##

mse = mean_squared_error(cars["mpg"], predictions)
rmse = mse ** (1/2)
print(rmse)