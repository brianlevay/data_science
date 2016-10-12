## 1. Introduction to the data ##

import pandas as pd
cars = pd.read_csv("auto.csv")

unique_regions = cars['origin'].unique()
print(unique_regions)

## 2. Dummy variables ##

dummy_cylinders = pd.get_dummies(cars["cylinders"], prefix="cyl")
cars = pd.concat([cars, dummy_cylinders], axis=1)
print(cars.head())

dummy_years = pd.get_dummies(cars['year'], prefix='year')
cars = pd.concat([cars, dummy_years], axis=1)
cars = cars.drop(['year','cylinders'],axis=1)
print(cars.head())

## 3. Multiclass classification ##

shuffled_rows = np.random.permutation(cars.index)
shuffled_cars = cars.iloc[shuffled_rows]

cutoff = int(round(len(shuffled_cars)*0.7,0))
train = shuffled_cars.iloc[0:cutoff]
test = shuffled_cars.iloc[cutoff:len(shuffled_cars)]

## 4. Training a multiclass logistic regression model ##

from sklearn.linear_model import LogisticRegression

unique_origins = cars["origin"].unique()
unique_origins.sort()

models = {}
cyl_year = train.iloc[:,6:]

for val in unique_origins:
    model = LogisticRegression()
    has_val = (train['origin'] == val)
    model.fit(cyl_year,has_val)
    models[val] = model

## 5. Testing the models ##

testing_probs = pd.DataFrame(columns=unique_origins)

for val in unique_origins:
    model = models[val]
    pred_probs = model.predict_proba(test.iloc[:,6:])
    testing_probs[val] = pred_probs[:,1]

## 6. Choose the origin ##

predicted_origins = testing_probs.idxmax(axis=1)
print(predicted_origins.head())