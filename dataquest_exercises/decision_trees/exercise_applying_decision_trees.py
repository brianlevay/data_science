## 2. Using decision trees with scikit-learn ##

from sklearn.tree import DecisionTreeClassifier

# A list of columns to train with.
# All columns have been converted to numeric.
columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

# Instantiate the classifier.
# Set random_state to 1 to keep results consistent.
clf = DecisionTreeClassifier(random_state=1)

# The variable income is loaded, and contains all the income data.
clf.fit(income[columns],income['high_income'])

## 3. Splitting the data into train and test sets ##

import numpy
import math

# Set a random seed so the shuffle is the same every time.
numpy.random.seed(1)

# Shuffle the rows.  This first permutes the index randomly using numpy.random.permutation.
# Then, it reindexes the dataframe with this.
# The net effect is to put the rows into random order.
income = income.reindex(numpy.random.permutation(income.index))

train_max_row = math.floor(income.shape[0] * .8)
train = income.iloc[0:train_max_row]
test = income.iloc[train_max_row:len(income)]

## 4. Evaluating error ##

from sklearn.metrics import roc_auc_score

clf = DecisionTreeClassifier(random_state=1)
clf.fit(train[columns], train["high_income"])

predictions = clf.predict(test[columns])

error = roc_auc_score(test["high_income"],predictions)
print(error)

## 5. Compute error on the training set ##

predictions = clf.predict(train[columns])
print(roc_auc_score(train["high_income"],predictions))

## 7. Building a shallower tree ##

# Decision trees model from the last screen.
clf = DecisionTreeClassifier(random_state=1,min_samples_split=13)
clf.fit(train[columns],train['high_income'])

train_pred = clf.predict(train[columns])
train_auc = roc_auc_score(train["high_income"],train_pred)

test_pred = clf.predict(test[columns])
test_auc = roc_auc_score(test["high_income"],test_pred)

print(train_auc)
print(test_auc)

## 8. More parameter tweaking ##

# First decision trees model we trained and tested.
clf = DecisionTreeClassifier(random_state=1,max_depth=7,min_samples_split=13)
clf.fit(train[columns], train["high_income"])
predictions = clf.predict(test[columns])
test_auc = roc_auc_score(test["high_income"], predictions)

train_predictions = clf.predict(train[columns])
train_auc = roc_auc_score(train["high_income"], train_predictions)

print(test_auc)
print(train_auc)

## 9. Tweaking the depth ##

# First decision trees model we trained and tested.
clf = DecisionTreeClassifier(random_state=1,max_depth=2,min_samples_split=100)
clf.fit(train[columns], train["high_income"])
predictions = clf.predict(test[columns])
test_auc = roc_auc_score(test["high_income"], predictions)

train_predictions = clf.predict(train[columns])
train_auc = roc_auc_score(train["high_income"], train_predictions)

print(test_auc)
print(train_auc)

## 12. Exploring decision tree variance ##

numpy.random.seed(1)

# Generate a column with random numbers from 0 to 4.
income["noise"] = numpy.random.randint(4, size=income.shape[0])

# Adjust columns to include the noise column.
columns = ["noise", "age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

# Make new train and test sets.
train_max_row = math.floor(income.shape[0] * .8)
train = income.iloc[:train_max_row]
test = income.iloc[train_max_row:]

# Initialize the classifier.
clf = DecisionTreeClassifier(random_state=1)
clf.fit(train[columns], train["high_income"])

predictions = clf.predict(test[columns])
test_auc = roc_auc_score(test["high_income"], predictions)

train_predictions = clf.predict(train[columns])
train_auc = roc_auc_score(train["high_income"], train_predictions)

print(test_auc)
print(train_auc)