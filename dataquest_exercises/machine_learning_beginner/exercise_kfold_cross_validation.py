## 2. Partititioning the data ##

import pandas as pd

admissions = pd.read_csv("admissions.csv")
admissions["actual_label"] = admissions["admit"]
admissions = admissions.drop("admit", axis=1)

shuffled_index = np.random.permutation(admissions.index)
shuffled_admissions = admissions.loc[shuffled_index]
admissions = shuffled_admissions.reset_index()

def fold(index):
    r_bin = [-1,128,257,386,514,644]
    fold = 0
    for i in range(5):
        if (index > r_bin[i]) & (index <= r_bin[i+1]):
            fold = i+1
    return fold

fold_vals = []
for row in admissions.iterrows():
    fold_vals.append(fold(row[0]))
    
admissions = admissions.assign(fold=fold_vals)
print(admissions.head())
print(admissions.tail())

## 3. First iteration ##

from sklearn.linear_model import LogisticRegression

def split_data(df,fold):
    for_train = (df["fold"] != fold)
    for_test = (df["fold"] == fold)
    train = df.loc[for_train]
    test = df.loc[for_test]
    return train, test

train, test = split_data(admissions,1)

model = LogisticRegression()
model.fit(train[["gpa"]],train["actual_label"])
labels = model.predict(test[["gpa"]])
matches = (test["actual_label"] == labels)
iteration_one_accuracy = len(test.loc[matches]) / len(test)

## 4. Function for training models ##

# Use np.mean to calculate the mean.
import numpy as np
fold_ids = [1,2,3,4,5]

def train_and_test(df,folds):
    accuracies = []
    model = LogisticRegression()
    for fold in folds:
        train, test = split_data(admissions,fold)
        model.fit(train[["gpa"]],train["actual_label"])
        labels = model.predict(test[["gpa"]])
        matches = (test["actual_label"] == labels)
        accuracy = len(test.loc[matches]) / float(len(test))
        accuracies.append(accuracy)
    return accuracies

accuracies = train_and_test(admissions,fold_ids)
average_accuracy = np.mean(accuracies)
print(accuracies)
print(average_accuracy)

## 5. Sklearn ##

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

admissions = pd.read_csv("admissions.csv")
admissions["actual_label"] = admissions["admit"]
admissions = admissions.drop("admit", axis=1)

kf = KFold(len(admissions), 5, shuffle=True, random_state=8)
lr = LogisticRegression()
accuracies = cross_val_score(lr, admissions[["gpa"]], admissions["actual_label"], scoring=None, cv=kf)
average_accuracy = np.mean(accuracies)
print(accuracies)
print(average_accuracy)