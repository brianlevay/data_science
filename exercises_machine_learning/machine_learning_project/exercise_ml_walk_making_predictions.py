## 1. Recap ##

import pandas as pd

loans = pd.read_csv('cleaned_loans_2007.csv')
print(loans.info())

## 2. Picking an error metric ##

import pandas as pd

is_tn = (predictions == 0) & (loans['loan_status'] == 0)
is_tp = (predictions == 1) & (loans['loan_status'] == 1)
is_fn = (predictions == 0) & (loans['loan_status'] == 1)
is_fp = (predictions == 1) & (loans['loan_status'] == 0)

tn = len(loans.loc[is_tn])
tp = len(loans.loc[is_tp])
fn = len(loans.loc[is_fn])
fp = len(loans.loc[is_fp])

## 3. Class imbalance ##

import pandas as pd
import numpy

# Predict that all loans will be paid off on time.
predictions = pd.Series(numpy.ones(loans.shape[0]))

is_tn = (predictions == 0) & (loans['loan_status'] == 0)
is_tp = (predictions == 1) & (loans['loan_status'] == 1)
is_fn = (predictions == 0) & (loans['loan_status'] == 1)
is_fp = (predictions == 1) & (loans['loan_status'] == 0)

tn = len(loans.loc[is_tn])
tp = len(loans.loc[is_tp])
fn = len(loans.loc[is_fn])
fp = len(loans.loc[is_fp])

fpr = fp / (fp + tn)
tpr = tp / (tp + fn)
print(fpr)
print(tpr)

## 4. Logistic Regression ##

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
cols = loans.columns
train_cols = cols.drop("loan_status")
features = loans[train_cols]
target = loans["loan_status"]
lr.fit(features, target)
predictions = lr.predict(features)

## 5. Cross Validation ##

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict, KFold
lr = LogisticRegression()
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr,features,target,cv=kf)
predictions = pd.Series(predictions)

is_tn = (predictions == 0) & (loans['loan_status'] == 0)
is_tp = (predictions == 1) & (loans['loan_status'] == 1)
is_fn = (predictions == 0) & (loans['loan_status'] == 1)
is_fp = (predictions == 1) & (loans['loan_status'] == 0)

tn = len(loans.loc[is_tn])
tp = len(loans.loc[is_tp])
fn = len(loans.loc[is_fn])
fp = len(loans.loc[is_fp])

fpr = fp / (fp + tn)
tpr = tp / (tp + fn)
print(fpr)
print(tpr)

## 6. Penalizing the classifier ##

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict

lr = LogisticRegression(class_weight="balanced")
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr,features,target,cv=kf)
predictions = pd.Series(predictions)

is_tn = (predictions == 0) & (loans['loan_status'] == 0)
is_tp = (predictions == 1) & (loans['loan_status'] == 1)
is_fn = (predictions == 0) & (loans['loan_status'] == 1)
is_fp = (predictions == 1) & (loans['loan_status'] == 0)

tn = len(loans.loc[is_tn])
tp = len(loans.loc[is_tp])
fn = len(loans.loc[is_fn])
fp = len(loans.loc[is_fp])

fpr = fp / (fp + tn)
tpr = tp / (tp + fn)
print(fpr)
print(tpr)

## 7. Manual penalties ##

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict

penalty = {
    0: 10,
    1: 1
}
lr = LogisticRegression(class_weight=penalty)
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr,features,target,cv=kf)
predictions = pd.Series(predictions)

is_tn = (predictions == 0) & (loans['loan_status'] == 0)
is_tp = (predictions == 1) & (loans['loan_status'] == 1)
is_fn = (predictions == 0) & (loans['loan_status'] == 1)
is_fp = (predictions == 1) & (loans['loan_status'] == 0)

tn = len(loans.loc[is_tn])
tp = len(loans.loc[is_tp])
fn = len(loans.loc[is_fn])
fp = len(loans.loc[is_fp])

fpr = fp / (fp + tn)
tpr = tp / (tp + fn)
print(fpr)
print(tpr)

## 8. Random forests ##

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict

lr = RandomForestClassifier(class_weight="balanced",random_state=1)
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr,features,target,cv=kf)
predictions = pd.Series(predictions)

is_tn = (predictions == 0) & (loans['loan_status'] == 0)
is_tp = (predictions == 1) & (loans['loan_status'] == 1)
is_fn = (predictions == 0) & (loans['loan_status'] == 1)
is_fp = (predictions == 1) & (loans['loan_status'] == 0)

tn = len(loans.loc[is_tn])
tp = len(loans.loc[is_tp])
fn = len(loans.loc[is_fn])
fp = len(loans.loc[is_fp])

fpr = fp / (fp + tn)
tpr = tp / (tp + fn)
print(fpr)
print(tpr)