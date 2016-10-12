## 1. Introduction to validation ##

import pandas as pd
from sklearn.linear_model import LogisticRegression

admissions = pd.read_csv("admissions.csv")
admissions["actual_label"] = admissions["admit"]
admissions = admissions.drop("admit", axis=1)

print(admissions.head())

## 2. Holdout validation ##

import numpy as np
np.random.seed(8)
admissions = pd.read_csv("admissions.csv")
admissions["actual_label"] = admissions["admit"]
admissions = admissions.drop("admit", axis=1)

rand_index = np.random.permutation(admissions.index)
shuffled_admissions = admissions.loc[rand_index]
train = shuffled_admissions.iloc[0:515]
test = shuffled_admissions.iloc[515:len(shuffled_admissions)]
print(shuffled_admissions.head())

## 3. Accuracy ##

shuffled_index = np.random.permutation(admissions.index)
shuffled_admissions = admissions.loc[shuffled_index]
train = shuffled_admissions.iloc[0:515]
test = shuffled_admissions.iloc[515:len(shuffled_admissions)]

model = LogisticRegression()
model.fit(train[["gpa"]],train["actual_label"])

labels = model.predict(test[["gpa"]])
test = test.assign(predicted_label = labels)
matches = (test["predicted_label"] == test["actual_label"])
accuracy = test.loc[matches].shape[0] / test.shape[0]
print(accuracy)

## 4. Sensitivity and specificity ##

model = LogisticRegression()
model.fit(train[["gpa"]], train["actual_label"])
labels = model.predict(test[["gpa"]])
test["predicted_label"] = labels
matches = test["predicted_label"] == test["actual_label"]
correct_predictions = test[matches]
accuracy = len(correct_predictions) / len(test)

is_true_pos = (test["predicted_label"] == 1) & (test["actual_label"] == 1)
is_true_neg = (test["predicted_label"] == 0) & (test["actual_label"] == 0)
is_false_pos = (test["predicted_label"] == 1) & (test["actual_label"] == 0)
is_false_neg = (test["predicted_label"] == 0) & (test["actual_label"] == 1)
true_pos = len(test.loc[is_true_pos])
true_neg = len(test.loc[is_true_neg])
false_pos = len(test.loc[is_false_pos])
false_neg = len(test.loc[is_false_neg])

sensitivity = true_pos / (true_pos + false_neg)
specificity = true_neg / (true_neg + false_pos)
print(sensitivity)
print(specificity)

## 6. ROC curve ##

import matplotlib.pyplot as plt
from sklearn import metrics
%matplotlib inline

pred_probs = model.predict_proba(test[["gpa"]])
fpr, tpr, thresholds = metrics.roc_curve(test["actual_label"],pred_probs[:,1])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(fpr,tpr)
plt.show()

## 7. Area under the curve ##

# Note the different import style!
from sklearn.metrics import roc_auc_score

auc_score = roc_auc_score(test["actual_label"],pred_probs[:,1])
print(auc_score)