import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# This section reads in the data, converts the date column to a datetime format, and then sorts the data from oldest to newest

df = pd.read_csv("sphist.csv")
date = pd.to_datetime(df["Date"])
df["Date"] = date
df.sort_values("Date",ascending=True,inplace=True)

# This section calculates the trailing averages and stdevs of closing prices

avg_5 = []
avg_30 = []
std_5 = []
std_30 = []
avg_365 = []
std_365 = []
year = []
dow = []

last_5 = []
last_30 = []
last_365 = []

for i in range(0,len(df)):
    if i >= 5:
        avg_5.append(np.mean(last_5))
        std_5.append(np.std(last_5))
        last_5.remove(last_5[0])
        last_5.append(df["Close"].iloc[i])
    else:
        avg_5.append(0)
        std_5.append(0)
        last_5.append(df["Close"].iloc[i])
    if i >= 30:
        avg_30.append(np.mean(last_30))
        std_30.append(np.std(last_30))
        last_30.remove(last_30[0])
        last_30.append(df["Close"].iloc[i])
    else:
        avg_30.append(0)
        std_30.append(0)
        last_30.append(df["Close"].iloc[i])
    if i >= 365:
        avg_365.append(np.mean(last_365))
        std_365.append(np.std(last_365))
        last_365.remove(last_365[0])
        last_365.append(df["Close"].iloc[i])
    else:
        avg_365.append(0)
        std_365.append(0)
        last_365.append(df["Close"].iloc[i])
    yr = df["Date"].iloc[i].year
    year.append(yr)
    dw = df["Date"].iloc[i].weekday()
    dow.append(dw)
    
df["avg_5"] = avg_5
df["std_5"] = std_5
df["avg_30"] = avg_30
df["std_30"] = std_30
df["avg_365"] = avg_365
df["std_365"] = std_365
df["year"] = year
df["dow"] = dow

# This section removes all rows where one or more of the trailing averages is 0 due to not enough data

df = df[df["Date"] > datetime(year=1951, month=1, day=2)]
df = df.dropna(axis=0)

# This section breaks the data into training and test sets

train = df[df["Date"] < datetime(year=2013, month=1, day=1)]
test = df[df["Date"] >= datetime(year=2013, month=1, day=1)]

# This section creates a linear regression model, trains it, and makes predictions for the test set

model = LinearRegression()
model.fit(train.iloc[:,7:],train.iloc[:,4])
predictions = model.predict(test.iloc[:,7:])

# This section calculates the mean absolute error for the model

mae = np.mean(abs(predictions - test["Close"]))
print(mae)
