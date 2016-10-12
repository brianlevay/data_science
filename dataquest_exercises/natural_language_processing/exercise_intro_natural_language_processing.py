## 2. Looking at the data ##

import pandas as pd
submissions = pd.read_csv("sel_hn_stories.csv")
submissions.columns = ["submission_time", "upvotes", "url", "headline"]
submissions = submissions.dropna()

## 3. Tokenization ##

def tokenize(row):
    words = row['headline'].split(" ")
    return words

tokenized_headlines = list(submissions.apply(tokenize,axis=1))

## 4. Preprocessing ##

punctuation = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]
clean_tokenized = []

for row in tokenized_headlines:
    new_row = []
    for word in row:
        new_word = ''.join(ch for ch in word if ch not in punctuation)
        new_word = new_word.lower()
        new_row.append(new_word)
    clean_tokenized.append(new_row)

## 5. Assembling a matrix ##

import numpy as np
unique_tokens = []
single_tokens = []
for tokens in clean_tokenized:
    for token in tokens:
        if token not in single_tokens:
            single_tokens.append(token)
        elif token in single_tokens and token not in unique_tokens:
            unique_tokens.append(token)

counts = pd.DataFrame(0, index=np.arange(len(clean_tokenized)), columns=unique_tokens)

## 6. Counting tokens ##

# clean_tokenized and counts have been loaded in.
for i, item in enumerate(clean_tokenized):
    for token in item:
        if token in unique_tokens:
            counts.iloc[i][token] += 1# clean_tokenized and counts have been loaded in.

## 7. Removing extraneous columns ##

# clean_tokenized and counts have been loaded in.
word_counts = counts.sum(axis=0)

counts = counts.loc[:,(word_counts >= 5) & (word_counts <= 100)]

## 8. Splitting the data ##

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(counts, submissions["upvotes"], test_size=0.2, random_state=1)

## 9. Making predictions ##

from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)

## 10. Calculating error ##

mse = sum((predictions - y_test)**2)/len(predictions)