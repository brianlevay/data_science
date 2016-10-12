## 1. A look at the data ##

import pandas
with open("nba_2013.csv", 'r') as csvfile:
    nba = pandas.read_csv(csvfile)

# The names of the columns in the data.
print(nba.columns.values)

## 3. Euclidean distance ##

selected_player = nba[nba["player"] == "LeBron James"].iloc[0]
distance_columns = ['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts']

def eucl_dist(row):
    tot = 0
    for col in distance_columns:
        tot += (selected_player[col] - row[col])**2
    dist = tot ** (1/2)
    return dist

lebron_distance = nba.apply(eucl_dist,axis=1)

## 4. Normalizing columns ##

nba_numeric = nba[distance_columns]

avgs = nba_numeric.mean()
stds = nba_numeric.std()

nba_normalized = (nba_numeric - avgs)/stds
nba_normalized

## 5. Finding the nearest neighbor ##

from scipy.spatial import distance

# Fill in NA values in nba_normalized
nba_normalized.fillna(0, inplace=True)

# Find the normalized vector for lebron james.
lebron_normalized = nba_normalized[nba["player"] == "LeBron James"]

# Find the distance between lebron james and everyone else.
euclidean_distances = nba_normalized.apply(lambda row: distance.euclidean(row, lebron_normalized), axis=1)

distance_frame = pandas.DataFrame(data={"dist":euclidean_distances,"idx":euclidean_distances.index})
distance_frame.sort_values("dist",ascending=True,inplace=True)
index = distance_frame["idx"].iloc[1]

most_similar_to_lebron = nba["player"].iloc[index]
print(most_similar_to_lebron)

## 6. Generating training and testing sets ##

import random
from numpy.random import permutation

# Randomly shuffle the index of nba.
random_indices = permutation(nba.index)
# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
test_cutoff = math.floor(len(nba)/3)
# Generate the test set by taking the first 1/3 of the randomly shuffled indices.
test = nba.loc[random_indices[1:test_cutoff]]
# Generate the train set with the rest of the data.
train = nba.loc[random_indices[test_cutoff:]]

## 7. Using sklearn ##

# The columns that we will be making predictions with.
x_columns = ['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf']
# The column that we want to predict.
y_column = ["pts"]

from sklearn.neighbors import KNeighborsRegressor
# Create the knn model.
knn = KNeighborsRegressor(n_neighbors=5)
# Fit the model on the training data.
knn.fit(train[x_columns], train[y_column])
# Make predictions on the test set using the fit model.
predictions = knn.predict(test[x_columns])

## 8. Computing error ##

actual = test[y_column]
mse = (((predictions - actual) ** 2).sum()) / len(predictions)