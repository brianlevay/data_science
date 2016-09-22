# This module reads in the HackerNews dataset and adds columns

import pandas as pd

def load_data():
    data = pd.read_csv("hn_stories.csv")
    data.columns = ["submission_time","upvotes","url","headline"]
    return data


if __name__ == "__main__":
    data = load_data()