# This module returns the most frequent words in HackerNews headline titles

import collections as coll
import read

def hundred_freq_words():
    # read the data and make a list from the headline column
    data = read.load_data()
    headlines = list(data["headline"])
    
    # filter out headlines that contain nan
    real_headlines = list(filter(lambda x: type(x) is str, headlines))
    
    # join the headlines and split on " " to generate a list of words
    merged_headlines = " ".join(real_headlines)
    words = merged_headlines.split(" ")
    
    # convert each word to lowercase
    lwr_words = list(map(lambda x: x.lower(), words))
    
    # use a Counter object to get the number of occurrences per unique word 
    counter = coll.Counter(lwr_words)
    most_common = counter.most_common(100)
    
    #print results
    print(most_common[0:10])
    return most_common
    
    
if __name__ == "__main__":
    hundred_freq_words()