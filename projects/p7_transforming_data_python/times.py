# This module returns the most common hours for HackerNews article submissions

import datetime
import dateutil
import read

def get_hour(timestamp):
    try:
        date_obj = dateutil.parser.parse(timestamp)
        return date_obj.hour
    except:
        return -1


def articles_by_hour():
    # read the data
    data = read.load_data()
    
    # returns a series containing only the hours of submission
    hours = data["submission_time"].apply(get_hour)
    
    # get the counts of each unique hour
    hour_cts = hours.value_counts(ascending=False)
    
    # print results
    for name, row in hour_cts.items():
        print("{0}: {1}".format(name,row))
    
    return hour_cts


if __name__ == "__main__":
    articles_by_hour()