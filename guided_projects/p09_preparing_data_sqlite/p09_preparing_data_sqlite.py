
# coding: utf-8

# Guided Project
# ----
# Preparing Data for SQLite

# **Part 1:** Introduction to the Data

# In[50]:

import pandas as pd
import sqlite3 as sql

academy_awards = pd.read_csv("academy_awards.csv", encoding="ISO-8859-1")

print("PREVIEW OF ROWS")
print(academy_awards.head(5))
print('\n')
print("SUMMARY OF UNNAMED COLUMN VALUES")
print(academy_awards["Unnamed: 5"].value_counts())
print(academy_awards["Unnamed: 6"].value_counts())
print(academy_awards["Unnamed: 7"].value_counts())
print(academy_awards["Unnamed: 8"].value_counts())
print(academy_awards["Unnamed: 9"].value_counts())
print(academy_awards["Unnamed: 10"].value_counts())


# **Part 2:** Filtering the Data

# In[51]:

academy_awards["Year"] = academy_awards["Year"].str[0:4]
academy_awards["Year"] = academy_awards["Year"].astype("int64")

later_than_2000 = academy_awards.loc[academy_awards["Year"] > 2000]

award_categories = ["Actor -- Leading Role", "Actor -- Supporting Role", "Actress -- Leading Role", "Actress -- Supporting Role"]
nominations = later_than_2000.loc[later_than_2000["Category"].isin(award_categories)]


# **Part 3:** Cleaning Up the Won? and Unnamed Columns

# In[52]:

replace_dict = {"YES": 1, "NO": 0}
new_won = nominations["Won?"].map(replace_dict)
nominations = nominations.assign(Won=new_won)

cols_to_drop = ["Won?", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10"]
final_nominations = nominations.drop(cols_to_drop, axis=1)
final_nominations


# **Part 4:** Cleaning Up the Additional Info Column

# In[53]:

def movie_char(row):
    parts = row.split(" {'")
    movie = parts[0]
    character = parts[1][0:len(parts[1])-2]
    vals = [movie, character]
    return vals

add_list = list(final_nominations["Additional Info"])
parts_list = [movie_char(row) for row in add_list]
movie_list = [x[0] for x in parts_list]
char_list = [x[1] for x in parts_list]

final_nominations = final_nominations.assign(Movie = movie_list)
final_nominations = final_nominations.assign(Character = char_list)
final_nominations = final_nominations.drop("Additional Info", axis=1)
final_nominations


# **Part 5:** Exporting to SQLite

# In[54]:

conn = sql.connect("nominations.db")
final_nominations.to_sql("nominations", conn, index=False, if_exists="replace")


# **Part 6:** Verifying in SQL

# In[55]:

c = conn.cursor()

query1 = "PRAGMA TABLE_INFO(nominations);"
c.execute(query1)
results = c.fetchall()
print("TABLE SCHEMA")
print(results)
print('\n')

query2 = "SELECT * FROM nominations LIMIT 10;"
c.execute(query2)
results = c.fetchall()
print("FIRST 10 ROWS")
print(results)

conn.close()


# **Part 7:** Next Steps
# 
# The suggestions for additonal work all center around the task of getting the entire dataset (not just recent entries, as we did above) into an SQL table with a consistent format. In order to make that happen, we need to understand how the data formats have changed through time.
# 
# For now, I'm going to park this project, but I may come back at a later time.
