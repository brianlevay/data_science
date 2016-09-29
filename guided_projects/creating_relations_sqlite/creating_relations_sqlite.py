
# coding: utf-8

# Guided Project
# ---
# Creating Relations in SQLite

# **Part 1:** Introduction to the Data

# In[2]:

import sqlite3 as sql
import pandas as pd

conn = sql.connect("nominations.db")
c = conn.cursor()

query1 = "PRAGMA TABLE_INFO(nominations);"
c.execute(query1)
schema = c.fetchall()
print("TABLE SCHEMA")
for row in schema:
    print(row)
print("\n")

query2 = "SELECT * FROM nominations LIMIT 10;"
c.execute(query2)
first_ten = c.fetchall()
print("FIRST TEN ROWS")
for row in first_ten:
    print(row)
print("\n")


# **Part 2:** Creating the Ceremonies Table

# In[7]:

create_query = "CREATE TABLE IF NOT EXISTS ceremonies( id INTEGER PRIMARY KEY, Year INTEGER, Host TEXT, UNIQUE(Year, Host) );"

c.execute(create_query)

years_hosts = [(2010, "Steve Martin"),
               (2009, "Hugh Jackman"),
               (2008, "Jon Stewart"),
               (2007, "Ellen DeGeneres"),
               (2006, "Jon Stewart"),
               (2005, "Chris Rock"),
               (2004, "Billy Crystal"),
               (2003, "Steve Martin"),
               (2002, "Whoopi Goldberg"),
               (2001, "Steve Martin"),
               (2000, "Billy Crystal"),
            ]

insert_query = "INSERT OR IGNORE INTO ceremonies (Year, Host) VALUES (?,?);"
conn.executemany(insert_query, years_hosts)

first_ten_query = "SELECT * FROM ceremonies LIMIT 10;"
c.execute(first_ten_query)
results = c.fetchall()
print("FIRST TEN ROWS OF CEREMONIES")
for row in results:
    print(row)
print('\n')

schema_query = "PRAGMA TABLE_INFO(ceremonies);"
c.execute(schema_query)
results = c.fetchall()
print("TABLE SCHEMA OF CEREMONIES")
for row in results:
    print(row)


# **Part 3:** Foreign Key Constraints

# In[9]:

fk_query = "PRAGMA foreign_keys = ON;"
c.execute(fk_query)
results = c.fetchall()


# **Part 4:** Setting Up One-To-Many

# In[11]:

create_query = "CREATE TABLE IF NOT EXISTS nominations_two( id INTEGER PRIMARY KEY, category TEXT, nominee TEXT, movie TEXT, character TEXT, won INTEGER, ceremony_id INTEGER REFERENCES ceremonies(id) );"

c.execute(create_query)
success = c.fetchall()

join_query = "SELECT nominations.category, nominations.nominee, nominations.movie, nominations.character, nominations.won, ceremonies.id FROM nominations INNER JOIN ceremonies ON nominations.year == ceremonies.year;"

c.execute(join_query)
joined_nominations = c.fetchall()

insert_nominations_two = "INSERT OR IGNORE INTO nominations_two (category, nominee, movie, character, won, ceremony_id) VALUES (?,?,?,?,?,?);"

conn.executemany(insert_nominations_two, joined_nominations)

preview_query = "SELECT * FROM nominations_two LIMIT 5;"
c.execute(preview_query)
preview = c.fetchall()
print("PREVIEW OF ROWS IN nominations_two")
for row in preview:
    print(row)


# **Part 6:** Deleting and Renaming Tables

# In[12]:

delete_query = "DROP TABLE nominations;"
c.execute(delete_query)
success = c.fetchall()

rename_query = "ALTER TABLE nominations_two RENAME TO nominations;"
c.execute(rename_query)
success = c.fetchall()


# **Part 6:** Creating a Join Table

# In[13]:

create_movies = "CREATE TABLE IF NOT EXISTS movies( id INTEGER PRIMARY KEY, movie TEXT );"

c.execute(create_movies)
success = c.fetchall()

create_actors = "CREATE TABLE IF NOT EXISTS actors( id INTEGER PRIMARY KEY, actor TEXT );"

c.execute(create_actors)
success = c.fetchall()

create_joint = "CREATE TABLE IF NOT EXISTS movies_actors( id INTEGER PRIMARY KEY, movie_id INTEGER REFERENCES movies(id), actor_id INTEGER REFERENCES actors(id) );"

c.execute(create_joint)
success = c.fetchall()


# **Part 7:** Next Steps
# 
# The suggestions for additional work are: (1) identify other datasets we can add to the database, and (2) determine how you would population the three new tables from nominations.
# 
# For now, however, I'm just going to park this project.
