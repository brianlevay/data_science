## 1. Introduction ##

import sqlite3
conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query = "SELECT AVG(population), AVG(population_growth), AVG(birth_rate), AVG(death_rate) FROM facts;"
c.execute(query)
results = c.fetchall()

pop_avg = results[0][0]
pop_growth_avg = results[0][1]
birth_rate_avg = results[0][2]
death_rate_avg = results[0][3]

## 2. Ranges ##

conn = sqlite3.connect("factbook.db")

averages = "select avg(population), avg(population_growth), avg(birth_rate), avg(death_rate), avg(migration_rate) from facts;"
avg_results = conn.execute(averages).fetchall()
pop_avg = avg_results[0][0]
pop_growth_avg = avg_results[0][1]
birth_rate_avg = avg_results[0][2]
death_rate_avg = avg_results[0][3]
mig_rate_avg = avg_results[0][4]

mins = "select min(population), min(population_growth), min(birth_rate), min(death_rate), min(migration_rate) from facts;"
min_results = conn.execute(mins).fetchall()
pop_min = min_results[0][0]
pop_growth_min = min_results[0][1]
birth_rate_min = min_results[0][2]
death_rate_min = min_results[0][3]
mig_rate_min = min_results[0][4]

maxes = "select max(population), max(population_growth), max(birth_rate), max(death_rate), max(migration_rate) from facts;"
max_results = conn.execute(maxes).fetchall()
pop_max = max_results[0][0]
pop_growth_max = max_results[0][1]
birth_rate_max = max_results[0][2]
death_rate_max = max_results[0][3]
mig_rate_max = max_results[0][4]

## 3. Filtering ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query = "\
SELECT MIN(population), MAX(population), MIN(population_growth), MAX(population_growth), \
MIN(birth_rate), MAX(birth_rate), MIN(death_rate), MAX(death_rate) \
FROM facts \
WHERE population < 2000000000 AND population > 0;"

c.execute(query)
results = c.fetchall()

pop_min = results[0][0]
pop_max = results[0][1]
pop_growth_min = results[0][2]
pop_growth_max = results[0][3]
birth_rate_min = results[0][4]
birth_rate_max = results[0][5]
death_rate_min = results[0][6]
death_rate_max = results[0][7]

## 4. Predicting future population growth ##

import sqlite3
conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query = "\
SELECT ROUND(population + (population * (population_growth/100)), 0) \
FROM facts \
WHERE (population is not null) AND (population_growth is not null) AND (population < 7000000000) AND (population > 0);"

c.execute(query)
projected_population = c.fetchall()

## 5. Exploring projected population ##

import sqlite3
conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query = "\
SELECT ROUND(MIN(population + (population * (population_growth/100))), 0),\
ROUND(MAX(population + (population * (population_growth/100))), 0), \
ROUND(AVG(population + (population * (population_growth/100))), 0) \
FROM facts \
WHERE (population is not null) AND (population_growth is not null) AND (population < 7000000000) AND (population > 0);"

c.execute(query)
results = c.fetchall()

pop_proj_min = results[0][0]
pop_proj_max = results[0][1]
pop_proj_avg = results[0][2]
