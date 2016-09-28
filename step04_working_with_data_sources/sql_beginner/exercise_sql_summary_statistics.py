## 1. Counting in Python ##

import sqlite3

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query = "SELECT * FROM facts;"
c.execute(query)
facts = c.fetchall()

conn.close()

print(facts)
facts_count = len(facts)
print(facts_count)

## 2. Counting in SQL ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()
query = "SELECT COUNT(birth_rate) FROM facts;"
c.execute(query)
result = c.fetchall()
conn.close()

birth_rate_count = result[0][0]
print(birth_rate_count)

## 3. Min and max in SQL ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query_min = "SELECT MIN(population_growth) FROM facts;"
c.execute(query_min)
results_min = c.fetchall()
min_population_growth = results_min[0][0]
print(min_population_growth)

query_max = "SELECT MAX(death_rate) FROM facts;"
c.execute(query_max)
results_max = c.fetchall()
max_death_rate = results_max[0][0]
print(max_death_rate)

conn.close()

## 4. Sum and average in SQL ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query_sum = "SELECT SUM(area_land) FROM facts;"
c.execute(query_sum)
results_sum = c.fetchall()
total_land_area = results_sum[0][0]
print(total_land_area)

query_avg = "SELECT AVG(area_water) FROM facts;"
c.execute(query_avg)
results_avg = c.fetchall()
avg_water_area = results_avg[0][0]
print(avg_water_area)

conn.close()

## 5. Multiple aggregation functions ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query = "SELECT AVG(population), SUM(population), MAX(birth_rate) FROM facts;"
c.execute(query)
facts_stats = c.fetchall()
print(facts_stats)

conn.close()

## 6. Conditional aggregation ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query = "SELECT AVG(population_growth) FROM facts WHERE population > 10000000;"
c.execute(query)
results = c.fetchall()

population_growth = results[0][0]
print(population_growth)

## 7. Selecting unique rows ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query = "SELECT DISTINCT birth_rate FROM facts;"
c.execute(query)
unique_birth_rates = c.fetchall()
print(unique_birth_rates)

conn.close()

## 8. Distinct aggregations ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query1 = "SELECT AVG(DISTINCT birth_rate) FROM facts WHERE population > 20000000;"
c.execute(query1)
results = c.fetchall()
average_birth_rate = results[0][0]
print(average_birth_rate)

query2 = "SELECT SUM(DISTINCT population) FROM facts WHERE area_land > 1000000;"
c.execute(query2)
results = c.fetchall()
sum_population = results[0][0]
print(sum_population)

conn.close()


## 9. Arithmetic in SQL ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()
query = "SELECT population_growth / 1000000.0 FROM facts;"
c.execute(query)
population_growth_millions = c.fetchall()
print(population_growth_millions)

## 10. Arithmetic between columns ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query = "SELECT (population * population_growth) + population FROM facts;"
c.execute(query)

next_year_population = c.fetchall()
print(next_year_population)

conn.close()