## 1. Introduction ##

import sqlite3
conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query1 = "\
EXPLAIN QUERY PLAN \
SELECT * FROM facts \
WHERE population > 1000000 AND population_growth < 0.05;"

c.execute(query1)
query_plan_one = c.fetchall()
print(query_plan_one)

conn.close()

## 2. Query plan for multi-column queries ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

create_pop_in = "CREATE INDEX IF NOT EXISTS pop_idx ON facts(population);"
c.execute(create_pop_in)
create_gro_in = "CREATE INDEX IF NOT EXISTS pop_growth_idx ON facts(population_growth);"

query2 = "\
EXPLAIN QUERY PLAN \
SELECT * FROM facts \
WHERE population > 1000000 AND population_growth < 0.05;"
c.execute(query2)
query_plan_two = c.fetchall()
print(query_plan_two)

conn.close()

## 5. Creating a multi-column index ##

conn = sqlite3.connect("factbook.db")
conn.execute("create index if not exists pop_pop_growth_idx on facts(population, population_growth);")
query_plan_three = conn.execute("explain query plan select * from facts where population > 1000000 and population_growth < 0.05;").fetchall()
print(query_plan_three)

## 6. Covering index ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()
c.execute("create index if not exists pop_pop_growth_idx on facts(population, population_growth);")

query4 = "EXPLAIN QUERY PLAN SELECT population, population_growth FROM facts WHERE population > 1000000 AND population_growth < 0.05;"
c.execute(query4)
query_plan_four = c.fetchall()
print(query_plan_four)

conn.close()

## 7. Covering index for single column ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()
c.execute("create index if not exists pop_pop_growth_idx on facts(population, population_growth);")

query5 = "EXPLAIN QUERY PLAN SELECT population FROM facts WHERE population > 1000000;"
c.execute(query5)
query_plan_five = c.fetchall()
print(query_plan_five)

conn.close()