## 1. Introduction ##

import sqlite3 as sql
conn = sql.connect("factbook.db")
c = conn.cursor()

schema_query = "PRAGMA TABLE_INFO(facts);"
c.execute(schema_query)
schema = c.fetchall()

for row in schema:
    print(row)
    
conn.close()

## 3. Explain query plan ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query1 = "\
EXPLAIN QUERY PLAN \
SELECT * FROM facts \
WHERE area > 40000;"

c.execute(query1)
query_plan_one = c.fetchall()

query2 = "\
EXPLAIN QUERY PLAN \
SELECT area FROM facts \
WHERE area > 40000;"

c.execute(query2)
query_plan_two = c.fetchall()

query3 = "\
EXPLAIN QUERY PLAN \
SELECT * FROM facts \
WHERE name == 'Czech Republic';"

c.execute(query3)
query_plan_three = c.fetchall()

conn.close()

print(query_plan_one)
print(query_plan_two)
print(query_plan_three)

## 5. Time complexity ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query = "\
EXPLAIN QUERY PLAN \
SELECT * FROM facts WHERE id == 20;"
c.execute(query)
query_plan_four = c.fetchall()

conn.close()

print(query_plan_four)

## 9. All together now ##

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query6 = "EXPLAIN QUERY PLAN SELECT * FROM facts WHERE population > 10000;"
c.execute(query6)
query_plan_six = c.fetchall()
print(query_plan_six)

add_index = "CREATE INDEX IF NOT EXISTS pop_idx ON facts(population);"
c.execute(add_index)

query7 = "EXPLAIN QUERY PLAN SELECT * FROM facts WHERE population > 10000;"
c.execute(query7)
query_plan_seven = c.fetchall()
print(query_plan_seven)

conn.close()