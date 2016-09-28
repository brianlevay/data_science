## 3. Connect to the database ##

import sqlite3

conn = sqlite3.connect("jobs.db")

## 6. Running a query ##

import sqlite3
conn = sqlite3.connect("jobs.db")
cursor = conn.cursor()

query = "select * from recent_grads;"
cursor.execute(query)
results = cursor.fetchall()
print(results[0:2])

query_my = "SELECT Major FROM recent_grads;"
cursor.execute(query_my)
majors = cursor.fetchall()
print(majors[0:3])

## 8. Fetching a specific number of results ##

import sqlite3
conn = sqlite3.connect("jobs.db")
cursor = conn.cursor()

query = "SELECT Major, Major_category FROM recent_grads;"
cursor.execute(query)
five_results = cursor.fetchmany(5)

## 9. Closing the connection ##

conn = sqlite3.connect("jobs.db")
conn.close()

## 10. Practice ##

conn = sqlite3.connect("jobs2.db")
cursor = conn.cursor()

query = "SELECT Major FROM recent_grads ORDER BY Major DESC;"
cursor.execute(query)
reverse_alphabetical = cursor.fetchall()

conn.close()
