## 3. Psycopg2 ##

import psycopg2
conn = psycopg2.connect("dbname=dq user=dq")
c = conn.cursor()
print(c)
conn.close()

## 4. Creating a table ##

conn = psycopg2.connect("dbname=dq user=dq")
c = conn.cursor()

create_query = "\
CREATE TABLE IF NOT EXISTS notes( \
id INTEGER PRIMARY KEY, \
body TEXT, \
title TEXT \
);"

c.execute(create_query)
conn.close()

## 5. SQL Transactions ##

conn = psycopg2.connect("dbname=dq user=dq")
c = conn.cursor()

create_query = "\
CREATE TABLE IF NOT EXISTS notes( \
id INTEGER PRIMARY KEY, \
body TEXT, \
title TEXT \
);"

c.execute(create_query)
conn.commit()
conn.close()

## 6. Autocommitting ##

conn = psycopg2.connect("dbname=dq user=dq")
conn.autocommit = True
c = conn.cursor()

create_query = "\
CREATE TABLE IF NOT EXISTS facts( \
id INTEGER PRIMARY KEY, \
country TEXT, \
value INTEGER \
);"

c.execute(create_query)
conn.close()

## 7. Executing queries ##

conn = psycopg2.connect("dbname=dq user=dq")
conn.autocommit = True
c = conn.cursor()

insert_query = "\
INSERT INTO notes \
SELECT 1, 'Do more missions on Dataquest.', 'Dataquest reminder' \
WHERE NOT EXISTS (SELECT * FROM notes WHERE id = 1);"
c.execute(insert_query)

preview_query = "SELECT * FROM notes;"
c.execute(preview_query)
results = c.fetchall()
print(results)

conn.close()

## 8. Creating a database ##

conn = psycopg2.connect("dbname=dq user=dq")
conn.autocommit = True
c = conn.cursor()

db_query = "CREATE DATABASE income OWNER dq;"
c.execute(db_query)

conn.close()

## 9. Deleting a database ##

conn = psycopg2.connect("dbname=dq user=dq")
conn.autocommit = True
c = conn.cursor()

db_query = "DROP DATABASE income;"
c.execute(db_query)

conn.close()