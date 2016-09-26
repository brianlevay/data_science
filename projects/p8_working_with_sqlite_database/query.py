import sqlite3

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query = "SELECT name FROM facts ORDER BY population ASC LIMIT 10;"
c.execute(query)
results = c.fetchall()

print(results)