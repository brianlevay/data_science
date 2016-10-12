## 2. Register DataFrame as a table ##

from pyspark.sql import SQLContext
sqlCtx = SQLContext(sc)
df = sqlCtx.read.json("census_2010.json")
df.registerTempTable('census2010')
tables = sqlCtx.tableNames()
print(tables)

## 3. Querying ##

query = "SELECT age FROM census2010"
results = sqlCtx.sql(query)
results.show()

## 4. Filtering ##

query = 'SELECT males, females FROM census2010 WHERE age > 5 AND age < 15'
results = sqlCtx.sql(query)
results.show()

## 5. Mixing functionality ##

query = 'SELECT males, females FROM census2010'
results = sqlCtx.sql(query)
description = results.describe()
description.show()

## 6. Multiple tables ##

from pyspark.sql import SQLContext
sqlCtx = SQLContext(sc)
df = sqlCtx.read.json("census_2010.json")
df.registerTempTable('census2010')
df = sqlCtx.read.json("census_1980.json")
df.registerTempTable('census1980')
df = sqlCtx.read.json("census_1990.json")
df.registerTempTable('census1990')
df = sqlCtx.read.json("census_2000.json")
df.registerTempTable('census2000')

tables = sqlCtx.tableNames()
print(tables)

## 7. Joins ##

query = """
SELECT census2010.total,census2000.total 
FROM census2000 
INNER JOIN census2010 
ON census2000.age = census2010.age
"""

results = sqlCtx.sql(query)
results.show()

## 8. SQL Functions ##

query = """
SELECT sum(census2010.total), sum(census2000.total), sum(census1990.total) 
FROM census2010
INNER JOIN census2000 ON census2010.age = census2000.age 
INNER JOIN census1990 ON census2000.age = census1990.age
"""

results = sqlCtx.sql(query)
results.show()