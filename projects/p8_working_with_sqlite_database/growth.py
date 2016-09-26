import sqlite3
import pandas as pd
import math

# This section reads in the data
conn = sqlite3.connect("factbook.db")
query = "SELECT * FROM facts;"
table = pd.read_sql_query(query, conn)
conn.close()

# This section removes rows that have missing data
table_clean = table.dropna(axis=0)
table_clean = table_clean.loc[table_clean["area_land"]!=0]

# This function calculates the projected population in 2050
def projected_pop(row):
    pop_2015 = row["population"]
    rate = row["population_growth"] / 100
    pop_2050 = pop_2015 * (math.e ** (rate * 35))
    return pop_2050

# This section applies the function to generate a new projected_pops Series
# Then creates a new DataFrame with the appended Series
# Then sorts by projected population in descending order
# Then prints the top 10 results
projected_pops = table_clean.apply(projected_pop, axis=1)
table_proj = table_clean.assign(population_2050 = projected_pops)
table_proj.sort_values("population_2050", inplace=True, ascending=False)
print(table_proj[["name","population_2050"]].head(10))
    