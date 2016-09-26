import sqlite3

conn = sqlite3.connect("factbook.db")
c = conn.cursor()

query_land = "SELECT SUM(area_land) FROM facts WHERE area_land != '';"
query_water = "SELECT SUM(area_water) FROM facts WHERE area_water != '';"

c.execute(query_land)
tot_land = c.fetchall()
c.execute(query_water)
tot_water = c.fetchall()

conn.close()

print(tot_land[0][0])
print(tot_water[0][0])

ratio_land_water = tot_land[0][0] / tot_water[0][0]
print(ratio_land_water)