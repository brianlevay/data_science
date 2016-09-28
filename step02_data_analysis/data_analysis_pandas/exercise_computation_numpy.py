## 2. Array comparisons ##

countries_canada = (world_alcohol[:,2] == "Canada")
years_1984 = (world_alcohol[:,0] == "1984")

## 3. Selecting elements ##

country_is_algeria = (world_alcohol[:,2] == "Algeria")
country_algeria = world_alcohol[country_is_algeria,:]

## 4. Comparisons with multiple conditions ##

is_algeria_and_1986 = (world_alcohol[:,0] == "1986") & (world_alcohol[:,2] == "Algeria")
rows_with_algeria_and_1986 = world_alcohol[is_algeria_and_1986,:]

## 5. Replacing values ##

has_1986 = world_alcohol[:,0] == "1986"
world_alcohol[has_1986,0] = "2014"

has_wine = world_alcohol[:,3] == "Wine"
world_alcohol[has_wine,3] = "Grog"

## 6. Replacing empty strings ##

is_value_empty = (world_alcohol[:,4] == '')
world_alcohol[is_value_empty, 4] = "0"

## 7. Converting data types ##

alcohol_consumption = world_alcohol[:,4]
alcohol_consumption = alcohol_consumption.astype(float)

## 8. Computing with NumPy ##

total_alcohol = alcohol_consumption.sum()
average_alcohol = alcohol_consumption.mean()

## 9. Total alcohol consumption in a year ##

has_canada_1986 = (world_alcohol[:,0] == "1986") & (world_alcohol[:,2] == "Canada")
canada_1986 = world_alcohol[has_canada_1986,:]
has_empty = (canada_1986[:,4] == '')
canada_1986[has_empty,4] = "0"
canada_alcohol = canada_1986[:,4].astype(float)
total_canadian_drinking = canada_alcohol.sum()

## 10. Calculating consumption for each country ##

totals = {}
has_year = (world_alcohol[:,0] == "1989")
year = world_alcohol[has_year,:]

for country in countries:
    has_country = (year[:,2] == country)
    country_consumption = year[has_country,4]
    has_empty = (country_consumption[:] == "")
    country_consumption[has_empty] = "0"
    country_consumption = country_consumption.astype(float)
    total_for_country = country_consumption.sum()
    totals[country] = total_for_country

## 11. Finding the country that drinks the most ##

highest_value = 0
highest_key = None

for key, val in totals.items():
    if (highest_key is None) | (val > highest_value):
        highest_value = val
        highest_key = key