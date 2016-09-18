## 1. Booleans ##

cat = True
dog = False
print(type(cat))

## 2. Boolean operators ##

print(cities)
first_alb = cities[0] == "Albuquerque"
second_alb = cities[1] == "Albuquerque"
first_last = cities[0] == cities[len(cities)-1]

## 3. Booleans with greater than ##

print(crime_rates)
first_500 = crime_rates[0] > 500
first_749 = crime_rates[0] >= 749
first_last = crime_rates[0] >= crime_rates[len(crime_rates)-1]

## 4. Booleans with less than ##

print(crime_rates)
second_500 = crime_rates[1] < 500
second_371 = crime_rates[1] <= 371
second_last = crime_rates[1] <= crime_rates[len(crime_rates)-1]

## 5. If statements ##

result = 0
if (cities[2] == "Anchorage"):
    result = 1

## 6. Nesting if statements ##

results = 0
if (crime_rates[0] > 500):
    if (crime_rates[1] > 300):
        results = 3

## 7. If statements and for loops ##

five_hundred_list = []
for cr in crime_rates:
    if (cr > 500):
        five_hundred_list.append(cr)

## 8. Find the highest crime rate ##

print(crime_rates)
highest = crime_rates[0]
for cr in crime_rates:
    if (cr > highest):
        highest = cr