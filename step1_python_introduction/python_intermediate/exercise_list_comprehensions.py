## 2. Enumerate ##

ships = ["Andrea Doria", "Titanic", "Lusitania"]
cars = ["Ford Edsel", "Ford Pinto", "Yugo"]

for i, ship in enumerate(ships):
    print(ship)
    print(cars[i])

## 3. Adding columns ##

things = [["apple", "monkey"], ["orange", "dog"], ["banana", "cat"]]
trees = ["cedar", "maple", "fig"]

for i, thing in enumerate(things):
    thing.append(trees[i])

## 4. List comprehensions ##

apple_prices = [100, 101, 102, 105]

apple_prices_doubled = [price*2 for price in apple_prices]
apple_prices_lowered = [price-100 for price in apple_prices]

## 5. Counting up female names ##

name_counts = {}
for leg in legislators:
    if (leg[3] == "F" and leg[7] > 1940):
        name = leg[1]
        if name in name_counts:
            name_counts[name] += 1
        else:
            name_counts[name] = 1

## 7. Comparing with None ##

values = [None, 10, 20, 30, None, 50]
checks = []

for val in values:
    check = val is not None and val > 30
    checks.append(check)

## 8. Highest female name count ##

max_value = None
for key in name_counts:
    count = name_counts[key]
    if max_value is None or count > max_value:
        max_value = count


## 9. The items method ##

plant_types = {"orchid": "flower", "cedar": "tree", "maple": "tree"}
for key, value in plant_types.items():
    print(key)
    print(value)

## 10. Finding the most common female names ##

top_female_names = []
for key, val in name_counts.items():
    if val == 2:
        top_female_names.append(key)


## 11. Finding the most common male names ##

top_male_names = []
male_name_counts = {}

for leg in legislators:
    if leg[3] == "M" and leg[7] > 1940:
        if leg[1] in male_name_counts:
            male_name_counts[leg[1]] += 1
        else:
            male_name_counts[leg[1]] = 1

highest_male_count = None
for key, val in male_name_counts.items():
    if highest_male_count is None or val > highest_male_count:
        highest_male_count = val
        top_male_names = [key]
    elif val == highest_male_count:
        top_male_names.append(key)

#for key, val in male_name_counts.items():
#    if val == highest_male_count:
#        top_male_names.append(key)