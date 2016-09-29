## 2. Sets ##

gender = []
for leg in legislators:
    gender.append(leg[3])

gender = set(gender)
print(gender)

## 3. Exploring the dataset ##

party = []
for leg in legislators:
    party.append(leg[6])
party = set(party)
print(party)
print(legislators)

## 4. Missing values ##

for leg in legislators:
    if (leg[3] == ""):
        leg[3] = "M"


## 5. Parsing birth years ##

birth_years = []
for leg in legislators:
    parts = leg[2].split("-")
    birth_years.append(parts[0])

## 6. Try/except blocks ##

try:
    float("hello")
except Exception:
    print("Error converting to float.")

## 7. Exception instances ##

try:
    int("")
except Exception as exc:
    print(type(exc))
    print(str(exc))

## 8. The pass keyword ##

converted_years = []
for year in birth_years:
    try:
        year = int(year)
    except Exception:
        pass
    converted_years.append(year)

## 9. Convert birth years to integers ##

for leg in legislators:
    birthday = leg[2].split("-")
    birth_year = birthday[0]
    try:
        birth_year = int(birth_year)
    except Exception:
        birth_year = 0
    leg.append(birth_year)


## 10. Fill in years without a value ##

last_value = 1
for leg in legislators:
    if (leg[7] == 0):
        leg[7] = last_value
    else:
        last_value = leg[7]