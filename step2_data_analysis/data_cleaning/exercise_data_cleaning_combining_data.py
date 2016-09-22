## 2. Condensing class size ##

class_size = data['class_size']
selection = (class_size['GRADE '] == '09-12') & (class_size['PROGRAM TYPE'] == 'GEN ED')

class_size = class_size.loc[selection]

print(class_size.head(5))

## 3. Computing average class sizes ##

import numpy

grouped_cs = class_size.groupby('DBN')
aggregate_cs = grouped_cs.agg(numpy.mean)

class_size = aggregate_cs
class_size.reset_index(inplace=True)

data['class_size'] = class_size

print(data['class_size'].head(5))

## 4. Condensing demographics ##

data['demographics'] = data['demographics'].loc[data['demographics']['schoolyear'] == 20112012]
print(data['demographics'].head(5))

## 5. Condensing graduation ##

selection = (data['graduation']['Cohort'] == "2006") & (data['graduation']['Demographic'] == "Total Cohort")
data['graduation'] = data['graduation'].loc[selection]

print(data['graduation'].head(5))


## 6. Converting AP test scores ##

cols = ['AP Test Takers ', 'Total Exams Taken', 'Number of Exams with scores 3 4 or 5']

for col in cols:
    data['ap_2010'][col] = pd.to_numeric(data['ap_2010'][col], errors="coerce")

print(data['ap_2010'].head(5))

## 8. Performing the left joins ##

combined = data["sat_results"]
combined = combined.merge(data["ap_2010"], on="DBN", how="left")
combined = combined.merge(data["graduation"], on="DBN", how="left")

print(combined.head(5))
print(combined.shape)

## 9. Performing the inner joins ##

combined = combined.merge(data["class_size"], on="DBN", how="inner")
combined = combined.merge(data["demographics"], on="DBN", how="inner")
combined = combined.merge(data["survey"], on="DBN", how="inner")
combined = combined.merge(data["hs_directory"], on="DBN", how="inner")

print(combined.head(5))
print(combined.shape)

## 10. Filling in missing values ##

means = combined.mean()
combined = combined.fillna(means)
combined = combined.fillna(0)

print(combined.head(5))

## 11. Adding a school district column ##

def first_two_chars(string):
    return string[0:2]

combined["school_dist"] = combined["DBN"].apply(first_two_chars)

print(combined["school_dist"].head(5))