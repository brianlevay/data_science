# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 10:20:01 2016

@author: Brian
"""
# 3. Read the file into a string
file = open("dq_unisex_names.csv", "r")
data = file.read()

# 4. Convert the string to a list
data_list = data.split("\n")
first_five = data_list[0:5]

# 5. Convert to list of lists
string_data = []
for item in data_list:
    comma_list = item.split(",")
    string_data.append(comma_list)

print(string_data[0:5])

# 6. Convert numerical values
numerical_data = []
for item in string_data:
    name = item[1]
    num = float(item[2])
    l = [name, num]
    numerical_data.append(l)

print(numerical_data[0:5])

# 7. Filter the list
thousand_or_greater = []
for item in numerical_data:
    if (item[1] >= 1000):
        thousand_or_greater.append(item[1])

print(thousand_or_greater[0:10])