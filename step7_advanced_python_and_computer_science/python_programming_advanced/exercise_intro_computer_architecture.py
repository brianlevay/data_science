## 1. Computer components ##

print("Hello World!")

## 2. Data storage ##

my_int = 12
int_addr = id(my_int)
my_str = "Dadadada Nonsense!"
str_addr = id(my_str)

## 4. Data storage in Python ##

import sys

my_int = 200
size_of_my_int = sys.getsizeof(my_int)

int1 = 10
int2 = 100000
str1 = "Hello"
str2 = "Hi"

int_diff = sys.getsizeof(int2) - sys.getsizeof(int1)
print(int_diff)

str_diff = sys.getsizeof(str2) - sys.getsizeof(str1)
print(str_diff)

## 6. Disk storage ##

import time
import csv

before_file = time.clock()
f = open("list.csv", "r")
list_from_file = list(csv.reader(f))
after_file = time.clock()
list_from_RAM = "1,2,3,4,5,6,7,8,9,10".split(",")
after_ram = time.clock()

file_time = after_file - before_file
RAM_time = after_ram - after_file
print(file_time)
print(RAM_time)

## 9. Binary ##

# num1 = (1*2^1) + (1*2^2) = 2 + 4 = 6
# num2 = (1*2^0) + (1*2^3) = 1 + 8 = 9
# num3 = (1*2^2) + (1*2^5) = 4 + 32 = 36
num1 = 6
num2 = 9
num3 = 36

## 10. Computation and control flow ##

a = 5
b = 10
print("On line 3")
if a == 5:
    print("On line 5")
else:
    print("On line 7")
if b < a:
    print("On line 9")
elif b == a:
    print("On line 11")
else:
    for i in range(3):
        print("On line 14")

printed_lines = [3, 5, 14, 14, 14]

## 11. Functions in memory ##

def my_func():
    print("On line 2")
a = 5
b = 10
print("On line 5")
my_func()
print("On line 7")
my_func()

printed_lines = [5, 2, 7, 2]