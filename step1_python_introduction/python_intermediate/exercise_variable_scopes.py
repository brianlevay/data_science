## 2. Built-in functions ##

total = sum([6,11])

## 3. Overwriting a built-in function ##

sum = sum(borrower_default_count_240)

test = sum(principal_outstanding_240)

## 4. Scopes ##

def find_average(column):
    length = len(column)
    total = sum(column)
    return total / length

total = sum(borrower_default_count_240)
average = find_average(principal_outstanding_240)
print(total)

## 5. Scope isolation ##

def find_average(column):
    length = len(column)
    total = sum(column)
    return total / length

def find_length(column):
    length = len(column)
    print(length)
    return length

length = len(borrower_default_count_240)
average = find_average(principal_outstanding_240)
principal_length = find_length(principal_outstanding_240)
print(length)

## 6. Scope inheritance ##

def find_average(column):
    total = sum(column)
    # In this function, we are going to pretend that we forgot to calculate the length
    return total / length

length = 10
average = find_average(principal_outstanding_240)

## 9. Global variables ##

def test():
    global b
    b = 20

test()
print(b)

## 10. Nested contexts ##

total = 20
def find_total(l):
    print(total)
    return total

def find_length(l):
    length = len(l)
    return length

def find_average(l):
    total = 10
    return find_total(l) / find_length(l)

find_average(principal_outstanding_240)
default_average = find_average(borrower_default_count_240)