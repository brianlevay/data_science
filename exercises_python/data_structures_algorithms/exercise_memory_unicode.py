## 2. Intro to binary ##

# Let's say a is a binary number.  In python, we have to store binary numbers as strings
# Trying to say b = 10 directly will assume base 10, so strings are needed
b = "10"

# We can convert b to a binary number from a string using the int function -- the optional second argument base is set to 2 (binary is base two)
print(int(b, 2))

base_10_100 = int("100",2)

## 3. Binary addition ##

# a is in base 10 -- because we have 10 possible digits, the highest value we can represent with one digit is 9
a = 9

# When we want to represent a value one higher, we need to add another digit
a += 1
# a now has two digits -- we incremented the invisible leading digit, which was 0 and is now 1, and set the last digit back to zero.
print(a)

# When we add 1 to 19, we increment the leading 1 by 1, and then set the last digit to 0, giving us 20.
a = 19
a += 1

# When we add 1 to 99, we increment the last digit by 1, and add 1 to the first digit, but the first digit is now greater than 9, so we have to increment the invisible leading digit.
a = 99
a += 1

# Binary addition works the exact same way, except the highest value any single digit can represent is 1.
b = "1"

# We'll add binary values using a binary_add function that was made just for this exercise
# It's not extremely important to know how it works right this second
def binary_add(a, b):
    return bin(int(a, 2) + int(b, 2))[2:]

c = binary_add(b, "1")

# We now see that c equals "10", which is exactly what happens in base 10 when we reach the highest possible digit.
print(c)

# c now equals "11"
c = binary_add(c, "1")
print(c)

# c now equals "100"
c = binary_add(c, "1")
print(c)

c = binary_add(c, "10")
print(c)

## 4. Converting binary values ##

def binary_add(a, b):
    return bin(int(a, 2) + int(b, 2))[2:]

# Start both at 0
a = 0
b = "0"

# Loop 10 times
for i in range(0, 10):
    # Add 1 to each
    a += 1
    b = binary_add(b, "1")

    # Check if they are equal
    print(int(b, 2) == a)

# The cool thing here is that a and b are always equal if you add the same amount to both
# This is because base 2 and base 10 are just ways to write numbers
# Counting 100 apples in base 2 or base 10 will always give you an equivalent result, you just have to convert between them
# We can represent any number in binary, we just need more digits than we would in base 10

base_10_1001 = int("1001",2)

## 5. Characters to binary ##

# We can use the ord() function to get the integer associated with an ascii character.
ord('a')

# Then we use the bin() function to convert to binary
# The bin function adds "0b" to the start of strings to indicate that they contain binary values
bin(ord('a'))

# ÿ is the "last" ascii character -- it has the highest integer value of any ascii character
# This is because 255 is the highest value that can be represented with 8 binary digits
ord('ÿ')
# As you can see, we get 8 1's, which shows that this is the highest possible 8 digit value
bin(ord('ÿ'))

# Why is this?  It's because a single binary digit is called a bit, and computers store values in sequences of bytes, which are 8 bits together.
# You might be more familiar with kilobytes or megabytes -- a kilobyte is 1000 bytes, and a megabyte is 1000 kilobytes.
# There are 256 different ascii symbols, because the largest amount of storage any single ascii character can take up is one byte.

binary_w = bin(ord("w"))
binary_bracket = bin(ord("}"))
print(binary_w)
print(binary_bracket)

## 6. Intro to unicode ##

# We can initialize unicode code points (the value for this code point is \u27F6, but you see it as a character because it is being automatically converted)
code_point = "⟶"

# This particular code point maps to a right arrow character
print(code_point)

# We can get the base 10 integer value of the code point with the ord function
print(ord(code_point))

# As you can see, this takes up a lot more than 1 byte
print(bin(ord(code_point)))

binary_1019 = bin(ord("\u1019"))
print(binary_1019)

## 7. Strings with unicode ##

s1 = "café"
# The \u prefix means "the next 4 digits are a unicode code point"
# It doesn't change the value at all (the last character in the string below is \u00e9)
s2 = "café"

# These strings are the same, because code points are equal to their corresponding unicode character.
# \u00e9 and é are equivalent.
print(s1 == s2)

s3 = "dog\u00e9"
print(s3)

## 8. The bytes type ##

# We can make a string with some unicode values
superman = "Clark Kent␦"
print(superman)

# This tells python to encode the string superman into unicode using the utf-8 encoding
# We end up with a sequence of bytes instead of a string
superman_bytes = "Clark Kent␦".encode("utf-8")

batman = "Bruce Wayne␦"
batman_bytes = batman.encode("utf-8")
print(batman_bytes)

## 10. Hexadecimal conversions ##

# F is the highest single digit in hexadecimal (base 16)
# Its value is 15 in base 10
print(int("F", 16))

# A in base 16 has the value 10 in base 10
print(int("A", 16))

# Just like the earlier binary_add function, this adds two hex numbers
def hexadecimal_add(a, b):
    return hex(int(a, 16) + int(b, 16))[2:]

# When we add 1 to 9 in hexadecimal, it becomes "a"
value = "9"
value = hexadecimal_add(value, "1")
print(value)

hex_ea = hexadecimal_add("ea","2")
print(hex_ea)
hex_ef = hexadecimal_add("e","f")
print(hex_ef)

## 11. Hex to binary ##

# One byte (8 bits) in hexadecimal (the value of the byte below is \xe2)
hex_byte = "â"

# Print the base 10 integer value for the hex byte
print(ord(hex_byte))

# This gives the exact same value -- remember than \x is just a prefix, and doesn't affect the value
print(int("e2", 16))

# Convert the base 10 integer to binary
print(bin(ord("â")))

binary_aa = bin(ord("\xaa"))
print(binary_aa)
binary_ab = bin(ord("\xab"))
print(binary_ab)

## 12. Bytes and strings ##

hulk_bytes = "Bruce Banner␦".encode("utf-8")

# We can't mix strings and bytes
# For instance, if we try to replace the unicode ␦ character as a string, it won't work, because that value has been encoded to bytes
try:
    hulk_bytes.replace("Banner", "")
except Exception:
    print("TypeError with replacement")

# We can create objects of the bytes datatype by putting a b in front of the quotation marks in a string
hulk_bytes = b"Bruce Banner"
# Now, instead of mixing strings and bytes, we can use the replace method with bytes objects instead
hulk_bytes.replace(b"Banner", b"")

thor_bytes = b"Thor"

## 13. Decode bytes to strings ##

# Make a bytes object with aquaman's secret identity
aquaman_bytes = b"Who knows?"

# Now, we can use the decode method, along with the encoding (utf-8) to turn it into a string.
aquaman = aquaman_bytes.decode("utf-8")

# We can print the value and type out to verify that it is a string.
print(aquaman)
print(type(aquaman))

morgan_freeman_bytes = b"Morgan Freeman"
morgan_freeman = morgan_freeman_bytes.decode('utf-8')
print(morgan_freeman)

## 14. Read in file data ##

# We can read our data in using csvreader
import csv
# When we open a file, we can specify the encoding that it's in.  In this case, utf-8
f = open("sentences_cia.csv", 'r', encoding="utf-8")
csvreader = csv.reader(f)
sentences_cia = list(csvreader)

# The data is two columns
# First column is year, second is a sentence from a CIA report written that year
# Print the first column of the second row
print(sentences_cia[1][0])

# Print the second column of the second row
print(sentences_cia[1][1])

sentences_ten = sentences_cia[9][1]
print(sentences_ten)

## 15. Convert to a dataframe ##

import csv
# Let's read in the legislators data from a few missions ago
f = open("legislators.csv", 'r', encoding="utf-8")
csvreader = csv.reader(f)
legislators = list(csvreader)

# Now, we can import pandas and use the DataFrame class to convert the list of lists to a dataframe
import pandas as pd

legislators_df = pd.DataFrame(legislators)

# As you can see, the first row is the headers, which we don't want (it's not actually data, it's just headers)
print(legislators_df.iloc[0,:])

# In order to remove the headers, we'll subset the df and pass them in separately
# This code removes the headers from legislators, and instead passes them into the columns argument
# The columns argument specifies column names
legislators_df = pd.DataFrame(legislators[1:], columns=legislators[0])
# We now have the right data in the first row, and the proper headers
print(legislators_df.iloc[0,:])

# The sentences_cia data from last screen is available.
sentences_cia_df = pd.DataFrame(sentences_cia[1:],columns=sentences_cia[0])

## 16. Clean up sentences ##

# The integer codes for all the characters we want to keep
good_characters = [48, 49, 50, 51, 52, 53, 54, 55, 56, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 32]

sentence_15 = sentences_cia["statement"][14]

# Iterate over the characters in the sentence, and only take those whose integer representations are in good_characters
# This will construct a list of single characters
cleaned_sentence_15_list = [s for s in sentence_15 if ord(s) in good_characters]

# Join the list together, separated by "" (no space), which creates a string again
cleaned_sentence_15 = "".join(cleaned_sentence_15_list)

def clean_statement(row):
    sentence = row["statement"]
    cleaned_sentence_list = [s for s in sentence if ord(s) in good_characters]
    cleaned_sentence = "".join(cleaned_sentence_list)
    return cleaned_sentence

cleaned_statement = sentences_cia.apply(clean_statement,axis=1)
sentences_cia["cleaned_statement"] = cleaned_statement

## 17. Tokenize statements ##

# We can use the .join() method on strings to join lists together.
# The string we use the method on will be used as the separator -- the character(s) between each string when they are joined.
combined_statements = " ".join(sentences_cia["cleaned_statement"])
statement_tokens = combined_statements.split(" ")

## 18. Filter the tokens ##

# statement_tokens has been loaded in.
filtered_tokens = [t for t in statement_tokens if len(t) >= 5]

## 19. Count the tokens ##

from collections import Counter
fruits = ["apple", "apple", "banana", "orange", "pear", "orange", "apple", "grape"]
fruit_count = Counter(fruits)

# Each of the items in the list has been counted up, and given a dictionary key
print(fruit_count)

# filtered_tokens has been loaded in
filtered_token_counts = Counter(filtered_tokens)
print(filtered_token_counts)

## 20. Most common tokens ##

from collections import Counter
fruits = ["apple", "apple", "banana", "orange", "pear", "orange", "apple", "grape"]
fruit_count = Counter(fruits)

# We can use the most_common method of a Counter class to get the most common items
# We pass in a number, which is the number of items we want to get
print(fruit_count.most_common(2))
print(fruit_count.most_common(3))

# filtered_token_counts has been loaded in
common_tokens = filtered_token_counts.most_common(3)
print(common_tokens)

## 21. Finding the most common tokens by year ##

# sentences_cia has been loaded in.
# It already has the cleaned_statement column.

def most_common(year_str):
    subset = sentences_cia["cleaned_statement"].loc[sentences_cia["year"] == year_str]
    combined = " ".join(subset)
    tokens = combined.split(" ")
    filtered_tokens = [t for t in tokens if len(t) >= 5]
    token_ct = Counter(filtered_tokens)
    common_tokens = token_ct.most_common(2)
    return common_tokens

common_2000 = most_common("2000")
common_2002 = most_common("2002")
common_2013 = most_common("2013")