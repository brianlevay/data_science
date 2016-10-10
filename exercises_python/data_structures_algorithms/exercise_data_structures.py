## 3. Dynamic arrays ##

# Retrieving an item in an array by index
retrieval_by_index = "constant"

# Searching for a value in an unordered array
search = "linear"

# Deleting an item from an array, and filling the gap
#     by shifting all later items back by one
deletion = "linear"

# Inserting an item into the array, and shifting forward
#     every item that comes after it
insertion = "linear"

## 4. Array insertion practice ##

players = ["Reggie Jackson"]
print(players)
players.insert(1, "C.J. Watson")
print(players)
players.insert(0, "Jeff Adrien")
print(players)
players.remove("Reggie Jackson")
print(players)
players.insert(0,"Quincy Acy")
print(players)
players.insert(2,"Evan Turner")
print(players)

## 6. 2-D Array Implementation ##

red_pieces = 0
black_pieces = 0

# Find how many red and black pieces there are
for i in range(len(checker_board)):
    for j in range(len(checker_board[i])):
        if checker_board[i][j] == "red":
            red_pieces += 1
        elif checker_board[i][j] == "black":
            black_pieces += 1

## 9. Dictionary Access ##

# Population of Rio de Janeiro
rio_population = city_populations["Rio de Janeiro"]
boston_population = city_populations["Boston"]
paris_population = city_populations["Paris"]
city_populations["Boston"] -= 1
city_populations["Beijing"] += 1

## 12. Choosing a data structure ##

# Scenario A: You need to keep track of people sitting in an auditorium for a play. You will have to identify which seats are empty, and sell tickets until the auditorium is completely full. How will you store the names of who is sitting where?
scenario_A_data_structure = "2d array"

# Scenario B: You are in charge of maintaining a guest list for a wedding. You are only concerned with a list of who is coming to the party. You have to add someone's name whenever they RSVP that they will be attending the wedding.
scenario_B_data_structure = "dynamic array"

# Scenario C: Now every person who RSVP's for the wedding indicates which meal they will be eating. You have to keep track of the person, and the meal order. You need to be able to quickly find out what meal a particular person ordered, so that the waiters don't delay too long when it comes time to eat.
scenario_C_data_structure = "hash table"