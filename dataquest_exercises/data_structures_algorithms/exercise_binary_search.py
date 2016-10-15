## 4. Implementing binary search -- Part 1 ##

# A function to extract a player's last name
def format_name(name):
    return name.split(" ")[1] + ", " + name.split(" ")[0]

# The length of the dataset
length = len(nba)

# Implement the player_age function. For now, simply return what is specified in the instructions
def player_age(name):
    # We need to appropriately format our name for successful comparison
    name = format_name(name)
    # First guess halfway through the list
    first_guess_index = math.floor(length/2)
    first_guess = format_name(nba[first_guess_index][0])
    # Check where we should continue searching
    if first_guess == name:
        return "found"
    elif first_guess > name:
        return "earlier"
    else:
        return "later"
    
johnson_odom_age = player_age("Darius Johnson-Odom")
young_age = player_age("Nick Young")
adrien_age = player_age("Jeff Adrien")
print(johnson_odom_age)
print(young_age)
print(adrien_age)

## 5. Implementing binary search -- Part 2 ##

# A function to extract a player's last name
def format_name(name):
    return name.split(" ")[1] + ", " + name.split(" ")[0]

# The length of the dataset
length = len(nba)

# Implement the player_age function. For now, simply return what is specified in the instructions
def player_age(name):
    # We need to appropriately format our name for successful comparison
    name = format_name(name)
    # Initial bounds of the search
    upper_bound = length - 1
    lower_bound = 0
    # Index of first split
    first_guess_index = math.floor(length/2)
    first_guess = format_name(nba[first_guess_index][0])
    # If the name comes before our guess
    if first_guess > name:
        # Adjust the bounds as needed
        upper_bound = first_guess_index - 1
    # Else if the name comes after our guess
    elif first_guess < name:
        # Adjust the bounds as needed
        lower_bound = first_guess_index + 1
    # Else
    else:
        # Player found, so return first guess
        return first_guess
    # Set the index of the second split
    second_guess_index = math.floor((upper_bound-lower_bound)/2 + lower_bound)
    # Find and format the second guess
    second_guess = format_name(nba[second_guess_index][0])
    # Return the second guess
    return second_guess

gasol_age = player_age("Pau Gasol")
pierce_age = player_age("Paul Pierce")

## 7. Implementing binary search -- Part 3 ##

# A function to extract a player's last name
def format_name(name):
    return name.split(" ")[1] + ", " + name.split(" ")[0]

# The length of the dataset
length = len(nba)

# Implement the player_age function. For now, simply return what is specified in the instructions
def player_age(name):
    # We need to appropriately format our name for successful comparison
    name = format_name(name)
    # Bounds of the search
    upper_bound = length - 1
    lower_bound = 0
    # Index of first split. It's important to understand how this is computed
    index = math.floor((upper_bound + lower_bound) / 2)
    # First guess halfway through the list
    guess = format_name(nba[index][0])
    # Keep guessing until the name is found. Use a while loop here
        # Check where our guess is in relation to the desired name,
        #     and adjust our bounds as necessary (multiple lines here).
        #     If we have found the name, we wouldn't be in this loop, so
        #     we shouldn't worry about that case
        # Find the new index of our guess
        # Find and format the new guess value
    # When our loop terminates, we have found the desired nba player's name

carmelo_age = "found"
## NOTES: This exercise timed out repeatedly on me, even when using the correct solution provided ##

## 8. Implementing binary search -- Part 4 ##

# A function to extract a player's last name
def format_name(name):
    return name.split(" ")[1] + ", " + name.split(" ")[0]

# The length of the dataset
length = len(nba)

# Implement the player_age function. For now, simply return what is specified in the instructions
def player_age(name):
    name = format_name(name)
    upper_bound = length - 1
    lower_bound = 0
    index = math.floor((upper_bound + lower_bound) / 2)
    guess = format_name(nba[index][0])
    while name != guess and upper_bound >= lower_bound:
        if name < guess:
            upper_bound = index - 1
        else:
            lower_bound = index + 1
        index = math.floor((lower_bound + upper_bound) / 2)
        guess = format_name(nba[index][0])
    if name == guess:
        return nba[index][2]
    else:
        return -1
    
curry_age = player_age("Stephen Curry")
griffin_age = player_age("Blake Griffin")
jordan_age = player_age("Michael Jordan")