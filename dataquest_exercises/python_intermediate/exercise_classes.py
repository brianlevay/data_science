## 3. Class syntax ##

# MY CODE #

class Car():
    def __init__(self):
        self.color = "black"
        self.make = "honda"
        self.model = "accord"

black_honda_accord = Car()

print(black_honda_accord.color)

class Team():
    def __init__(self):
        self.name = "Tampa Bay Buccaneers"

bucs = Team()
print(bucs.name)

## 4. Instance methods and __init__ ##

# MY CODE #

class Team():
    def __init__(self, name):
        self.name = name

giants = Team("New York Giants")

## 6. More instance methods ##

import csv

f = open("nfl.csv", 'r')
nfl = list(csv.reader(f))

# The nfl data is loaded into the nfl variable.
class Team():
    def __init__(self, name):
        self.name = name

    def print_name(self):
        print(self.name)
        
    # MY CODE #
    def count_total_wins(self):
        wins = 0
        for game in nfl:
            if (game[2] == self.name):
                wins += 1
        return wins
    
    
bucs = Team("Tampa Bay Buccaneers")
bucs.print_name()

broncos_wins = Team("Denver Broncos").count_total_wins()
chiefs_wins = Team("Kansas City Chiefs").count_total_wins()

f.close()


## 7. Adding to the init function ##

import csv
class Team():
    def __init__(self, name):
        self.name = name
        f = open("nfl.csv")
        self.nfl = list(csv.reader(f))
        f.close()

    def count_total_wins(self):
        count = 0
        for row in self.nfl:
            if row[2] == self.name:
                count = count + 1
        return count

jaguars_wins = Team("Jacksonville Jaguars").count_total_wins()

## 8. Wins in a year ##

import csv
class Team():
    def __init__(self, name):
        self.name = name
        f = open("nfl.csv", 'r')
        csvreader = csv.reader(f)
        self.nfl = list(csvreader)

    def count_total_wins(self):
        count = 0
        for row in self.nfl:
            if row[2] == self.name:
                count = count + 1
        return count
    
    def count_wins_in_year(self, year_str):
        count = 0
        for row in self.nfl:
            if row[2] == self.name and row[0] == year_str:
                count = count + 1
        return count

niners_wins_2013 = Team("San Francisco 49ers").count_wins_in_year("2013")