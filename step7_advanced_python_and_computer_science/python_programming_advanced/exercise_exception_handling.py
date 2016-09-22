## 2. Organizing our code ##

# Define the Trial class here
class Trial:
    def __init__(self, row):
        self.efficiency = float(row[0])
        self.individual = int(row[1])
        self.chopstick_length = int(row[2])

first_trial = Trial(chopsticks[0])

## 3. The Chopstick class ##

class Trial(object):
    def __init__(self, datarow):
        self.efficiency = float(datarow[0])
        self.individual = int(datarow[1])
        self.chopstick_length = int(datarow[2])
first_trial = Trial(chopsticks[0])

# Define the Chopstick class here
class Chopstick(object):
    def __init__(self, length):
    	self.length = length

mini_chopstick = Chopstick(100)

## 4. Storing the trials ##

class Trial(object):
    def __init__(self, datarow):
        self.efficiency = float(datarow[0])
        self.individual = int(datarow[1])
        self.chopstick_length = int(datarow[2])

class Chopstick(object):
    def __init__(self, length):
        self.length = length
        # Start our trial list empty
        self.trials = []
        # Now, fill our list with relevant trials
        for trial in chopsticks:
            if int(trial[2]) == self.length:
                self.trials.append(int(trial[2]))

medium_chopstick = Chopstick(240)

## 5. Average Efficiency ##

import math

class Trial(object):
    def __init__(self, datarow):
        self.efficiency = float(datarow[0])
        self.individual = int(datarow[1])
        self.chopstick_length = int(datarow[2])

class Chopstick(object):
    def __init__(self, length):
        self.length = length
        self.trials = []
        for row in chopsticks:
            if int(row[2]) == self.length:
                self.trials.append(Trial(row))
    def num_trials(self):
        return len(self.trials)
    
    def avg_efficiency(self):
        return math.fsum([trial.efficiency for trial in self.trials]) / self.num_trials()
    
avg_eff_210 = Chopstick(210).avg_efficiency()

## 8. Bad Data ##

class Trial(object):
    def __init__(self, datarow):
        try:
            self.efficiency = float(datarow[0])
        except:
            self.efficiency = -1
        try:
            self.individual = int(datarow[1])
        except:
            self.individual = -1
        try:
            self.chopstick_length = int(datarow[2])
        except:
            self.chopstick_length = -1

bad_trial = Trial(chopsticks[-1])

## 9. Bad Data - Part 2 ##

class Trial(object):
    def __init__(self, datarow):
        try:
            self.efficiency = float(datarow[0])
            self.individual = int(datarow[1])
            self.chopstick_length = int(datarow[2])
        except:
            self.efficiency = -1
            self.individual = -1
            self.chopstick_length = -1

class Chopstick(object):
    def __init__(self, length):
        self.length = length
        self.trials = []
        for row in chopsticks:
            if int(row[2]) == self.length:
                trial = Trial(row)
                if ((trial.efficiency != -1) & (trial.individual != -1) & (trial.chopstick_length != -1)):
                    self.trials.append(trial)
    def num_trials(self):
        return len(self.trials)
    def avg_efficiency(self):
        efficiency_sum = 0
        for trial in self.trials:
            efficiency_sum += trial.efficiency
        return efficiency_sum / self.num_trials()
                    
bad_chopstick = Chopstick(400)

## 10. Division By Zero ##

class Trial(object):
    def __init__(self, datarow):
        try:
            self.efficiency = float(datarow[0])
            self.individual = int(datarow[1])
            self.chopstick_length = int(datarow[2])
        except:
            self.efficiency = -1
            self.individual = -1
            self.chopstick_length = -1

class Chopstick(object):
    def __init__(self, length):
        self.length = length
        self.trials = []
        for row in chopsticks:
            if int(row[2]) == self.length:
                trial = Trial(row)
                if trial.individual >= 0:
                    self.trials.append(trial)
    def num_trials(self):
        return len(self.trials)
    def avg_efficiency(self):
        efficiency_sum = 0
        for trial in self.trials:
            efficiency_sum += trial.efficiency
        try:
            return efficiency_sum / self.num_trials()
        except:
            return -1

bad_average = Chopstick(100).avg_efficiency()

## 11. Most Efficient Chopsticks ##

class Trial(object):
    def __init__(self, datarow):
        try:
            self.efficiency = float(datarow[0])
            self.individual = int(datarow[1])
            self.chopstick_length = int(datarow[2])
        except:
            self.efficiency = -1
            self.individual = -1
            self.chopstick_length = -1

class Chopstick(object):
    def __init__(self, length):
        self.length = length
        self.trials = []
        for row in chopsticks:
            if int(row[2]) == self.length:
                trial = Trial(row)
                if trial.individual >= 0:
                    self.trials.append(trial)
    def num_trials(self):
        return len(self.trials)
    def avg_efficiency(self):
        efficiency_sum = 0
        for trial in self.trials:
            efficiency_sum += trial.efficiency
        try:
            return efficiency_sum / self.num_trials()
        except ZeroDivisionError:
            return -1.0
        
        
chopstick_lengths = [180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330]
chopstick_list = [Chopstick(length) for length in chopstick_lengths]

## 12. Most Efficient Chopsticks - Part 2 ##

class Trial(object):
    def __init__(self, datarow):
        try:
            self.efficiency = float(datarow[0])
            self.individual = int(datarow[1])
            self.chopstick_length = int(datarow[2])
        except:
            self.efficiency = -1
            self.individual = -1
            self.chopstick_length = -1

class Chopstick(object):
    def __init__(self, length):
        self.length = length
        self.trials = []
        for row in chopsticks:
            if int(row[2]) == self.length:
                trial = Trial(row)
                if trial.individual >= 0:
                    self.trials.append(trial)
    def num_trials(self):
        return len(self.trials)
    def avg_efficiency(self):
        efficiency_sum = 0
        for trial in self.trials:
            efficiency_sum += trial.efficiency
        try:
            return efficiency_sum / self.num_trials()
        except ZeroDivisionError:
            return -1.0
    def __lt__(self, other):
        return self.avg_efficiency() < other.avg_efficiency()
    def __gt__(self, other):
        return self.avg_efficiency() > other.avg_efficiency()
    def __le__(self, other):
        return self.avg_efficiency() <= other.avg_efficiency()
    def __ge__(self, other):
        return self.avg_efficiency() >= other.avg_efficiency()
    def __eq__(self, other):
        return self.avg_efficiency() == other.avg_efficiency()
    def __ne__(self, other):
        return self.avg_efficiency() != other.avg_efficiency()
    
        
chopstick_lengths = [180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330]

chopstick_list = [Chopstick(length) for length in chopstick_lengths]
most_efficient = max(chopstick_list)