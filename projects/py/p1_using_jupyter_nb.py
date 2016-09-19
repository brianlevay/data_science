
# coding: utf-8

# In[4]:

import pandas as pd
white_house = pd.read_csv("data/2015_white_house.csv")
print(white_house.shape)


# In[2]:

print(white_house.iloc[0])
print(white_house.iloc[len(white_house.index)-1])


# In[3]:

white_house


# In[4]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.hist(white_house["Salary"])
plt.show()


# **Jupyter Notebook Guided Project**
# 
# This project is a great introduction to using Jupyter notebooks for data science. It emphasizes the ability to mix code, formated results, and graphics to build a *story* that others can follow.

# **Additional Questions:**
#     
# * How does length of employee titles correlate to salary?
# * How much does the White House pay in total salary?
# * Who are the highest and lowest paid staffers?
# * What words are the most common in titles?

# **Question 1: Title Length vs Salary**

# In[10]:

title_lengths = white_house['Position Title'].apply(lambda x: len(x))

plt.scatter(title_lengths, white_house['Salary'])
plt.xlim(0, 160)
plt.ylim(0, 200000)
plt.xlabel('Length of Job Title (chars)')
plt.ylabel('Annual Salary (USD)')
plt.show()


# **Answer 1:** There appears to be a very weak positive correlation between job title and annual salary

# **Question 2: Total Salary Payments**

# In[14]:

total_salary = white_house['Salary'].sum()
num_employees = len(white_house.index)
ave_salary = total_salary / num_employees

print('Total Salary Payments (USD): %i' % total_salary)
print('Number of Employees: %i' % num_employees)
print('Average Salary (USD): %i' % ave_salary)


# **Answer 2:** The White House spends ~40 million dollars a year on salary for its 474 employees. The average salary is ~$84,000.

# **Question 3: Highest and Lowest Paid Staffers**

# In[20]:

# finds the minimum salary and the employee(s) who make it
min_salary = white_house['Salary'].min()
employees_have_min = (white_house['Salary'] == min_salary)
employees_with_min = white_house[['Name','Position Title']].loc[employees_have_min]
print('Minimum Salary: %i' % min_salary)
print(employees_with_min)
print('\n')

# finds the maximum salary and the employee(s) who make it
max_salary = white_house['Salary'].max()
employees_have_max = (white_house['Salary'] == max_salary)
employees_with_max = white_house[['Name','Position Title']].loc[employees_have_max]
print('Maximum Salary: %i' % max_salary)
print(employees_with_max)


# **Answer 3:** The lowest paid staffers don't get paid at all, and the highest paid staffers receive a salary of ~$174,000. See the lists above for the names and job titles of the individuals in each category.

# **Question 4: Most Common Words in Job Titles**

# In[38]:

words = {}
positions = set(list(white_house['Position Title']))

for pos in positions:
    title_parts = pos.split(' ')
    for part in title_parts:
        if part in words:
            words[part] += 1
        else:
            words[part] = 1

word_text = list(words.keys())
word_count = list(words.values())

word_series = pd.Series(index=word_text, data=word_count)
word_series.sort_values(ascending=False, inplace=True)
print(word_series[0:10])
    


# **Answer 4:** The most common words in White House job titles are articles. Beyond those (they are irrelevant to this analysis), the five most common words are "Assistant", "Director", "President", "Special" and "Deputy". 
