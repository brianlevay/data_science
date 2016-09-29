
# coding: utf-8

# **Guided Project**: Police killings
# 
# This project will look at a dataset containing records of police shootings from January to June 2015

# **Parts 1 & 2:** Introduction and Setup

# In[2]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

police_killings = pd.read_csv('police_killings.csv', encoding='ISO-8859-1')

print(police_killings.columns)
print(police_killings.head(5))


# **Part 3:** Shootings by Race

# In[3]:

# US Population Stats (2015, from US Census)
race_per = {'White': 61.6, 'Black': 13.3, 'Hispanic/Latino': 17.6, 'Asian/Pacific Islander': 5.8, 'Native American': 1.2}

# This creates a DataFrame for counts of police shootings per ethnicity
by_race = police_killings['raceethnicity'].value_counts().to_frame('pct')
by_race = by_race[by_race.index != 'Unknown']

# This creates a DataFrame for relative percentages, excluding "Unknown"
by_race_per = (100*by_race) / by_race.sum()
print('Percentage of Police Shootings Belonging to Each Race/Ethnicity')
print(by_race_per)
print('\n')

# This creates a DataFrame for differences between an ethnicity's percentage of the population
# percentage of police shootings
def percent_diffs(row):
    if row.name in race_per:
        return row - race_per[row.name]
    else:
        return 0
    
relative = by_race_per.apply(percent_diffs, axis=1)
print('Differences Between Percentage of Police Shootings and Population Percentage for Each Race/Ethnicity')
print(relative)
print('\n')

# This creates the plots
fig = plt.figure(figsize=(10,10))
ax_1 = fig.add_subplot(2,1,1)
sns.barplot(x=by_race.index, y=by_race['pct'], ax=ax_1)
ax_1.set_xlabel('Ethnicity')
ax_1.set_ylabel('Police Shootings from Jan-Jun 2015')

ax_2 = fig.add_subplot(2,1,2)
sns.barplot(x=relative.index, y=relative['pct'], ax=ax_2)
ax_2.set_xlabel('Ethnicity')
ax_2.set_ylabel("Percent of Police Shootings - Population Percent")
plt.show()


# **Discussion:** White people are the ones killed most often from police shootings, but when adjusted for population percentages, blacks are far more likely (on a per capita basis) to be shot.
# 
# **Notes:** The "Unknown" category was left out of the analysis above, because it provides no insight into ethnicity trends. The US statistics above also included ~2.6% of the population that identified as more than one race (counted separated), but that category was ignored for the comparison as well due to a lack of 1:1 category match.

# **Part 4:** Shootings by Regional Income

# In[4]:

# US Median Personal Income in 2015
us_median = 30240
income = police_killings['p_income'].loc[police_killings['p_income'] != '-'].astype(int)
income_rel = income - us_median

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
income.hist(ax=ax1)
income_rel.hist(ax=ax2)
ax1.set_xlabel('Median Income (USD) Where Shooting Took Place')
ax1.set_ylabel('Number of Shootings')
ax2.set_xlabel('Difference Between Median Income Where Shooting Took Place and Overall US Median')
ax2.set_ylabel('Number of Shootings')
plt.show()


# **Discussion:** The majority of shootings happened in poorer communities, relative to the US as a whole. Personal income is based on census-tract level data for the location of the killing and is not necessarily indicative of the income for the individual or their community. Many suspects are killed while fleeing the police, and the locations where they are stopped may have little to do with the locations where they live.

# **Part 5:** Shootings By State

# In[16]:

state_pop = pd.read_csv('state_population.csv')

counts = police_killings['state_fp'].value_counts()
states = pd.DataFrame({"STATE": counts.index, "shootings": counts})
states = states.merge(state_pop, on='STATE')
states = states.assign(pop_millions = states['POPESTIMATE2015'] / 1000000)
states = states.assign(rate = states['shootings'] / states['pop_millions'])
sorted_states = states[['STATE','NAME','pop_millions','shootings','rate']].sort_values(by='rate', ascending=False)

print(sorted_states)

sns.lmplot(x='pop_millions', y='rate', data=states)
plt.show()
sns.lmplot(x='pop_millions', y='shootings', data=states)
plt.show()


# **Discussion:** The state with the highest number of people shot by police per capita is Oklahoma. The lowest rate is observed in Connecticut. The differences likely reflect economic conditions in the states. However, it's worth being cautious about reading too much into rates when the absolute numbers are small. If you look at the plots of rate vs population and number of shootings vs population, you can see that states such as Oklahoma and Arizona are lower in population and therefore random variations are more likely to skew the rates significantly. It's possible that the ~10 extra shootings (relative to the baseline national trend) were related to specific events or were clustered somehow. One way to test the significance of those numbers is to look for geographic or temporal clusters and see if the shootings were more likely to be grouped than in other states closer to the baseline.

# **Part 6:** State By State Differences

# In[31]:

all_real_vals = (police_killings['share_white'] != '-') & (police_killings['share_black'] != '-') & (police_killings['share_hispanic'] != '-')

pk = police_killings.loc[all_real_vals]
pk = pk.assign(share_white = pk['share_white'].astype(float))
pk = pk.assign(share_black = pk['share_black'].astype(float))
pk = pk.assign(share_hispanic = pk['share_hispanic'].astype(float))

removed = police_killings.loc[all_real_vals != True]
print('REMOVED ROWS')
print(removed[['streetaddress','city']])
print('\n')

top_10_states = sorted_states['STATE'].head(10)
bottom_10_states = sorted_states['STATE'].tail(10)
is_in_top_10 = pk['state_fp'].isin(top_10_states)
is_in_bottom_10 = pk['state_fp'].isin(bottom_10_states)
desired_cols = ['state','share_white','share_black','share_hispanic','urate','college']
shootings_in_top_10 = pk[desired_cols].loc[is_in_top_10]
shootings_in_bottom_10 = pk[desired_cols].loc[is_in_bottom_10]

print('STATE-LEVEL AVERAGES OF COMMUNITIES WHERE SHOOTINGS OCCCURRED (10 STATES WITH HIGHEST RATES)')
print(pd.pivot_table(data=shootings_in_top_10, index='state'))
print('\n')
print('STATE-LEVEL AVERAGES OF COMMUNITIES WHERE SHOOTINGS OCCCURRED (10 STATES WITH LOWEST RATES)')
print(pd.pivot_table(data=shootings_in_bottom_10, index='state'))
print('\n')

print('AVERAGE COMMUNITY COMPOSITION FOR LOCATIONS WHERE SHOOTINGS OCCURRED (10 STATES WITH HIGHEST RATES)')
print(shootings_in_top_10.mean())
print('\n')
print('AVERAGE COMMUNITY COMPOSITION FOR LOCATIONS WHERE SHOOTINGS OCCURRED (10 STATES WITH LOWEST RATES)')
print(shootings_in_bottom_10.mean())
print('\n')


# **Discussion:** Shootings that happen in the states with the highest overall rates also tend to happen in communities with higher proportions of hispanic individuals. The communities where shootings occur also tend to have slightly lower unemployment rates and slightly higher college attendance rates.
# 
# **Additional Points:** The two samples that were removed due to incomplete ethnicity data were both events that happened at airports. Removing them should have no impact on the analysis. The other point, about using county data, has merit. Right now, county data are simply averaged without population weighting. This biases the data so that rural county percentages are given equal importance. In reality, we would want to convert county fractions to county numbers, add the numbers up by state, and then convert to state fractions.

# **Part 7:** Next Steps
# 
# In the future, I may come back to further explore this dataset. For now, I will park this analysis.
