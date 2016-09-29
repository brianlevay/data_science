## 1. Probability basics ##

# Print the first two rows of the data.
print(flags[:2])

most_bars_country = flags["name"].loc[flags["bars"].idxmax()]
highest_population_country = flags["name"].loc[flags["population"].idxmax()]

## 2. Calculating probability ##

total_countries = flags.shape[0]
orange_probability = len(flags.loc[flags["orange"] == 1]) / total_countries
stripe_probability = len(flags.loc[flags["stripes"] > 1]) / total_countries

## 3. Conjunctive probabilities ##

five_heads = .5 ** 5
ten_heads = 0.5 ** 10
hundred_heads = 0.5 ** 100

## 4. Dependent probabilities ##

# Remember that whether a flag has red in it or not is in the `red` column.
t = flags.shape[0]
n = len(flags.loc[flags["red"]==1])

three_red = (n/t) * ((n-1)/(t-1)) * ((n-2)/(t-2))

## 5. Disjunctive probability ##

start = 1
end = 18000

hundred_prob = (end / 100) / end
seventy_prob = (end / 70) / end

## 6. Disjunctive dependent probabilities ##

stripes_or_bars = None
red_or_orange = None

num_tot = len(flags)
num_red = len(flags.loc[flags["red"]==1])
num_orange = len(flags.loc[flags["orange"]==1])
num_red_and_orange = len(flags.loc[(flags["red"]==1)&(flags["orange"]==1)])
red_or_orange = (num_red + num_orange - num_red_and_orange)/num_tot

num_stripes = len(flags.loc[flags["stripes"]>0])
num_bars = len(flags.loc[flags["bars"]>0])
num_stripes_and_bars = len(flags.loc[(flags["stripes"]>0)&(flags["bars"]>0)])
stripes_or_bars = (num_stripes + num_bars - num_stripes_and_bars)/num_tot

## 7. Disjunctive probabilities with multiple conditions ##

all_tails = 0.5 ** 3
heads_or = 1 - all_tails