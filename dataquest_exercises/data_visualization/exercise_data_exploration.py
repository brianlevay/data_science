# -*- coding: utf-8 -*-

# 1. Data loading
import pandas as pd

recent_grads = pd.read_csv('recent_grads.csv')

# 2. Histograms
# Demo code to show Pandas .hist() method (requires matplotlib underneath)
import matplotlib.pyplot as plt

columns = ['Median','Sample_size']
recent_grads.hist(column=columns)

# 3. Customizing histograms
# Demo code to show style options
recent_grads.hist(column=columns, layout=(2,1), grid=False)

# 4. Practice: histograms
recent_grads.hist(column='Sample_size', bins=50)

# 5. Box plots
# Demo code
sample_size = recent_grads[['Sample_size', 'Major_category']]
sample_size.boxplot(by='Major_category')
plt.xticks(rotation=90)
plt.show()

# 6. Explanation: box plots
# No code

# 7. Practice: box plots
recent_grads[['Sample_size', 'Major_category']].boxplot(by='Major_category')
plt.xticks(rotation=90)
plt.show()

recent_grads[['Total', 'Major_category']].boxplot(by='Major_category')
plt.xticks(rotation=90)
plt.show()

# 8. Multiple plots on one chart
plt.scatter(recent_grads['Unemployment_rate'], recent_grads['Median'], color='red')
plt.scatter(recent_grads['ShareWomen'], recent_grads['Median'], color='blue')
plt.show()

# 9. Practice: multiple plots on one chart
plt.scatter(recent_grads['Unemployment_rate'], recent_grads['P25th'], color='red')
plt.scatter(recent_grads['ShareWomen'], recent_grads['P25th'], color='blue')
plt.show()



