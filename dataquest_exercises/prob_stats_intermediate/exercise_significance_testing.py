## 3. Statistical significance ##

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

mean_group_a = np.mean(weight_lost_a)
mean_group_b = np.mean(weight_lost_b)
print(mean_group_a)
print(mean_group_b)
plt.hist(weight_lost_a)
plt.show()
plt.hist(weight_lost_b)
plt.show()

## 4. Test statistic ##

mean_difference = mean_group_b - mean_group_a
print(mean_difference)

## 5. Permutation test ##

mean_difference = 2.52
print(all_values)

mean_differences = []
for i in range(1000):
    group_a = []
    group_b = []
    for val in all_values:
        num = np.random.rand()
        if num >= 0.5:
            group_a.append(val)
        else:
            group_b.append(val)
    a_mean = np.mean(group_a)
    b_mean = np.mean(group_b)
    iteration_mean_difference = b_mean - a_mean
    mean_differences.append(iteration_mean_difference)

plt.hist(mean_differences)
plt.show()

## 7. Dictionary representation of a distribution ##

sampling_distribution = {}

for diff in mean_differences:
    if sampling_distribution.get(diff, False):
        val = sampling_distribution.get(diff)
        inc = val + 1
        sampling_distribution[diff] = inc
    else:
        sampling_distribution[diff] = 1



## 8. P value ##

frequencies = []
for key in sampling_distribution:
    if key >= 2.52:
        frequencies.append(sampling_distribution[key])
p_value = np.sum(frequencies) / 1000