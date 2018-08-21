import numpy as np
import matplotlib.pyplot as plt

# Data
x = [i/10 for i in range(5, 11)]
y_addition = [[5, 5, 5, 5, 5, 5], [0, 1, 2, 3, 4, 5]]
y_removal = [[5, 4, 3, 2, 1, 0], [5, 5, 5, 5, 5, 5]]

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.tight_layout()

plt.subplot(2, 1, 1)
plt.stackplot(x, y_removal, labels=['Organic Data', 'Artificial Data'])
plt.ylabel('Total Size of Training Set')
plt.xlabel('Augmentation Ratio')
plt.title('Removal of Organic Data')
plt.legend(loc='upper left')

plt.subplot(2, 1, 2)
plt.stackplot(x, y_addition, labels=['Organic Data', 'Artificial Data'])
plt.ylabel('Total Size of Training Set')
plt.xlabel('Augmentation Ratio')
plt.title('Addition of Artificial Data')
plt.legend(loc='upper left')

plt.savefig('../out/figures/experiment_3_ar_illustration.png')

plt.show()