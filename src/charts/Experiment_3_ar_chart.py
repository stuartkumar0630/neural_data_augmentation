import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="darkgrid")

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")

# 0.5x

positive_d = pd.DataFrame({'F1 Score': [0.6151, 0.6229, 0.6462, 0.6346, 0.6386],
     'Augmentation Ratio': [0.1, 0.2, 0.3, 0.4, 0.5],
      'Classification Objective': ['Positive', 'Positive', 'Positive', 'Positive', 'Positive']})

negative_d = pd.DataFrame({'F1 Score': [0.8995, 0.8946, 0.9080, 0.9040, 0.9016],
     'Augmentation Ratio': [0.1, 0.2, 0.3, 0.4, 0.5],
      'Classification Objective': ['Negative', 'Negative', 'Negative', 'Negative', 'Negative']})

data = pd.concat([positive_d, negative_d])

# Plot the responses for different events and regions
plot = sns.pointplot(x="Augmentation Ratio", y="F1 Score", hue='Classification Objective', data=data)

plt.ylim(0.6, 1)
plt.xlim(-1, 5)

plt.title('F1 Score in Sentiment Classification By Augmentation Ratio')
plot.figure.savefig('./resources/experiment_3_ar.png')

plt.show()