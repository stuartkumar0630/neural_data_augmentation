import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="darkgrid")

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")

positive_d = pd.DataFrame({'F1 Score': [0.5789, 0.6464, 0.6225, 0.6229],
     'Temperature': [0.4, 0.8, 1.2, 1.6],
      'Classification Objective': ['Positive', 'Positive', 'Positive', 'Positive']})

negative_d = pd.DataFrame({'F1 Score': [0.8844, 0.9080, 0.9048, 0.9006],
     'Temperature': [0.4, 0.8, 1.2, 1.6],
      'Classification Objective': ['Negative', 'Negative', 'Negative', 'Negative']})

data = pd.concat([positive_d, negative_d])

# Plot the responses for different events and regions
plot = sns.pointplot(x="Temperature", y="F1 Score", hue='Classification Objective', data=data)

plt.ylim(0.55, 1)
plt.xlim(-1, 4)

plt.title('F1 Score in Sentiment Classification By Temperature')
plot.figure.savefig('./resources/experiment_4_temp.png')

plt.show()