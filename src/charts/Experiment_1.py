import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


organic_d = pd.DataFrame({'F1 Score': [0.8978, 0.5923],
     'Sentiment': ['Negative', 'Positive'],
      'Augmentation Type': ['Duplicate', 'Duplicate']})

heuristic_augmentation_d = pd.DataFrame({'F1 Score': [0.9156, 0.6666],
      'Sentiment': ['Negative', 'Positive'],
      'Augmentation Type': ['Heuristic', 'Heuristic']})

generative_augmentation_d = pd.DataFrame({'F1 Score': [0.9013, 0.6191],
      'Sentiment': ['Negative', 'Positive'],
      'Augmentation Type': ['Generative', 'Generative']})

data = pd.concat([organic_d, heuristic_augmentation_d, generative_augmentation_d])

plot = sns.barplot(x="Sentiment", y="F1 Score", hue="Augmentation Type", data=data)
plot.figure.savefig('../out/figures/experiment_1_lstm_scores.png')

plt.title('F1 Score in Sentiment Classification By Augmentation Type')
plt.show()