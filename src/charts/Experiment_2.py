import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="darkgrid")


organic_d = pd.DataFrame({'epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     'loss': [0.4131, 0.2993, 0.2517, 0.2215, 0.1922, 0.1759, 0.1541, 0.1402, 0.1271, 0.1221],
      'Augmentation Type': ['duplicate', 'duplicate','duplicate', 'duplicate', 'duplicate', 'duplicate','duplicate', 'duplicate','duplicate', 'duplicate']})

heuristic_augmentation_d = pd.DataFrame({'epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     'loss': [0.4084, 0.3011, 0.2680, 0.2372, 0.2105, 0.1887,  0.1736, 0.1619, 0.1436, 0.1323],
      'Augmentation Type': ['heuristic', 'heuristic','heuristic', 'heuristic', 'heuristic',
                            'heuristic','heuristic', 'heuristic','heuristic', 'heuristic']})

generative_augmentation_d = pd.DataFrame({'epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     'loss': [0.4281, 0.3012, 0.2622, 0.2406, 0.2216, 0.1976,  0.1872, 0.1705, 0.1607, 0.1510],
      'Augmentation Type': ['generative', 'generative','generative', 'generative', 'generative',
                            'generative','generative', 'generative','generative', 'generative']})

data = pd.concat([organic_d, heuristic_augmentation_d, generative_augmentation_d])


plot = sns.pointplot(x="epoch", y="loss", hue='Augmentation Type', data=data)
plot.figure.savefig('../out/figures/experiment_2_learning_curves.png')

plt.ylim(0, 0.5)
plt.xlim(-1, 10)

plt.show()