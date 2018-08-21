# Merges Postitive and Negative generative artificial tweets into a single artificial dataset

import pandas as pd

artificial_data_pos = pd.read_csv('./resources/generative_pos_big_0_8.csv')
artificial_data_pos = artificial_data_pos[['text', 'sentiment']]

artificial_data_neg = pd.read_csv('./resources/generative_neg_big_0_8.csv')
artificial_data_neg = artificial_data_neg[['text', 'sentiment']]

frames = [artificial_data_pos, artificial_data_neg]
augmented_data = pd.concat(frames)

augmented_data.to_csv("./resources/artificial_data/generative_big_0_8.csv", encoding='utf-8', index=False)