# For Training and Getting Tweets from a Postive Tweet Generator

from textgenrnn import textgenrnn
import pandas as pd
import time

data = pd.read_csv('./resources/Sentiment.csv')
data = data[['text','sentiment']]

data_pos = data[ data['sentiment'] == 'Positive']
data_neg = data[ data['sentiment'] == 'Negative']
data_neut = data[ data['sentiment'] == 'Neutral']

# Instantiating textgenrn
postive_textgen = textgenrnn()


positive_texts = data_pos['text'].values
postive_textgen.train_on_texts(positive_texts[0:4000], num_epochs=5,  gen_epochs=2)

# If weights from previously saved model are available
#postive_textgen = textgenrnn('pos_w.hdf5')

# Generating Artificial Tweets

t0 = time.time()

# Adding the artificial negative tweets back to the positive tweets df
artificial_positive_tweets_content = postive_textgen.generate(n=3500, temperature=0.8, return_as_list=True)
artificial_positive_tweets = pd.DataFrame({"text":artificial_positive_tweets_content, "sentiment":"Positive"})
data_pos = artificial_positive_tweets

t1 = time.time()

total = t1-t0

# This run took 3221.604505300522 seconds
print("Tweet generation took " + str(total) + " units of time")

data_pos.to_csv("./resources/artificial_data/generative_neg_big_0_8.csv", encoding='utf-8', index=False)