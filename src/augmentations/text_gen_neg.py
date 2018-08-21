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
negative_textgen = textgenrnn()

negative_texts = data_neg['text'].values
negative_textgen.train_on_texts(negative_texts[0:4000], num_epochs=5,  gen_epochs=2)

# If weights from previously saved model are available
#negative_textgen = textgenrnn('neg_w.hdf5')

# Generating Artificial Tweets
t0 = time.time()

# Adding the artificial negative tweets back to the negaive tweets df
artificial_negative_tweets_content = negative_textgen.generate(n=3500, temperature=0.8, return_as_list=True)
artificial_negative_tweets = pd.DataFrame({"text":artificial_negative_tweets_content, "sentiment":"Negative"})
data_neg = artificial_negative_tweets

t1 = time.time()

total = t1-t0

# This run took 4007.6340339183807 seconds
print("Tweet generation took " + str(total) + " units of time")

data_neg.to_csv("./out/artificial_data/generative_neg_big_0_8.csv", encoding='utf-8', index=False)