from textgenrnn import textgenrnn
import pandas as pd
import time

######### Loading Data

data = pd.read_csv('./resources/Sentiment.csv')
data = data[['text','sentiment']]

data_pos = data[ data['sentiment'] == 'Positive']
data_neg = data[ data['sentiment'] == 'Negative']
data_neut = data[ data['sentiment'] == 'Neutral']

######### Generating Positive Tweets

# Instantiating textgenrn
postive_textgen = textgenrnn()

# If weights from previously saved model are available
postive_textgen = textgenrnn('./models/pos_w.hdf5')

# If a new model has to be trained
# positive_texts = data_pos['text'].values
# postive_textgen.train_on_texts(positive_texts[0:4000], num_epochs=5,  gen_epochs=2)

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

data_pos.to_csv("../resources/artificial_data/generative_neg_big_0_8.csv", encoding='utf-8', index=False)


######### Generating Negative Tweets

# Instantiating textgenrn
negative_textgen = textgenrnn()

# If weights from previously saved model are available
negative_textgen = textgenrnn('../models/neg_w.hdf5')

# If a new model has to be trained
# negative_texts = data_neg['text'].values
# negative_textgen.train_on_texts(negative_texts[0:4000], num_epochs=5,  gen_epochs=2)

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

data_neg.to_csv("../out/artificial_data/generative_neg_big_0_8.csv", encoding='utf-8', index=False)

######### # Merges Postitive and Negative generative artificial tweets into a single artificial dataset

artificial_data_pos = pd.read_csv('../resources/generative_pos_big_0_8.csv')
artificial_data_pos = artificial_data_pos[['text', 'sentiment']]

artificial_data_neg = pd.read_csv('../resources/generative_neg_big_0_8.csv')
artificial_data_neg = artificial_data_neg[['text', 'sentiment']]

frames = [artificial_data_pos, artificial_data_neg]
augmented_data = pd.concat(frames)

augmented_data.to_csv("../resources/artificial_data/generative_big_0_8.csv", encoding='utf-8', index=False)