import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split  
from sklearn.metrics import precision_recall_fscore_support as score


nltk.download('stopwords')

# Loading organic data
organic_data = pd.read_csv('/home/ubuntu/resources/Sentiment.csv')
# Keeping only the neccessary columns
organic_data = organic_data[['text', 'sentiment']]

# Splitting the dataset into train and test set
train, test = train_test_split(organic_data, test_size=0.1)

# Loading and preparing artificial data
artificial_data = pd.read_csv('/home/ubuntu/resources/heuristic_augmented_sentiment.csv')
artificial_data = artificial_data[['text', 'sentiment']]

# Removing neutral sentiments
train = train[train.sentiment != "Neutral"]
test = test[test.sentiment != "Neutral"]
artificial_data = artificial_data[artificial_data.sentiment != "Neutral"]

# Appending artifical data
print("There are " + str(len(train)) + " items in the organic training data")
frames = [shuffle(train)[0:700], shuffle(artificial_data)[0:300]]#shuffle(train)[0:300]]
train = pd.concat(frames)
print("There are " + str(len(train)) + " items in the augmented training data")

train_pos = train[ train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neg = train[ train['sentiment'] == 'Negative']
train_neg = train_neg['text']

# Cleaning Data
tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
                     if 'http' not in word
                     and not word.startswith('@')
                     and not word.startswith('#')
                     and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.sentiment))

test_pos = test[ test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[ test['sentiment'] == 'Negative']
test_neg = test_neg['text']

# Extracting word features
def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features


w_features = get_word_features(get_words_in_tweets(tweets))


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

y_pred = [classifier.classify(extract_features(obj.split())) for obj in test['text']]
y_test = test['sentiment']

precision, recall, fscore, support = score(y_test, y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))