# For creating artificial tweets with heuristic modifications in spelling and hashtag rearragnment

import numpy as np
import pandas as pd
from random import shuffle
import itertools

data = pd.read_csv('../resources/Sentiment.csv')
data = data[['text','sentiment']]

def get_miss_spellings():

    miss_spelling_dict = {}
    file = open("../resources/spelling_mistakes.txt", "r")
    words = [word.rstrip() for word in file.readlines()]

    correct_spelling = words[0]

    for word in words:
        if "$" in word:
            correct_spelling = word.replace("$", "")
            miss_spelling_dict[correct_spelling] = []
        else:
            miss_spelling_dict[correct_spelling].append(word)

    return miss_spelling_dict

# Returns an artifial tweet with a spelling mistake added
def miss_spelt_tweet(tweet):

    words = tweet.split()
    miss_spelt_dict = get_miss_spellings()
    new_tweet_contents = [miss_spelt_dict[word][0] if word in miss_spelt_dict.keys() else word for word in words]
    new_tweet_text = ' '.join(new_tweet_contents)

    return new_tweet_text

# Returns an artifial tweet with hashtags rearranged
def transformed_hashtag_arrangement(tweet):

    words  = tweet.split()
    hashtags = set()

    for first, second in zip(words, words[1:]):
        if first.count("#") > 0 and second.count("#")>0:
            hashtags.add(first)
            hashtags.add(second)

    words = [word for word in words if word not in hashtags]
    hashtag_arrangements = list(itertools.permutations(hashtags))

    new_tweets = []

    for hastag_arrangement in hashtag_arrangements:
        new_tweet_contents = words + list(hastag_arrangement)
        new_tweet_text = ' '.join(new_tweet_contents)
        new_tweets.append(new_tweet_text)

    if len(new_tweets) > 0:
        return new_tweets
    else:
        return tweet

def augmented_dataset(organic_train):
    
    augmented_train = organic_train
    
    for index, row in augmented_train.iterrows():
        
        print(str(index/len(organic_train)*100),  end="\r", flush=True)
        
        # Spelling Mistakes
        artificial_tweet_spelling_mistake = miss_spelt_tweet(row.text)
        augmented_train = augmented_train.append({"text": artificial_tweet_spelling_mistake, "sentiment": row.sentiment}, ignore_index=True)

        # Hashtag Rearranged
        artificial_tweet_hashtag_rearranged = transformed_hashtag_arrangement(row.text)[0]
        augmented_train = augmented_train.append({"text": artificial_tweet_hashtag_rearranged, "sentiment": row.sentiment},
                             ignore_index=True)

    return augmented_train

print("There were originally " + str(len(data)) + " observations")
augmented_data = augmented_dataset(data)
print("There are now " + str(len(augmented_data)) + " observations")

augmented_data.to_csv("../out/artificial_data/heuristic_augmented_sentiment.csv", encoding='utf-8', index=False)