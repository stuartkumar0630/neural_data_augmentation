import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support as score

import re

# Loading data
data = pd.read_csv('/home/ubuntu/resources/Sentiment.csv')
# Keeping only the neccessary columns
data = data[['text', 'sentiment']]

# Loading artificial data
artificial_data = pd.read_csv('/home/ubuntu/resources/generative_big_0_8.csv')
# Keeping only the neccessary columns
artificial_data = artificial_data[['text', 'sentiment']]

# Data Cleaning
data = data[data.sentiment != "Neutral"]
print("There are " + str(len(data)) + " non-neutral tweets")
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')
    
# Artificial Data Cleaning
artificial_data = artificial_data[artificial_data.sentiment != "Neutral"]
artificial_data['text'] = artificial_data['text'].apply(lambda x: str(x).lower())
artificial_data['text'] = artificial_data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in artificial_data.iterrows():
    row[0] = row[0].replace('rt', ' ')
    
# Augmentation Ratio
#data = shuffle(data)
print('The full organic dataset contains ' + str(len(data)) + ' observations')

#artificial_data = shuffle(artificial_data)
print('The full artificial dataset contains ' + str(len(artificial_data)) + ' observations')

# Model Hyperparameter Setup
max_fatures = 2000
embed_dim = 128
lstm_out = 196
batch_size = 32
validation_size = 1500

# Text Preparation
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(pd.concat([artificial_data['text'], data['text']]).values)

# Artificial Text Preparation
X_artificial = tokenizer.texts_to_sequences(artificial_data['text'].values)
X_artificial = pad_sequences(X_artificial)

X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)


#X_artificial = np.pad(X_artificial, [(0, 0),(len(X[0]) - len(X_artificial[0]), 0)], mode='constant')  
X = np.pad(X, [(0, 0),(len(X_artificial[0]) - len(X[0]), 0)], mode='constant')

print(X_artificial[0])
print(X[0])

print(str(len(X_artificial[0])))
print(str(len(X[0])))

# Model Configuration
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

# Test-Train Split
Y = pd.get_dummies(data['sentiment']).values
Y_artificial = pd.get_dummies(artificial_data['sentiment']).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

# Appending artificial data
Y_train = np.concatenate((Y_artificial[0:7000], Y_train[0:7000]), axis=0)
X_train = np.concatenate((X_artificial[0:7000], X_train[0:7000]), axis=0)

# Model Training
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2)

# Model Evaluation on Validation Data
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

y_pred = []
y_test = []

for x in range(len(X_validate)):

    pred_result = model.predict(X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]
    
    if np.argmax(Y_validate[x]) == 0:
        y_test.append('Negative')
    else:
        y_test.append('Positive')
            
    if np.argmax(pred_result) == 0:
        y_pred.append('Negative')
    else:
        y_pred.append('Positive')
        
        
precision, recall, fscore, support = score(y_test, y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))