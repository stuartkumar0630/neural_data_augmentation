# Neural Text Generation For Data Augmentation

These directories contain the source code, resources, and output created for this dissertation. 

## Source Code

All source code can be found in the directory. The augmentations subdirectory contains the code associated with creating artificial data, both for the novel generative augmentation and the heuristic benchmark. Implementations of the standard sentiment analysis models adapted from the Kaggle Challenge are located in the sentiment_analysis subdirectory. The code used to create the charts and figures in the report is in the Charts subdirectory.

## Resources

The organic tweet dataset and the spelling mistakes dataset are located in the resources folder.

## Output

The artificial data for all parts of this investigation can be found in the out/artificial_data directory. The adjacent figures folder contains the images files used in the report. Raw results from the experiments are presented in text files in out/raw_experimental_results

## Running Instructions

Each python file is capable of being run independenedly. The code creating artifical dataset with TextGenRNN is located in generative_augmentation.py.

## Authors
Stuart Kumar

## Key References
* TextGenRNN library - https://github.com/minimaxir/textgenrnn
* Organic Twitter Dataset - https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment
* Spelling Mistakes Dataset - Birkbeck spelling error corpus
* Standard LSTM model - https://www.kaggle.com/ngyptr/lstm-sentiment-analysis-keras
* Standard Naive Bayes Model - https://www.kaggle.com/ngyptr/python-nltk-sentiment-analysis
