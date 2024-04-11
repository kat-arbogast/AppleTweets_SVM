'''
Author: Katrina Arbogast

Firstly, data preprocessing techniques such as tokenization, stop-word removal, and stemming
will be applied to clean and prepare the tweet data for analysis. Feature extraction methods, including
bag-of-words representation or TF-IDF (Term Frequency-Inverse Document Frequency), will then be
utilized to convert the text data into numerical vectors suitable for SVM training. Next, SVM models
will be trained given the target labels of negative, neutral, and positive on a subset of the dataset.
For a more comprehensive analysis multiple kernels will be used (linear, polynomial, and radial basis
function) to classify tweets into the sentiment categories.
'''

import pandas as pd    
import numpy as np
from pandas import DataFrame
import os

## To tokenize and vectorize text type data
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import sklearn

def basket_word_column(df):
    '''
    Make a basket datset.
    Agrs:
        - vecotrized text dataframe
    Returns:
        - a dataframe where 1 is the columns name and 0 becomes an empty string
    '''
    df2 = df.copy()
    
    for col in df2.columns[1:]:
        df2[col] = df2[col].apply(lambda x: col if x != 0 else '')
            
    df2 = df2.drop(df2.columns[0], axis=1)

    return df2

## Tokenize and Vectorize the Apple Tweet Sentiments

RAW_DF=pd.read_csv("./Data/Apple.csv", error_bad_lines=False)

text_df = RAW_DF.copy()    
text_df = text_df.dropna()                                                                  # REMOVE any rows with NaN in them

tweetsLIST=[]
labelLIST=[]
for nexttweet, nextlabel in zip(text_df["tweets"], text_df["labels"]):
    tweetsLIST.append(nexttweet)
    labelLIST.append(nextlabel)

### Vectorize

## Instantiate your CV
MyCountV=CountVectorizer(
    input="content",                                                                        # because we have a csv file
    lowercase=True,                                                                         # make all words lower case
    stop_words = "english",                                                                 # remove stop words
    max_features=100                                                                        # maximum number of unique words
    )

MyDTM = MyCountV.fit_transform(tweetsLIST)                                                  # create a sparse matrix

ColumnNames = MyCountV.get_feature_names_out()

MyDTM_DF = pd.DataFrame(MyDTM.toarray(),columns=ColumnNames)                                # Build the data frame
Labels_DF = pd.DataFrame(labelLIST,columns=['labels'])                                      # Convert the labels from list to df
My_Orig_DF = MyDTM_DF                                                                       # Save original DF - without the lables
dfs = [Labels_DF, MyDTM_DF]                                                                 # Now - let's create a complete and labeled dataframe

appleTweets_Labeled = pd.concat(dfs,axis=1, join='inner')
appleTweets_Labeled.drop(columns=['aapl', 'apple'], inplace=True)

appleTweets_basket = basket_word_column(appleTweets_Labeled)                                # basketize the words - good for association rule mining

appleTweets_Labeled.to_csv(f"./Data/appleTweets_vectorized.csv", index=False)
appleTweets_basket.to_csv(f"./Data/appleTweets_basket.csv", index=False)

## Word Cloud

# Iterate over unique labels
for label in appleTweets_Labeled['labels'].unique():
    filtered_df = appleTweets_Labeled[appleTweets_Labeled['labels'] == label]               # Filter the DataFrame for the current label
    text = ' '.join(filtered_df.columns[1:])                                                # Concatenate all the text data associated with the current label
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)   # Generate a word cloud
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')                                         # Display the word cloud
    plt.title('Word Cloud for Label ' + str(label))
    plt.axis('off')
    plt.savefig(f"./CreatedVisuals/WordCloud/wordCloud_{label}.png", dpi = 300)

print("Made it out alive")