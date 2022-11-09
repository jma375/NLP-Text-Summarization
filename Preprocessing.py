#James Alfano
import pandas as pd
import numpy as np
import sys
import contractions
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

#Run these commands when running for first time
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet') 
#!{sys.executable} -m pip install contractions

#-----------------Preprocssing Functions---------------------

#Function to plot word frequencies 
def wrd_freq_plot(df,title):
    # use explode to expand the lists into separate rows
    dfe = df.text.explode().to_frame().reset_index(drop=True)
    # groupby the values in the column, get the count and sort
    dfg = dfe.groupby('text').text.count() \
                                .reset_index(name='count') \
                                .sort_values(['count'], ascending=False) \
                                .head(25).reset_index(drop=True)                           
    dfg.plot.bar(x='text',color='c',alpha=0.65)
    plt.title(title)


#Each article begins with a location, source, and --
#e.g. "MINNEAPOLIS, Minnesota (CNN) --" or "WASHINGTON (CNN) --"
#This function seeks to remove the intro
def rvm_article_intro(df):
    for i in range(len(df)):
        df.iloc[i,0] = df.iloc[i,0][df.iloc[i,0].find('-')+3:] #+3 to remove the "-- " 


#Expand contractions 
#e.g. they're -> they are
def expand_contractions(df):
    for i in range(len(df)):
        df.iloc[i,0] = contractions.fix(df.iloc[i,0])
        df.iloc[i,1] = contractions.fix(df.iloc[i,1])


#Lower Case Function
def lower_case_text(df):
    df["text"] = df["text"].str.lower()
    df["y"] = df["y"].str.lower()


#Function to tokenize df 
def token(df):
    df['text'] = df['text'].apply(nltk.word_tokenize)
    df['y'] = df['y'].apply(nltk.word_tokenize)
    #for i in range(len(df)):
        #df.iloc[i,0] = nltk.word_tokenize(df.iloc[i,0])
        #df.iloc[i,1] = nltk.word_tokenize(df.iloc[i,1])


#Removing stop words/punctuation
#e.g. 'a', 'an', 'not', 'do', 'over', 'themselves', "--", "''", ":", and "."
def rmv_stop_wrds(df):
    stop_words = set(stopwords.words('english'))
    char_rmv = ["'",",","``","`","-", "--","''",":",".","'s","said","$","(",")"]
    stop_words.update(char_rmv) #Adding extra stopwords
    df['text'] = df['text'].apply(lambda x: [item for item in x if item not in stop_words])
    df['y'] = df['y'].apply(lambda x: [item for item in x if item not in stop_words])


#Function to lemmatize 
#i.e. transform the word to its root
def lemmatize_wrds(df):
    wnl = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda lst:[wnl.lemmatize(word) for word in lst])
    df['y'] = df['y'].apply(lambda lst:[wnl.lemmatize(word) for word in lst])


#Function to get number average number of words in article/summary
def avg_num_words(df):
    count = []
    for i in range(len(df)):
        count.append(len(df[i]))
    avg = int(np.round(np.mean(count)))
    return (avg, count)


#Function to plot Average Number of words in Summaries and Articles 
def plot_counts(avg_smry, smry_counts, avg_text, text_counts):
    #First plot for word count in summaries 
    plt.subplot(1, 2, 1)
    plt.gcf().set_size_inches(12, 6)
    plt.hist(smry_counts, bins=10, color='c', edgecolor='k', alpha=0.65)
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.title("Word Count Per Summary")
    plt.axvline(avg_smry, color='k', linestyle='dashed', linewidth=1,label = "Average Count = {0}".format(avg_smry)) #Add Average to hist
    plt.legend()
    #Second plot for word count in articles (i.e. "texts") 
    plt.subplot(1, 2, 2)
    plt.hist(text_counts, bins=10, color='c', edgecolor='k', alpha=0.65)
    plt.xlabel("Word Count")
    plt.title("Word Count Per Article")
    plt.axvline(avg_text, color='k', linestyle='dashed', linewidth=1, label = "Average Count = {0}".format(avg_text)) #Add Average to hist
    plt.legend()
    plt.show


#Function to add "_stop_" and "_start_" to summaries
#stop/start used for training models
def add_stop_start(df):
    for i in range(len(df)):
        df.iloc[i,1]= ['_start_'] + df.iloc[i,1] + ['_stop_']
    
