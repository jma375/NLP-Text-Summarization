#James Alfano
import pandas as pd
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
    dfg.plot.bar(x='text')
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
    for i in range(len(df)):
        df.iloc[i,0] = nltk.word_tokenize(df.iloc[i,0])
        df.iloc[i,1] = nltk.word_tokenize(df.iloc[i,1])


#Removing stop words/punctuation
#e.g. 'a', 'an', 'not', 'do', 'over', 'themselves', "--", "''", ":", and "."
def rmv_stop_wrds(df):
    stop_words = set(stopwords.words('english'))
    char_rmv = ["'",",","``","`","-", "--","''",":",".","'s","said","$","(",")"]
    stop_words.update(char_rmv) #Adding extra stopwords
    for i in range(len(df)):
        df.iloc[i,0] = [w for w in df.iloc[i,0] if not w in stop_words] 
        df.iloc[i,1] = [w for w in df.iloc[i,1] if not w in stop_words] 


#Function to lemmatize 
#i.e. transform the word to its root
def lemmatize_wrds(df):
    wnl = WordNetLemmatizer()
    for i in range(len(df)):
        df.iloc[i,0] = [wnl.lemmatize(w) for w in df.iloc[i,0]] 
        df.iloc[i,1] = [wnl.lemmatize(w) for w in df.iloc[i,1]] 