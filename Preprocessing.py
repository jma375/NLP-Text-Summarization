#James Alfano
import pandas as pd
import numpy as np
import sys
import contractions
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from datasets.dataset_dict import DatasetDict
from datasets import Dataset


#-----------------Preprocssing Functions---------------------



#Each article begins with a location and source
#e.g. "MINNEAPOLIS, Minnesota (CNN) " or "WASHINGTON (CNN)"
#This function seeks to remove the intro
def rvm_article_intro(df):
    for i in range(len(df)):
        df.iloc[i,0] = df.iloc[i,0][df.iloc[i,0].find('(cnn)')+5:] #Looks to remove "(CNN)" and everything before


#Expand contractions 
#e.g. they're -> they are
def expand_contractions(df):
    for i in range(len(df)):
        df.iloc[i,0] = contractions.fix(df.iloc[i,0])


#Lower Case Function
def lower_case_text(df):
    df["text"] = df["text"].str.lower()


#Function to tokenize df 
def token(df):
    df['text'] = df['text'].apply(nltk.word_tokenize)
    df['y'] = df['y'].apply(nltk.word_tokenize)


#Removing special characters
#e.g. "'","``","`","-", and "--"
def rmv_special_chars(df):
    char_rmv = ["'","``","`","-", "--","''",":","'s","$","(",")","?",".",",","’","‘"]
    df['text'] = df['text'].apply(lambda x: [item for item in x if item not in char_rmv])


#Function to lemmatize 
#i.e. transform the word to its root, or lemma 
def lemmatize_wrds(df):
    wnl = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda lst:[wnl.lemmatize(word) for word in lst])


#Function to get number average number of words in article/summary
def avg_num_words(df):
    count = []
    for i in range(len(df)):
        count.append(len(df[i]))
    avg = int(np.round(np.mean(count)))
    return (avg, count)


#Function to plot Average Number of words in Summaries and Articles 
def plot_counts(df):
    #get average and list of word counts  
    avg_smry, smry_counts = avg_num_words(df["y"])
    avg_text, text_counts = avg_num_words(df["text"])
    
    #Get max length for each
    print("The longest summary has {0} words".format(max(smry_counts)))
    print("The longest article has {0} words".format(max(text_counts)))

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


#Function to plot word frequencies 
def wrd_freq_plot(df,title):
    #Set size
    fig = plt.figure(figsize=(8, 6))
    temp = df.text.explode().to_frame().reset_index(drop=True)
    # groupby the values in the column, get the count and sort descending
    temp = temp.groupby('text').text.count() \
                                    .reset_index(name='count') \
                                    .sort_values(['count'], ascending=False) \
                                    .head(25).reset_index(drop=True) #top 25 most freq words
    plt.bar(temp["text"], temp["count"],color='c',alpha=0.65)
    plt.title("Word Frequencies")
    plt.title('{0}: Word Frequencies'.format(title))
    plt.tick_params(axis='x', rotation=45,right= True) #Rotate x axis labels         



#Function to aggregate all preprocessing functions into one
def preprocess(df, type):
    expand_contractions(df)
    lower_case_text(df)
    rvm_article_intro(df)
    token(df)
    rmv_special_chars(df)
    wrd_freq_plot(df,type)
    #lemmatize_wrds(df)
    


#Recombine data to be in correct format
def combine_data(subset_train,subset_test,subset_valid):
    #Here we rejoin the texts    
    subset_valid["text"] = subset_valid["text"].apply(lambda x: ' '.join(wrd for wrd in x))
    subset_valid["y"] = subset_valid["y"].apply(lambda x: ' '.join(wrd for wrd in x))

    subset_test["text"] = subset_test["text"].apply(lambda x: ' '.join(wrd for wrd in x))
    subset_test["y"] = subset_test["y"].apply(lambda x: ' '.join(wrd for wrd in x))

    subset_train["text"] = subset_train["text"].apply(lambda x: ' '.join(wrd for wrd in x))
    subset_train["y"] = subset_train["y"].apply(lambda x: ' '.join(wrd for wrd in x))

    #here we recombine our df to 'datasets'
    d = {'train':Dataset.from_dict({"article":subset_train["text"],"highlights":subset_train["y"],"id":subset_train["id"]}),
        'validation':Dataset.from_dict({"article":subset_valid["text"],"highlights":subset_valid["y"],"id":subset_valid["id"]}),
        'test':Dataset.from_dict({"article":subset_test["text"],"highlights":subset_test["y"],"id":subset_test["id"]})
        }

    dataset = DatasetDict(d)
    return dataset
