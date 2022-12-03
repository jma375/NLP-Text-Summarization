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


#Removing stop words/punctuation
#e.g. 'a', 'an', 'not', 'do', 'over', 'themselves', "--", "''", ":", and "."
def rmv_stop_wrds(df):
    stop_words = set(stopwords.words('english'))
    char_rmv = ["'","``","`","-", "--","''",":","'s","said","$","(",")","?",".",",","’","‘"]
    stop_words.update(char_rmv) #Adding extra stopwords
    df['text'] = df['text'].apply(lambda x: [item for item in x if item not in char_rmv])


#Function to lemmatize 
#i.e. transform the word to its root
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
    #plot 1
    fig, (ax1, ax2) = plt.subplots(2)
    plt.gcf().set_size_inches(14, 8)
    # use explode to expand the lists into separate rows
    temp = df.text.explode().to_frame().reset_index(drop=True)

    # groupby the values in the column, get the count and sort
    temp = temp.groupby('text').text.count() \
                                    .reset_index(name='count') \
                                    .sort_values(['count'], ascending=False) \
                                    .head(25).reset_index(drop=True)
                            
    ax1.bar(temp["text"], temp["count"],color='c',alpha=0.65)
    ax1.set_title("Word Frequency Before Removing Stopwords and and Performing Lemmatization")
    ax1.tick_params(axis='x', rotation=45,right= True)

    #remove stop words and lemmatize before ploting again 
    rmv_stop_wrds(df)
    lemmatize_wrds(df)

    #plot 2 i.e. after we removed stop words and lemmatized
    temp = df.text.explode().to_frame().reset_index(drop=True)
    # groupby the values in the column, get the count and sort
    temp = temp.groupby('text').text.count() \
                                    .reset_index(name='count') \
                                    .sort_values(['count'], ascending=False) \
                                    .head(25).reset_index(drop=True)
                            
    ax2.bar(temp["text"], temp["count"],color='c',alpha=0.65)
    ax2.set_title("Word Frequency After Removing Stopwords and Performing Lemmatization")
    ax2.tick_params(axis='x', rotation=45,right= True)
    fig.suptitle('{0}: Word Frequencies'.format(title))
    fig.tight_layout(pad=1.0)

#Function to aggregate all preprocessing functions into one
def preprocess(df, type, plots):
    expand_contractions(df)
    lower_case_text(df)
    rvm_article_intro(df)
    token(df)
    if plots == 1:
        wrd_freq_plot(df,type)
    elif plots == 0:
        rmv_stop_wrds(df)
        lemmatize_wrds(df)


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
