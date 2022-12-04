#James Alfano
import re
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
def rvm_article_intro(dataset):
    return {"article": dataset["article"][dataset["article"].find('(cnn)')+5:]}


#Expand contractions 
#e.g. they're -> they are
def expand_contractions(dataset):
    return {"article": contractions.fix(dataset["article"])}


#Lower Case Function
def lower_case_text(dataset):
    return {"article": dataset["article"].lower()}


#Function to split strings into lists of words 
def split_str(dataset):
    return {"article": re.findall(r"[\w']+|[.,!?;]",dataset["article"])}
    

#Removing special characters
def rmv_special_chars(dataset):
    return {"article": re.sub('[^A-Za-z0-9 ]+', '',dataset["article"])}

#Function to lemmatize 
#i.e. transform the word to its root, or lemma 
#def lemmatize_wrds(df):
#    wnl = WordNetLemmatizer()
#    #df['text'] = df['text'].apply(lambda lst:[wnl.lemmatize(word) for word in lst])
#    dataset['article'] = dataset['article'].apply(lambda lst:[wnl.lemmatize(word) for word in lst])


#Function to get number average number of words in article/summary
def avg_num_words(df):
    count = []
    for i in range(len(df)):
        count.append(len(df[i]))
    avg = int(np.round(np.mean(count)))
    return (avg, count)


#Function to plot Average Number of words in Summaries and Articles 
def plot_counts(dataset,split):
    
    #Convert dataset to df
    splt_data = dataset.map(split_str)
    lst_dics = [dic for dic in splt_data[split]]
    df = pd.DataFrame(lst_dics).rename(columns={"article":"text", 
      "highlights":"y","id":"id"})[["text","y","id"]]

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
    return avg_smry, smry_counts, avg_text, text_counts


#Function to plot word frequencies 
def wrd_freq_plot(dataset,split,title):
    #Convert dataset to df
    splt_data = dataset.map(split_str)
    lst_dics = [dic for dic in splt_data[split]]
    df = pd.DataFrame(lst_dics).rename(columns={"article":"text", 
      "highlights":"y","id":"id"})[["text","y","id"]]
    
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
def preprocess(dataset):
    dataset = dataset.map(expand_contractions)#
    dataset = dataset.map(lower_case_text)
    dataset = dataset.map(rvm_article_intro)
    dataset = dataset.map(rmv_special_chars) 
    return dataset