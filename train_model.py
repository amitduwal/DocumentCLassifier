import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import string
import nltk
from nltk.corpus import stopwords
import fitz
import pickle

nltk.download('stopwords')
vectorizer  = CountVectorizer()

def pre_process_df():
    f_df=pd.DataFrame(columns=['Text','label'])
    df= pd.read_csv('Dataset.csv')
    f_df['Text']=df['Text']
    f_df['label']=df['label']
    return f_df

def input_process(text):
    translator = str.maketrans('','',string.punctuation)  #remove punctuatons--/,
    nopunc = text.translate(translator)                     #and giberrish words
    words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')] #list comprehension
    # print(' '.join(words))
    return ' '.join(words)  #join ords with space in between

def remove_stop_words(input):
    final_input=[]
    for line in input:
        line=input_process(line)
        final_input.append(line)
    return final_input

def train_model(df):
    input = df['Text']
    output = df['label']
    input=remove_stop_words(input)
    df['text'] = input
    input= vectorizer.fit_transform(input)
    nb = MultinomialNB()
    nb.fit(input, output)
    return nb

if __name__=='__main__':
    df=pre_process_df()
    model = train_model(df)
    pickle.dump(model, open('Classifier.model', 'wb'))
    pickle.dump(vectorizer, open('Vectorizer.pickle', 'wb'))
    