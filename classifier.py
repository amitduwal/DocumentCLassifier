import sklearn
import nltk
import pickle
import fitz
from sklearn.feature_extraction.text import CountVectorizer
from train_model import input_process

def load_model_and_vectorizer():
    model=pickle.load(open("classifier.model","rb"))
    vectorizer=pickle.load(open("vectorizer.pickle", "rb"))
    return model,vectorizer

if __name__ == "__main__":
    model,vectorizer = load_model_and_vectorizer()
    path = input("Enter path of file: ")
    doc=fitz.open(path)
    content=''
    for page in range(len(doc)):
        content=content+doc[page].get_text()

    content=input_process(content)
    content=vectorizer.transform([content])
    pred=model.predict(content)
    if pred[0]==1:
        print('This document is about AI')
    else:
        print('This document is about WEB')