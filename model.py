"""
Kumparan's Model Interface

This is an interface file to implement your model.

You must implement `train` method and `predict` method.

`train` is a method to train your model. You can read
training data, preprocess and perform the training inside
this method.

`predict` is a method to run the prediction using your
trained model. This method depends on the task that you
are solving, please read the instruction that sent by
the Kumparan team for what is the input and the output
of the method.

In this interface, we implement `save` method to helps you
save your trained model. You may not edit this directly.

You can add more initialization parameter and define
new methods to the Model class.

Usage:
Install `kumparanian` first:

    pip install kumparanian

Run

    python model.py

It will run the training and save your trained model to
file `model.pickle`.
"""

from kumparanian import ds

import csv    
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
from sklearn import metrics


# Import your libraries here
# Example:
# import torch


class Model:

    def __init__(self):
        """
        You can add more parameter here to initialize your model
        """
        # Adding Stopword List
        self.file_sw = open('stopwordbahasa.csv', 'r')
        self.stopword = self.file_sw.read()
        self.arrayStopword = self.stopword.split()
        self.vectorizer = TfidfVectorizer(stop_words=self.arrayStopword)
        self.model = MultinomialNB()
        pass


    def cleaning(self,data):
        data['article_content'] = data['article_content'].str.replace(
            r'(((http(s)?:\/\/)|(www\.))[0-9a-z\.\/_\-+\(\)\$\#\&\!\?]+)', ' ', regex=True)
        data['article_content'] = data['article_content'].str.replace(r'[-*#:]?([\d]+([^ ]?[a-z\.\)/]*))+', ' ',
                                                                      regex=True)
        data['article_content'] = data['article_content'].str.replace(r'([a-z]*[-*&+/]{1}[a-z0-9]*)', ' ', regex=True)
        return data



    def train(self):
        
        rawData=pd.read_csv('data.csv')

        rawData=self.cleaning(rawData)
       
        df_x = rawData['article_content']
        df_y = rawData['article_topic']


        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)

        featured_train_vectorized = self.vectorizer.fit_transform(x_train.values.astype('U'))

        x_test_dtm=self.vectorizer.transform(x_test.values.astype('U'))



        self.model.fit(featured_train_vectorized,y_train)

        y_pred_class=self.model.predict(x_test_dtm)


        #accuracy=metrics.accuracy_score(y_test,y_pred_class)
        #accuracy=accuracy*100




    def predict(self, input):

       stringInput=input
       stringTranform=self.vectorizer.transform([stringInput])
       articile_topic=self.model.predict(stringTranform)

       return str(articile_topic[0])


    def save(self):
        """
        Save trained model to model.pickle file.
        """
        ds.model.save(self, "model.pickle")


if __name__ == '__main__':
    # NOTE: Edit this if you add more initialization parameter
    model = Model()

    # Train your model
    model.train()

    # Save your trained model to model.pickle
    model.save()






