# Kumparan Internship - Data Scientist Assessment V2

## **Description** :

This is topic extractor Model that will extract and determine the article topic based on the article content . By using _Naive Bayesian_ method .

## **Prerequisites**

- Jupyter Notebook/PyCharm
- Sci-Kit Learn
- Pandas
- Regex
- Sastrawi Stemmer
- [List of Tala stopwords](<[https://www.google.com](https://www.kaggle.com/oswinrh/indonesian-stoplist)>)

## **Instalation**

- ### Installation library

```python
pip install -U scikit-learn
# Installation SciKit Learn
pip install pandas
#Installation Pandas
pip install Sastrawi
#Installation Sastrawi Stemmer
pip install regex
#Installation Regex
```

- # Inside Class Model

```python
    def __init__(self):
        """
        You can add more parameter here to initialize your model
        """
        # Open and append  the Stopword to List
        """
        Im using stopword list from Tala because it has a bunch of stopword that i need.
        """
        self.file_sw = open('stopwordbahasa.csv', 'r')
        self.stopword = self.file_sw.read()
        self.arrayStopword = self.stopword.split()

        #instantiate the TfIdfvectorizer
        self.vectorizer = TfidfVectorizer(stop_words=self.arrayStopword)
        """
        Im using TfIdfVectorizer will convert a collection of raw documents (ex:['article_content']) into a matrix of TF-IDF feature
        """
        #instantiate a Multinomial Naive Bayes model
        self.model = MultinomialNB()
        pass
```

```python

"""
Cleaning Method will remove the unnecessary text/symbol/number and link by using Regex
"""
 def cleaning(self,data):
        #Regex for deleting Link
        data['article_content'] = data['article_content'].str.replace(
            r'(((http(s)?:\/\/)|(www\.))[0-9a-z\.\/_\-+\(\)\$\#\&\!\?]+)', ' ', regex=True)

        #Regex for deleting Number
        data['article_content'] = data['article_content'].str.replace(r'[-*#:]?([\d]+([^ ]?[a-z\.\)/]*))+', ' ',regex=True)

        #Regex for deleting Symbol
        data['article_content'] = data['article_content'].str.replace(r'([a-z]*[-*&+/]{1}[a-z0-9]*)', ' ', regex=True)

        #Returning data that already cleaning
        return data
```

```python
    def train(self):
        #Read the csv data and assign to rawData variable
        rawData=pd.read_csv('data.csv')

        #Cleaning the data by calling cleaning method
        rawData=self.cleaning(rawData)

        """Seperate the article content and article topic by using indexing in rawData , variable df_x contain bunch of article content , and df_y contain article topic
        """
        df_x = rawData['article_content']
        df_y = rawData['article_topic']

        """
        Im using train_test_split method from scikitLearn because it has the quick utility that split dataFrame into random datatraining and datatesting
        """
        #Param1: The variabel/dataFrame that want to split
        #Param2: Represent the proportion of the datatesting (0.2 = 20% from all data)
        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)

        """
        Learn datatraining vocabulary , and use it to create into a document-term matrix . Combine fit and transform into a single syntax instead calling the fit method and tranform method one-by-one , because more faster.
        """
        #param1: The variable that want to train (Im using .values.astype('U') because the param only accept the str only for lowercase.)
        featured_train_vectorized = self.vectorizer.fit_transform(x_train.values.astype('U'))

        """
        Transforming the datatesting by using fitted vocabulary into document matrix , so the number of columns same as what already been learn above in featured_train_vectorized.
        """
        #Param1: The variable that want to transform.
        x_test_dtm=self.vectorizer.transform(x_test.values.astype('U'))

        #Training the model by using featured_train_vectorized variable
        self.model.fit(featured_train_vectorized,y_train)

        #Create a variable that predict for the transformed datatesting
        y_pred_class=self.model.predict(x_test_dtm)

        #accuracy=metrics.accuracy_score(y_test,y_pred_class)
        #accuracy=accuracy*100
```

```python
def predict(self, input):
       #Create Variable that take the input parameter as stringInput
       stringInput=input

       """
       Create variable stringTransform , calling transform method based on TfIdfVectorizer , I only using transform method (not fit or fit_transform) because it give me an error [INVALID] dimension mismatch when i run ds verify model.pickle "
       """
       stringTranform=self.vectorizer.transform([stringInput])

       """
       Create variable article_topic that contains 2-D mapping feature matrix from stringTransform variable
       """
       articile_topic=self.model.predict(stringTranform)

        #Return the predicted topic , using index[0] so returning the string type
       return str(articile_topic[0])
```

## **Note** : If I have more time, I want to improve and correct my stemming algorithm, because if I use my stemming method from Sastrawi, it will take more than 2 hours to stem. For now, I am not include the stemmer yet. And i want to try using another library such as Tensorflow/Pytorch.
