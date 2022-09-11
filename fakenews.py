import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

from google.colab import files
files.upload()

df = pd.read_csv('Fake_News.csv')
df.head()
df.shape
df.drop_duplicates(inplace=True)
df.shape
df.isnull().sum()
df.dropna(axis=0, inplace=True)
df.shape
#combine colums
df['combined'] = df['author'] + ' ' + df['title']
df.head
nltk.download('stopwords')
def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return clean_words

df['combined'].head().apply(process_text)

from sklearn.feature_extraction.text import CountVectorizer
message_bow = CountVectorizer(analyzer = process_text).fit_transform(df['combined'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(message_bow, df['label'], test_size = 0.20, random_state = 0)
message_bow.shape

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
print(classifier.predict(X_train))
print(y_train.values)

from sklearn.metrics import classification_report
pred = classifier.predict(X_train)
print(classification_report(y_train, pred))

from sklearn.metrics import classification_report
pred = classifier.predict(X_test)
print(classification_report(y_test, pred))
