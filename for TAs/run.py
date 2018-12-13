#for TAs
# we have provided the test set as two CSVs, one is Xtest and other is Ytest.
# to reproduce results, follow these steps
#first extract the test files from the zip file and place them in this folder.

#1. import all packages
#2. load pickle file, Xtest and Ytest files
#3. run the model
#4. print the accuracy

#STEP-1
# import basic libraries
import re
import string
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

#import nlp libraries
import nltk
import ftfy
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

#import ml libraries
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss,accuracy_score
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#STEP-2
#load pickle file, Xtest and Ytest files
Xtest = pd.read_csv('Xtest.csv')
ytest = pd.read_csv('Ytest.csv')
lr_clf_model_pkl = open('lr_clf.pkl', 'rb')
lr_clf_model = pickle.load(lr_clf_model_pkl)

#STEP-3 run model
print ("Loaded Logistic Regression model :: ", lr_clf_model)
predicted = lr_clf_model.predict(Xtest)

#STEP-4 print accuracy
print(("Accuracy  ") + str(accuracy_score(ytest, predicted, normalize=False)))