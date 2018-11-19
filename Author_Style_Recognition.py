#import basic libraries
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

#import data
train = pd.read_csv("TRAIN.csv")

#dataviz - word cloud creation
#create a new stopwords set because existing ones aren't that good
eng_stopwords = set(stopwords.words("english"))
STOPWORDS = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"])
wc_text = ' '.join(word for word in (re.split(':|, |,|!|! |; | |: |;|.,"', (' '.join(text for text in train.text[train.author == 4])))))
wordcloud = WordCloud(background_color="white", stopwords = ENGLISH_STOP_WORDS, max_words=5000).generate(wc_text)
plt.figure(figsize=(16,13))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#dataviz - author to data distribution
plt.figure(figsize = (9,4))
author_ctns = train.author.value_counts()
sns.barplot(author_ctns.index, author_ctns.values, alpha = 0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Author Name', fontsize=12)
plt.show()


#clean data with regex.
#because the data has a lot of mojibake, we use regex and ftfy library to clean it up.
train.text = train.text.apply(lambda x: re.sub(r'\"', r"'", re.sub("†", "", ftfy.fix_encoding(re.sub("�", " ", re.sub(" {2,}", " ", re.sub("\([0-9]*\)|\[[0-9]*\]", " ", re.sub("\r|\n", " ", x))))))))


#feature engineering
#total no of words in the dataset
num_words = train.text.apply(lambda x : len(str(x).split()))

#calculation of fractions of new feature columns
train['fraction_unique_words'] = train.text.apply(lambda x : len(set(str(x).split()))) / num_words
train['fraction_stopwords'] = train.text.apply(lambda x: len([l for l in str(x).lower().split() if l in eng_stopwords])) / num_words
train['fraction_punctuations'] = train.text.apply(lambda x: len([ch for ch in str(x) if ch in string.punctuation])) / num_words
train['fraction_nouns'] = train.text.apply(lambda x: len([poc for poc in nltk.pos_tag(str(x).split()) if poc[1] in ('NN','NNP','NNPS','NNS')])) / num_words
train['fraction_adj'] = train.text.apply(lambda x: len([poc for poc in nltk.pos_tag(str(x).split()) if poc[1] in ('JJ','JJR','JJS')])) / num_words
train['fraction_verbs'] = train.text.apply(lambda x: len([poc for poc in nltk.pos_tag(str(x).split()) if poc[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])) / num_words
train['informality'] = train.text.apply(lambda x: len(re.findall("(\. ){2,}|(\.){2,}", x)))
train['perspective'] = train.text.apply(lambda x: len(re.findall("\“", x)))
del num_words


#tf-idf feature creation with 1-4 ngram range
tfidf_vec = TfidfVectorizer(
    max_df = 0.3,
    min_df = 3,
    lowercase = True,
    stop_words = eng_stopwords,
    ngram_range = (1,4),
    analyzer = 'char'
)
train_tfidf = tfidf_vec.fit_transform(train.text)
indices = pd.DataFrame(tfidf_vec.get_feature_names())


# svd-based feature dimension reduction
n_comp = 500
svd_obj = TruncatedSVD(n_components = n_comp, algorithm = 'arpack')
svd_obj.fit(train_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
train_svd.columns = ['svd_char_' + str(i) for i in range(n_comp)]
train = pd.concat([train, train_svd], axis=1)
del train_tfidf, train_svd


#dataviz - violin plots
plt.figure(figsize=(12,8))
sns.violinplot(x = 'author', y = 'fraction_unique_words', data = train)
plt.xlabel('Author Name', fontsize = 12)
plt.ylabel('Fraction of Unique words in text', fontsize=12)
plt.title("Unique Words", fontsize=15)
plt.show()

plt.figure(figsize=(12,8))
sns.violinplot(x = 'author', y = 'fraction_stopwords', data = train)
plt.xlabel('Author Name', fontsize = 12)
plt.ylabel('Fraction of Stop words in text', fontsize=12)
plt.title("Stop Words", fontsize=15)
plt.show()


# train.num_words.loc[train.num_words > 85] = 85
plt.figure(figsize=(12,8))
sns.violinplot(x = 'author', y = 'fraction_punctuations', data = train)
plt.xlabel('Author Name', fontsize = 12)
plt.ylabel('Fraction of Punctuation in text', fontsize=12)
plt.title("Punctuations", fontsize=15)
plt.show()

plt.figure(figsize=(12,8))
sns.violinplot(x='author', y='fraction_nouns', data=train)
plt.xlabel('Author Name', fontsize=12)
plt.ylabel('Fraction of Nouns in the text', fontsize=12)
plt.title("Nouns", fontsize=15)
plt.show()


# train.num_punctuations.loc[train.num_punctuations > 13] = 13
plt.figure(figsize=(12,8))
sns.violinplot(x = 'author', y = 'fraction_adj', data = train)
plt.xlabel('Author Name', fontsize = 12)
plt.ylabel('Fractions of Adjective in text', fontsize=12)
plt.title("Adjectives", fontsize=15)
plt.show()

plt.figure(figsize=(12,8))
sns.violinplot(x = 'author', y = 'fraction_verbs', data = train)
plt.xlabel('Author Name', fontsize = 12)
plt.ylabel('Fractions of verbs in text', fontsize=12)
plt.title("Verbs", fontsize=15)
plt.show()


#split train data for training and testing
y = train.author
train.drop(columns = ['author','text'],axis= 1, inplace = True)
Xtrain, Xtest, ytrain, ytest = train_test_split(train, y, test_size = 0.2)


#naive bayes clf
# mnb_clf = MultinomialNB()
# mnb_clf.fit(Xtrain, ytrain)
# predicted = mnb_clf.predict(Xtest)
# print(("Accuracy ") + str(accuracy_score(ytest, predicted)))

#Logistic regression clf
lr_clf = LogisticRegression(
    random_state = 200,
    max_iter = 500,
    verbose = 1,
    n_jobs = -1
)
lr_clf.fit(Xtrain, ytrain)
predicted = lr_clf.predict(Xtest)
print(("Accuracy  ") + str(accuracy_score(ytest, predicted)))


#model pickelisation
#picklise the logistic regression model
lr_clf_pkl_filename = 'lr_clf.pkl'
lr_clf_pkl = open(lr_clf_pkl_filename, 'wb')
pickle.dump(lr_clf, lr_clf_pkl)
lr_clf_pkl.close()

#open the pickle file and run the model om test data
lr_clf_model_pkl = open(lr_clf_pkl_filename, 'rb')
lr_clf_model = pickle.load(lr_clf_model_pkl)
print ("Loaded Logistic Regression model :: ", lr_clf_model)
predicted = lr_clf_model.predict(Xtest)
print(("Accuracy  ") + str(accuracy_score(ytest, predicted)))
