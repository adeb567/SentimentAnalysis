from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
import re
import pandas as pd
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LinearRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.ensemble import RandomForestClassifier


raw_text = pd.read_csv("F:\\COLLEGE STUFF\\SENTIMENT ANALYSIS\\amazon_review_polarity_csv\\test.csv")[:30000]

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

#stop_words=list(stopwords.words('English'))
grammer={'JJR','JJS','JJ','RB','VB','VBD','VBG','VBN','VBP','RBR','RBS','VBZ'}
stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
            'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
            'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
            'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
            ]


lemmatizer=WordNetLemmatizer()

list_reviews=[]
list_words=[]

for raw_review in raw_text.review:
    review= re.sub('[^a-zA-Z]', ' ', raw_review)
    splitted=review.split()
    words = []

    for word in splitted:
        if word in contractions:
            contracted_word=contractions[word]
            for cw in contracted_word.split(" "):
                words.append(cw)
        else:
            words.append(word)

    tagged = nltk.pos_tag(words)
    filtered=[]
    for w, g in tagged:
        if w not in stop_words and g in grammer:
            #filtered.append(ps.stem(w))
            filtered.append(lemmatizer.lemmatize(w.lower()))
            list_words.append(lemmatizer.lemmatize(w.lower()))

    list_reviews.append(filtered)

list_ratings=list(raw_text.rating)
all_words=nltk.FreqDist(list_words)
word_features=list(all_words.keys())[:3000]

def find_features(review):
    words = set(review)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

feature_set=[]
for rev,rat in zip(list_reviews,list_ratings):
    feature_set.append((find_features(rev),rat))
random.shuffle(feature_set)

training_set=feature_set[:25000]
testing_set=feature_set[25000:]

NB_classifier=nltk.NaiveBayesClassifier.train(training_set)
print("NB accuracy %: ", (nltk.classify.accuracy(NB_classifier,testing_set))*100)

MNB_classifier=SklearnClassifier(MultinomialNB()).train(training_set)
print("MNB accuracy %: ", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)

BNB_classifier=SklearnClassifier(BernoulliNB()).train(training_set)
print("BNB accuracy %: ", (nltk.classify.accuracy(BNB_classifier,testing_set))*100)

SVC_classifier=SklearnClassifier(SVC(C=1 , kernel='linear')).train(training_set)
print("SVC accuracy %: ", (nltk.classify.accuracy(SVC_classifier,testing_set))*100)

LinearSVC_classifier=SklearnClassifier(LinearSVC(penalty='l2', loss='hinge')).train(training_set)
print("LSVC accuracy %: ", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

NuSVC_classifier=SklearnClassifier(NuSVC()).train(training_set)
print("NSVC accuracy %: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)

RF_classifier=SklearnClassifier(RandomForestClassifier(n_estimators= 200, criterion='entropy')).train(training_set)
print("RF accuracy %: ", (nltk.classify.accuracy(RF_classifier,testing_set))*100)


class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers= classifiers

    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidance(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        votes_choice=votes.count(mode(votes))
        conf=votes_choice/len(votes)
        return conf


Voted_classifier=VoteClassifier(NB_classifier,MNB_classifier,BNB_classifier,SVC_classifier,RF_classifier)
print("Voted accuracy %: ", (nltk.classify.accuracy(Voted_classifier,testing_set))*100)


def sentiment(text):
    feats=find_features(text)
    
    return Voted_classifier.classify(feats),Voted_classifier.confidance(feats)

print("liked it very much:",sentiment("liked it very much"))
