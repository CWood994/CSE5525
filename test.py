from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer #pip install
import os
from sklearn.cluster import KMeans
import numpy as np
from sets import Set
import random
import re
from scipy import sparse
from sklearn.naive_bayes import BernoulliNB
import nltk #import nltk and do the next line... might be able to stop after a bit, only need some of it
# nltk.download('all')



class NaiveBayes:
    """docstring for ClassName"""
    def __init__(self, data, classes):
        self.data = data
        self.gold = classes

        self.splitData()
        self.train()
        self.test()

    def splitData(self):
        PERCENT_TO_TRAIN_ON = .60

        self.trainData = self.data[0:int(self.data.shape[0]*PERCENT_TO_TRAIN_ON),:]
        self.testData = self.data[int(self.data.shape[0]*PERCENT_TO_TRAIN_ON):,:]


    def train(self):
        self.clf = BernoulliNB()
        self.clf.fit(self.trainData.toarray(), self.gold[0:self.trainData.shape[0]])

    def test(self):
        self.guesses = self.clf.predict(self.testData.toarray())

    def accuracy(self):
        correct = 0

        for i in range(len(self.gold[self.trainData.shape[0]:])):
            if self.gold[self.trainData.shape[0] + i] == self.guesses[i]:
                correct += 1
        return correct / len(self.guesses)


# NER to find nouns
# combine ones with all same nouns
# combine unclassified ones to class if all its nnouns are in class
# repeat until converge 
# find closest nouns then assign
# if no nouns similar then NB on words

#combine similar classes based on similarity of total words if very similar
class mySweetAssAlgorithm:
    def __init__(self, x, vocab):
        self.vocab = vocab
        self.x = x

        self.classify()


    def classify(self):
        NNP = []
        groupIndexes = []

        for i in range(len(self.x)):
            text = re.sub(',', ' ', self.x[i])
            text = re.sub('@', '', text)
            tags = nltk.pos_tag(nltk.word_tokenize(text))

            verbs = []
            for tag in tags:
                if tag[1][0:2] == "VB":
                    verbs.append(tag[0])
            if verbs in NNP:
                index = NNP.index(verbs)
                groupIndexes[index].append(i)
            else:
                NNP.append(verbs)
                groupIndexes.append([i])

        print len(self.x)
        print len(groupIndexes)





    def accuracy(self):
        return 100

#unclassified tweets
def loadAllData():
    tweets = []

    for line in open("twitter-1.17.1/all.out"):
        words = line.strip().split()
        tweets.append(",".join(words))

    random.shuffle(tweets)

    vec = CountVectorizer()
    vocab = vec.fit_transform(tweets)
    return vocab

#classified tweets
def loadCategoryData():
    tweets = []
    goldCategories = []

    categories = os.listdir("twitter-1.17.1/data/")
    for cat in categories:
        if cat[0] == ".":
            categories.remove(cat)

    for file in categories:
        for line in open("twitter-1.17.1/data/"+file):
            words = line.strip().split()
            tweets.append(",".join(words))
            goldCategories.append(file[0:len(file)-4])

    c = list(zip(tweets, goldCategories))
    random.shuffle(c)
    tweets, goldCategories = zip(*c)

    vec = CountVectorizer()
    vocab = vec.fit_transform(tweets)
    return tweets,vocab,goldCategories

def kmeans(vocab):
    km = KMeans()
    km.fit(vocab)
    return km.labels_

def kmeansKnown(vocab):
    km = KMeans(15)
    km.fit(vocab)
    return km.labels_


def accuracy(gold , pred):
    numberToCategory = []
    for i in set(pred):
        indexes = [n for (n, e) in enumerate(pred) if e == i]
        guesses = [gold[i] for i in indexes]
        guess = max(set(guesses), key=guesses.count)
        numberToCategory.append(guess)
    print numberToCategory

    correct = 0
    for i in range(len(gold)):
        if gold[i] == numberToCategory[pred[i]]:
            correct += 1
    return correct / len(gold)


def dropcols_coo(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)   
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()


if __name__ == "__main__":
    x,vocab,y_gold = loadCategoryData()
    #vocab = loadAllData()
    
    #drop lowest percent of data... worsens data rn
    sumCol = vocab.sum(axis=0)
    PERCENT_TO_DROP = .30
    count_to_drop = int(vocab.shape[1] * PERCENT_TO_DROP)
    indexes_to_delete= sumCol.argsort().tolist()[0][0:count_to_drop]

    indexes_to_delete.sort()

    for index in reversed(indexes_to_delete): 
        vocab = dropcols_coo(vocab, index) # this is slow, see if faster way/ figure out dropcols_coo (S.O.)
    

    print "\nKMeans with unknown K\n"
    y_pred = kmeans(vocab)
    print accuracy(y_gold, y_pred)


    print "\nKMeans with known K\n"
    y_pred = kmeansKnown(vocab)
    print accuracy(y_gold, y_pred)


    print "\nSupervised NaiveBayes\n"
    NB = NaiveBayes(vocab, y_gold)
    print NB.accuracy()


    print "\nMy Sweet Ass Algorithm with unknown K\n"
    CW = mySweetAssAlgorithm(x, vocab)
    print CW.accuracy()





















