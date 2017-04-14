from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer #pip install
import os
from sklearn.cluster import KMeans
import numpy as np
from sets import Set
import random
from scipy import sparse
from sklearn.naive_bayes import BernoulliNB



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
        #print self.gold[0:self.trainData.shape[0]]
        self.clf.fit(self.trainData.toarray(), self.gold[0:self.trainData.shape[0]])

    def test(self):
        self.guesses = self.clf.predict(self.testData.toarray())

    def accuracy(self):
        correct = 0

        for i in range(len(self.gold[self.trainData.shape[0]:])):
            if self.gold[i] == self.guesses[i]:
                correct += 1
        return correct / len(self.guesses)



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
    return vocab,goldCategories

def kmeans(vocab):
    km = KMeans()
    km.fit(vocab)
    return km.labels_

def kmeansKnown(vocab):
    km = KMeans(15)
    km.fit(vocab)
    return km.labels_

def accuracy(gold, pred):
    correct_array = []
    incorrect_array = []

    categories = Set(pred)
    for cat in categories:
        indexes = [n for (n, e) in enumerate(pred) if e == cat]
        guesses = [gold[index] for index in indexes]
        guess = max(set(guesses), key=guesses.count)
        correct = guesses.count(guess)
        incorrect = len(guesses) - correct
        correct_array.append(correct)
        incorrect_array.append(incorrect)

    print correct_array
    print incorrect_array

    #accuracy_array = [correct_array[i]/incorrect_array[i] for i in range(len(correct_array))]

    return sum(correct_array) / (sum(incorrect_array) + sum(correct_array))

def dropcols_coo(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)   
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()


if __name__ == "__main__":
    vocab,y_gold = loadCategoryData()
    #vocab = loadAllData()
    
    sumCol = vocab.sum(axis=0)
    PERCENT_TO_DROP = .30
    count_to_drop = int(vocab.shape[1] * PERCENT_TO_DROP)
    indexes_to_delete= sumCol.argsort().tolist()[0][0:count_to_drop]

    indexes_to_delete.sort()

    for index in reversed(indexes_to_delete): 
        vocab = dropcols_coo(vocab, index) # this is slow, see if faster way/ figure out dropcols_coo (S.O.)


    y_pred = kmeans(vocab)
    print accuracy(y_gold, y_pred)


    y_pred = kmeansKnown(vocab)
    print accuracy(y_gold, y_pred)

    NB = NaiveBayes(vocab, y_gold)
    print NB.accuracy()





















