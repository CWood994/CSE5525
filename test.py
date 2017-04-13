from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer #pip install
import os
from sklearn.cluster import KMeans
import numpy as np
from sets import Set
from sklearn.utils import shuffle
from scipy import sparse





#unclassified tweets
def loadAllData():
    tweets = []

    for line in open("twitter-1.17.1/all.out"):
        words = line.strip().split()
        tweets.append(",".join(words))

    shuffle(tweets)

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

    shuffle(tweets, goldCategories)

    vec = CountVectorizer()
    vocab = vec.fit_transform(tweets)
    return vocab,goldCategories

def kmeans(vocab):
    km = KMeans()
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

    return sum(correct_array) / sum(incorrect_array)

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

    old_shape = vocab.shape

    sumCol = vocab.sum(axis=0)
    PERCENT_TO_DROP = .30
    count_to_drop = int(vocab.shape[1] * PERCENT_TO_DROP)
    indexes_to_delete= sumCol.argsort().tolist()[0][0:count_to_drop]

    indexes_to_delete.sort()

    for index in reversed(indexes_to_delete): 
        vocab = dropcols_coo(vocab, index) # this is slow, see if faster way/ figure out dropcols_coo (S.O.)


    y_pred = kmeans(vocab)
    print accuracy(y_gold, y_pred)




















