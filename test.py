from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer #pip install
import os
from sklearn.cluster import KMeans
import numpy as np
from sets import Set



#unclassified tweets
def loadAllData():
    tweets = []

    for line in open("twitter-1.17.1/all.out"):
        words = line.strip().split()
        tweets.append(",".join(words))

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

    vec = CountVectorizer()
    vocab = vec.fit_transform(tweets)
    return vocab,goldCategories

def kmeans(vocab):
    km = KMeans(15)
    km.fit(vocab)
    return km.labels_

def accuracy(gold, pred):
    print gold
    print pred
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

if __name__ == "__main__":
   vocab,y_gold = loadCategoryData()
   #vocab = loadAllData()
   y_pred = kmeans(vocab)
   print accuracy(y_gold, y_pred)




















