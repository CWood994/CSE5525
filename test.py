from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
from sklearn.cluster import KMeans
import numpy as np
from sets import Set
import random
import re
from scipy import sparse
from sklearn.naive_bayes import BernoulliNB
from sklearn.cluster import DBSCAN
from sklearn import linear_model
import sys
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

        classes,best_groups = self.classify1()

        self.predicted = classes
        #self.classify2(classes,best_groups)


    def classify1(self):
        NNP = []
        groupIndexes = []

        for i in range(len(self.x)):
            text = re.sub(',', ' ', self.x[i])
            text = re.sub('@', '', text)
            text = re.sub('is', '', text)
            text = re.sub('\'s', '', text)
            text = re.sub('has', '', text)
            text = re.sub('are', '', text)
            text = re.sub('got', '', text)
            text = re.sub('be', '', text)


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

        sizes = []
        for group in groupIndexes:
            sizes.append(len(group))

        index_to_sort = np.array(sizes).argsort()

        PERCENT_TO_CLASSIFY = .90

        best_group_indexes = index_to_sort[int(PERCENT_TO_CLASSIFY*len(groupIndexes)):]

        best_groups_NNP = []
        best_groups = []
        for i in best_group_indexes:
            best_groups_NNP.append(NNP[i])
            best_groups.append(groupIndexes[i])

        converged = False

        temp = index_to_sort[0:int(PERCENT_TO_CLASSIFY*len(groupIndexes))]
        NNP_to_classify = []
        indexes_to_classidy = []

        for i in temp:
            NNP_to_classify.append(NNP[i])
            indexes_to_classidy.append(groupIndexes[i])

        passes = 0
        cant_classify = []
        while len(indexes_to_classidy) > 0:
            new_list = []
            for i in range(len(indexes_to_classidy)):
                if len(NNP_to_classify[i]) - passes > 1:
                    temp = [i_NNP for i_NNP in best_groups_NNP if NNP_to_classify[i][0:len(NNP_to_classify[i]) - passes] in i_NNP]
                    if len(temp) > 0:
                        best_groups[best_groups_NNP.index(temp[0])].append(index_to_sort[indexes_to_classidy[i]])
                    else:
                        new_list.append(indexes_to_classidy[i])
                else:
                    cant_classify.append(indexes_to_classidy[i])
            indexes_to_classidy = list(new_list)
            passes += 1

        index = 0
        for NNP_temp in range(len(best_groups_NNP)):
            if best_groups_NNP[NNP_temp] == []:
                index = NNP_temp
                break

        classes = [-1 for i in range(len(self.x))]
        for i in range(len(best_groups)):
            if i is not index:
                for j in best_groups[i]:
                    classes[j] = i

        to_pop = best_groups[index]
        for i in cant_classify:
            for j in i:
                to_pop.append(j)
        to_pop.sort()

        for i in reversed(to_pop):
            classes.pop(i)

        best_groups.pop(index)

        test = list(to_pop)
        train = self.vocab.toarray()

        tempVocabArray = self.vocab.toarray()

        test.sort()
        for i in reversed(test):
            train = np.delete(train, i, 0)

        clf = linear_model.SGDClassifier()
        clf.fit(train, classes)

        lalaland = np.array([tempVocabArray[i] for i in test])

        guesses = clf.predict(lalaland)

        classes = range(len(self.x))
        for i in range(len(best_groups)):
            if i is not index:
                for j in best_groups[i]:
                    classes[j] = i

        for i in range(len(test)):
            classes[test[i]] = guesses[i]


        return classes, best_groups 

    def classify2(self, classes, best_groups):

        # unassigned = [temp]
        #while unassigned is not empty

        km = KMeans(int(len(best_groups)/2))
        km.fit(self.vocab, classes)
        km_results = km.labels_

        new_classes = range(len(classes))
        for i in set(classes):
            indexes = [n for (n, e) in enumerate(classes) if e == i]
            km_guesses = [km_results[ind] for ind in indexes]
            guess = max(set(km_guesses), key=km_guesses.count)

            for ind in indexes:
                new_classes[ind] = guess

        self.predicted = new_classes


    def guess(self):
        return self.predicted

#unclassified tweets
def loadAllData():
    tweets = []

    for line in open("twitter-1.17.1/all.out"):
        words = line.strip().split()
        tweets.append(",".join(words))

    random.shuffle(tweets)

    vec = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=10000,
                                   stop_words='english')
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
    vec = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=10000,
                                   stop_words='english')
    vocab = vec.fit_transform(tweets)
    return tweets,vocab,goldCategories

def kmeansKnown(vocab):
    km = KMeans(30)
    km.fit(vocab)
    return km.labels_

def dbscanunknown(vocab):
    db = DBSCAN(eps=.9).fit(vocab)
    lb =  db.labels_
    return lb

def accuracy(gold , pred):
    numberToCategory = {}
    for i in set(pred):
        indexes = [n for (n, e) in enumerate(pred) if e == i]
        guesses = [gold[j] for j in indexes]
        guess = max(set(guesses), key=guesses.count)
        numberToCategory[i] = guess
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

def lda():

    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    # stemming doesn't seem to actually help
    p_stemmer = PorterStemmer()

    tweets = []
    categories = os.listdir("twitter-1.17.1/data/")

    for category in categories:
        for line in open("twitter-1.17.1/data/"+category):
            tweets.append(line)
    data = []
    for i in tweets:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        data.append(stemmed_tokens) # data.append(stemmed_tokens)

    dictionary = corpora.Dictionary(data)
    corpus = [dictionary.doc2bow(tweet) for tweet in data]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=15, id2word = dictionary, passes=100)

    # https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
    for i in range(0,ldamodel.num_topics):
        print "\nTopic", i
        for j in ldamodel.get_topic_terms(i, 5):
            word = dictionary[j[0]]
            probability = j[1]
            print word,  "=",  probability


if __name__ == "__main__":
    x,vocab,y_gold = loadCategoryData()
    #vocab = loadAllData()
    
    #drop lowest percent of data... 
    sumCol = vocab.sum(axis=0)
    PERCENT_TO_DROP = 0
    count_to_drop = int(vocab.shape[1] * PERCENT_TO_DROP)
    indexes_to_delete= sumCol.argsort().tolist()[0][0:count_to_drop]
    indexes_to_delete.sort()
    for index in reversed(indexes_to_delete): 
        vocab = dropcols_coo(vocab, index) 


    print "\nClustering with unknown K\n"
    y_pred = dbscanunknown(vocab)
    print accuracy(y_gold, y_pred)


    print "\nKMeans with known K\n"
    y_pred = kmeansKnown(vocab)
    print accuracy(y_gold, y_pred)


    print "\nSupervised NaiveBayes\n"
    NB = NaiveBayes(vocab, y_gold)
    print NB.accuracy()
    

    print "\nMy Sweet Ass Algorithm with unknown K\n"
    MSAA = mySweetAssAlgorithm(x, vocab)
    print accuracy(y_gold, MSAA.guess() )



