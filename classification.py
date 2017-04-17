from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.cluster import KMeans
import os
import random
from sklearn import tree
from sklearn.cluster import AgglomerativeClustering


def loadCategoryData(folder="data"):
    tweets = []
    goldCategories = []

    categories = os.listdir("twitter-1.17.1/"+folder+"/")
    for cat in categories:
        if cat[0] == ".":
            categories.remove(cat)

    for file in categories:
        file_data = []
        for line in open("twitter-1.17.1/"+folder+"/"+file):
            tweets.append(line)
            # file_data.append(line)
            goldCategories.append(file[0:len(file)-4])
        # new_data = set(file_data)
        # print new_data
        # target = open("twitter-1.17.1/data_unique/"+file, 'w')
        # for item in new_data:
        # 	target.write("%s" % item)
        # target.close()
    c = list(zip(tweets, goldCategories))
    random.shuffle(c)
    tweets, goldCategories = zip(*c)

    return tweets,goldCategories


def accuracy(data,categories,pred):
	maj_class = {}
	for x in range(0,len(data)):
		if  pred[x] not in maj_class:
			maj_class[pred[x]] = {}
		if categories[x] not in maj_class[pred[x]]:
			maj_class[pred[x]][categories[x]] = 0
		maj_class[pred[x]][categories[x]] += 1

	classification = []
	for k,v in maj_class.iteritems():
		count = 0
		tag = None
		for key,val in v.iteritems():
			if val > count:
				tag = key
				count = val
		classification.append(tag)
	correct = 0
	for i in range(len(data)):
		if categories[i] == classification[pred[i]]:
		    correct += 1
	return correct*1.0 / len(data)

def decision_accuracy(pred,actual):
	correct = 0;
	for i in range(len(pred)):
		if actual[i] == pred[i]:
			correct += 1
	return correct*1.0 / len(pred)


n_features = 10000

# tweets,goldCategories = loadCategoryData()

tweets,goldCategories = loadCategoryData("data_unique")
# tweets,goldCategories = loadCategoryData("data_big")


train  = tweets[0:int(len(tweets)*.7)]
test  = tweets[int(len(tweets)*.7):len(tweets)]

train_class = goldCategories[0:int(len(tweets)*.7)]
test_class  = goldCategories[int(len(tweets)*.7):len(tweets)]


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')

tfidf_mat = tfidf_vectorizer.fit_transform(train)

tfidf_mat_test = tfidf_vectorizer.transform(test)

###### Kmeans
 

kmeans = KMeans(n_clusters=30, random_state=0, n_jobs=-1).fit(tfidf_mat)

print accuracy(train,train_class,kmeans.predict(tfidf_mat))

print accuracy(test,test_class,kmeans.predict(tfidf_mat_test))



### Decission trees

# clf = tree.DecisionTreeClassifier(max_depth=10)
# clf = clf.fit(tfidf_mat, train_class)

# train_result = clf.predict(tfidf_mat)

# print decision_accuracy(train_result,train_class)

# test_result = clf.predict(tfidf_mat_test)
# print decision_accuracy(test_result,test_class)

#### Heirarchical clustering

# model = AgglomerativeClustering(n_clusters=30)
# model.fit(tfidf_mat.toarray(),train_class)
# print model.labels_
# print accuracy(train,train_class,model.labels_)

