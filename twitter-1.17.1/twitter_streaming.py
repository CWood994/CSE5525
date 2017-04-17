# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
import sys


# Variables that contains the user credentials to access Twitter API 
ACCESS_TOKEN = '302739969-dlFZ55JJT59jTowrxOhBH49QBu8PiOLwD3aj5VrQ'
ACCESS_SECRET = 'lgmYjzcn1m08UBX8jN5R2v9XHAh9Rx9eQA4dhEGJtL94A'
CONSUMER_KEY = 'a8QvO4l69U66dNs3s3UerYn5B'
CONSUMER_SECRET = 'bxyW8zcAVfUdMpYU16FKdHYAbj5VIkLNAksNnKqPyOd8gXhtFD'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

# Initiate the connection to Twitter Streaming API
twitter_stream = TwitterStream(auth=oauth)

# Get a sample of the public data following through Twitter
#iterator = twitter_stream.statuses.filter(track="#family", language="en")

twitter = Twitter(auth=oauth)
HASH_WORD = ["#syria", "#easter", "#baseball", "#gym", "#unitedairlines", "#terror", "#stocks", "#nba", "#tech", "#prisonbreak", "#food", "Birthday", "NASA", "#PET", "#country"]
FILE_NAME = ["syria", "easter", "baseball", "gym", "united", "terror", "stocks", "nba", "tech", "prisonbreak", "food", "birthday" , "nasa", "pet", "country"]
tweet_count = 1000 #max that can be received is 100 tho... 



for i in range(len(HASH_WORD)):
    
    iterator = twitter.search.tweets(q=HASH_WORD[i], result_type = 'recent', lang='en', count=tweet_count)

    print len(iterator['statuses'])
    import re

    def clean(text):
        text = re.sub(HASH_WORD[i]+"[^ ]*", '', text.lower())
        text = re.sub('@', ' @ ', text)
        text = re.sub("[^a-zA-Z' @0-9]","", text)
        text = re.sub("https[a-z0-9]+"," ",text)
        text = re.sub("\s+" , " ", text)
        text = text.strip()
        return text

    data = []
    print "pehle"   
    for tweet in iterator['statuses']:

        tweet = json.loads(json.dumps(tweet) )
        if 'text' in tweet: # only messages contains 'text' field is a tweet
            #print tweet['id'] # This is the tweet's id
            #print tweet['created_at'] # when the tweet posted

            lengthofHashTags = 0
            for hashtag in tweet['entities']['hashtags']:
                lengthofHashTags += len(hashtag['text'])

            cleaned = clean(tweet['text']) # content of the tweet
            if (len(cleaned) - lengthofHashTags) > 15 and cleaned[0:2] != "rt":
                print tweet['text'].encode('utf-8')
                print cleaned
                data.append(cleaned)
                print "\n"
                        
            #print tweet['user']['id'] # id of the user who posted the tweet
            #print tweet['user']['name'] # name of the user, e.g. "Wei Xu"
            #print tweet['user']['screen_name'] # name of the user account, e.g. "cocoweixu"
    path = ""       
    # path = "/Users/Saurabh/Desktop/study/SLP/Project/CSE5525/twitter-1.17.1/"  
    print path+"data_big/" + FILE_NAME[i] + ".out"     
    with open(path+"data_big/" + FILE_NAME[i] + ".out", "a") as text_file:
        print text_file
        for result in data:
            text_file.write(str(result) + "\n")
'''






#for all data


iterator = twitter_stream.statuses.sample(language="en")

#print len(iterator)
import re

def clean(text):
    text = text.lower()
    text = re.sub('@', ' @ ', text)
    text = re.sub("[^a-zA-Z' @0-9]","", text)
    text = re.sub("https[a-z0-9]+"," ",text)
    text = re.sub("\s+" , " ", text)
    text = text.strip()
    return text

with open("data/all.out", "a") as text_file:
    for tweet in iterator:
        tweet = json.loads(json.dumps(tweet) )
        if 'text' in tweet: 
            cleaned = clean(tweet['text'])
            print tweet['text']
            print cleaned
            text_file.write(str(cleaned) + "\n")
            print "\n"
'''


                    












