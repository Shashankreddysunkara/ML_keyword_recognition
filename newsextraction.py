from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import json, pickle, requests, pymongo
from nltk.corpus import stopwords
import time

def geturl(url):
    res=requests.get(url)
    return res.text
with open('data/cleannews.json') as f:
    corpus=json.load(f)
#
# ##Creating a list of stop words and adding custom stopwords
# stop_words = set(stopwords.words("english"))
# ##Creating a list of custom stopwords
# new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown", "headline_text"]
# stop_words = stop_words.union(new_words)
# cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
# X=cv.fit_transform(corpus)
# tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
# tfidf_transformer.fit(X)
#
# print ("save tfid vector")
# pickle.dump(cv.vocabulary_,open("data/newstransform.pkl","wb"))

#Load it later

def get_tfid():
    print ("resuse vector")
    transformer = TfidfTransformer()
    cv = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("data/newstransform.pkl", "rb")))
    X=cv.fit_transform(corpus)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(X)



    # doc="Chandrayaan 1: The Story of India’s Budget Moon Spaceship - The Weather Channel"


    return cv, tfidf_transformer

def getkeywords(cv, doc, tfidf_transformer):
    feature_names=cv.get_feature_names()
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
    #Function for sorting tf_idf in descending order
    from scipy.sparse import coo_matrix
    def sort_coo(coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def extract_topn_from_vector(feature_names, sorted_items, topn=3):
        """get the feature names and tf-idf score of top n items"""

        #use only topn items from vector
        sorted_items = sorted_items[:topn]

        score_vals = []
        feature_vals = []

        # word index and corresponding tf-idf score
        for idx, score in sorted_items:

            #keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        #create a tuples of feature,score
        #results = zip(feature_vals,score_vals)
        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]

        return results
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,5)

    return keywords



## analyse news similarity

import gensim
from gensim.models.callbacks import CallbackAny2Vec
from color import *

class EpochLogger(CallbackAny2Vec):
    """docstring for ."""
    def __init__(self):

        self.epoch = 0

    def on_epoch_begin(self, model):
        print ("Epoch # {} start".format(self.epoch))

    def on_epoch_end(self, model):
        print ("Epoch # {} end".format(self.epoch))
        self.epoch+=1


model = gensim.models.Word2Vec.load("models/kipgrams_newsheadline.model")

cats=['banking', 'finance', 'health', 'fitness', 'entertainment', 'shopping', 'travel', 'sports',
'lifestyle', 'technology']

def getsimilarity(keywords):
    # print ("start checking similarity==============================================")
    # print (keywords)
    similaritydata=[]
    for x in cats:
        try:
            for k,v in keywords.items():
                temp={}
                similarity=model.similarity(x, k)
                text="similarity between {} and {} is {}".format(x,k,similarity)
                if (similarity>0.5):
                    temp['key']=k
                    temp['category']=x
                    temp['similarity']=similarity
                    similaritydata.append(temp)

                # if (0.6>similarity>0.5):
                #     prYellow(text)
                # elif (0.7>similarity>0.6):
                #     prGreen(text)
                # elif (similarity>0.7):
                #     prRed(text)
        except Exception as e:
            print ("similarity between {} and {} is {}".format(x,k,str(e)))
    return similaritydata
    # print ("end checking similarity==============================================")



# print (keywords)

currentnews=geturl("https://newsapi.org/v2/top-headlines?country=in&apiKey=45b15d84ea944fb09bbb7baa11ebb9bf")
articles=json.loads(currentnews)['articles']
cv, tfidf_transformer =get_tfid()
newsdict=[]
for art in articles:

    keywords=getkeywords(cv,art['title'], tfidf_transformer)
    sim=getsimilarity(keywords)
    temp={'title':art['title'], 'organic':keywords, 'similaritydata':sim}
    if (len(temp)>0):
        newsdict.append(temp)

def appendkey(key,temp):
    if key!="":
        temp.append(key)

    return temp

def datafinalisation(simdata):

    sim=simdata['similaritydata']
    sim=sorted(sim, key = lambda i: i['similarity'])
    heavykey={}
    for x in sim:
        if x['key'] in heavykey:
            heavykey[x['key']]+=x['similarity']
        else:
            heavykey[x['key']]=x['similarity']

    # print (heavykey)
    q=-1
    for y,v in heavykey.items():
        if q<v:
            q=v
            hv=y

    # print (hv, q)
    fd_data=[]
    for x in sim:
        if x['key']==hv:
            fd_data.append(x['category'])
    key1=""
    key2=""
    if (len(fd_data)>2):
        key1=fd_data[0]
        key2=fd_data[1]
    elif (len(fd_data)>1):
        key1=fd_data[0]
        key2=""
    groups=['healthfitness', 'financebanking']
    keys=[]
    if (key1+key2) in (groups):
        keys.append(key1+key2)
    else:
        keys=appendkey(key1, keys)
        keys=appendkey(key2, keys)
        # print (key1, key2)
    if len(keys)>0:
        print (keys, simdata['title'])

    return keys, simdata['title'], ' '.join(simdata['organic'].keys())

sendkeydata=[]
for news in newsdict:
    keys, title, organic = datafinalisation(news)
    print (organic)
    # now find sponsored_keywords
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["pushnotif"]
    mycol = mydb["keywords"]
    notifdata=[]
    base_url="https://news.google.com/search?q="
    for x in keys:
        spkeywords = mycol.find({"category":x}).limit(1).skip(9)
        for sp in spkeywords:
            temp={}
            temp={'title':'Daily News', 'url':base_url+sp['keyword'], 'description':title, 'category':x}
            notifdata.append(temp)
    notifdata.append({'title':'Daily News', 'url':base_url+organic, 'description':title})
    # if len(notifdata)>0:
        # print (notifdata)
    sendkeydata+=notifdata


print (sendkeydata)


# print (sendkeydata)
def sendpost(sendata):

    # print (data)
    headers={
        "Authorization":"Token 55e1daeca5b7857f7fd5fbbc489642375272e372",
        "Content-Type":"application/json"
    }

    url='https://devnotify.expressnotify.com/api/upload/'
    # imeis=['861638039239913']
    imeis=['861638039239913', '861736034841525', '911511400208330', '867729030675645']
    pkg='com.gionee.gnservice'
    for x in imeis:
        data={'imei':x, 'packageName':pkg, 'tag':'Notification Plain'}
        data.update(sendata)
        data=json.dumps(data)
        req=requests.post(url, data=data, headers=headers)
        print (req.text, data)



def pushdemo(sendata):

    for idx, data in enumerate(sendata):
        if (idx!=0 and idx%4==0):
            time.sleep(60)
            print ("wait for a minute")
        else:
            sendpost(data)
            time.sleep(60)


pushdemo(sendkeydata)
