import csv, pymongo
import datetime
cat=['entertainment', 'fashion', 'financebanking', 'food', 'healthfitness', 'shopping', 'sports', 'technology', 'travel']
def pushdata(category, mycol):
    with open('data/sponsored_keywords/{}.csv'.format(category)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            temp={'category':category, 'keyword':row[0], 'created':datetime.datetime.now()}
            # print (temp)
            x = mycol.insert_one(temp)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["pushnotif"]
mycol = mydb["keywords"]
mycol.drop()


for y in cat:
    pushdata(y, mycol)
    print ("done for {}".format(y))
