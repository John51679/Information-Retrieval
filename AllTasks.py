from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
import re
import string
from sklearn.cluster import KMeans
from keras_preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import time
import os

warnings.filterwarnings("ignore")
es = Elasticsearch()

ds = pd.read_csv("ratings.csv")

rt = ds.iloc[:,:].values

#clusters = []

def Search1(search_bar):
    start = time.time()
    results = []

    body = {
        "from":0,
        "size":9000,
        "query": {
            "match": {
                "movie":search_bar
            }
        }
    }

    res = es.search(body=body, index="dataset")

    for i in range(len(res.get("hits").get("hits"))):
        results += [[res.get("hits").get("hits")[i].get("_source").get("movie")]]
        results[i] += [res.get("hits").get("hits")[i].get("_score")]
    end = time.time()
    print("total time for function Search1 is {0}".format(end - start))
    return results

def Search2(search_bar,userID, ratings = None, neuralArray = None):
    start = time.time()
    hasRatings = False
    hasNeuralArray = False
    if (type(ratings).__name__ != 'NoneType'):
        cols = ratings.columns
        cols = cols.to_list()
        ratings = ratings.iloc[:, :].values
        hasRatings = True
    if (type(neuralArray).__name__ != 'NoneType'):
        hasNeuralArray = True
    if (type(neuralArray).__name__ == 'DataFrame'):
        neuralArray = neuralArray.iloc[:,:].values
        hasNeuralArray = True
    results = []
    body = {
        "from":0,
        "size":9000,
        "query": {
            "match": {
                "movie":search_bar
            }
        }
    }

    res = es.search(body=body, index="dataset")

    for i in range(len(res.get("hits").get("hits"))):
        avg = []
        results += [[res.get("hits").get("hits")[i].get("_source").get("movie")]]
        results[i] += [res.get("hits").get("hits")[i].get("_score")]
        for j in range(len(rt[:])):
            if (rt[j,0] == userID + 1 and hasRatings == False):
                results[i] += [0]
                break
            if (rt[j,0] == userID + 1 and hasRatings == True):
                rate = 0

                TYPE = res.get("hits").get("hits")[i].get("_source").get("movie type")
                TYPE = re.sub("[" + string.punctuation[-3] + "]", ' ',TYPE).split()
                for k in TYPE:
                    index = cols.index(k)
                    rate += ratings[int(userID - 1),index]
                rate = rate / len(TYPE)
                results[i] += [rate]
                break
            if (rt[j,0]==userID and rt[j,1] == int(res.get("hits").get("hits")[i].get("_id"))):
                results[i] += [rt[j,2]]
                break
        if  (hasRatings):
            TYPE = res.get("hits").get("hits")[i].get("_source").get("movie type")
            TYPE = re.sub("[" + string.punctuation[-3] + "]", ' ', TYPE).split()
            allRates = []
            for j in range(len(ratings[:])):
                calcRate = 0
                for k in TYPE:
                    index = cols.index(k)
                    calcRate += ratings[j,index]
                calcRate = calcRate / len(TYPE)
                allRates += [calcRate]
            SUM = sum(allRates)
            avg = SUM / len(allRates)
            results[i] += [avg]
        elif (hasNeuralArray):
            id = int(res.get("hits").get("hits")[i].get("_id"))
            seq_no = es.get('dataset',id)
            seq_no = seq_no.get("_seq_no")
            avg = np.mean(neuralArray[:,seq_no])
            results[i][2] = neuralArray[userID - 1,seq_no]
            results[i] += [avg]
        else:
            for j in range(len(rt[:])):
                if (rt[j,1] == int(res.get("hits").get("hits")[i].get("_id"))):
                    avg += [rt[j,2]]
            length = len(avg)
            avg = sum(avg)
            results[i] += [avg / length]
        print("personal rating = {0} and movieid = {1}".format(results[i][2],res.get("hits").get("hits")[i].get("_id")))
        results[i][1] = results[i][1] + results[i][2] + results[i][3]
        results[i].pop(3)
        results[i].pop(2)

    end = time.time()
    print("total time for function Search2 is {0}".format(end - start))
    return results


def Search3(forSearch4 = False):
    start = time.time()
    unique = []
    movie_names = []
    ratings = []
    clusters = []
    for i in range(es.count(index="dataset").get("count")):
        body = {
            "from":i,
            "size":1
        }
        cat = es.search(index = "dataset",body=body)
        name = cat.get("hits").get("hits")[0].get("_source").get("movie")
        cat = re.sub("[" + string.punctuation[-3] + "]",' ',cat.get("hits").get("hits")[0].get("_source").get("movie type")).split()
        for j in cat:
            if j not in unique:
                unique += [j]
                movie_names += [[]]
                ratings += [[]]
            index = unique.index(j)
            movie_names[index] += [name]

    unique.pop(-1)
    unique.pop(-1)
    ratings.pop(-1)
    ratings.pop(-1)
    movie_names.pop(-1)
    movie_names.pop(-1)
    unique[-1] = "no genres listed"
    array = np.zeros([int(max(rt[:,0])),len(unique)+1])
    kmArray = np.zeros([int(max(rt[:,0])),len(unique)])
    counter = 0
    for i in range(len(array[:])):
        array[i,0] = i + 1

        temp = [x[:] for x in ratings]
        while (counter < len(rt[:]) and rt[counter,0] == i+1):
            name = es.get_source(index = "dataset",id = int(rt[counter,1])).get("movie")
            for j in range(len(unique)):
                try:
                    movie_names[j].index(name)
                    temp[j].append(rt[counter,2])
                except ValueError: continue
            counter += 1
        for j in range(len(temp)):
            if len(temp[j]) != 0:
                temp[j] = sum(temp[j])/len(temp[j])
            else: temp[j] = None
        SUM = 0
        for j in range(len(temp)):
            if temp[j] != None:
                SUM += temp[j]
        array[i,1:] = temp
        SUM = SUM / (len(temp) - temp.count(None))
        for j in range(len(temp)):
            if temp[j] == None:
                temp[j] = SUM
        kmArray[i,:] = temp
    k = 15
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(kmArray)
    result = kmeans.predict(kmArray)
    for i in range(k):
        clusters.append([])
    for i in range(len(result)):
        clusters[result[i]] += [i]

    if forSearch4:
        return clusters

    for i in range(len(clusters)):
        for j in range(1,len(array[0,:])):
            temp = []
            mean = 0
            counter = 0
            for k in clusters[i]:
                temp.append(array[int(k-1),j])
            for k in temp:
                if not np.isnan(k):
                    mean += float(k)
                    counter += 1
            try:
                mean = mean / counter
            except ZeroDivisionError:
                mean = 3.0
            for k in clusters[i]:
                if np.isnan(array[int(k-1),j]):
                    array[int(k-1),j] = mean
    unique.insert(0,'id')
    ds = pd.DataFrame(data=array,columns=unique)
    ds.to_csv(os.path.abspath(os.getcwd()) + '\\search3.csv',index=False)
    end = time.time()
    print("total time for function Search3 is {0}".format(end - start))
    return ds

def Search4(withSearch3 = False):
    start = time.time()
    MTlist = []
    MovieList = []
    ids = []

    for i in range(es.count(index="dataset").get("count")):
        body = {
            "from": i,
            "size": 1
        }
        cat = es.search(index="dataset", body=body)
        id = cat.get("hits").get("hits")[0].get("_id")
        ids += [id]
        name = cat.get("hits").get("hits")[0].get("_source").get("movie")
        cat = re.sub("[" + string.punctuation[-3] + "]", ' ',cat.get("hits").get("hits")[0].get("_source").get("movie type"))
        MTlist += [cat]
        MovieList += [name]
    for i in range(len(MTlist)):
        for j in range(len(MTlist[i])):
            try:
                if (MTlist[i][j] == '-'):
                    MTlist[i] = MTlist[i].replace(MTlist[i][j], '')
            except IndexError: break
    oh = [one_hot(i, 30) for i in MTlist]
    temp = pad_sequences(oh, maxlen=10, padding='pre')
    array = np.zeros([len(temp),1])
    array = np.insert(array,0, ids, axis=1)
    array = np.delete(array,1,axis = 1)

    filtered_data = []
    for i in range(len(MovieList)):
        tokens = word_tokenize(MovieList[i])
        for j in range(len(tokens)):
            tokens[j] = tokens[j].lower()
        filtered_data += [tokens]

    for i in range(len(filtered_data)):
        error = 0
        for j in range(len(filtered_data[i])):
            if (filtered_data[i][j - error] in string.punctuation):
                filtered_data[i].remove(filtered_data[i][j - error])
                error = error + 1
    size = 100
    model = Word2Vec(filtered_data, min_count=1,size=size)
    array = np.append(array,np.zeros([len(array[:]),size]),1)
    for i in range(len(filtered_data)):
        a = np.zeros([100])
        for j in range(len(filtered_data[i])):
            a += model.wv.__getitem__(filtered_data[i][j])
        array[i,1:] = a
    array = np.append(array,temp,1)


    """Test Neural"""
    totalRatings = np.copy(rt)
    """de 3erw"""
    temp = totalRatings[:,2]
    temp = np.transpose(temp)
    Decoder = np.unique(temp)
    Decoder = Decoder.reshape(1,-1)
    temp = temp.tolist()
    le = LabelEncoder()
    label_encoder = np.array(le.fit_transform(temp))
    Decoder = np.append(Decoder,np.unique(label_encoder).reshape(1,-1),axis=0)
    totalRatings = np.insert(totalRatings,2,label_encoder,axis=1)
    totalRatings = np.append(totalRatings,np.zeros([1,len(totalRatings[0,:])]),axis=0)
    """akoma de 3erw"""
    sorted_array = array[np.argsort(array[:,0])]
    if withSearch3:
        resultArray = np.zeros([1, len(sorted_array[:, 0]) + 1])
        clusters = Search3(True)
        for i in range(len(clusters)):
            counterI = 0
            inputTesting = np.copy(sorted_array)
            outputTraining = []
            inputTraining = np.zeros([1, len(array[0, :])])
            for j in range(len(clusters[i])):
                counterJ = 0

                while (int(totalRatings[counterI, 0]) != clusters[i][j] + 1):
                    counterI += 1
                while (int(totalRatings[counterI, 0]) == clusters[i][j] + 1):
                    if (int(inputTesting[counterJ, 0]) == int(totalRatings[counterI, 1])):
                        inputTraining = np.append(inputTraining, inputTesting[counterJ, :].reshape(1, -1), axis=0)
                        inputTesting = np.delete(inputTesting, counterJ, axis=0)
                        outputTraining += [[totalRatings[counterI, 2]]]
                        totalRatings = np.delete(totalRatings, counterI, axis=0)
                        counterJ -= 1
                    elif (int(inputTesting[counterJ, 0]) > int(totalRatings[counterI, 1])):
                        totalRatings = np.delete(totalRatings, counterI, axis=0)
                        counterJ -= 1
                    counterJ += 1
            inputTraining = np.delete(inputTraining, 0, axis=0)

            outputTraining = np.array(outputTraining).reshape(-1, 1)
            mlp = MLPClassifier(hidden_layer_sizes=(15, 10, 5), max_iter=200)
            mlp.fit(inputTraining[:, 1:], outputTraining)

            length = len(inputTesting[:, 0])
            inputTesting = np.append(inputTesting, inputTraining, axis=0)
            IndexSort = np.argsort(inputTesting[:, 0])
            for k in range(len(clusters[i])):
                ot = np.copy(outputTraining)
                prediction = mlp.predict(inputTesting[:length, 1:])

                prediction = np.reshape(prediction, [-1, 1])
                for j in range(len(IndexSort)):
                    if (int(IndexSort[j]) >= int(length)):
                        prediction = np.insert(prediction, j, ot[0], axis=0)
                        ot = np.delete(ot, 0, axis=0)

                    prediction[j] = Decoder[0, int(prediction[j])]
                prediction = np.insert(prediction, 0, clusters[i][k], axis=0)
                resultArray = np.append(resultArray, prediction.transpose(), axis=0)
        resultArray = resultArray[np.argsort(resultArray[:,0])]
        resultArray = np.delete(resultArray,0,axis=1)

    else:
        resultArray = np.zeros([1, len(sorted_array[:, 0])])
        for i in range(int(max(totalRatings[:,0]))):
            inputTesting = np.copy(sorted_array)
            counter = 0
            outputTraining = []
            inputTraining = np.zeros([1, len(array[0,:])])
            """while (int(inputTesting[counter,0]) != int(totalRatings[0,1])):

                if int(inputTesting[counter,0]) == int(totalRatings[0,1]):
                    inputTraining = np.insert(inputTraining, 0, inputTesting[counter,:], axis = 0)
                    inputTraining = np.delete(inputTraining, 1, axis=0)
                    inputTesting = np.delete(inputTesting,counter,axis=0)
                    #counter = 0
                    outputTraining += [[totalRatings[0,2]]]
                    totalRatings = np.delete(totalRatings, 0, axis=0)
                    break
                counter += 1"""

            while(int(totalRatings[0,0]) == i + 1):
                if (int(inputTesting[counter, 0]) == int(totalRatings[0, 1])):
                    inputTraining = np.append(inputTraining,inputTesting[counter,:].reshape(1,-1),axis=0)
                    inputTesting = np.delete(inputTesting, counter, axis=0)
                    outputTraining += [[totalRatings[0,2]]]
                    totalRatings = np.delete(totalRatings, 0, axis=0)
                    counter -= 1
                counter += 1

            inputTraining = np.delete(inputTraining,0,axis=0)

            outputTraining = np.array(outputTraining).reshape(-1,1)
            mlp = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=200)
            mlp.fit(inputTraining[:,1:], outputTraining)
            prediction = mlp.predict(inputTesting[:,1:])
            length = len(inputTesting[:,0])
            inputTesting = np.append(inputTesting,inputTraining,axis=0)
            IndexSort = np.argsort(inputTesting[:, 0])
            prediction = np.reshape(prediction, [-1, 1])
            for j in range(len(IndexSort)):
                if (int(IndexSort[j]) >= int(length)):
                    prediction = np.insert(prediction,j,outputTraining[0],axis = 0)
                    outputTraining = np.delete(outputTraining,0,axis= 0)

                prediction[j] = Decoder[0,int(prediction[j])]
            resultArray = np.append(resultArray,prediction.transpose(),axis=0)
    """End of test"""
    resultArray = np.delete(resultArray,0,axis=0)
    ds = pd.DataFrame(resultArray)
    ds.to_csv(os.path.abspath(os.getcwd()) + '\\search4.csv', index=False)
    end = time.time()
    print("total time for function Search4 is {0}".format(end - start))
    return resultArray

""" FOR TASK 1"""

s = input("What are you looking for?\n")
id = int(input("id = "))

#res = Search3()
#res = Search2(s,id,res)
#res = Search3()
#res = Search4(True)
res = pd.read_csv("search4.csv")
res = Search2(s,id,neuralArray=res)
#res = Search4()

"""END"""

for i in res:
    print(i)