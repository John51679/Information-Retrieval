from elasticsearch import Elasticsearch
import pandas as pd


es = Elasticsearch()

df = pd.read_csv("movies.csv")

movies = df.iloc[:,:].values
for i in range(len(movies[:])):
    mov = {"movie": movies[i,1],"movie type": movies[i,2]}
    es.index(index="dataset",doc_type="type",id=movies[i,0],body=mov)