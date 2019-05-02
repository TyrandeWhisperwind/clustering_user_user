import pandas as pd
from scipy.spatial import distance
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import *
import numpy as np
import math 

#np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})

def meanRatings(usageMatrix):
    listeMeanRatings=[]
    for i in range(len(usageMatrix)):
        rating=np.sum(np.trim_zeros(usageMatrix[i]))/np.count_nonzero(usageMatrix[i])
        listeMeanRatings.append(rating)
    return listeMeanRatings
#########################################################################
#read a file and create matrice user user with cosin similarity
def creatMatrice(usagearrays):
    for i in range(len(usagearrays)):
        avg=sum(usagearrays[i])/np.count_nonzero(usagearrays[i])
        for j in range(len(usagearrays[i])):
            if(usagearrays[i,j]!=0):
                usagearrays[i,j]=usagearrays[i,j]-avg        
    matrice = cosine_similarity(usagearrays) #if similarity we use max if distance we use min 
    matrice=(1.-matrice)/2.
    np.fill_diagonal(matrice, np.nan)#not taking the element itself when counting
    return matrice
#################################################################################################
#read a test file and create a dictionary of movies that we need to guess
def createDictTestMovies(testFile):
    # test file 
    movieDict =  defaultdict(list)#list of a couple 
    with open(testFile, mode='r', encoding='UTF-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            userID = int(fields[0])-1 #users are from 0 to 942
            movieID = fields[1]
            rating = fields[2]
            movieDict[userID].append({ movieID:rating })
    return movieDict
#################################################################################################
def getNeighbours(k,matrice):
    userNeighbours=defaultdict(list)
    for x in range(len(matrice)):#0~942 users
        ligne=matrice[x]
        for cpt in range(k):
            userNeighbours[x].append(np.nanargmin(ligne))#if similarity we use max if distance we use min 
            ligne[np.nanargmin(ligne)]=np.nan #remove max element from the list to get the next max element 
    return userNeighbours
#################################################################################################
def knn(k,baseFile,testFile):
    ratings = pd.read_csv(baseFile,sep='\t',names=['user','movie','rating','time'])
    usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating').fillna(0) 
    usagearrays=usagematrix.values

    #print("matrice d'usage")
    #print(usagearrays)
    listeMeanRatings=meanRatings(usagearrays)
    userNeighbours= getNeighbours(k,creatMatrice(usagearrays))
    movieDict=createDictTestMovies(testFile)
  
    aze= ratings.pivot_table(index='user', columns='movie', values='rating').fillna(0) 
    aze=aze.values
    realValue=[]
    predection=[]
    for x in range(len(aze)):#0~942 users ids
        #get the id of movies in test file of user x
            listOfMovies=movieDict[x]#0~942 users ids: get the movies of user x
            for element in listOfMovies:
                for val in element:
                    #print("movie",int(val)-1)
                    realValue.append(element[val])#element[val] is rating and int(val)-1 is the id of the movie
                    rating=0
                    #print("neighbours of the user",x,"=",userNeighbours[x])
                    for j in userNeighbours[x]:#get the neighbours of  user x
                        #test if rating is zero 
                            if (aze[j][int(val)-1]==0.):
                                rating=rating+listeMeanRatings[x]
                            else:
                                rating=rating+aze[j][int(val)-1]#remove 1 cuz in matrix movies are from 0 to ...
                    rating=rating/k#average rating of neighobrs for a given movie 
                    if rating >5.: 
                        rating=5.
                    predection.append(round(rating))

    realValue=list(np.float_(realValue))
    mae=mean_absolute_error(realValue,predection)
    rmse=math.sqrt(mean_squared_error(realValue,predection))
    resutls=[]
    resutls.append(mae)
    resutls.append(rmse)
    #print("real values:",realValue)
    #print("predection:",predection)
    #print("-----------------------")
    print("mean_absolute_error and mean_squared_error=",resutls)
    return resutls
#################################################################################################
knn(60,"ua.base","ua.test")
