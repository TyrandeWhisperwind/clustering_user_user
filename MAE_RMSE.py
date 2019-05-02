import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

##############################################################################
def getRating(clusterI,idMovie,idUser,matrice):
    rating=0
    for cpt in clusterI:
        rating=rating+matrice[cpt][idMovie]
    #remove rating of the user i'm predecting (didn't want to test if cpt!=idUser...)
    rating=rating-matrice[idUser][idMovie]#i know it is equal to zero but in the futur we might replace them with something (._. who knows !!)
    return rating
#######################################################
#create a dictionary of movies that we need to guess
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
############################################################################################
def MAE_RMSE(ratings,Clusters,testFile):
    usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating')
    usagematrix=usagematrix.apply(lambda usagematrix: usagematrix.fillna(usagematrix.mean()), axis=1)
    matrice=usagematrix.values
    realValue=[]
    predection=[]
    movieDict=createDictTestMovies(testFile)
    for label in range(len(Clusters)): 
        for idUser in Clusters[label]:
                listOfMovies=movieDict[idUser]
                for element in listOfMovies:
                    for idMovie in element:
                        #print(idUser+1,idMovie)
                        realValue.append(element[idMovie])#element[val] is rating and int(idMovie)-1 is the id of the movie
                        rating=getRating(Clusters[label],int(idMovie)-1,idUser,matrice)
                        if len(Clusters[label])==1:#there is only one element in the cluster no predection can be done
                            rating =0 #maybe we can add them to the nearest cluster ... we 'll see later(case kmedoids)
                            predection.append(rating)
                        else : 
                            rating=rating/(len(Clusters[label])-1)
                            if rating >5.:
                                rating=5.
                            predection.append(round(rating))

    realValue=list(np.float_(realValue))
    mae=mean_absolute_error(realValue,predection)
    rmse=sqrt(mean_squared_error(realValue,predection))
    resutls=[]
    resutls.append(mae)
    resutls.append(rmse)
    #print("real values:",realValue)
    #print("predection:",predection)
    print("mean_absolute_error and mean_squared_error=",resutls)
    return resutls