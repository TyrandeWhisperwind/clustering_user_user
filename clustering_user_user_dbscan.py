from dbscan import dbscan
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.insert(0, 'C:/Users/HP/Desktop/RM')
from MAE_RMSE import MAE_RMSE

def creatMatrice(usagearrays):
    for i in range(len(usagearrays)):
        avg=sum(usagearrays[i])/np.count_nonzero(usagearrays[i])
        for j in range(len(usagearrays[i])):
            if(usagearrays[i,j]!=0):
                usagearrays[i,j]=usagearrays[i,j]-avg        
    matrice = cosine_similarity(usagearrays) 
    matrice=(1.-matrice)/2.
    return matrice
##################################################################################
ratings = pd.read_csv('ua.base',sep='\t',names=['user','movie','rating','time'])
usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating')
usagematrix=usagematrix.apply(lambda usagematrix: usagematrix.fillna(usagematrix.mean()), axis=1)
matrice=creatMatrice(usagematrix.values)

dbscan_instance = dbscan(matrice, 0.4,20,ccore=False,data_type='distance_matrix')
dbscan_instance.process()
clusters = dbscan_instance.get_clusters()
noise = dbscan_instance.get_noise()

print(len(clusters))
print(len(noise))   

MAE_RMSE(ratings,clusters,'ua.test')



