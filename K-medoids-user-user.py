
from kmedoids import kmedoids
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.insert(0, 'C:/Users/HP/Desktop/RM')
from MAE_RMSE import MAE_RMSE
from pandas import DataFrame

def creatMatrice(usagearrays):
    for i in range(len(usagearrays)):
        avg=sum(usagearrays[i])/np.count_nonzero(usagearrays[i])
        for j in range(len(usagearrays[i])):
            if(usagearrays[i,j]!=0):
                usagearrays[i,j]=usagearrays[i,j]-avg        
    matrice = cosine_similarity(usagearrays) 
    matrice=(1.-matrice)/2.
    return matrice

############################################################################################

ratings = pd.read_csv('ua.base',sep='\t',names=['user','movie','rating','time'])
usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating')
usagematrix=usagematrix.apply(lambda usagematrix: usagematrix.fillna(usagematrix.mean()), axis=1)

matrice=creatMatrice(usagematrix.values)

StartMedoids=[275,   9 ]
kmedoids_instance = kmedoids(matrice, StartMedoids,ccore=False,data_type='distance_matrix')
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

print(medoids)

MAE_RMSE(ratings,clusters,"ua.test")
