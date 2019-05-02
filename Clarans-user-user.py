from Clarans import clarans
import pandas as pd
from pyclustering.utils import timedcall
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def creatMatrice(usagearrays):
    for i in range(len(usagearrays)):
        avg=sum(usagearrays[i])/np.count_nonzero(usagearrays[i])
        for j in range(len(usagearrays[i])):
            if(usagearrays[i,j]!=0):
                usagearrays[i,j]=usagearrays[i,j]-avg        
    return usagearrays

#the function changes the value of the matrix that's why it works good ._.!

#a = np.array([[4.0,0.0,0.0,5.0,1.0,0.0,0.0],[5.0,5.0,4.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,2.0,4.0,5.0,0.0],[0.0,3.0,0.0,0.0,0.0,0.0,3.0]])
#matrice=creatMatrice(a)

ratings = pd.read_csv('ua.base',sep='\t',names=['user','movie','rating','time'])
usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating').fillna(0) 
matrice=creatMatrice(usagematrix.values)
print(matrice)

clarans_instance = clarans(matrice, 6, 5, 100)
(ticks, result) = timedcall(clarans_instance.process)
clusters = clarans_instance.get_clusters()
print(clusters)



