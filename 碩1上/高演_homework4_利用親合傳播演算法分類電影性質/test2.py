import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score


#讀取movies的CSV檔案
movies = pd.read_csv("movies.csv", encoding='ISO-8859-1')

#分割電影性質
genres = movies['genres'].str.split("|",expand= True)
num_genres = pd.DataFrame(np.array(genres).reshape(192130,1))
# print(pd.Series(num_genres[0]).value_counts()) #列出各個性質共有幾部片

#分割維度轉向量 使用one-hot encoding
cleaned = movies.set_index('movieId').genres.str.split('|', expand=True).stack()
feature_g =pd.get_dummies(cleaned,prefix='g').groupby(level=0).sum()
feature_g = pd.DataFrame(feature_g.values,columns=['g_(no genres listed)','g_Action','g_Adventure','g_Animation','g_Children','g_Comedy', 'g_Crime','g_Documentary','g_Drama','g_Fantasy', 'g_Film-Noir', 'g_Horror','g_IMAX','g_Musical','g_Mystery','g_Romance', 'g_Sci-Fi', 'g_Thriller', 'g_War','g_Western'])
movies_feature = pd.concat([movies,feature_g],axis=1)
genres_features = movies_feature[['g_(no genres listed)', 'g_Action', 'g_Adventure', 'g_Animation', 'g_Children', 'g_Comedy', 'g_Crime', 'g_Documentary', 'g_Drama', 'g_Fantasy', 'g_Film-Noir', 'g_Horror', 'g_IMAX', 'g_Musical', 'g_Mystery', 'g_Romance', 'g_Sci-Fi', 'g_Thriller', 'g_War', 'g_Western']]
print(genres_features)

