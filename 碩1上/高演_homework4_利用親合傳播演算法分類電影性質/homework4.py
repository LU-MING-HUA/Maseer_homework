import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

#讀取movies的CSV檔案
movies = pd.read_csv("movies.csv", encoding='ISO-8859-1')

#分割電影性質
genres = movies['genres'].str.split("|",expand= True)
num_genres = pd.DataFrame(np.array(genres).reshape(192130,1))
print(pd.Series(num_genres[0]).value_counts()) #列出各個性質共有幾部片

#分割維度轉向量 使用one-hot encoding
cleaned = movies.set_index('movieId').genres.str.split('|', expand=True).stack()
feature_g =pd.get_dummies(cleaned,prefix='g').groupby(level=0).sum()
feature_g = pd.DataFrame(feature_g.values,columns=['g_(no genres listed)','g_Action','g_Adventure','g_Animation','g_Children','g_Comedy', 'g_Crime','g_Documentary','g_Drama','g_Fantasy', 'g_Film-Noir', 'g_Horror','g_IMAX','g_Musical','g_Mystery','g_Romance', 'g_Sci-Fi', 'g_Thriller', 'g_War','g_Western'])
movies_feature = pd.concat([movies,feature_g],axis=1)
genres_features = movies_feature[['g_(no genres listed)', 'g_Action', 'g_Adventure', 'g_Animation', 'g_Children', 'g_Comedy', 'g_Crime', 'g_Documentary', 'g_Drama', 'g_Fantasy', 'g_Film-Noir', 'g_Horror', 'g_IMAX', 'g_Musical', 'g_Mystery', 'g_Romance', 'g_Sci-Fi', 'g_Thriller', 'g_War', 'g_Western']]

#計算歐基里德距離矩陣
euclidean_matrix = euclidean_distances(genres_features)

#將結果轉換為 DataFrame 方便查看
euclidean_df = pd.DataFrame(euclidean_matrix, index=movies_feature['movieId'], columns=movies_feature['movieId'])

#顯示歐基里德距離矩陣
#print(euclidean_df)

#親和傳播模型
prefer = np.mean(euclidean_matrix) #將
model = AffinityPropagation(affinity = "precomputed", preference = prefer)
model.fit(euclidean_matrix)
labels = model.labels_
#將 labels 存進 excel
labels_df = pd.DataFrame({'Labels': labels})
labels_df.to_excel('Labels.xlsx', index=False)

#劃出電影性質數量圖
# sns.countplot(num_genres[0])
# plt.show()

#計算 silhouette 分數
silhouette_avg = silhouette_score(euclidean_matrix, labels)
print("Silhouette分數為：",silhouette_avg)

#降維
scaler = StandardScaler()
one_hot_vectors_scaled = scaler.fit_transform(feature_g)
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(one_hot_vectors_scaled)
tsne_df = pd.DataFrame(data=tsne_result, columns=['Dimension 1', 'Dimension 2'])
tsne_df['Labels'] = labels

plt.figure(figsize=(12, 8))
sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Labels', data=tsne_df, palette='viridis', legend='full')
plt.title('t-SNE Scatter Plot with Clusters')
plt.show()