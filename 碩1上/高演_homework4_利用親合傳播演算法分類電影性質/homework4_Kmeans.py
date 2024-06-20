import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 讀取movies的CSV檔案
movies = pd.read_csv("movies.csv", encoding='ISO-8859-1')

# 分割電影性質
genres = movies['genres'].str.split("|", expand=True)
num_genres = pd.DataFrame(np.array(genres).reshape(192130, 1))
# print(pd.Series(num_genres[0]).value_counts())  # 列出各個性質共有幾部片

# 分割維度轉向量 使用one-hot encoding
cleaned = movies.set_index('movieId').genres.str.split('|', expand=True).stack()
feature_g = pd.get_dummies(cleaned, prefix='g').groupby(level=0).sum()
feature_g = pd.DataFrame(feature_g.values, columns=['g_(no genres listed)','g_Action','g_Adventure','g_Animation','g_Children','g_Comedy', 'g_Crime','g_Documentary','g_Drama','g_Fantasy', 'g_Film-Noir', 'g_Horror','g_IMAX','g_Musical','g_Mystery','g_Romance', 'g_Sci-Fi', 'g_Thriller', 'g_War','g_Western'])
movies_feature = pd.concat([movies, feature_g], axis=1)
genres_features = movies_feature[['g_(no genres listed)', 'g_Action', 'g_Adventure', 'g_Animation', 'g_Children', 'g_Comedy', 'g_Crime', 'g_Documentary', 'g_Drama', 'g_Fantasy', 'g_Film-Noir', 'g_Horror', 'g_IMAX', 'g_Musical', 'g_Mystery', 'g_Romance', 'g_Sci-Fi', 'g_Thriller', 'g_War', 'g_Western']]

# 選擇 K 值
k_value = 19  # 可根據實際情況調整

# K-means 演算法
kmeans = KMeans(n_clusters=k_value, random_state=42)
labels = kmeans.fit_predict(genres_features)

# 將 labels 存進 Excel
labels_df = pd.DataFrame({'Labels': labels})
labels_df.to_excel('KMeans_Labels.xlsx', index=False)

# 計算 silhouette 分數
silhouette_avg = silhouette_score(genres_features, labels)
print("Silhouette分數為:", silhouette_avg)

# 降維
scaler = StandardScaler()
one_hot_vectors_scaled = scaler.fit_transform(feature_g)
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(one_hot_vectors_scaled)
tsne_df = pd.DataFrame(data=tsne_result, columns=['Dimension 1', 'Dimension 2'])
tsne_df['Labels'] = labels

# 繪製 t-SNE 散點圖
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Labels', data=tsne_df, palette='viridis', legend='full')
plt.title('t-SNE Scatter Plot with K-means Clusters')
plt.show()
