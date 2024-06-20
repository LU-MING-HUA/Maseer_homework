import pandas as pd

# 假設 movies 表格包含電影信息，其中包含唯一標識符 MovieID
movies = pd.read_csv('movies.csv', encoding='ISO-8859-1')  # 替換為實際的文件路徑

# 獲取電影的數量
num_movies = len(movies['movieId'].unique())

print(f"ml-25m 共有 {num_movies} 部電影。")




# 檢查 genres 列的不同值
unique_genres = movies['genres'].unique()

# 計算 genres 列的不同值的總數
num_unique_genres = len(unique_genres)

print(f"ml-25m 中的 movies 的 genres 有 {num_unique_genres} 種。")

# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.cluster import AffinityPropagation
# from sklearn.metrics import silhouette_score

# #讀取movies的CSV檔案
# movies = pd.read_csv("movies.csv", encoding='ISO-8859-1')

# #分割電影性質
# genres = movies['genres'].str.split("|",expand= True)
# num_genres = pd.DataFrame(np.array(genres).reshape(192130,1))
# print(pd.Series(num_genres[0]).value_counts()) #列出各個性質共有幾部片