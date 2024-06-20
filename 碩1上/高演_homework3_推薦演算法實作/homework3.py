import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 讀取數據
ratings = pd.read_excel('ratings.xlsx')

# 創建用戶-物品評分矩陣 第一步驟
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# 建立物品相似性矩陣（使用餘弦相似度）第二步驟
item_similarity = cosine_similarity(user_item_matrix.T)

# 將物品相似性矩陣轉換為 DataFrame
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# 定義一個函數來生成用戶推薦
def get_user_based_recommendations(user_id, user_item_matrix, item_similarity_df, num_recommendations=5):
    # 獲取用戶評分的物品分數列表
    user_rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0]
    
    # 獲取用戶評分的物品列表
    user_rated_items = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index

    # 獲取用戶尚未評分的物品列表
    user_unrated_items = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] == 0].index


    sum = [] #創建空list用於存放分數
    for movie_id in user_unrated_items:
        #遍歷該用戶所有尚未評分的電影並計算該電影與已評分電影的相似度
        target_movie_similarity = item_similarity_df[movie_id][user_rated_items]
        
        #預估用戶對候選電影的興趣
        scores = target_movie_similarity.iloc[:] * user_rated.iloc[:]
        #將預估完的分數一一寫入list
        sum.append(scores.sum())

    #將預估完的分數放回尚未評分的電影表裡
    user_unrated_items = pd.Series(sum, index=user_unrated_items)

    #按照分數從高到低排序
    user_unrated_items = user_unrated_items.sort_values(ascending=False)
    
    #返回分數最高的前x個電影
    recommendations = user_unrated_items.head(num_recommendations).index
    return recommendations

user_id_to_recommend = 16  # 更換為你想要推薦的用戶的 userId
recommendations = get_user_based_recommendations(user_id_to_recommend, user_item_matrix, item_similarity_df)

print(f"推薦給用戶 {user_id_to_recommend} 的前 5 部相似電影：")
print(recommendations)