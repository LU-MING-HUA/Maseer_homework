import pandas as pd

stockdf = pd.read_csv("dataset.csv", index_col=0)
stockdf.dropna(how='any', inplace=True)

print(stockdf)

from sklearn import preprocessing
from keras.utils import to_categorical
min_max_scaler = preprocessing.MinMaxScaler()

newdf = stockdf.copy()
flagdf = stockdf.copy()

newdf['open'] = min_max_scaler.fit_transform(stockdf.open.values.reshape(-1,1))
newdf['low'] = min_max_scaler.fit_transform(stockdf.low.values.reshape(-1,1))
newdf['high'] = min_max_scaler.fit_transform(stockdf.high.values.reshape(-1,1))
newdf['close'] = min_max_scaler.fit_transform(stockdf.close.values.reshape(-1,1))
newdf['volume'] = min_max_scaler.fit_transform(stockdf.volume.values.reshape(-1,1))


print(stockdf)
print(newdf)