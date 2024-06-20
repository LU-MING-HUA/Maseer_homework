import twstock
stock = twstock.Stock("2331")

import pandas as pd
x = {
    "open":stock.open,
    "high":stock.high,
    "low":stock.low,
    "close":stock.close,
    "capacity":stock.capacity,
}

y = pd.DataFrame(x)
y.to_csv("s2331.csv")
print(y)


