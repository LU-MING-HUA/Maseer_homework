from sklearn import datasets
iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target

from sklearn.model_selection import train_test_split
(train_data, test_data, train_lable, test_lable) = train_test_split(iris_data, iris_label, test_size=0.2)

from sklearn import linear_model
lr = linear_model.LogisticRegression()
lr.fit(train_data, train_lable)

predicted = lr.predict(test_data)
print(predicted)
print(test_lable)

from sklearn import metrics
print(metrics.accuracy_score(test_lable, predicted))