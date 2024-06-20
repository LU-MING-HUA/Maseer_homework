import numpy as np
import scipy.io as scio
from sklearn import neighbors 

def Read():
    offline_data = scio.loadmat("offline_data_random.mat")
    online_data = scio.loadmat("online_data.mat")
    Y_train,X_train = offline_data["offline_location"],offline_data["offline_rss"]
    Y_test,X_test = online_data["trace"],online_data["rss"]
    #print("X_train:",X_train)
    #print("Y_train:",Y_train)
    #print("X_test:",X_test)
    #print("Y_test:",Y_test)
    
    return Y_train, X_train,Y_test,X_test

def KNN(Y_train, X_train,Y_test,X_test):
    labels = np.round(Y_train[:,0]/100)*100 + np.round(Y_train[:,1]/100)
    print(Y_train)
    print(labels)
    knn  = neighbors.KNeighborsClassifier(n_neighbors = 10 )
    knn.fit(X_train,labels)
    print(knn)
    return knn

def Predict(knn,Y_train, X_train,Y_test,X_test):
    p = knn.predict(X_test)
    print("Predict:", p)
    print("Y_test:", np.round(Y_test[:,0]/100)*100 + np.round(Y_test[:,1]/100))
    errors = location_error(np.array(p), np.array(Y_test))
    print("Errors:", errors)
    mse = compute_mse(errors)
    print("MSE:", mse)
    rmse = compute_rmse(errors)
    print("RMSE:", rmse)

def compute_mse(errors):
    return np.mean(errors**2)
def compute_rmse(errors):
    mse = np.mean(errors**2)
    return np.sqrt(mse)

def location_error(predicted, actual):
    actual_encoded = np.round(actual[:,0]/100)*100 + np.round(actual[:,1]/100)
    
    pred_x, pred_y = np.divmod(predicted, 100)
    actual_x, actual_y = np.divmod(actual_encoded, 100)
    
    diff_x = pred_x - actual_x
    diff_y = pred_y - actual_y

    errors = np.sqrt(diff_x**2 + diff_y**2)
    
    return errors

    

if __name__ == '__main__':
    Y_train, X_train,Y_test,X_test = Read()
    knn= KNN(Y_train, X_train,Y_test,X_test)
    Predict(knn,Y_train, X_train,Y_test,X_test)
