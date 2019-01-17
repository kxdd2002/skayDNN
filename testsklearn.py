from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 

def initData():
	iris = datasets.load_iris()
	iris_X = iris.data
	iris_y = iris.target
	print('-----data input example like this -----')
	print(iris_X[:3, :])
	print('------lables like this -------')
	print(iris_y)
	X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
	return X_train, X_test, y_train, y_test

def knn(X_train, X_test, y_train, y_test):
	knn = KNeighborsClassifier()
	knn.fit(X_train, y_train)
	y_predict = knn.predict(X_test)
	print('-----predict value is ------')
	print(y_predict)
	print('-----actual value is -------')
	print(y_test)
	count = 0
	for i in range(len(y_predict)):
		if y_predict[i] == y_test[i]:
			count += 1
	print('accuracy is %0.2f%%'%(100*count/len(y_predict)))


# ---------------------------------------------------------  main funciton  ------------------------------------------------------

if __name__ == '__main__':
	X_train, X_test, y_train, y_test = initData()
	knn(X_train, X_test, y_train, y_test)
