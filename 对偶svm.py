#python3
#wsq

from sklearn import svm

X=[[1,1],[2,2],[2,0],[0,0],[1,0],[0,1]]
Y=[1,1,1,0,0,0]
clf = svm.SVC()
clf.fit(X,Y)
'''SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)'''
