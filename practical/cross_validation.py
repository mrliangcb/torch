#cross validation
from sklearn.cross_validation import cross_val_score
knn=KNeighborsClassifer(n_neighbors=5)

#参数:knn=[0,1,2,3]一个batch的预测结果
#X=
#Y=[0,1,2,3]标准答案

scores=cross_val_score(knn,X,y,cv=5,scoring='accuracy')#分成5组，这是分类
#回归的时候，用scoring='mean_squared_error'
print(scores)