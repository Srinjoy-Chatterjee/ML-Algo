import numpy as np
from regression import (LinearRegression,
                        RidgeRegression,
                        LassoRegression,
                        ElasticNetRegression,
                        PolinomialRegression
                        )
from score import Score
from sklearn.model_selection import train_test_split,cross_val_score,KFold

lr = LinearRegression()
rr = RidgeRegression()
lr = LassoRegression()
enr = ElasticNetRegression(epochs=1000,l1_lambda=0.1,l2_lambda=0.1)
pr = PolinomialRegression(degree=2)

np.random.seed(0)

# 100 samples, 3 features
X = np.random.rand(100, 3)
#X = np.random.rand(100, 1)
# True weights
true_theta = np.array([2, -1, 3])
#true_theta = np.array([2])
# Generate target
y = X @ true_theta + 4 + np.random.randn(100) * 0.5
X_train,X_test,y_train,y_test = train_test_split(X,y)

# print(X_train)
pr.fit(X_train,y_train)
# y_pred = pr.predict(X_test)

# y_pred = y_pred.reshape(1,-1)

# print(Error.rsquare(y_test,y_pred))