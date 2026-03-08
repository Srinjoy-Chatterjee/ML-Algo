import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(0)

# 100 samples, 3 features
X = np.random.rand(100, 3)
# True weights
true_theta = np.array([2, -1, 3])
# Generate target
y = X @ true_theta + 4 + np.random.randn(100) * 0.5
X_train,X_test,y_train,y_test = train_test_split(X,y)


#Polinomial
from regression import PolinomialRegression
from score import Score
max_degree = 15
mses = []
for degree in range(1,max_degree+1):
    pr = PolinomialRegression(degree=degree)
    pr.fit(X_train,y_train)
    y_pred = pr.predict(X_test)
    mses.append(Score.mean_squared_error(y_test,y_pred))

mini = min(mses)
print("minimum MSE: ",mini)
print("Degree ",mses.index(mini)+1)
plt.plot(range(1,max_degree+1),mses)
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.title("Bias-Variance Tradeoff")
plt.show()