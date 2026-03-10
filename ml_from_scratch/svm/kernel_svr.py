import numpy as np
from ml_from_scratch.base import Regressor
from utils.helper import Kernel

class KernelSVR(Regressor):
    def __init__(self, epoch=1000, lr=0.001, C=100, epsilon=0.1, 
                 Kernel=Kernel.LINEAR, **kwargs):
        
        self.epoch = epoch
        self.lr = lr
        self.C = C
        self.epsilon = epsilon
        self.Kernel = Kernel
        self.kwargs = kwargs

    def fit(self, X, y):
        self.X_train = X
        y = y.reshape(-1,1)
        self.y_train = y
        
        n = X.shape[0]
        
        # Two alpha vectors
        self.alpha = np.zeros((n,1))
        self.alpha_star = np.zeros((n,1))
        
        # Kernel matrix
        K = self.Kernel(X, X, **self.kwargs)
        
        for _ in range(self.epoch):
            
            beta = self.alpha - self.alpha_star   # (n,1)
            f = K @ beta                          # (n,1)
            
            # Gradients from SVR dual
            grad_alpha = y - f - self.epsilon
            grad_alpha_star = -y + f - self.epsilon
            
            # Gradient ascent
            self.alpha += self.lr * grad_alpha
            self.alpha_star += self.lr * grad_alpha_star
            
            # Box constraints
            self.alpha = np.clip(self.alpha, 0, self.C)
            self.alpha_star = np.clip(self.alpha_star, 0, self.C)
            
            # Enforce equality constraint: sum(alpha - alpha*) = 0
            beta = self.alpha - self.alpha_star
            correction = np.sum(beta) / n
            
            self.alpha -= correction / 2
            self.alpha_star += correction / 2
            
            # Clip again
            self.alpha = np.clip(self.alpha, 0, self.C)
            self.alpha_star = np.clip(self.alpha_star, 0, self.C)
        
        # Final beta
        self.beta = self.alpha - self.alpha_star
        
        # -------- Compute bias b --------
        
        f_train = K @ self.beta
        
        idx1 = np.where((self.alpha > 1e-5) & 
                        (self.alpha < self.C-1e-5))[0]
        
        idx2 = np.where((self.alpha_star > 1e-5) & 
                        (self.alpha_star < self.C-1e-5))[0]
        
        b_vals = []
        
        for i in idx1:
            b_vals.append(y[i] - self.epsilon - f_train[i])
        
        for i in idx2:
            b_vals.append(y[i] + self.epsilon - f_train[i])
        
        self.b = np.mean(b_vals) if b_vals else 0

    def predict(self, X):
        K = self.Kernel(self.X_train, X, **self.kwargs)
        return (K.T @ self.beta + self.b)
