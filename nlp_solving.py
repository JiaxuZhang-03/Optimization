import numpy as np

class Solver:
    def __init__(self,f,grad_f,start_point,alpha = 0.001,eps = 1e-6,max_iter = 100000):
        self.f = f
        self.grad_f = grad_f
        self.x0 = start_point
        self.alpha = alpha
        self.eps = eps
        self.max_iter = max_iter
    def gradient_descent(self):
        x = self.x0.copy()
        for k in range(self.max_iter):
            grad = self.grad_f(x)
            if np.linalg.norm(grad) < self.eps:
                break
            x = x - self.alpha * grad
        return x,self.f(x),k
    def approximate_hessian(self,x,h = 1e-5):
        n = len(x)
        H = np.zeros((n,n))
        grad0 = self.grad_f(x)
        for j in range(n):
            x_step = x.copy()
            x_step[j] += h
            grad_step = self.grad_f(x_step)
            H[:,j] = (np.array(grad_step).flatten() - np.array(grad0).flatten())/h
        return H
    def newton_method(self):
        x = self.x0.copy()

        for k in range(self.max_iter):
            grad = self.grad_f(x)
            if np.linalg.norm(grad) < self.eps:
                break
            H = self.approximate_hessian(x)
            x = x - np.linalg.inv(H) @ grad
        return x,self.f(x),k
    

if __name__ == "__main__":
    def f(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def grad_f(x):
        return np.array([
            -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
            200*(x[1] - x[0]**2)
        ])

    solver = Solver(f,grad_f,start_point=[0,0])
    print("GD:",solver.gradient_descent())
    print("NM:",solver.newton_method())    


