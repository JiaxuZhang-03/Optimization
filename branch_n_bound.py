import numpy as np
from scipy.optimize import linprog
import math
from copy import deepcopy

# c = np.array([-3.-2])
# A = np.array([[1,1]])
# b = np.array([4])
class ILP:
    class Node:
        def __init__(self,bounds,level = 0):
            self.bounds = bounds
            self.level = level
            self.value = None
            self.solution = None
    def __init__(self,c,A,b):
        self.c = -np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)
        self.best_value = -np.inf
        self.best_solution = None

    def solve_LP(self,bounds):
        res = linprog(self.c,A_ub = self.A,b_ub = self.b,bounds = bounds,method = 'highs')
        if not res.success:
            return None,None
        return res.x, -res.fun
    
    def is_integer_solution(self,x,tol = 1e-6):
        return all(abs(xi-round(xi))<tol for xi in x)
    
    def branch_and_bound(self,node):
        x,value = self.solve_LP(node.bounds)
        if x is None:
            return
        if value <= self.best_value:
            return
        if self.is_integer_solution(x):
            if value > self.best_value:
                self.best_value = value

                self.best_solution = np.round(x)
            return
        
        for i,xi in enumerate(x):
            if abs(xi-round(xi)) > 1e-6:
                break
        left_bounds = deepcopy(node.bounds)
        left_bounds[i] = (left_bounds[i][0],math.floor(xi))
        right_bounds = deepcopy(node.bounds)
        right_bounds[i] = (math.ceil(xi),right_bounds[i][1])
        
        # Recursion
        self.branch_and_bound(ILP.Node(left_bounds,node.level + 1))
        self.branch_and_bound(ILP.Node(right_bounds,node.level + 1))
    def solve(self,init_bounds):
        root = ILP.Node(bounds = init_bounds,level = 0)
        self.branch_and_bound(root)
        return self.best_solution,self.best_value
    

if __name__ == "__main__":
    c = [3,2]
    A = [[1,1]]
    b = [4]

    model = ILP(c,A,b)
    bounds = [(0,None),(0,None)]
    sol,val = model.solve(bounds)
    print("Optimal Integer Solution:",sol)
    print("Optimal Objective Value:",val)

