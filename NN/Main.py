import numpy as np
from matplotlib import pyplot as plt
import MPA as MPA
import numpy as np
import time
from OptimizationTestFunctions import Sphere, Ackley, AckleyTest, Rosenbrock, Fletcher, Griewank, Penalty2, Quartic, Rastrigin, SchwefelDouble, SchwefelMax, SchwefelAbs, SchwefelSin, Stairs, Abs, Michalewicz, Scheffer, Eggholder, Weierstrass
import random

#start_time = time.time()

def Test_Fun(X):
    dim = len(X)
    res = np.sum(100 * (X[1:dim] - (X[:dim-1] ** 2)) ** 2 + (X[:dim-1] - 1) ** 2)
    return res


def rastrigin(X:list, A=10):
    n = len(X)
    return A * n + np.sum(X**2 - A * np.cos(2 * np.pi * X))


def himmelblau(x):
    n = len(x)
    result = 0
    for i in range(0, n, 2):
        result += (x[i]**2 + x[i+1] - 11)**2 + (x[i] + x[i+1]**2 - 7)**2
    return result


def rosenbrock(x):
    n = len(x)
    result = 0
    for i in range(n-1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return result

def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    return X, lb, ub

# 设置海洋捕食者参数
pop = 300  # 种群数量
MaxIter = 100  # 最大迭代次数
dim = 2 # 维度
lb = np.array([-6]*dim )  # 下边界
#lb = -30
ub = np.array([6]*dim)  # 上边界
#ub = 30
fobj = himmelblau
GbestScore, GbestPositon, Curve = MPA.MPA(pop, dim, lb, ub, MaxIter, fobj)

#end_time = time.time()
#elapsed_time = end_time - start_time

#print(f"Running Time：{elapsed_time:.4f} Seconds")
print('Best result：', GbestScore)
print('Best X：', GbestPositon)
print('Curve：', Curve)