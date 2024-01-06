import numpy as np
import random
import math
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''' 种群初始化函数 '''


def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    return X, lb, ub


'''边界检查函数'''


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''计算适应度函数'''


def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


'''适应度排序'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


''' Levy飞行'''


def Levy(d):
    beta = 3 / 2
    sigma = (math.gamma(1 + beta) * np.sin(math.pi * beta / 2)) / (
                math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)) ** (1 / beta)
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = u / np.abs(v) ** (1 / beta)
    L = 0.05 * step
    return L


'''海洋捕食者算法'''


def MPA(pop, dim, lb, ub, MaxIter, fun):
    P = 0.5
    FADS = 0.2
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    #fitness = fun
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([MaxIter, 1])
    Xnew = copy.copy(X)
    for t in range(MaxIter):
        print("          ")
        print("Iteration：" + str(t) + " Times")
        RB = np.random.randn(pop, dim)
        L = Levy(dim)
        CF = (1 - t / MaxIter) ** (2 * t / MaxIter)
        for i in range(pop):
            for j in range(dim):
                if t < MaxIter / 3:  # 前期搜索
                    stepSize = RB[i, j] * (GbestPositon[0, j] - RB[i, j] * X[i, j])
                    Xnew[i, j] = X[i, j] + P * np.random.random() * stepSize
                if t >= MaxIter / 3 and t <= 2 * MaxIter / 3:  # 中期搜索
                    if i < pop / 2:
                        stepSize = L[0, j] * (GbestPositon[0, j] - L[0, j] * X[i, j])
                        Xnew[i, j] = X[i, j] + P * np.random.random() * stepSize
                    else:
                        stepSize = RB[i, j] * (RB[i, j] * GbestPositon[0, j] - X[i, j])
                        Xnew[i, j] = X[i, j] + P * CF * stepSize
                if t > 2 * MaxIter / 3:  # 后期搜索
                    stepSize = L[0, j] * (L[0, j] * GbestPositon[0, j] - X[i, j])
                    Xnew[i, j] = X[i, j] + P * CF * stepSize

        Xnew = BorderCheck(Xnew, ub, lb, pop, dim)
        fitnessNew = CaculateFitness(Xnew, fun)  # 计算适应度值
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:
                fitness[i] = copy.copy(fitnessNew[i])
                X[i, :] = copy.copy(Xnew[i, :])

        indexBest = np.argmin(fitness)
        if (fitness[indexBest] < GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[indexBest])
            GbestPositon[0, :] = copy.copy(X[indexBest, :])
        #print("X best for now B4: ", X[indexBest, :])
        #print("Fitness best for now B4: ", GbestScore)

        # 涡流效应
        for i in range(pop):
            for j in range(dim):
                if np.random.random() < FADS:
                    U = np.random.random() < FADS
                    Xnew[i, j] = X[i, j] * CF * (lb[j] + np.random.random() * (ub[j] - lb[j]) * U)
                else:
                    r = np.random.random()
                    stepsize = (FADS * (1 - r) + r) * (X[np.random.randint(pop), j] - X[np.random.randint(pop), j])
                    Xnew[i, j] = X[i, j] + stepsize
        Xnew = BorderCheck(Xnew, ub, lb, pop, dim)
        fitnessNew = CaculateFitness(Xnew, fun)  # 计算适应度值
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:
                fitness[i] = fitnessNew[i].item()
                X[i, :] = copy.copy(Xnew[i, :])

        indexBest = np.argmin(fitness)
        if (fitness[indexBest] < GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[indexBest])
            GbestPositon[0, :] = copy.copy(X[indexBest, :])

        #print("X best for now: ", X[indexBest, :])
        #print("Fitness best for now: ", GbestScore)

        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve









