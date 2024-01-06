import CMA
import numpy as np
import cma
import gym
import MPA as MPA
import numpy as np
import time

np.random.seed(123)
env = gym.make('Acrobot-v1', render_mode='rgb_array')
ann = CMA.NeuralNetwork(env.observation_space.shape[0], env.action_space.n)

def Get_fit(X):
    ann.set_params(X)
    return -CMA.evaluate(ann, env, visul=False)

##########
# 设置海洋捕食者参数
pop = 40  # 种群数量
MaxIter = 5  # 最大迭代次数
dim = len(ann.get_params()) # 维度
lb = np.array([-1]*dim )  # 下边界
ub = np.array([1]*dim)  # 上边界
fobj = Get_fit

GbestScore, GbestPositon, Curve = MPA.MPA(pop, dim, lb, ub, MaxIter, fobj)

print('Best result：', GbestScore)
print('Best X：', GbestPositon)
print('Curve：', -Curve)
