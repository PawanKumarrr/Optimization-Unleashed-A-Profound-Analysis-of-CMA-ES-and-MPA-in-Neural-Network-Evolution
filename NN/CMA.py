import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import gym
from pyvirtualdisplay import Display
import IPython
from IPython import display
import matplotlib.pyplot as plt
import cma

class NeuralNetwork(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_shape, 32)
        self.l2 = nn.Linear(32, 32)
        self.lout = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x.float()))
        x = F.relu(self.l2(x))
        return self.lout(x)

    def get_params(self):
        p = np.empty((0,))
        for n in self.parameters():
            p = np.append(p, n.flatten().cpu().detach().numpy())
        return p

    def set_params(self, x):
        start = 0
        for p in self.parameters():
            e = start + np.prod(p.shape)
            p.data = torch.FloatTensor(x[start:e]).reshape(p.shape)
            start = e

def evaluate(ann, env, visul=True):
    obs, info = env.reset(seed=0)
    if visul:
        img = plt.imshow(env.render())
        plt.show()
    total_reward = 0
    while True:
        # Output of the neural net
        net_output = ann(torch.tensor(obs))
        # the action is the value clipped returned by the nn
        action = net_output.data.cpu().numpy().argmax()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if visul:
            img.set_data(env.render())
            plt.axis('off')
            display.display(plt.gcf())
            display.clear_output(wait=True)
        if done or truncated:
            break
    return total_reward

def fitness(x, ann, env, visul=False):
    ann.set_params(x)
    return -evaluate(ann, env, visul=visul)

def mu_lambda(x, fitness, gens=200, lam=10, alpha=0.2, verbose=False):
    x_best = x
    f_best = fitness(x)
    fits = np.zeros(gens)
    for g in range(gens):
        N = np.random.normal(size=(lam, len(x)))
        F = np.zeros(lam)
        for i in range(lam):
            ind = x + N[i, :]
            F[i] = fitness(ind)
            if F[i] < f_best:
                f_best = F[i]
                x_best = ind
                if verbose:
                   print(g, " ", f_best)
        fits[g] = f_best
        mu_f = np.mean(F)
        std_f = np.std(F)
        A = F
        if std_f != 0:
            A = (F - mu_f) / std_f
        x = x - alpha * np.dot(A, N) / lam
    return fits, x_best


# np.random.seed(654)
# env = gym.make('Acrobot-v1', render_mode='rgb_array')
# ann = NeuralNetwork(env.observation_space.shape[0], env.action_space.n)
# x = np.random.randn(len(ann.get_params()))
# f = lambda x : fitness(x, ann, env)
# fits, x = mu_lambda(x, f, gens=100, lam=10, alpha=0.1, verbose=True)
# plt.plot(fits)
# -fitness(x, ann, env, visul=True)

# np.random.seed(123)
# env = gym.make('Acrobot-v1', render_mode='rgb_array')
# ann = NeuralNetwork(env.observation_space.shape[0], env.action_space.n)
# es = cma.CMAEvolutionStrategy(len(ann.get_params()) * [0], 0.1, {'popsize': 40})
#
# for i in range(50):
#     solutions = np.array(es.ask())
#     fits = [fitness(x, ann, env) for x in solutions]
#     es.tell(solutions, fits)
#     es.disp()
#
# x = es.result[0]
# -fitness(x, ann, env, visul=True)
#
# np.shape(es.sm.C)