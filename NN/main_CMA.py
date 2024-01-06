import CMA
import numpy as np
import cma
import gym


np.random.seed(123)
env = gym.make('Acrobot-v1', render_mode='rgb_array')
ann = CMA.NeuralNetwork(env.observation_space.shape[0], env.action_space.n)
es = cma.CMAEvolutionStrategy(len(ann.get_params()) * [0], 0.2, {'popsize': 40})

for i in range(5):
    solutions = np.array(es.ask())
    fits = [CMA.fitness(x, ann, env) for x in solutions]
    es.tell(solutions, fits)
    es.disp()

    x = es.result[0]
    print(x)
    var = -CMA.fitness(x, ann, env, visul=False)
    print(var)



#np.shape(es.sm.C)