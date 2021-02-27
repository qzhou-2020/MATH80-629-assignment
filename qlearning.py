from collections import defaultdict	# hashing with __missing__()
import numpy as np
import gym


class DiscretizedObservationWrapper(gym.ObservationWrapper):

    """wrapper that converts a Box observation into a single integer
    
    source:
    https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html#naive-q-learning
    """

    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        
        # get range of the continuous observations
        low = self.observation_space.low if low is None else np.asarray(low)
        high = self.observation_space.high if high is None else np.asarray(high)
        
        self.n_bins = n_bins
        self.val_bins = [
            np.linspace(l, h, n_bins + 1) for l, h in zip(
                low.flatten(), high.flatten()
            )
        ]
        self.observation_space = gym.spaces.Discrete(
            n_bins ** low.flatten().shape[0]
        )
        
    def _convert_to_one_number(self, digits):
        return sum([
            d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)
        ])
    
    def observation(self, observation):
        digits = [
            np.digitize([x], bins)[0] for x, bins in zip(
                observation.flatten(), self.val_bins
            )
        ]
        return self._convert_to_one_number(digits)


class BaseAgent:

    """base class for agents"""

    def __init__(self, actions, lr, gma):
        self.actions = actions
        self.lr = lr
        self.gma = gma
        self.Q = defaultdict(float)
        self.rewards = []   # type: float

    def learn(self, obv, r, act, obv_, done):
        raise NotImplementedError

    def choose_action(self, obv):
        raise NotImplementedError

    def fit(self, env, nb_episodes=1000):
        """train the agent"""
        for _ in range(nb_episodes):
            obv = env.reset()
            reward = 0.0
            while True:
                act = self.choose_action(obv)
                obv_, r, done, info = env.step(act)
                self.learn(obv, r, act, obv_, done)
                reward += r
                if done:
                    self.rewards.append(reward)
                    reward = 0.0
                    obv = env.reset()
                    break
                else:
                    obv = obv_

    def evaluate(self, env, nb_episodes=100):
        """test the performance"""
        rewards = np.zeros((nb_episodes,))
        for i in range(nb_episodes):
            reward = 0
            obv = env.reset()
            while True:
                act = self.choose_action(obv)
                obv_, r, done, _ = env.step(act)
                reward += r
                if done:
                    rewards[i] = reward
                    break
                obv = obv_
        return rewards


class EpsilonQlearningAgent(BaseAgent):
    
    """constant epsilon-greedy Q-learning agent"""

    def __init__(self, actions, lr=0.01, gma=0.99, epsilon=0.1):
        super(EpsilonQlearningAgent, self).__init__(actions, lr, gma)
        self.epsilon = epsilon

    def learn(self, obv, r, act, obv_, done):
        max_q_ = max([self.Q[obv_, act_] for act_ in self.actions])
        self.Q[obv, act] += self.lr * (
            r + self.gma * max_q_ * (1 - done) - self.Q[obv, act]
        )

    def choose_action(self, obv):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        return self._best_action(obv)

    def _best_action(self, obv):
        """off-policy action"""
        q_vals = {act: self.Q[obv, act] for act in self.actions}
        q_max = max(q_vals.values())
        # handle tie actions
        act_q_max = [act for act, q in q_vals.items() if q == q_max]
        return np.random.choice(act_q_max)


class AnnealEpsilonQlearningAgent(BaseAgent):

    """annealing epsilon greedy Q-learning agent.
    
        reduce epsilon from start value to some small number based 
        on the times a state is visited.
    """

    def __init__(self, actions, lr=0.01, gma=0.99, epsilon=1.0):
        super(AnnealEpsilonQlearningAgent, self).__init__(actions, lr, gma)
        self.epsilon = epsilon
        self.visits = defaultdict(int)

    def learn(self, obv, r, act, obv_, done):
        max_q_ = max([self.Q[obv_, act_] for act_ in self.actions])
        self.Q[obv, act] += self.lr * (
            r + self.gma * max_q_ * (1 - done) - self.Q[obv, act]
        )
        # remember the number of visits
        self.visits[obv] += 1

    def choose_action(self, obv):
        eps = self.epsilon / (self.visits[obv] + 1)
        if np.random.random() < eps:
            return np.random.choice(self.actions)
        return self._best_action(obv)

    def _best_action(self, obv):
        """off-policy action"""
        q_vals = {act: self.Q[obv, act] for act in self.actions}
        q_max = max(q_vals.values())
        # handle tie actions
        act_q_max = [act for act, q in q_vals.items() if q == q_max]
        return np.random.choice(act_q_max)


class RandomQlearningAgent(BaseAgent):

    """random policy Q-learning agent"""

    def __init__(self, actions, lr=0.01, gma=0.99):
        super(RandomQlearningAgent, self).__init__(actions, lr, gma)

    def learn(self, obv, r, act, obv_, done):
        # no need to learn
        pass

    def choose_action(self, obv):
        return np.random.choice(self.actions)


class BoltzmannQlearningAgent(BaseAgent):

    """Boltzmann policy Q-learning agent"""

    def __init__(self, actions, lr=0.01, gma=0.99, temperature=1.0):
        super(BoltzmannQlearningAgent, self).__init__(actions, lr, gma)
        self.tau = temperature
        self.epss = defaultdict(lambda: 1.0)

    def learn(self, obv, r, act, obv_, done):
        max_q_ = max([self.Q[obv_, act_] for act_ in self.actions])
        temp_diff = r + self.gma * max_q_ * (1 - done) - self.Q[obv, act]
        self.Q[obv, act] += self.lr * temp_diff
        # update eps based on VDBE_boltzmann
        self._vdbe_boltzmann(obv, temp_diff)

    def choose_action(self, obv):
        if np.random.random() < self.epss[obv]:
            return np.random.choice(self.actions)
        return self._best_action(obv)

    def _best_action(self, obv):
        """off-policy action"""
        q_vals = {act: self.Q[obv, act] for act in self.actions}
        q_max = max(q_vals.values())
        # handle tie actions
        act_q_max = [act for act, q in q_vals.items() if q == q_max]
        return np.random.choice(act_q_max)

    def _vdbe_boltzmann(self, obv, td, sigma=1.0):
        td = np.fabs(td)
        delta = 1.0 / len(self.actions)
        f = (1. - np.exp(-td / sigma)) / (1. + np.exp(-td / sigma))
        eps_ = max(delta * f + (1. - delta) * self.epss[obv], 0)
        self.epss[obv] = eps_


class BayesianQlearningAgent(BaseAgent):

    """Bayesian exploration Q-learning agent"""

    def __init__(self, actions, lr=0.01, gma=0.99, epsilon=1.0):
        super(EpsilonQlearningAgent, self).__init__(actions, lr, gma)
        self.epsilon = epsilon
        self.visits = defaultdict(int)

    def learn(self, obv, r, act, obv_, done):
        max_q_ = max([self.Q[obv_, act_] for act_ in self.actions])
        self.Q[obv, act] += self.lr * (
            r + self.gma * max_q_ * (1 - done) - self.Q[obv, act]
        )
        # remember the number of visits
        self.visits[obv] += 1

    def choose_action(self, obv):
        eps = self.epsilon / (self.visits[obv] + 1)
        if np.random.random() < eps:
            return np.random.choice(self.actions)
        return self._best_action(obv)

    def _best_action(self, obv):
        """off-policy action"""
        q_vals = {act: self.Q[obv, act] for act in self.actions}
        q_max = max(q_vals.values())
        # handle tie actions
        act_q_max = [act for act, q in q_vals.items() if q == q_max]
        return np.random.choice(act_q_max)


class EpsilonSARSAAgent(BaseAgent):

    def __init__(self, actions, lr=0.01, gma=0.99, epsilon=0.1):
        super(EpsilonSARSAAgent, self).__init__(actions, lr, gma)
        self.epsilon = epsilon

    def learn(self, obv, r, act, obv_, done):
        # call choose_action function for SARSA,
        # since on-policy algorithm follows epsilon greedy
        act_ = self.choose_action(obv_)
        q_ = self.Q[obv_, act_]
        self.Q[obv, act] += self.lr * (
            r + self.gma * q_ * (1 - done) - self.Q[obv, act]
        )

    def choose_action(self, obv):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        return self._best_action(obv)

    def _best_action(self, obv):
        """off-policy action"""
        q_vals = {act: self.Q[obv, act] for act in self.actions}
        q_max = max(q_vals.values())
        # handle tie actions
        act_q_max = [act for act, q in q_vals.items() if q == q_max]
        return np.random.choice(act_q_max)


class BoltzmannSARSAAgent(BaseAgent):

    def __init__(self, actions, lr=0.01, gma=0.99):
        super(BoltzmannSARSAAgent, self).__init__(actions, lr, gma)
        self.epss = defaultdict(lambda: 1.0)

    def learn(self, obv, r, act, obv_, done):
        # call choose_action function for SARSA,
        # since on-policy algorithm follows epsilon greedy
        act_ = self.choose_action(obv_)
        q_ = self.Q[obv_, act_]
        temp_diff = r + self.gma * q_ * (1 - done) - self.Q[obv, act]
        self.Q[obv, act] += self.lr * temp_diff
        # update eps based on VDBE_boltzmann
        self._vdbe_boltzmann(obv, temp_diff)

    def choose_action(self, obv):
        if np.random.random() < self.epss[obv]:
            return np.random.choice(self.actions)
        return self._best_action(obv)

    def _best_action(self, obv):
        """off-policy action"""
        q_vals = {act: self.Q[obv, act] for act in self.actions}
        q_max = max(q_vals.values())
        # handle tie actions
        act_q_max = [act for act, q in q_vals.items() if q == q_max]
        return np.random.choice(act_q_max)

    def _vdbe_boltzmann(self, obv, td, sigma=1.0):
        td = np.fabs(td)
        delta = 1.0 / len(self.actions)
        f = (1. - np.exp(-td / sigma)) / (1. + np.exp(-td / sigma))
        eps_ = max(delta * f + (1. - delta) * self.epss[obv], 0)
        self.epss[obv] = eps_