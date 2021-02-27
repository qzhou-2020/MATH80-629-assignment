"""REINFORCE algorithm, a.k.a, the plain policy gradient algorithm"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp


class ReinforceAgent:

	"""vanilla policy gradient agent"""

	def __init__(self, obv_dim, act_dim, lr=0.001, gma=0.99):
		self.action_dim = act_dim
		self.obv_dim = obv_dim
		self.lr = lr
		self.gma = gma
		self.optimizer = tf.keras.optimizers.Adam(lr)
		self.model = self._build_model() # policy
		self.rewards = []	# total rewards for each episodes in training

	def _build_model(self):
		"""create a nn for policy"""

		# two hidden layers with ReLU activation
		# the output layer doesn't have activation function since
		# I'm using sparse_categorical_crossentropy() to calculate
		# the negative likelihood.

		model = tf.keras.models.Sequential([
			tf.keras.layers.Dense(128, activation="relu", input_shape = (self.obv_dim,)),
			tf.keras.layers.Dense(128, activation="relu"),
			tf.keras.layers.Dense(self.action_dim, activation=None)
		])
		return model

	def _discount_and_normalize(self, rewards):
		G = []
		Gt = 0
		for r in rewards:
			Gt = Gt * self.gma + r
			G.append(Gt)
		G = np.array(G[::-1]).astype(np.float32)
		G = (G - np.mean(G)) / np.std(G)
		return G
	
	def sample_trajectory(self, env) -> tuple:
		"""generate one trajectory"""
		obv = env.reset()
		states, actions, rewards = [], [], []
		done = False
		while not done:
			policy_params = self.policy_params(obv[np.newaxis,:])
			action = self.sample_action(policy_params)
			obv_, reward, done, _ = env.step(action)
			# save
			states.append(obv)
			actions.append(action)
			rewards.append(reward)
			# update obv
			obv = obv_
		return (states, actions, rewards)

	def sample_action(self, probs):
		sampled_action = tf.random.categorical(probs, 1)
		return sampled_action[0][0].numpy()

	def policy_params(self, obv):
		"""return the raw logits"""
		return self.model(obv)

	def log_prob(self, policy_params, actions):
		"""return negative likelihood"""
		return tf.keras.losses.sparse_categorical_crossentropy(
			y_true=actions, y_pred=policy_params, from_logits=True 
		)

	def learn(self, states, actions, rewards):
		"""update policy nn parameters using gradient ascending"""
		rs = self._discount_and_normalize(rewards)
		# automatic differentiation
		# https://www.tensorflow.org/api_docs/python/tf/GradientTape
		with tf.GradientTape() as tape:
			loss = 0
			for s, a, r in zip(states, actions, rs):
				logits = self.policy_params(s[np.newaxis,:])
				nll = self.log_prob(logits, a)
				loss += tf.squeeze(nll) * r
		train_vars = self.model.trainable_variables
		grads = tape.gradient(loss, train_vars)
		self.optimizer.apply_gradients(zip(grads, train_vars))

	def fit(self, env, nb_episodes=1000, log_interval=100):
		"""train the model"""
		total_rewards = []
		for t in range(nb_episodes):
			s, a, r = self.sample_trajectory(env)
			self.learn(s, a, r)
			self.rewards.append(np.sum(r))
			if t % log_interval == 0:
				avgr = np.mean(self.rewards[-10:])
				print(f"episode {t}, rewards {avgr}")

	def evaluate(self, env, nb_episodes=100):
		rewards = []
		for _ in range(nb_episodes):
			s, a, r = self.sample_trajectory(env)
			rewards.append(np.sum(r))
		return rewards

					
def test():
	import gym
	import tensorflow as tf

	env = gym.make("CartPole-v1")
	agent = ReinforceAgent(
		env.observation_space.shape[0], 
		env.action_space.n, 
		lr=0.001
	)
	agent.fit(env, nb_episodes=500, log_interval=10)


if __name__ == "__main__":
	test()
