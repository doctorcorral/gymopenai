import numpy as np

MAX_NUM_EPISODES = 100
STEPS_PER_EPISODE = 50
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30


class Q_learner(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.obs_high
        self.obs_low = env.observation_space.low
        self.obs_bin = NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n
        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1,
                           self.action_shape))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0

    def discretize(self, obs):
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        discretized_obs = self.discretize(obs)
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discretized_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * np.max(
            self.Q[discretized_next_obs]
        )
        td_error = td_target - self.Q[discretized_obs][action]
        self.Q[discretized_obs][action] += self.alpha * td_error

    def train(self,agent, env):
        best_reward = -float('inf')
        for episode in range(MAX_NUM_EPISODES):
            done = False
            obs = env.reset()
            total_reward = 0.0
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                agent.learn(obs, action, reward, next_obs)
                obs = next_obs
                total_reward += reward
            if total_reward > best_reward:
                best_reward = total_reward
            print("Episode # : {}  reward : {} best_reward : {} eps : {}".format(
                episode, total_reward, best_reward, agent.epsilon))
        return np.argmax(agent.Q, axis=2)

    