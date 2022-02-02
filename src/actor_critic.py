# Imports
import sys
import numpy as np

PARALLEL = 10

class ActorCritic():
  # ---------------------------
  # Begin Public Methods
  # ----------------------------
  
  def __init__(self):
    # a_w: learning rate for Value estimator (critic)
    self.alpha_w = 0.01
    # a_theta: Learning rate for the Policy estimator (actor)
    self.alpha_theta = 0.01

    # Number of discrete state segments. smaller is faster
    self.segments = 100

    # w: weights for the value estimator
    self.ws = [np.zeros(2 * self.segments)] * PARALLEL
    # theta: weights for the policy estimator
    self.thetas = [np.zeros(2 * self.segments * 3)] * PARALLEL

    # Keep track of average reward earned for each episode
    self.rewards = []

  # Actor estimates the probability that an action is good
  # Critic esimates the expected reward given current data
  def update(self, states_vector, actions_vector, rewards_vector, next_states_vector):
    total_reward = 0
    total_w = []
    total_theta = []

    # TODO: Parallelize both of these loops in CUDA
    for i in range(len(states_vector)):
      states = states_vector[i]
      actions = actions_vector[i]
      rewards = rewards_vector[i]
      next_states = next_states_vector[i]

      ep_rewards = 0
      w = self.ws[i]
      theta = self.thetas[i]

      for j in range(len(states)):
        curr_state = states[j]
        action = actions[j]
        reward = rewards[j]
        next_state = next_states[j]

        curr_state = (0,0)
        action = 0
        reward = 1
        next_state = (0,0)
        for i in range(len(w)):
          w[i] = 0

        # Discretize the states, makes things faster
        curr_discrete = self.segment_state(curr_state)
        next_discrete = self.segment_state(next_state)

        # Estimate the TD error (advantage)
        delta_t = reward + self.approximate_value(next_discrete, w) - self.approximate_value(curr_discrete, w)

        # update the weights
        w += self.alpha_w * delta_t * self.approximate_value_gradient(curr_discrete)
        theta += self.alpha_theta * delta_t * self.calc_policy_gradient(action, curr_discrete, theta)

        ep_rewards += reward
      
      total_w.append(w)
      total_theta.append(theta)

      total_reward += ep_rewards
    
    average_w = np.average(total_w, axis=0)
    average_theta = np.average(total_theta, axis=0)

    print(total_reward / PARALLEL)
    self.rewards.append(total_reward / PARALLEL)
    
    self.ws = [average_w] * PARALLEL
    self.thetas = [average_theta] * PARALLEL

  # Given a state, return an action
  def act(self, i_env, curr_state):
    discrete_state = self.segment_state(curr_state)

    # calculate an action probability distribution
    policy = self.calc_policy(discrete_state, self.thetas[i_env])
    return np.random.choice([0, 1, 2], p = policy)

  # -----------------------------------
  # Begin Private Methods
  # -----------------------------------

  # Discretize the state into a bunch of smaller segments. makes things faster
  def segment_state(self, curr_state):
    pos_segment = (0.6 + 1.2) / self.segments
    vel_segment = (0.7 + 0.7) / self.segments

    discrete_state = np.zeros(2 * self.segments)
    
    coarse_pos_index = int((curr_state[0] + 1.2) / pos_segment)
    coarse_vel_index = int((curr_state[1] + 0.7) / vel_segment)

    discrete_state[coarse_pos_index] = 1
    discrete_state[coarse_vel_index + self.segments] = 1

    return discrete_state

  # Calculate a vector of preferences and its gradient
  def calc_preference(self, discrete_state, theta):
    # Preferences should be 3 long because there are 3 possible actions
    preferences = np.zeros(3)
    preferences_grad = []

    for a in range(3):
      # For each action, construct an action vector 
      action_vector = np.zeros(3)
      action_vector[a] = 1
      s_a = np.zeros(len(theta))

      # Construct a state vector [s0, 0, 0, 0, s0, 0, 0, 0, s0, s1, 0, 0, ...]
      # Basically repeat a pattern of 9 elements where the first 3 elements contain
      # [0 sn 0], the next 3 elements are [0 sn 0], and the last 3 are [0, 0, sn]
      index = 0
      for s_i in discrete_state:
        for a_i in action_vector:
          s_a[index] = s_i * a_i
          index += 1

      # Then we can calculate the preferences by using this state vector to multiply
      # with the theta weights. This gives us the actor's model's preference for picking
      # action a given state s_a.
      preferences[a] = np.dot(s_a, theta)

      # The gradient of a variable times a constant is just the variable
      preferences_grad.append(s_a)

    return preferences, preferences_grad

  # Calculate a policy action distribution (actor)
  # The probability vector gives us the actors preference for picking an action given a state.
  # Convert the preferences to a probability vector with a softmax
  def calc_policy(self, discrete_state, theta):
    policy, _ = self.calc_preference(discrete_state, theta)

    # https://en.wikipedia.org/wiki/Softmax_function
    numerator = np.exp(policy - np.max(policy))
    denominator = np.sum(numerator, axis=-1)
    return numerator / denominator

  # Need to calculate the derivative of a softmax function
  def calc_policy_gradient(self, action, discrete_state, theta):
    # First get the derivative of the preferences
    _, x_s = self.calc_preference(discrete_state, theta)

    h_s_a_w = x_s * theta

    f = np.exp(h_s_a_w[action])
    f_prime = np.exp(h_s_a_w[action]) * x_s[action]
    g = np.sum(np.exp(h_s_a_w), axis = 0)
    g_prime = np.sum(np.exp(h_s_a_w) * x_s, axis = 0)

    numerator = f_prime * g - f * g_prime
    denominator = g ** 2

    return numerator / denominator

  # Calculate the approximate value (critic) with a simple linear model
  # Just multiply the given state by a matrix
  def approximate_value(self, discrete_state, w):
    return np.dot(discrete_state, w)

  # The gradient of a * x = x
  def approximate_value_gradient(self, discrete_state):
    return discrete_state