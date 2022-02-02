import cuda_bindings
from actor_critic import ActorCritic
import gym
import time
import matplotlib.pyplot as plt

PARALLEL = 10

# action can be 0, 1, or 2.
# if action is 0, then accelerate to the left
# if action is 1, then don't accelerate
# if action is 2, then accelerate to the right

# reward = 0 if agent reached flag
# reward = -1 if agent did not reach flag

# State is a tuple of two numbers
# The first number is the car position in the range [-1.2, 0.6]
# The second number is the car velocity in the range [-0.7, 0.7]

# Episode terminates when the car reaches the flag
# or the car takes more than 200 actions and still fails

reward_plot = []

# Setup parallel envs
envs = []
for _ in range(PARALLEL):
  envs.append(gym.make('MountainCar-v0'))

# Run the environment
cuda_bindings.init()
# model = ActorCritic()

# Train for 1000 episodes
for i_episode in range(1000):
  states_vector = []
  actions_vector = []
  rewards_vector = []
  next_states_vector = []

  for i_env, env in enumerate(envs):
    states = [];
    actions = [];
    rewards = [];
    next_states = [];

    curr_state = env.reset()
    done = False

    # Run each episode for 200 time steps
    is_done = False
    for _ in range(200):
      if not is_done:
          # Take an action
          action = cuda_bindings.act(curr_state)
          # action = model.act(i_env, curr_state)
          next_state, reward, done, _ = env.step(action)

          states.append(curr_state)
          actions.append(action)
          rewards.append(reward)
          next_states.append(next_state)

          # Update state
          curr_state = next_state

          if done:
            is_done = True

      else:
          states.append((0,0))
          actions.append(0)
          rewards.append(0)
          next_states.append((0,0))

    states_vector.append(states)
    actions_vector.append(actions)
    rewards_vector.append(rewards)
    next_states_vector.append(next_states)
  
  # Update model once per episode
  start = time.perf_counter_ns()
  cuda_bindings.update(states_vector, actions_vector, rewards_vector, next_states_vector)
  # model.update(states_vector, actions_vector, rewards_vector, next_states_vector)
  end = time.perf_counter_ns()
  # reward_plot.append(average_reward)
  print("Ep Complete: " + str(i_episode + 1) + " / 1000")
  print("Training Time in ns: " + str(end - start))

# Demo only, remove this loop later so that was can cleanup and free all memory
while True:
  curr_state = env.reset()
  done = False
  while not done:
    env.render()
    action = cuda_bindings.act(curr_state)
    curr_state, _, done, _ = env.step(action)

# Cleanup and free all memory
for env in envs:
  env.close()

cuda_bindings.free()