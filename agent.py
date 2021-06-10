import numpy as np
import random
import time
import os

import tensorflow as tf
from model import build_model
from tensorflow.keras import optimizers, losses, models

class Agent:
  def __init__ (self, state_shape, actions):
    # set model file name
    self.model_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    self.model_path = './models/{name}'.format(name=self.model_name)
    self.checkpoints_path = '{m}/checkpoints'.format(m=self.model_path)

    # create models folder to save later
    if not os.path.isdir('./models'):
      os.mkdir('./models')

    if not os.path.isdir(self.model_path):
      os.mkdir(self.model_path)

    if not os.path.isdir(self.checkpoints_path):
      os.mkdir(self.checkpoints_path)

    self.state_shape = state_shape
    self.actions = actions
    self.actions_n = len(self.actions)

    self.gamma = 0.9
    self.epsilon = 1.0
    self.epsilon_min = 0.1
    self.epsilon_max = 1.0
    self.epsilon_interval = (self.epsilon_max - self.epsilon_min)
    self.epsilon_random_frames = 50000
    self.epsilon_greedy_frames = 100000


    self.batch_size = 32

    self.model = build_model(self.state_shape, self.actions_n)
    self.model_target = build_model(self.state_shape, self.actions_n)
    self.loss_function = losses.Huber()

    self.learn_after_actions = 4
    self.update_target_model = 10000
    self.max_memory_length = 100000
    self.checkpoint_each_episode = 100

    self.memory_action         = []
    self.memory_state          = []
    self.memory_state_next     = []
    self.memory_rewards        = []
    self.memory_done           = []

  @staticmethod
  def replay (env, path, checkpooint=False, amount=1, fps=180):
    state = env.reset()

    if checkpooint:
      model = build_model(np.shape(state), len(env.actions))
      model.load_weights(path)
    else:
      model = models.load_model(path)

    for _ in range(amount):
      while True:
        # Simulate FPS
        time.sleep(1/fps)

        # Render Env
        env.render()

        # Get best action for current state
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()

        # Perform action
        next_state, _, done, _ = env.step(action)
        
        # Update Env state
        state = next_state

        if done:
          break

    while True:
      continue

  def add (self, experience):
    self.memory_action.append(experience[0])
    self.memory_state.append(experience[1])
    self.memory_state_next.append(experience[2])
    self.memory_rewards.append(experience[3])
    self.memory_done.append(experience[4])

    if (len(self.memory_rewards) > self.max_memory_length):
      del self.memory_rewards[:1]
      del self.memory_state[:1]
      del self.memory_state_next[:1]
      del self.memory_action[:1]
      del self.memory_done[:1]

  def run (self, state, frame_count):
    if (frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]):
      action = random.randint(0, self.actions_n - 1)      
    else:
      state_tensor = tf.convert_to_tensor(state)
      state_tensor = tf.expand_dims(state_tensor, 0)
      action_probs = self.model(state_tensor, training=False)
      action = tf.argmax(action_probs[0]).numpy()

    self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
    self.epsilon = max(self.epsilon, self.epsilon_min)

    return action

  def learn (self, frame_count,):
    optimizer = optimizers.Adam(learning_rate=0.00001, clipnorm=1.0)

    if (frame_count % self.learn_after_actions == 0 and len(self.memory_done) > self.batch_size):
      # Get indices of samples for replay buffers
      indices = np.random.choice(range(len(self.memory_done)), size=self.batch_size)

      # Using list comprehension to sample from replay buffer
      state_sample = np.array([
        self.memory_state[i] for i in indices
      ])

      state_next_sample = np.array([
        self.memory_state_next[i] for i in indices
      ])

      rewards_sample = np.array([
        self.memory_rewards[i] for i in indices
      ])

      action_sample = np.array([
        self.memory_action[i] for i in indices
      ])

      done_sample = tf.convert_to_tensor([
        float(self.memory_done[i]) for i in indices
      ])

      # Build the updated Q-values for the sampled future states
      # Use the target model for stability
      future_rewards = self.model_target.predict(state_next_sample)

      # Q value = reward + discount factor * expected future reward
      updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)

      # If final frame set the last value to -1
      updated_q_values = updated_q_values * (1 - done_sample) - done_sample

      # Create a mask so we only calculate loss on the updated Q-values
      masks = tf.one_hot(action_sample, self.actions_n)

      with tf.GradientTape() as tape:
        # Train the model on the states and updated Q-values
        q_values = self.model(state_sample)

        # Apply the masks to the Q-values to get the Q-value for action taken
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

        # Calculate loss between new Q-value and old Q-value
        loss = self.loss_function(updated_q_values, q_action)

      # Backpropagation
      grads = tape.gradient(loss, self.model.trainable_variables)
      optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    if (frame_count % self.update_target_model == 0):
      # update the the target network with new weights
      self.model_target.set_weights(self.model.get_weights())

  def save (self):
    self.model.save(self.model_name + '.model')

  def checkpoint (self, episode_count):
    if (episode_count % self.checkpoint_each_episode == 0):
      self.model.save_weights('{p}/{e}.check'.format(p=self.checkpoints_path, e=episode_count))