import numpy as np
import random
import time

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses

class Agent:
  def __init__ (self, state_shape, actions):
    self.state_shape = state_shape
    self.actions = actions
    self.actions_n = len(self.actions)

    self.gamma = 0.90
    self.epsilon = 1
    self.epsilon_min = 0.1
    self.epsilon_decay = 0.9990

    self.batch_size = 32

    self.model = self.build_model()
    self.model_target = self.build_model()
    self.loss_function = losses.Huber()

    self.update_after_actions = 4
    self.update_target_model = 10000
    self.max_memory_length = 100000

    self.memory_action         = []
    self.memory_state          = []
    self.memory_state_next     = []
    self.memory_rewards        = []
    self.memory_done           = []

  def build_model (self):
    inputs = layers.Input(self.state_shape)

    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(self.actions_n, activation="linear")(layer5)

    return Model(inputs=inputs, outputs=action)

  def add (self, experience):
    self.memory_action.append(experience[0])
    self.memory_state.append(experience[1])
    self.memory_state_next.append(experience[2])
    self.memory_rewards.append(experience[3])
    self.memory_done.append(experience[4])

  def run (self, state):
    if np.random.rand() < self.epsilon:
      action = random.randint(0, self.actions_n - 1)
    else:
      state_tensor = tf.convert_to_tensor(state)
      state_tensor = tf.expand_dims(state_tensor, 0)
      action_probs = self.model(state_tensor, training=False)
      action = tf.argmax(action_probs[0]).numpy()

    self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon, self.epsilon_min)

    return action

  def learn (self, frame_count):
    optimizer = optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    if (frame_count % self.update_after_actions == 0 and len(self.memory_done) > self.batch_size):
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

      # if (frame_count % self.update_target_model == 0):
      #   # update the the target network with new weights
      #   self.model_target.set_weights(self.model.get_weights())

      # # Limit the state and reward history
      # if (len(self.memory_rewards) > self.max_memory_length):
      #   del self.memory_rewards[:1]
      #   del self.memory_state[:1]
      #   del self.memory_state_next[:1]
      #   del self.memory_action[:1]
      #   del self.memory_done[:1]

  # def replay(self, env, model_path, n_replay, plot):
  #   ckpt = tf.train.latest_checkpoint(model_path)
  #   saver = tf.train.import_meta_graph(ckpt + '.meta')
  #   graph = tf.get_default_graph()

  #   input = graph.get_tensor_by_name('input:0')
  #   output = graph.get_tensor_by_name('online/output/BiasAdd:0')

  #   # Replay RL agent
  #   state = env.process_frame(env.reset())
  #   total_reward = 0
  #   with tf.Session() as sess:
  #     saver.restore(sess, ckpt)
  #     for _ in range(n_replay):
  #       step = 0
  #       while True:
  #         time.sleep(0.01)
  #         env.render()
  #         # Plot
  #         if plot:
  #           if step % 100 == 0:
  #             self.visualize_layer(session=sess, layer=self.conv_2, state=state, step=step)
  #         # Action
  #         if np.random.rand() <= 0:
  #           action = self.actions.sample()
  #         else:
  #           q = sess.run(fetches=output, feed_dict={input: np.expand_dims(state, 0)})
  #           action = np.argmax(q)

  #         next_state, reward, done, info = env.step(action)
  #         total_reward += reward
  #         state = env.process_frame(next_state)
  #         step += 1

  #         if done:
  #           break

  #   env.close()