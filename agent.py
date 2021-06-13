import tensorflow as tf
import numpy as np

import random
import time
import os

from model import build_model
from tensorflow.keras import optimizers, losses, models

class Agent:
  def __init__ (self, state_shape, actions):
    # Seta os caminhos para salvar os checkpoints e o model
    self.model_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    self.model_path = './models/{name}'.format(name=self.model_name)
    self.checkpoints_path = '{m}/checkpoints'.format(m=self.model_path)

    # Cria as pastas de models e checkpoints
    if not os.path.isdir('./models'):
      os.mkdir('./models')

    if not os.path.isdir(self.model_path):
      os.mkdir(self.model_path)

    if not os.path.isdir(self.checkpoints_path):
      os.mkdir(self.checkpoints_path)

    # Atributos para o modelo
    self.state_shape = state_shape
    self.actions = actions
    self.actions_n = len(self.actions)

    # Hiperparametros
    self.gamma = 0.9
    self.epsilon = 1.0
    self.epsilon_min = 0.3
    self.epsilon_max = 1.0
    self.epsilon_interval = (self.epsilon_max - self.epsilon_min)
    self.epsilon_random_frames = 10000
    self.epsilon_greedy_frames = 10000

    # Builda os modelos
    self.model = build_model(self.state_shape, self.actions_n)
    self.model_target = build_model(self.state_shape, self.actions_n)
    self.loss_function = losses.Huber()

    # Parâmetros para controle de iterçações
    self.learn_after_actions = 4
    self.update_target_model = 10000
    self.max_memory_length = 100000
    self.checkpoint_each_episode = 100

    # Memórias
    self.batch_size = 32
    self.memory_action         = []
    self.memory_state          = []
    self.memory_state_next     = []
    self.memory_rewards        = []
    self.memory_done           = []

  # Método estático para fazer o replay de um modelo ou checkpoint
  @staticmethod
  def replay (env, path, checkpooint=False, amount=1, fps=120):
    if path == '':
      print("Para fazer o replay de um modelo ou checkpoint, passe o caminho dele!")
      exit()

    state = env.reset()

    # No caso de um checkpoint, cria um modelo e carrega os pesos
    if checkpooint:
      model = build_model(np.shape(state), len(env.actions))
      model.load_weights(path)
    # No caso do model, carrega ele todo
    else:
      model = models.load_model(path)

    # Quantidade de replays do modelo/checkpoint
    for _ in range(amount):
      while True:
        # Simulate o FPS no jogo
        time.sleep(1/fps)

        env.render()

        # Pega a melhor ação para o estado atual
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()

        next_state, _, done, _ = env.step(action)
        state = next_state

        if done:
          break

    while True:
      continue

  # Método para incrementar a memória do agente
  def add (self, experience):
    self.memory_action.append(experience[0])
    self.memory_state.append(experience[1])
    self.memory_state_next.append(experience[2])
    self.memory_rewards.append(experience[3])
    self.memory_done.append(experience[4])

    # Limita a memmória para um tamanho específico
    if (len(self.memory_rewards) > self.max_memory_length):
      del self.memory_rewards[:1]
      del self.memory_state[:1]
      del self.memory_state_next[:1]
      del self.memory_action[:1]
      del self.memory_done[:1]

  # Método para pegar a melhor ação para um estado ou uma ação aleatória
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

  # Método para treinar o modelo
  def learn (self, frame_count, episode_count):
    optimizer = optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    if (frame_count % self.learn_after_actions == 0 and len(self.memory_done) > self.batch_size):
      # Pega uma parte dos dados salvos na memória
      indices = np.random.choice(range(len(self.memory_done)), size=self.batch_size)

      state_sample = tf.convert_to_tensor([
        self.memory_state[i] for i in indices
      ])

      state_next_sample = tf.convert_to_tensor([
        self.memory_state_next[i] for i in indices
      ])

      rewards_sample = tf.convert_to_tensor([
        self.memory_rewards[i] for i in indices
      ])

      action_sample = tf.convert_to_tensor([
        self.memory_action[i] for i in indices
      ])

      done_sample = tf.convert_to_tensor([
        float(self.memory_done[i]) for i in indices
      ])

      # Faz o cálculo do q learning
      future_rewards = self.model_target.predict(state_next_sample)
      updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)
      updated_q_values = updated_q_values * (1 - done_sample) - done_sample

      # Propaga o gradiente para o modelo
      masks = tf.one_hot(action_sample, self.actions_n)
      with tf.GradientTape() as tape:
        q_values = self.model(state_sample)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = self.loss_function(updated_q_values, q_action)

      grads = tape.gradient(loss, self.model.trainable_variables)
      optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    # Quando for o caso, atualiza os pesos da rede alvo
    if (frame_count % self.update_target_model == 0):
      self.model_target.set_weights(self.model.get_weights())

    # Quando for o caso, salva os pesos do modelo em um checkpoint
    if (episode_count % self.checkpoint_each_episode == 0):
      self.checkpoint(episode_count)

  # Método para salvar o modelo
  def save (self):
    self.model.save('{p}/agent.model'.format(p=self.model_path))

  # Método para salvar um checkpoint do modelo
  def checkpoint (self, episode_count):
    self.model.save_weights('{p}/{e}.check'.format(p=self.checkpoints_path, e=episode_count))