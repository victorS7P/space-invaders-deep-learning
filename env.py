import random
import retro
import numpy
import cv2

from utils import to_bin, to_int

# Cria uma representação discreta de algumas ações possíveis
# Aqui, deixamos de usar as 2^9 ações do ambiente para apenas 2^3 ações
discrete_actions = [x for x in range(0, 8)]

class Env():
  def __init__ (self):
    self.env = retro.make(game='SpaceInvaders-Nes')
    self.actions = discrete_actions

  # Reseta o ambiente
  def reset (self):
    return self.process_frame(self.env.reset())

  # Converte uma ação do ambiente para nossa representação
  def filter_action (self, action):
    return to_int([action[0], action[6], action[7]])

  # Converte nossa representação para a forma como o ambiente aceita ela (array binário)
  def expand_action (self, discrete_action):
    # 0 = tiro
    # 6 = esquerda
    # 7 = direita

    binary = to_bin(discrete_action)
    return [binary[0], 0, 0, 0, 0, 0, binary[1], binary[2], 0, 0]

  # Performa uma ação no ambiente
  def step (self, discrete_action):
    next_state, reward, done, info = self.env.step(self.expand_action(discrete_action))
    return self.process_frame(next_state), reward, done, info

  # Retorna uma ação aleatória do ambiente
  def random (self):
    return discrete_actions[random.randint(0, 7)]
  
  # Renderiza o ambiente
  def render (self):
    return self.env.render()

  # Faz o processamento do estado do ambiente, para salvar processamento
  def process_frame (self, state):
    frame = state.astype(numpy.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    frame = frame[20:len(frame)-13, 0:len(frame[0])-50]

    frame = cv2.resize(frame, (80, 80), interpolation=cv2.INTER_NEAREST)
    frame = numpy.reshape(frame, (80, 80, 1))

    return numpy.array(frame / 255)