import random
import retro
import numpy
import cv2

from utils import to_bin, to_int
discrete_actions = [x for x in range(0, 8)]

class Env():
  def __init__ (self):
    self.env = retro.make(game='SpaceInvaders-Nes')

  def reset (self):
    return self.process_frame(self.env.reset())

  def filter_action (self, action):
    return to_int([action[0], action[6], action[7]])

  def expand_action (self, discrete_action):
    # 0 = tiro
    # 6 = esquerda
    # 7 = direita

    binary = to_bin(discrete_action)
    return [binary[0], 0, 0, 0, 0, 0, binary[1], binary[2], 0, 0]

  def step (self, discrete_action):
    next_state, reward, done, info = self.env.step(self.expand_action(discrete_action))
    return self.process_frame(next_state), reward, done, info

  def random (self):
    return discrete_actions[random.randint(0, 7)]
  
  def render (self):
    return self.env.render()

  def pygame_state (self, state):
    return numpy.flip(numpy.rot90(state), 0)

  def process_frame (self, state):
    frame = state.astype(numpy.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    frame = frame[20:len(frame)-13, 0:len(frame[0])-50]

    frame = cv2.resize(frame, (85, 85), interpolation=cv2.INTER_NEAREST)
    frame = numpy.reshape(frame, (85, 85, 1))

    return numpy.array(frame)