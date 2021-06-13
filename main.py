import time
import numpy as np

from agent import Agent
from env import Env, discrete_actions

from config import should_replay_checkpoint, should_replay_model, should_render, path

# Builda o ambiente
env = Env()

# com a flag '-c' faz o replay de um checkpoint
if should_replay_checkpoint:
  Agent.replay(env, path, True)

# com a flag '-m' faz o replay de um model
if should_replay_model:
  Agent.replay(env, path, False)

agent = Agent(state_shape=np.shape(env.reset()), actions=discrete_actions)

max_steps_per_episode = 10000
memory_episode_reward = []

frame_count     = 0
running_reward  = 0
episode_count   = 0
episode_log     = 1

while True:
  state = env.reset()

  episode_reward = 0
  episode_frame_count = 0

  start = time.time()

  for timestamp in range(1, max_steps_per_episode):
    frame_count += 1
    episode_frame_count += 1

    # Renderiza o treinamento quando passamos a flag '-r'
    if should_render:
      env.render()

    action = agent.run(state, frame_count)
    next_state, reward, done, info = env.step(action)

    # Também encerramos um episódio caso o agente perca uma vida
    done = done or info['lives'] < 2

    episode_reward += reward

    agent.add([action, state, next_state, reward, done])
    agent.learn(frame_count, episode_count)

    state = next_state

    if done:
      break

  memory_episode_reward.append(episode_reward)

  if (len(memory_episode_reward) > agent.max_memory_length):
    del memory_episode_reward[:1]

  running_reward = np.mean(memory_episode_reward)
  episode_count += 1

  if (episode_count % episode_log == 0):
    print(
      'Episode {e:5} - '
      'Frame       {f:7d} - '
      'Frames/sec  {fs:9.2f} - '
      'Epsilon     {eps:9.2f} - '
      'Mean Reward {r:9.2f}'.format(
        e=episode_count,
        f=frame_count,
        fs=episode_frame_count / (time.time() - start),
        eps=agent.epsilon,
        r=running_reward)
      )

  # Só acaba o treinamento quando o agente termina a fase sem morrer
  if done and info['lives'] == 2:
    print("Solved at episode {}!".format(episode_count))

    agent.save()
    print("Agent saved with success!")

    exit()