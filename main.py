import time
import numpy as np

from env import Env, discrete_actions
from agent import Agent
from config import should_replay_checkpoint, checkpoint_path

# Environment
env = Env()
if should_replay_checkpoint:
  Agent.replay(env, checkpoint_path, True)

  exit()

# Agent
agent = Agent(state_shape=np.shape(env.reset()), actions=discrete_actions)
# agent.replay(env, './models', 1, False)

# configs
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

    # Show env
    env.render()

    # Run agent
    action = agent.run(state, frame_count)

    # Perform action
    next_state, reward, done, info = env.step(action)
    done = done or info['lives'] < 2

    # Update episode requerd
    episode_reward += reward

    # Add experience to agent
    agent.add([action, state, next_state, reward, done])

    # Learn with new experience
    agent.learn(frame_count)

    # Update state
    state = next_state

    # If done break loop
    if done:
      break

  # Update running reward to check condition for solving
  memory_episode_reward.append(episode_reward)

  if (len(memory_episode_reward) > 100):
    del memory_episode_reward[:1]

  running_reward = np.mean(memory_episode_reward)
  episode_count += 1

  if (episode_count % episode_log == 0):
    print(
      '\n'
      'Episode {e}\n'
      'Frame       {f:7d}\n'
      'Frames/sec  {fs:9.2f}\n'
      'Epsilon     {eps:9.2f}\n'
      'Mean Reward {r:9.2f}\n'.format(
        e=episode_count,
        f=frame_count,
        fs=episode_frame_count / (time.time() - start),
        eps=agent.epsilon,
        r=running_reward
      )
    )

  agent.checkpoint(episode_count)

  if done and info['lives'] == 2:
    print("Solved at episode {}!".format(episode_count))

    agent.save()
    print("Agent saved with success!")

    exit()