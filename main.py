import time
import numpy as np

from env import Env, discrete_actions
from agent import Agent
from config import should_replay_checkpoint, checkpoint_path

# Environment
env = Env()
if should_replay_checkpoint:
  Agent.replay(env, checkpoint_path, 1, 60, True)

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
frame_count     = 0

while True:
  state = env.reset()
  episode_reward = 0

  start = time.time()
  for timestamp in range(1, max_steps_per_episode):
    frame_count += 1

    # Show env
    # env.render()

    # Run agent
    action = agent.run(state)

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

  if (episode_count % episode_log == 0):
    print(
      'Episode {e:6d} - '
      'Frame {f:7d} - '
      'Frames/sec {fs:7.2f} - '
      'Epsilon {eps:7.2f} - '
      'Mean Reward {r:7.2f}'.format(
        e=episode_count,
        f=frame_count,
        fs=frame_count / (time.time() - start),
        eps=agent.epsilon,
        r=running_reward
      )
    )

  # Update running reward to check condition for solving
  memory_episode_reward.append(episode_reward)

  if (len(memory_episode_reward) > 100):
    del memory_episode_reward[:1]

  running_reward = np.mean(memory_episode_reward)
  episode_count += 1

  if done and info['lives'] == 2:
    print("Solved at episode {}!".format(episode_count))

    agent.save()
    print("Agent saved with success!")

    exit()