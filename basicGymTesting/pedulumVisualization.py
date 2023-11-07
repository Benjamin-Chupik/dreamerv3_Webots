import gym
import numpy as np
import dreamerv3
from dreamerv3 import embodied
from embodied.envs import from_gym


# Configuration
logdir = embodied.Path('~/logdir/pendulum')
print("1")
config = embodied.Config.load(logdir / 'config.yaml')
print("2")
# Making env
env = gym.make("Pendulum-v1")
env = from_gym.FromGym(env, obs_key='vector')  # I found I had to specify a different obs_key than the default of 'image'
env = dreamerv3.wrap_env(env, config)


print("Trying to import agent from checkpoint")
# Trying to import agent from checkpoint
step = embodied.Counter()
agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
checkpoint = embodied.Checkpoint()
checkpoint.agent = agent
checkpoint.load('/Users/guidoinsinger/Documents/GitHub/dreamerv3_webots/logdir/pendulum/checkpoint.ckpt', keys=['agent'])
print("loaded agent")
state = None

act = {'action': env.act_space['action'].sample(), 'reset': np.array(True)}
N = 1000

print("Starting movement")
for step in np.arange(N):
    obs = env.step(act)
    print("got step",obs)
    act, state = agent.policy(obs, state, mode='eval')
    print("got actions")
    act = {'action': act['action'][0], 'reset': obs['is_last'][0]}
    print(obs)

env.close()