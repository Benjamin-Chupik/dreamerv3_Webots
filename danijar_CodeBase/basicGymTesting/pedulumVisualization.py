import gym
import numpy as np
import dreamerv3
from dreamerv3 import embodied
from embodied.envs import from_gym


# Configuration
logdir = embodied.Path('/logdir/pendulum')

config = embodied.Config(dreamerv3.configs["defaults"])
config = config.update(dreamerv3.configs["medium"])
config = config.update(
    {
        "logdir": f"logdir/pendulum",  # this was just changed to generate a new log dir every time for testing
        "run.train_ratio": 64,
        "run.log_every": 30,
        "batch_size": 16,
        "jax.prealloc": False,
        "encoder.mlp_keys": ".*",
        "decoder.mlp_keys": ".*",
        "encoder.cnn_keys": "$^",
        "decoder.cnn_keys": "$^",
        "jax.platform": "cpu",  # I don't have a gpu locally
    }
)
config = embodied.Flags(config).parse()
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
    obs = {k: v[None] for k, v in obs.items()}
    act, state = agent.policy(obs, state, mode='eval')
    act = {'action': act['action'][0], 'reset': obs['is_last'][0]}
env.close()